"""
Punto de entrada reproducible del laboratorio de predicción de congestión LAN.
Ejecuta la canalización completa desde los archivos crudos (trazas PCAP/CSV,
registros de cola TBF y salidas JSON de iperf3) hasta la evaluación en prueba
y generación de gráficas.

Uso:
    python main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analisis_red.config import (
    COLUMNAS_PREDICTORAS,
    CORRIDAS_ENTRENAMIENTO,
    CORRIDAS_PRUEBA,
    CORRIDAS_TODAS,
    CORRIDAS_VALIDACION,
    DATOS_CRUDOS_DIR,
    DATOS_PROCESADOS_DIR,
    RESULTADOS_DIR,
    UMBRAL_ALERTA_ESTANDAR,
)
from analisis_red.evaluacion import (
    analizar_inicios_y_persistencia,
    analizar_rampas_vs_saltos,
    calcular_metricas_alertas,
    calcular_metricas_regresion,
    construir_tabla_comparativa,
    extraer_casos_falsa_alarma,
    extraer_casos_inicio_omitido,
    extraer_tabla_inicios,
)
from analisis_red.ingesta import leer_resumen_iperf_todas
from analisis_red.modelos import ModeloRegresionCongestion, ReglaUmbralTasa
from analisis_red.series import (
    construir_dataset_supervisado,
    construir_ventanas_todas,
    dividir_particiones,
)
from analisis_red.series.caracteristicas import resumen_particiones
from analisis_red.visualizacion import (
    graficar_calibracion_validacion,
    graficar_inicios_y_persistencia,
    graficar_matrices_confusion,
    graficar_saturacion_traza,
    graficar_serie_temporal_corrida,
)


def _sanear_json(obj: Any) -> Any:
    """Convierte tipos NumPy/Pandas a tipos nativos serializables en JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanear_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanear_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _sanear_json(obj.tolist())
    return obj


def ejecutar_pipeline(verbose: bool = True) -> Dict[str, Any]:
    """Ejecuta toda la canalización de datos, modelado, evaluación y gráficas."""
    if verbose:
        print("=" * 78)
        print("  LABORATORIO DE ANÁLISIS DE DATOS EN REDES")
        print("  Predicción de congestión en un enlace LAN (Captura PCAP + Regresión Lineal)")
        print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Comprobación complementaria con salidas JSON de iperf3 (emisor vs B)
    # ------------------------------------------------------------------
    df_iperf = leer_resumen_iperf_todas(DATOS_CRUDOS_DIR, list(CORRIDAS_TODAS))
    if not df_iperf.empty:
        df_iperf.to_csv(DATOS_PROCESADOS_DIR / "resumen_iperf3.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Alineación temporal de ventanas [t_{k-1}, t_k)
    # ------------------------------------------------------------------
    ruta_ventanas = DATOS_PROCESADOS_DIR / "ventanas.csv"
    df_ventanas = construir_ventanas_todas(
        directorio_crudos=DATOS_CRUDOS_DIR,
        corridas=CORRIDAS_TODAS,
        ruta_salida_csv=ruta_ventanas,
    )
    n_vent_total = int(len(df_ventanas))
    n_vent_invalidas = int((~df_ventanas["valida"]).sum())
    n_vent_vacias = int((df_ventanas["valida"] & (df_ventanas["n_k"] == 0)).sum())

    if verbose:
        print(f"\n[1/5] Ventanas construidas: {n_vent_total} total | "
              f"{n_vent_invalidas} marcadas anómalas | {n_vent_vacias} válidas vacías")
        print(f"      Guardado en: {ruta_ventanas}")

    # ------------------------------------------------------------------
    # 3. Construcción del dataset supervisado y división estricta por corrida
    # ------------------------------------------------------------------
    ruta_sup = DATOS_PROCESADOS_DIR / "dataset_supervisado.csv"
    df_sup = construir_dataset_supervisado(df_ventanas, ruta_salida_csv=ruta_sup)
    df_train, df_val, df_test = dividir_particiones(df_sup)
    tabla_particiones = resumen_particiones(df_sup)

    if verbose:
        print(f"\n[2/5] Ejemplos supervisados: {len(df_sup)} "
              f"(Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)})")
        print(tabla_particiones.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Entrenamiento de LinearRegression y calibración en validación (c04)
    # ------------------------------------------------------------------
    modelo = ModeloRegresionCongestion(
        columnas_predictoras=COLUMNAS_PREDICTORAS,
        umbral_alerta=UMBRAL_ALERTA_ESTANDAR,
    )
    ajuste = modelo.ajustar(df_train, col_objetivo="y_k")

    umbral_reg_opt, tabla_cal_reg = modelo.calibrar_umbral_validacion(
        df_val, col_objetivo="y_k", fijar_umbral=False
    )
    regla_base = ReglaUmbralTasa()
    tau_base_opt, tabla_cal_base = regla_base.calibrar_en_validacion(
        df_val, col_objetivo="y_k"
    )

    # Guardar predicciones en validación (c04)
    df_val_eval = df_val.copy()
    df_val_eval["y_hat_bruto"] = modelo.predecir_bruto(df_val)
    df_val_eval["y_hat_recortado"] = modelo.predecir_descartes(df_val)
    df_val_eval["alerta_regresion_std"] = modelo.emitir_alerta(df_val, umbral=UMBRAL_ALERTA_ESTANDAR)
    df_val_eval["alerta_regresion_opt"] = modelo.emitir_alerta(df_val, umbral=umbral_reg_opt)
    df_val_eval["alerta_regresion"] = df_val_eval["alerta_regresion_opt"]
    df_val_eval["alerta_base"] = regla_base.emitir_alerta(df_val)

    if verbose:
        print("\n[3/5] Modelo ajustado en entrenamiento (c01–c03):")
        print(f"      {modelo.ecuacion_texto()}")
        print(f"      Invariancia con StandardScaler (max |diff|): {ajuste.diferencia_max_escalado:.2e}")
        print(f"      R² train={ajuste.r2_entrenamiento:.4f} | MAE train={ajuste.mae_entrenamiento:.2f} pkt/ventana")
        print(f"      Umbral estándar: theta={UMBRAL_ALERTA_ESTANDAR:g} | "
              f"Umbral calibrado en validación (c04): theta*={umbral_reg_opt:.2f}")
        print(f"      Umbral regla base fijado en validación (c04): tau*={tau_base_opt:.2f} Mbit/s")

    # ------------------------------------------------------------------
    # 5. Evaluación en corridas de prueba reservadas (c05 y c06)
    # ------------------------------------------------------------------
    df_test_eval = df_test.copy()
    df_test_eval["y_hat_bruto"] = modelo.predecir_bruto(df_test)
    df_test_eval["y_hat_recortado"] = modelo.predecir_descartes(df_test)
    df_test_eval["alerta_regresion_std"] = modelo.emitir_alerta(df_test, umbral=UMBRAL_ALERTA_ESTANDAR)
    df_test_eval["alerta_regresion_opt"] = modelo.emitir_alerta(df_test, umbral=umbral_reg_opt)
    # Se usa el umbral calibrado en validación (fijado antes de prueba) como alerta principal
    # y también se reporta el desempeño con el umbral canónico theta = 1.0.
    df_test_eval["alerta_regresion"] = df_test_eval["alerta_regresion_opt"]
    df_test_eval["alerta_base"] = regla_base.emitir_alerta(df_test)

    df_test_eval.to_csv(DATOS_PROCESADOS_DIR / "predicciones_prueba.csv", index=False)

    # Métricas de regresión (global y por corrida de prueba)
    reg_val = calcular_metricas_regresion(
        df_val_eval["y_k"].to_numpy(),
        df_val_eval["y_hat_recortado"].to_numpy(),
        df_val_eval["y_hat_bruto"].to_numpy(),
    )
    reg_test = calcular_metricas_regresion(
        df_test_eval["y_k"].to_numpy(),
        df_test_eval["y_hat_recortado"].to_numpy(),
        df_test_eval["y_hat_bruto"].to_numpy(),
    )

    # Métricas de alerta en prueba y validación
    al_val_std = calcular_metricas_alertas(df_val_eval["y_k"].to_numpy(), df_val_eval["alerta_regresion_std"].to_numpy())
    al_val_opt = calcular_metricas_alertas(df_val_eval["y_k"].to_numpy(), df_val_eval["alerta_regresion_opt"].to_numpy())
    al_val_base = calcular_metricas_alertas(df_val_eval["y_k"].to_numpy(), df_val_eval["alerta_base"].to_numpy())

    al_test_std = calcular_metricas_alertas(df_test_eval["y_k"].to_numpy(), df_test_eval["alerta_regresion_std"].to_numpy())
    al_test_opt = calcular_metricas_alertas(df_test_eval["y_k"].to_numpy(), df_test_eval["alerta_regresion_opt"].to_numpy())
    al_test_base = calcular_metricas_alertas(df_test_eval["y_k"].to_numpy(), df_test_eval["alerta_base"].to_numpy())

    # Inicios vs persistencia en prueba
    ini_test_std = analizar_inicios_y_persistencia(df_test_eval, "alerta_regresion_std")
    ini_test_opt = analizar_inicios_y_persistencia(df_test_eval, "alerta_regresion_opt")
    ini_test_base = analizar_inicios_y_persistencia(df_test_eval, "alerta_base")

    # Rampas vs saltos bruscos
    rs_test_std = analizar_rampas_vs_saltos(df_test_eval, "alerta_regresion_std")
    rs_test_opt = analizar_rampas_vs_saltos(df_test_eval, "alerta_regresion_opt")
    rs_test_base = analizar_rampas_vs_saltos(df_test_eval, "alerta_base")

    # Evaluación desagregada por corrida de prueba (c05 y c06)
    metricas_por_corrida: Dict[str, Dict[str, Any]] = {}
    for rid in CORRIDAS_PRUEBA:
        sub = df_test_eval[df_test_eval["run_id"] == rid]
        metricas_por_corrida[rid] = {
            "regresion": calcular_metricas_regresion(
                sub["y_k"].to_numpy(),
                sub["y_hat_recortado"].to_numpy(),
                sub["y_hat_bruto"].to_numpy(),
            ),
            "alerta_std": calcular_metricas_alertas(sub["y_k"].to_numpy(), sub["alerta_regresion_std"].to_numpy()),
            "alerta_opt": calcular_metricas_alertas(sub["y_k"].to_numpy(), sub["alerta_regresion_opt"].to_numpy()),
            "alerta_base": calcular_metricas_alertas(sub["y_k"].to_numpy(), sub["alerta_base"].to_numpy()),
            "inicios_std": analizar_inicios_y_persistencia(sub, "alerta_regresion_std"),
            "inicios_opt": analizar_inicios_y_persistencia(sub, "alerta_regresion_opt"),
            "inicios_base": analizar_inicios_y_persistencia(sub, "alerta_base"),
        }

    # Casos de estudio concretos (inicios, falsas alarmas, inicios omitidos)
    tabla_inicios_test = extraer_tabla_inicios(df_test_eval, col_alerta="alerta_regresion_opt")
    tabla_inicios_test["estado_inicio_std"] = df_test_eval.loc[
        df_test_eval["es_inicio_observado"] == 1, "alerta_regresion_std"
    ].map({1: "Anticipado", 0: "Omitido"}).values
    tabla_inicios_test["estado_inicio_base"] = df_test_eval.loc[
        df_test_eval["es_inicio_observado"] == 1, "alerta_base"
    ].map({1: "Anticipado", 0: "Omitido"}).values

    casos_fa_opt = extraer_casos_falsa_alarma(df_test_eval, col_alerta="alerta_regresion_opt", max_casos=3)
    casos_fa_std = extraer_casos_falsa_alarma(df_test_eval, col_alerta="alerta_regresion_std", max_casos=3)
    casos_omi_opt = extraer_casos_inicio_omitido(df_test_eval, col_alerta="alerta_regresion_opt", max_casos=3)
    casos_omi_std = extraer_casos_inicio_omitido(df_test_eval, col_alerta="alerta_regresion_std", max_casos=3)

    tabla_inicios_test.to_csv(DATOS_PROCESADOS_DIR / "inicios_prueba.csv", index=False)

    tabla_comp = construir_tabla_comparativa(
        [
            {"Modelo": f"Regresión Lineal (theta={UMBRAL_ALERTA_ESTANDAR:g})", "Partición": "Prueba (c05-c06)", **al_test_std, **ini_test_std},
            {"Modelo": f"Regresión Lineal (theta*={umbral_reg_opt:.2f}, val.)", "Partición": "Prueba (c05-c06)", **al_test_opt, **ini_test_opt},
            {"Modelo": f"Regla Base (r_k >= {tau_base_opt:.2f} Mbit/s)", "Partición": "Prueba (c05-c06)", **al_test_base, **ini_test_base},
        ]
    )
    tabla_comp.to_csv(DATOS_PROCESADOS_DIR / "comparativa_prueba.csv", index=False)

    if verbose:
        print("\n[4/5] Resultados en Prueba (c05 + c06):")
        print(f"      MAE regresión (recortado >= 0): {reg_test['mae_paquetes']:.2f} paquetes/ventana "
              f"(RMSE={reg_test['rmse_paquetes']:.2f}, R²={reg_test['r2_recortado']:.4f})")
        cols_mostrar = [
            "Modelo", "VP", "FP", "FN", "VN", "precision", "sensibilidad", "f1",
            "inicios_anticipados", "inicios_omitidos", "frac_inicios_anticipados", "sensibilidad_persistencia",
        ]
        print(tabla_comp[cols_mostrar].to_string(index=False))
        print("\n      Catálogo de inicios observados en prueba:")
        print(
            tabla_inicios_test[
                ["run_id", "ventana_idx_k1", "tipo_transicion_k1", "r_k", "n_k", "y_k", "y_hat_recortado", "estado_inicio", "estado_inicio_std", "estado_inicio_base"]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 6. Generación de figuras reproducibles
    # ------------------------------------------------------------------
    figuras: Dict[str, str] = {}

    for rid, etiqueta_split in [("c04", "Validación"), ("c05", "Prueba 1"), ("c06", "Prueba 2")]:
        df_ev_src = df_val_eval if rid == "c04" else df_test_eval
        p_fig = graficar_serie_temporal_corrida(
            df_ventanas_corrida=df_ventanas[df_ventanas["run_id"] == rid],
            df_eval_corrida=df_ev_src[df_ev_src["run_id"] == rid],
            run_id=rid,
            tau_base_mbps=tau_base_opt,
            umbral_reg=umbral_reg_opt,
            ruta_salida=RESULTADOS_DIR / f"serie_temporal_{rid}.png",
            col_alerta_reg="alerta_regresion_opt",
            titulo_extra=etiqueta_split,
        )
        figuras[f"serie_{rid}"] = str(p_fig)

    figuras["calibracion_validacion"] = str(
        graficar_calibracion_validacion(
            tabla_reg=tabla_cal_reg,
            umbral_reg_std=UMBRAL_ALERTA_ESTANDAR,
            umbral_reg_opt=umbral_reg_opt,
            tabla_base=tabla_cal_base,
            tau_opt=tau_base_opt,
            ruta_salida=RESULTADOS_DIR / "calibracion_validacion.png",
        )
    )

    figuras["matrices_confusion"] = str(
        graficar_matrices_confusion(
            metricas_modelos=[
                (f"Regresión Lineal ($\\theta={UMBRAL_ALERTA_ESTANDAR:g}$)", al_test_std),
                (f"Regresión Lineal ($\\theta^*={umbral_reg_opt:.1f}$)", al_test_opt),
                (f"Regla Base ($r_k \\geq {tau_base_opt:.2f}$)", al_test_base),
            ],
            ruta_salida=RESULTADOS_DIR / "matrices_confusion_prueba.png",
        )
    )

    figuras["inicios_vs_persistencia"] = str(
        graficar_inicios_y_persistencia(
            resumen_std=ini_test_std,
            resumen_opt=ini_test_opt,
            resumen_base=ini_test_base,
            df_rs_std=rs_test_std,
            df_rs_opt=rs_test_opt,
            df_rs_base=rs_test_base,
            ruta_salida=RESULTADOS_DIR / "inicios_vs_persistencia.png",
        )
    )

    figuras["saturacion_traza"] = str(
        graficar_saturacion_traza(
            df_ventanas=df_ventanas,
            ruta_salida=RESULTADOS_DIR / "saturacion_traza_tbf.png",
        )
    )

    resumen = {
        "ventanas": {
            "total": n_vent_total,
            "invalidas": n_vent_invalidas,
            "validas_vacias": n_vent_vacias,
            "delta_mediana_s": float(df_ventanas.loc[df_ventanas["valida"], "delta_s"].median()),
            "delta_min_valida_s": float(df_ventanas.loc[df_ventanas["valida"], "delta_s"].min()),
            "delta_max_valida_s": float(df_ventanas.loc[df_ventanas["valida"], "delta_s"].max()),
        },
        "particiones": tabla_particiones.to_dict(orient="records"),
        "modelo_regresion": {
            "ecuacion": modelo.ecuacion_texto(),
            "intercepto": ajuste.intercepto,
            "coeficientes": ajuste.coeficientes,
            "coeficientes_estandarizados": ajuste.coeficientes_estandarizados,
            "diferencia_max_escalado": ajuste.diferencia_max_escalado,
            "r2_entrenamiento": ajuste.r2_entrenamiento,
            "mae_entrenamiento": ajuste.mae_entrenamiento,
            "umbral_estandar": UMBRAL_ALERTA_ESTANDAR,
            "umbral_calibrado_validacion": umbral_reg_opt,
        },
        "regla_base": {
            "tau_validacion_mbps": tau_base_opt,
        },
        "validacion_c04": {
            "regresion": reg_val,
            "alerta_std": al_val_std,
            "alerta_opt": al_val_opt,
            "alerta_base": al_val_base,
        },
        "prueba_c05_c06": {
            "regresion": reg_test,
            "alerta_std": al_test_std,
            "alerta_opt": al_test_opt,
            "alerta_base": al_test_base,
            "inicios_std": ini_test_std,
            "inicios_opt": ini_test_opt,
            "inicios_base": ini_test_base,
            "rampas_vs_saltos_std": rs_test_std.to_dict(orient="records"),
            "rampas_vs_saltos_opt": rs_test_opt.to_dict(orient="records"),
            "rampas_vs_saltos_base": rs_test_base.to_dict(orient="records"),
            "por_corrida": metricas_por_corrida,
        },
        "inicios_prueba": tabla_inicios_test.to_dict(orient="records"),
        "casos_falsa_alarma_opt": [c.to_dict(orient="records") for c in casos_fa_opt],
        "casos_falsa_alarma_std": [c.to_dict(orient="records") for c in casos_fa_std],
        "casos_inicio_omitido_opt": [c.to_dict(orient="records") for c in casos_omi_opt],
        "casos_inicio_omitido_std": [c.to_dict(orient="records") for c in casos_omi_std],
        "figuras": figuras,
    }

    ruta_resumen = DATOS_PROCESADOS_DIR / "resumen_resultados.json"
    ruta_resumen.write_text(json.dumps(_sanear_json(resumen), indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"\n[5/5] Figuras generadas en {RESULTADOS_DIR}:")
        for clave, ruta in figuras.items():
            print(f"      - {clave}: {Path(ruta).name}")
        print(f"      Resumen estructurado guardado en: {ruta_resumen}")
        print("\n" + "=" * 78)
        print("  CANALIZACIÓN COMPLETADA CON ÉXITO")
        print("=" * 78)

    return resumen


if __name__ == "__main__":
    ejecutar_pipeline(verbose=True)
