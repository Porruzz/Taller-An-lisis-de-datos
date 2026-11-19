"""
Construcción de ejemplos supervisados por corrida y partición estricta por
orden de adquisición (entrenamiento: c01-c03, validación: c04, prueba: c05-c06).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..config import (
    COLUMNAS_PREDICTORAS,
    CORRIDAS_ENTRENAMIENTO,
    CORRIDAS_PRUEBA,
    CORRIDAS_VALIDACION,
)


def construir_dataset_supervisado(
    df_ventanas: pd.DataFrame,
    ruta_salida_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Al cierre de la ventana k, construye el ejemplo supervisado:
      - Entradas (exclusivamente de la traza): r_k, r_{k-1}, r_{k-2}, n_k
      - Objetivo (de la cola en la ventana siguiente): y_k = d_{k+1}

    Los rezagos y objetivos se generan estrictamente por corrida y por segmento
    contiguo válido (`segmento_id >= 0`), descartando extremos sin historia o
    sin etiqueta y evitando cruzar cualquier tramo anómalo.
    """
    validas = df_ventanas[df_ventanas["valida"] & (df_ventanas["segmento_id"] >= 0)].copy()

    bloques = []
    for (run_id, seg_id), grp in validas.groupby(["run_id", "segmento_id"], sort=False):
        g = grp.sort_values("ventana_idx").reset_index(drop=True)
        if len(g) < 4:
            # Se necesitan al menos ventanas k-2, k-1, k y k+1
            continue

        g["r_k_1"] = g["r_k"].shift(1)
        g["r_k_2"] = g["r_k"].shift(2)
        g["y_k"] = g["d_k"].shift(-1)

        # Metadatos de la ventana pronosticada (k+1) para ubicación exacta en gráficas y análisis
        g["ventana_idx_k1"] = g["ventana_idx"].shift(-1)
        g["t_inicio_k1"] = g["t_inicio"].shift(-1)
        g["t_fin_k1"] = g["t_fin"].shift(-1)
        g["t_rel_inicio_k1_s"] = g["t_rel_inicio_s"].shift(-1)
        g["t_rel_fin_k1_s"] = g["t_rel_fin_s"].shift(-1)
        g["delta_k1_s"] = g["delta_s"].shift(-1)
        g["r_k1_real"] = g["r_k"].shift(-1)
        g["fase_k1_interpretacion"] = g["fase_interpretacion"].shift(-1)
        g["tipo_transicion_k1"] = g["tipo_transicion"].shift(-1)
        g["tasa_ofrecida_k1_mbps"] = g["tasa_ofrecida_mbps"].shift(-1)

        # Descartar filas de los extremos sin historia (k-1, k-2) o sin etiqueta (k+1)
        g = g.dropna(subset=COLUMNAS_PREDICTORAS + ["y_k"]).copy()
        if g.empty:
            continue

        g["ventana_idx_k1"] = g["ventana_idx_k1"].astype(int)
        g["d_k"] = g["d_k"].astype(int)
        g["y_k"] = g["y_k"].astype(int)
        g["n_k"] = g["n_k"].astype(int)

        # Etiquetas binarias de evento y distinción inicio vs persistencia
        g["evento_congestion_k1"] = (g["y_k"] >= 1).astype(int)
        g["es_inicio_observado"] = ((g["d_k"] == 0) & (g["y_k"] > 0)).astype(int)
        g["es_persistencia"] = ((g["d_k"] > 0) & (g["y_k"] > 0)).astype(int)

        bloques.append(g)

    if not bloques:
        raise ValueError("No se pudieron construir ejemplos supervisados válidos.")

    df_sup = pd.concat(bloques, ignore_index=True)

    # Etiquetar partición respetando corridas completas y orden de adquisición
    def _asignar_split(rid: str) -> str:
        if rid in CORRIDAS_ENTRENAMIENTO:
            return "entrenamiento"
        if rid in CORRIDAS_VALIDACION:
            return "validacion"
        if rid in CORRIDAS_PRUEBA:
            return "prueba"
        return "otro"

    df_sup["particion"] = df_sup["run_id"].map(_asignar_split)

    columnas_ordenadas = [
        "run_id",
        "particion",
        "segmento_id",
        "ventana_idx",
        "ventana_idx_k1",
        "t_inicio",
        "t_fin",
        "t_rel_inicio_s",
        "t_rel_fin_s",
        "delta_s",
        "t_inicio_k1",
        "t_fin_k1",
        "t_rel_inicio_k1_s",
        "t_rel_fin_k1_s",
        "delta_k1_s",
        "r_k",
        "r_k_1",
        "r_k_2",
        "n_k",
        "d_k",
        "y_k",
        "evento_congestion_k1",
        "es_inicio_observado",
        "es_persistencia",
        "r_k1_real",
        "fase_interpretacion",
        "fase_k1_interpretacion",
        "tipo_transicion",
        "tipo_transicion_k1",
        "tasa_ofrecida_mbps",
        "tasa_ofrecida_k1_mbps",
    ]
    df_sup = df_sup[[c for c in columnas_ordenadas if c in df_sup.columns]]

    if ruta_salida_csv is not None:
        ruta_salida_csv = Path(ruta_salida_csv)
        ruta_salida_csv.parent.mkdir(parents=True, exist_ok=True)
        df_sup.to_csv(ruta_salida_csv, index=False)

    return df_sup


def dividir_particiones(
    df_supervisado: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna (df_train, df_val, df_test) manteniendo corridas completas y
    estricto orden cronológico de adquisición.
    """
    df_train = df_supervisado[df_supervisado["run_id"].isin(CORRIDAS_ENTRENAMIENTO)].copy().reset_index(drop=True)
    df_val = df_supervisado[df_supervisado["run_id"].isin(CORRIDAS_VALIDACION)].copy().reset_index(drop=True)
    df_test = df_supervisado[df_supervisado["run_id"].isin(CORRIDAS_PRUEBA)].copy().reset_index(drop=True)

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError(
            f"Partición vacía detectada: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
        )

    return df_train, df_val, df_test


def resumen_particiones(df_supervisado: pd.DataFrame) -> pd.DataFrame:
    """Genera un resumen tabular por corrida y partición."""
    filas = []
    for run_id, grp in df_supervisado.groupby("run_id", sort=False):
        filas.append(
            {
                "run_id": run_id,
                "particion": str(grp["particion"].iloc[0]),
                "ejemplos": int(len(grp)),
                "ventanas_congestion": int(grp["evento_congestion_k1"].sum()),
                "pct_congestion": float(100.0 * grp["evento_congestion_k1"].mean()),
                "inicios_observados": int(grp["es_inicio_observado"].sum()),
                "ventanas_persistencia": int(grp["es_persistencia"].sum()),
                "descartes_totales": int(grp["y_k"].sum()),
                "r_k_promedio_mbps": float(grp["r_k"].mean()),
                "r_k_max_mbps": float(grp["r_k"].max()),
            }
        )
    return pd.DataFrame(filas)
