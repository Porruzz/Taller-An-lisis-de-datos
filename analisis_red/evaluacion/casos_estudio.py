"""
Extracción detallada de casos concretos de interés en prueba:
  - catálogo completo de inicios observados;
  - análisis contextual de falsas alarmas (d_{k+1} = 0 y alerta = 1);
  - análisis contextual de inicios omitidos (d_k = 0, d_{k+1} > 0 y alerta = 0).
"""

from __future__ import annotations

from typing import List

import pandas as pd


COLUMNAS_CONTEXTO = [
    "run_id",
    "ventana_idx",
    "ventana_idx_k1",
    "t_rel_fin_s",
    "t_rel_fin_k1_s",
    "r_k_2",
    "r_k_1",
    "r_k",
    "n_k",
    "d_k",
    "y_k",
    "y_hat_bruto",
    "y_hat_recortado",
    "alerta_regresion",
    "alerta_base",
    "fase_interpretacion",
    "fase_k1_interpretacion",
    "tipo_transicion_k1",
    "tasa_ofrecida_mbps",
    "tasa_ofrecida_k1_mbps",
]


def _filtrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUMNAS_CONTEXTO if c in df.columns]
    return df[cols].copy()


def extraer_tabla_inicios(df_eval: pd.DataFrame, col_alerta: str = "alerta_regresion") -> pd.DataFrame:
    """Devuelve todos los inicios observados (d_k = 0 y d_{k+1} > 0) con su diagnóstico."""
    inicios = df_eval[df_eval["es_inicio_observado"] == 1].copy()
    inicios["estado_inicio"] = inicios[col_alerta].map({1: "Anticipado", 0: "Omitido"})
    cols = [c for c in COLUMNAS_CONTEXTO if c in inicios.columns] + ["estado_inicio"]
    return inicios[cols].reset_index(drop=True)


def extraer_casos_falsa_alarma(
    df_eval: pd.DataFrame,
    col_alerta: str = "alerta_regresion",
    ventana_vecindad: int = 2,
    max_casos: int = 3,
) -> List[pd.DataFrame]:
    """
    Localiza las falsas alarmas más pronunciadas (mayor y_hat_recortado con
    y_k = 0) y retorna tablas con las ventanas vecinas para interpretar la causa.
    """
    fa = df_eval[(df_eval[col_alerta] == 1) & (df_eval["y_k"] == 0)].copy()
    if fa.empty:
        return []

    if "y_hat_recortado" in fa.columns:
        fa = fa.sort_values("y_hat_recortado", ascending=False)

    casos: List[pd.DataFrame] = []
    usados = set()

    for row in fa.itertuples(index=False):
        clave = (row.run_id, int(row.ventana_idx))
        # Evitar repetir ventanas inmediatamente adyacentes en el mismo evento
        if any(r == clave[0] and abs(v - clave[1]) <= ventana_vecindad for r, v in usados):
            continue
        usados.add(clave)

        sub = df_eval[
            (df_eval["run_id"] == row.run_id)
            & (df_eval["ventana_idx"] >= int(row.ventana_idx) - ventana_vecindad)
            & (df_eval["ventana_idx"] <= int(row.ventana_idx) + ventana_vecindad)
        ].copy()
        sub["es_evento_foco"] = sub["ventana_idx"] == int(row.ventana_idx)
        casos.append(_filtrar_columnas(sub).assign(es_evento_foco=sub["es_evento_foco"].values))
        if len(casos) >= max_casos:
            break

    return casos


def extraer_casos_inicio_omitido(
    df_eval: pd.DataFrame,
    col_alerta: str = "alerta_regresion",
    ventana_vecindad: int = 2,
    max_casos: int = 3,
) -> List[pd.DataFrame]:
    """
    Localiza inicios observados que el modelo no anunció (d_k = 0, d_{k+1} > 0,
    alerta = 0) y retorna su entorno temporal cercano para diagnosticar la omisión.
    """
    omi = df_eval[(df_eval["es_inicio_observado"] == 1) & (df_eval[col_alerta] == 0)].copy()
    if omi.empty:
        return []

    omi = omi.sort_values("y_k", ascending=False)
    casos: List[pd.DataFrame] = []

    for row in omi.itertuples(index=False):
        sub = df_eval[
            (df_eval["run_id"] == row.run_id)
            & (df_eval["ventana_idx"] >= int(row.ventana_idx) - ventana_vecindad)
            & (df_eval["ventana_idx"] <= int(row.ventana_idx) + ventana_vecindad)
        ].copy()
        sub["es_evento_foco"] = sub["ventana_idx"] == int(row.ventana_idx)
        casos.append(_filtrar_columnas(sub).assign(es_evento_foco=sub["es_evento_foco"].values))
        if len(casos) >= max_casos:
            break

    return casos
