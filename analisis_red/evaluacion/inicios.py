"""
Evaluación diferenciada entre inicio de episodios de congestión (d_k = 0 y
d_{k+1} > 0) y persistencia de episodios ya en curso (d_k > 0 y d_{k+1} > 0),
así como comparación entre transiciones en rampa y saltos bruscos.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def analizar_inicios_y_persistencia(
    df_eval: pd.DataFrame,
    col_alerta: str,
) -> Dict[str, float]:
    """
    Cuantifica cuántos inicios observados (d_k = 0 y d_{k+1} > 0) fueron
    anunciados al cierre de k frente a cuántos se omitieron, y compara esa
    fracción con la sensibilidad sobre ventanas de persistencia (d_k > 0, d_{k+1} > 0).
    """
    alerta = df_eval[col_alerta].to_numpy(dtype=int)
    es_inicio = df_eval["es_inicio_observado"].to_numpy(dtype=int) == 1
    es_persist = df_eval["es_persistencia"].to_numpy(dtype=int) == 1

    inicios_tot = int(np.sum(es_inicio))
    inicios_ant = int(np.sum(es_inicio & (alerta == 1)))
    inicios_omi = int(np.sum(es_inicio & (alerta == 0)))
    frac_inicios = (inicios_ant / inicios_tot) if inicios_tot > 0 else float("nan")

    persist_tot = int(np.sum(es_persist))
    persist_det = int(np.sum(es_persist & (alerta == 1)))
    persist_omi = int(np.sum(es_persist & (alerta == 0)))
    frac_persist = (persist_det / persist_tot) if persist_tot > 0 else float("nan")

    return {
        "inicios_evaluables": inicios_tot,
        "inicios_anticipados": inicios_ant,
        "inicios_omitidos": inicios_omi,
        "frac_inicios_anticipados": float(frac_inicios),
        "persistencia_evaluable": persist_tot,
        "persistencia_detectada": persist_det,
        "persistencia_omitida": persist_omi,
        "sensibilidad_persistencia": float(frac_persist),
    }


def analizar_rampas_vs_saltos(
    df_eval: pd.DataFrame,
    col_alerta: str,
) -> pd.DataFrame:
    """
    Agrupa los inicios observados según el tipo de transición del perfil de
    carga en la ventana k+1 (rampa vs salto_brusco vs otro) usando la columna
    puramente interpretativa `tipo_transicion_k1`.
    """
    inicios = df_eval[df_eval["es_inicio_observado"] == 1].copy()
    if inicios.empty:
        return pd.DataFrame(
            columns=[
                "tipo_transicion_k1",
                "inicios_evaluables",
                "inicios_anticipados",
                "inicios_omitidos",
                "frac_anticipados",
                "r_k_media_previa_mbps",
                "y_hat_media",
            ]
        )

    filas = []
    for tipo, grp in inicios.groupby("tipo_transicion_k1", sort=True):
        n_tot = int(len(grp))
        n_ant = int((grp[col_alerta] == 1).sum())
        n_omi = n_tot - n_ant
        filas.append(
            {
                "tipo_transicion_k1": str(tipo),
                "inicios_evaluables": n_tot,
                "inicios_anticipados": n_ant,
                "inicios_omitidos": n_omi,
                "frac_anticipados": float(n_ant / n_tot) if n_tot > 0 else float("nan"),
                "r_k_media_previa_mbps": float(grp["r_k"].mean()),
                "y_hat_media": float(grp["y_hat_recortado"].mean()) if "y_hat_recortado" in grp.columns else float("nan"),
            }
        )

    return pd.DataFrame(filas)
