"""
Cálculo de métricas de regresión (MAE en paquetes por ventana, RMSE, R²) y de
evaluación binaria de alertas (matriz de confusión, precisión, sensibilidad y F1).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas_regresion(
    y_real: np.ndarray,
    y_pred_recortado: np.ndarray,
    y_pred_bruto: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Calcula MAE en paquetes/ventana sobre las estimaciones continuas recortadas
    en cero (max(0, y_hat_k)), junto con RMSE, R² y métricas complementarias.
    """
    y_r = np.asarray(y_real, dtype=np.float64)
    y_p = np.asarray(y_pred_recortado, dtype=np.float64)

    res = {
        "n_ejemplos": int(len(y_r)),
        "mae_paquetes": float(mean_absolute_error(y_r, y_p)),
        "rmse_paquetes": float(np.sqrt(mean_squared_error(y_r, y_p))),
        "r2_recortado": float(r2_score(y_r, y_p)) if len(np.unique(y_r)) > 1 else float("nan"),
        "media_real_paquetes": float(np.mean(y_r)),
        "media_pred_paquetes": float(np.mean(y_p)),
    }
    if y_pred_bruto is not None:
        y_b = np.asarray(y_pred_bruto, dtype=np.float64)
        res["mae_bruto_paquetes"] = float(mean_absolute_error(y_r, y_b))
        res["r2_bruto"] = float(r2_score(y_r, y_b)) if len(np.unique(y_r)) > 1 else float("nan")
        res["pct_pred_negativas"] = float(100.0 * np.mean(y_b < 0.0))
    return res


def calcular_metricas_alertas(
    y_real_descartes: np.ndarray,
    alerta_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compara las alertas emitidas al tiempo t_k con el evento observado d_{k+1} >= 1.
    Retorna conteos de matriz de confusión (VP, FP, FN, VN), precisión,
    sensibilidad (recall), especificidad, F1 y exactitud.
    """
    y_bin = (np.asarray(y_real_descartes, dtype=np.float64) >= 1.0).astype(int)
    a_bin = (np.asarray(alerta_pred, dtype=int) >= 1).astype(int)

    vp = int(np.sum((a_bin == 1) & (y_bin == 1)))
    fp = int(np.sum((a_bin == 1) & (y_bin == 0)))
    fn = int(np.sum((a_bin == 0) & (y_bin == 1)))
    vn = int(np.sum((a_bin == 0) & (y_bin == 0)))
    total = vp + fp + fn + vn

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0.0
    exactitud = (vp + vn) / total if total > 0 else 0.0
    f1 = (2.0 * precision * sensibilidad / (precision + sensibilidad)) if (precision + sensibilidad) > 0 else 0.0

    return {
        "n_ventanas": total,
        "positivos_reales": vp + fn,
        "alertas_emitidas": vp + fp,
        "VP": vp,
        "FP": fp,
        "FN": fn,
        "VN": vn,
        "precision": float(precision),
        "sensibilidad": float(sensibilidad),
        "especificidad": float(especificidad),
        "exactitud": float(exactitud),
        "f1": float(f1),
    }


def construir_tabla_comparativa(
    registros: List[Dict[str, object]],
) -> pd.DataFrame:
    """Organiza una lista de diccionarios de evaluación en un DataFrame tabular limpio."""
    return pd.DataFrame(registros)
