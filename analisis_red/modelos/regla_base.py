"""
Regla simple de referencia (baseline) basada únicamente en la tasa capturada r_k:
emite alerta al cierre de k para la ventana k+1 si r_k >= tau, donde tau se
elige exclusivamente en el conjunto de validación (c04) y se fija antes de prueba.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


class ReglaUmbralTasa:
    """Clasificador de referencia basado en umbral directo sobre r_k (Mbit/s)."""

    def __init__(self, umbral_tasa_mbps: float = 9.0) -> None:
        self.umbral_tasa_mbps = float(umbral_tasa_mbps)

    def calibrar_en_validacion(
        self,
        df_val: pd.DataFrame,
        candidatos: Optional[Iterable[float]] = None,
        col_objetivo: str = "y_k",
    ) -> Tuple[float, pd.DataFrame]:
        """
        Explora umbrales candidatos de tasa r_k sobre validación (c04), elige el
        de mayor F1 (desempatando por sensibilidad y precisión) y congela el valor.
        """
        r_val = df_val["r_k"].to_numpy(dtype=np.float64)
        y_real_bin = (df_val[col_objetivo].to_numpy(dtype=np.float64) >= 1).astype(int)
        es_inicio = df_val["es_inicio_observado"].to_numpy(dtype=int) if "es_inicio_observado" in df_val.columns else None

        if candidatos is None:
            grilla = np.round(np.arange(4.0, 10.55, 0.1), 2)
        else:
            grilla = np.array(list(candidatos), dtype=np.float64)

        filas = []
        for tau in grilla:
            alerta = (r_val >= tau).astype(int)
            vp = int(np.sum((alerta == 1) & (y_real_bin == 1)))
            fp = int(np.sum((alerta == 1) & (y_real_bin == 0)))
            fn = int(np.sum((alerta == 0) & (y_real_bin == 1)))
            vn = int(np.sum((alerta == 0) & (y_real_bin == 0)))

            prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
            sens = vp / (vp + fn) if (vp + fn) > 0 else 0.0
            f1 = (2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

            if es_inicio is not None and np.sum(es_inicio == 1) > 0:
                inicios_ant = int(np.sum((alerta == 1) & (es_inicio == 1)))
                frac_inicios = inicios_ant / int(np.sum(es_inicio == 1))
            else:
                inicios_ant = 0
                frac_inicios = float("nan")

            filas.append(
                {
                    "tau_mbps": float(tau),
                    "VP": vp,
                    "FP": fp,
                    "FN": fn,
                    "VN": vn,
                    "precision": prec,
                    "sensibilidad": sens,
                    "f1": f1,
                    "inicios_anticipados": inicios_ant,
                    "frac_inicios_anticipados": frac_inicios,
                }
            )

        tabla = pd.DataFrame(filas).sort_values("tau_mbps").reset_index(drop=True)
        mejor = tabla.sort_values(
            by=["f1", "sensibilidad", "precision"],
            ascending=[False, False, False],
        ).iloc[0]

        self.umbral_tasa_mbps = float(mejor["tau_mbps"])
        return self.umbral_tasa_mbps, tabla

    def emitir_alerta(self, df: pd.DataFrame, umbral_tasa_mbps: Optional[float] = None) -> np.ndarray:
        """Emite alerta para la ventana k+1 si r_k >= tau."""
        tau = self.umbral_tasa_mbps if umbral_tasa_mbps is None else float(umbral_tasa_mbps)
        return (df["r_k"].to_numpy(dtype=np.float64) >= tau).astype(int)
