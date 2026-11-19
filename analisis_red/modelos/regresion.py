"""
Modelo de regresión lineal múltiple para estimar los descartes de la ventana
siguiente a partir de la tasa capturada, dos rezagos y el conteo de paquetes:

    y_hat_k = beta_0 + beta_1 * r_k + beta_2 * r_{k-1} + beta_3 * r_{k-2} + beta_4 * n_k
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..config import COLUMNAS_PREDICTORAS, UMBRAL_ALERTA_ESTANDAR


@dataclass
class ResumenAjuste:
    """Parámetros aprendidos en entrenamiento y verificación de escalamiento."""

    intercepto: float
    coeficientes: Dict[str, float]
    coeficientes_estandarizados: Dict[str, float]
    diferencia_max_escalado: float
    r2_entrenamiento: float
    mae_entrenamiento: float


class ModeloRegresionCongestion:
    """
    Encapsula `sklearn.linear_model.LinearRegression` con:
      - ajuste estricto sobre corridas de entrenamiento;
      - verificación opcional con `StandardScaler` ajustado solo en entrenamiento;
      - recorte de predicciones negativas a cero para el reporte continuo de descartes;
      - calibración de umbral de alerta en validación (fijado antes de prueba).
    """

    def __init__(
        self,
        columnas_predictoras: Optional[List[str]] = None,
        umbral_alerta: float = UMBRAL_ALERTA_ESTANDAR,
    ) -> None:
        self.columnas_predictoras = list(columnas_predictoras or COLUMNAS_PREDICTORAS)
        self.umbral_alerta = float(umbral_alerta)
        self._modelo = LinearRegression()
        self._scaler = StandardScaler()
        self._modelo_escalado = LinearRegression()
        self._ajustado = False

    def ajustar(self, df_train: pd.DataFrame, col_objetivo: str = "y_k") -> ResumenAjuste:
        """Ajusta la regresión lineal y su contraparte escalada únicamente con entrenamiento."""
        X_train = df_train[self.columnas_predictoras].to_numpy(dtype=np.float64)
        y_train = df_train[col_objetivo].to_numpy(dtype=np.float64)

        self._modelo.fit(X_train, y_train)

        # Escalamiento ajustado exclusivamente con entrenamiento (comprobación de invariancia OLS)
        X_train_std = self._scaler.fit_transform(X_train)
        self._modelo_escalado.fit(X_train_std, y_train)
        self._ajustado = True

        y_hat_bruto = self._modelo.predict(X_train)
        y_hat_std = self._modelo_escalado.predict(X_train_std)
        diff_max = float(np.max(np.abs(y_hat_bruto - y_hat_std)))

        y_hat_rec = np.maximum(0.0, y_hat_bruto)
        mae_train = float(np.mean(np.abs(y_train - y_hat_rec)))
        r2_train = float(self._modelo.score(X_train, y_train))

        coefs = {
            col: float(val)
            for col, val in zip(self.columnas_predictoras, self._modelo.coef_)
        }
        coefs_std = {
            col: float(val)
            for col, val in zip(self.columnas_predictoras, self._modelo_escalado.coef_)
        }

        return ResumenAjuste(
            intercepto=float(self._modelo.intercept_),
            coeficientes=coefs,
            coeficientes_estandarizados=coefs_std,
            diferencia_max_escalado=diff_max,
            r2_entrenamiento=r2_train,
            mae_entrenamiento=mae_train,
        )

    def _verificar_ajustado(self) -> None:
        if not self._ajustado:
            raise RuntimeError("El modelo debe ajustarse antes de predecir.")

    def predecir_bruto(self, df: pd.DataFrame) -> np.ndarray:
        """Retorna la estimación lineal sin recortar (y_hat_k)."""
        self._verificar_ajustado()
        X = df[self.columnas_predictoras].to_numpy(dtype=np.float64)
        return self._modelo.predict(X)

    def predecir_descartes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Retorna descartes pronosticados continuos llevando salidas negativas a cero:
            max(0.0, y_hat_k)
        """
        return np.maximum(0.0, self.predecir_bruto(df))

    def emitir_alerta(
        self,
        df: pd.DataFrame,
        umbral: Optional[float] = None,
    ) -> np.ndarray:
        """
        Emite alerta binaria al cierre de la ventana k para la ventana k+1 si
        y_hat_k >= umbral.
        """
        thr = self.umbral_alerta if umbral is None else float(umbral)
        return (self.predecir_bruto(df) >= thr).astype(int)

    def calibrar_umbral_validacion(
        self,
        df_val: pd.DataFrame,
        candidatos: Optional[Iterable[float]] = None,
        col_objetivo: str = "y_k",
        fijar_umbral: bool = False,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Evalúa un barrido de umbrales de alerta estrictamente en el conjunto de
        validación (c04) y retorna el umbral óptimo según F1 (con desempate por
        sensibilidad y cercanía al umbral estándar 1.0) junto con la tabla completa.
        """
        self._verificar_ajustado()
        y_hat = self.predecir_bruto(df_val)
        y_real_bin = (df_val[col_objetivo].to_numpy(dtype=np.float64) >= 1).astype(int)
        es_inicio = df_val["es_inicio_observado"].to_numpy(dtype=int) if "es_inicio_observado" in df_val.columns else None

        if candidatos is None:
            grilla = np.unique(
                np.concatenate(
                    [
                        np.array([0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]),
                        np.linspace(1.0, max(20.0, float(np.percentile(y_hat, 90))), 40),
                    ]
                )
            )
        else:
            grilla = np.array(list(candidatos), dtype=np.float64)

        filas = []
        for thr in grilla:
            alerta = (y_hat >= thr).astype(int)
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
                    "umbral": float(thr),
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

        tabla = pd.DataFrame(filas).sort_values("umbral").reset_index(drop=True)
        # Selección por mayor F1, desempatando por mayor sensibilidad y menor distancia a 1.0
        mejor_fila = tabla.assign(
            dist_std=(tabla["umbral"] - UMBRAL_ALERTA_ESTANDAR).abs()
        ).sort_values(
            by=["f1", "sensibilidad", "dist_std"],
            ascending=[False, False, True],
        ).iloc[0]

        umbral_opt = float(mejor_fila["umbral"])
        if fijar_umbral:
            self.umbral_alerta = umbral_opt
        return umbral_opt, tabla

    def ecuacion_texto(self) -> str:
        """Formatea la ecuación ajustada en texto legible."""
        self._verificar_ajustado()
        b0 = float(self._modelo.intercept_)
        terminos = [f"{b0:.4f}"]
        nombres = {"r_k": "r_k", "r_k_1": "r_{k-1}", "r_k_2": "r_{k-2}", "n_k": "n_k"}
        for col, c in zip(self.columnas_predictoras, self._modelo.coef_):
            signo = "+" if c >= 0 else "-"
            terminos.append(f" {signo} {abs(float(c)):.4f}*{nombres.get(col, col)}")
        return "y_hat_k = " + "".join(terminos)
