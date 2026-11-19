"""
Alineación temporal entre las muestras del contador de cola TBF (`cXX_cola.log`)
y las tramas capturadas (`cXX_paquetes.csv`) para formar ventanas [t_{k-1}, t_k).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import CORRIDAS_TODAS, DELTA_MAX_S, DELTA_MIN_S
from ..ingesta.cola_parser import leer_registro_cola
from ..ingesta.iperf_parser import leer_calendario_fases
from ..ingesta.paquetes_parser import leer_csv_paquetes


def _asignar_fase_interpretacion(
    t_inicio: float,
    t_fin: float,
    fases_corrida: pd.DataFrame,
    d_k: float = 0.0,
) -> Tuple[str, str, float]:
    """
    Asigna la fase del calendario correspondiente al intervalo [t_inicio, t_fin)
    únicamente con fines descriptivos y de análisis posterior (fuera de predictores).

    En ventanas de frontera donde ocurren descartes (d_k > 0) y se solapan dos
    fases (p. ej., fin de Base/Ascenso e inicio de Sobrecarga), se atribuye la
    fase de sobrecarga solapada (tasa_mbps >= 10.0) que causó los descartes; en
    los demás casos se selecciona la fase con mayor solapamiento temporal.
    """
    if fases_corrida.empty:
        return "sin_calendario", "desconocido", float("nan")

    mejor_solape = 0.0
    mejor_nombre = "reposo"
    mejor_tipo = "reposo"
    mejor_tasa = 0.0

    sobrecarga_solape = 0.0
    sobrecarga_info: Optional[Tuple[str, str, float]] = None

    for row in fases_corrida.itertuples(index=False):
        ini = max(t_inicio, float(row.t_inicio_epoch))
        fin = min(t_fin, float(row.t_fin_epoch))
        solape = fin - ini
        if solape > 0.0:
            tasa_row = float(row.tasa_mbps)
            if solape > mejor_solape:
                mejor_solape = solape
                mejor_nombre = str(row.fase_nombre)
                mejor_tipo = str(row.tipo_transicion)
                mejor_tasa = tasa_row
            if tasa_row >= 10.0 and solape > sobrecarga_solape:
                sobrecarga_solape = solape
                sobrecarga_info = (str(row.fase_nombre), str(row.tipo_transicion), tasa_row)

    if np.isfinite(d_k) and d_k > 0.0 and sobrecarga_info is not None:
        return sobrecarga_info

    return mejor_nombre, mejor_tipo, mejor_tasa


def construir_ventanas_corrida(
    df_cola: pd.DataFrame,
    df_paquetes: pd.DataFrame,
    run_id: str,
    df_calendario: Optional[pd.DataFrame] = None,
    delta_min_s: float = DELTA_MIN_S,
    delta_max_s: float = DELTA_MAX_S,
) -> pd.DataFrame:
    """
    Construye la serie temporal de ventanas [t_{k-1}, t_k) para una corrida:
      - delta_k = t_k - t_{k-1}
      - r_k = (8 * B_k) / (10^6 * delta_k)  [Mbit/s]
      - d_k = D(t_k) - D(t_{k-1})           [paquetes descartados]
      - n_k = conteo de paquetes en [t_{k-1}, t_k)

    Detecta y marca anomalías (muestra corrupta, reinicio de contador, pausa
    anormal o interrupción de captura) asignando un `segmento_id` contiguo que
    impide construir rezagos o etiquetas cruzando un tramo afectado.
    """
    if len(df_cola) < 2:
        raise ValueError(f"La corrida {run_id} tiene menos de dos muestras de cola.")

    cola = df_cola.reset_index(drop=True)
    tiempos_pkt = df_paquetes["time_epoch"].to_numpy(dtype=np.float64) if not df_paquetes.empty else np.array([], dtype=np.float64)
    lens_pkt = df_paquetes["frame_len"].to_numpy(dtype=np.int64) if not df_paquetes.empty else np.array([], dtype=np.int64)
    cumsum_bytes = np.concatenate(([0], np.cumsum(lens_pkt, dtype=np.int64)))

    # Referencia temporal inicial de la corrida (primera muestra válida)
    ts_validos = cola.loc[cola["sample_ok"], "timestamp"].dropna()
    t0_corrida = float(ts_validos.iloc[0]) if not ts_validos.empty else 0.0

    # Filtrar calendario solo para esta corrida (uso meramente interpretativo)
    fases_corrida = pd.DataFrame()
    if df_calendario is not None and not df_calendario.empty:
        fases_corrida = df_calendario[df_calendario["run_id"] == run_id].copy()

    # Rango de captura observada para detectar posibles interrupciones graves
    t_pkt_min = float(tiempos_pkt[0]) if len(tiempos_pkt) > 0 else float("nan")
    t_pkt_max = float(tiempos_pkt[-1]) if len(tiempos_pkt) > 0 else float("nan")

    filas: List[Dict[str, object]] = []
    segmento_id = 0
    prev_invalida = False

    for k in range(1, len(cola)):
        prev = cola.iloc[k - 1]
        curr = cola.iloc[k]

        motivos: List[str] = []
        if not bool(prev["sample_ok"]) or not bool(curr["sample_ok"]):
            motivos.append("muestra_faltante_o_invalida")

        t_prev = float(prev["timestamp"]) if pd.notna(prev["timestamp"]) else float("nan")
        t_curr = float(curr["timestamp"]) if pd.notna(curr["timestamp"]) else float("nan")
        delta_k = t_curr - t_prev if (np.isfinite(t_prev) and np.isfinite(t_curr)) else float("nan")

        if not np.isfinite(delta_k) or delta_k <= 0.0:
            motivos.append("delta_no_positivo")
        elif delta_k < delta_min_s or delta_k > delta_max_s:
            motivos.append(f"pausa_anormal({delta_k:.3f}s)")

        d_prev = float(prev["drops_cum"]) if pd.notna(prev["drops_cum"]) else float("nan")
        d_curr = float(curr["drops_cum"]) if pd.notna(curr["drops_cum"]) else float("nan")
        d_k = d_curr - d_prev if (np.isfinite(d_prev) and np.isfinite(d_curr)) else float("nan")

        if not np.isfinite(d_k):
            motivos.append("contador_no_disponible")
        elif d_k < 0:
            motivos.append("reinicio_contador")

        # Conteo de paquetes y suma de bytes en [t_{k-1}, t_k)
        if np.isfinite(t_prev) and np.isfinite(t_curr) and t_curr > t_prev and len(tiempos_pkt) > 0:
            i_izq = int(np.searchsorted(tiempos_pkt, t_prev, side="left"))
            i_der = int(np.searchsorted(tiempos_pkt, t_curr, side="left"))
            n_k = i_der - i_izq
            bytes_k = int(cumsum_bytes[i_der] - cumsum_bytes[i_izq])
        else:
            n_k = 0
            bytes_k = 0

        # Detección de interrupción de captura: tráfico reportado en qdisc o descartes
        # durante una ventana sin ningún paquete capturado después del cierre de la traza
        if (
            not motivos
            and n_k == 0
            and d_k > 0
            and np.isfinite(t_pkt_max)
            and t_prev > t_pkt_max + delta_max_s
        ):
            motivos.append("interrupcion_captura")

        valida = len(motivos) == 0
        if not valida:
            prev_invalida = True
            seg_asignado = -1
        else:
            if prev_invalida:
                segmento_id += 1
                prev_invalida = False
            seg_asignado = segmento_id

        # Tasa r_k = 8 * B_k / (10^6 * delta_k) en Mbit/s; 0 si ventana válida vacía
        if valida and delta_k > 0.0:
            r_k = (8.0 * bytes_k) / (1e6 * delta_k)
            d_k_val: float = float(int(round(d_k)))
        else:
            r_k = float("nan")
            d_k_val = float("nan")

        fase_nombre, tipo_trans, tasa_ofrecida = (
            _asignar_fase_interpretacion(t_prev, t_curr, fases_corrida, d_k=d_k_val)
            if (np.isfinite(t_prev) and np.isfinite(t_curr))
            else ("invalida", "invalida", float("nan"))
        )

        filas.append(
            {
                "run_id": run_id,
                "ventana_idx": k,
                "segmento_id": seg_asignado,
                "t_inicio": t_prev,
                "t_fin": t_curr,
                "t_rel_inicio_s": (t_prev - t0_corrida) if np.isfinite(t_prev) else float("nan"),
                "t_rel_fin_s": (t_curr - t0_corrida) if np.isfinite(t_curr) else float("nan"),
                "delta_s": delta_k,
                "bytes_k": bytes_k,
                "n_k": int(n_k),
                "r_k": r_k,
                "drops_cum_prev": d_prev,
                "drops_cum": d_curr,
                "d_k": d_k_val,
                "congestion_k": int(d_k_val >= 1) if np.isfinite(d_k_val) else -1,
                "valida": valida,
                "motivo_anomalia": ";".join(motivos),
                # Columnas solo interpretativas (nunca se usan como predictores)
                "fase_interpretacion": fase_nombre,
                "tipo_transicion": tipo_trans,
                "tasa_ofrecida_mbps": tasa_ofrecida,
            }
        )

    return pd.DataFrame(filas)


def construir_ventanas_todas(
    directorio_crudos: Path,
    corridas: Iterable[str] = CORRIDAS_TODAS,
    ruta_salida_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Construye y concatena el CSV maestro de ventanas para todas las corridas en
    orden de adquisición, verificando anomalías por corrida.
    """
    directorio_crudos = Path(directorio_crudos)
    calendario = leer_calendario_fases(directorio_crudos / "calendario_fases.csv")

    partes: List[pd.DataFrame] = []
    for run_id in corridas:
        ruta_cola = directorio_crudos / f"{run_id}_cola.log"
        ruta_pkts = directorio_crudos / f"{run_id}_paquetes.csv"
        df_cola = leer_registro_cola(ruta_cola, run_id=run_id)
        df_pkts = leer_csv_paquetes(ruta_pkts)
        df_ventanas = construir_ventanas_corrida(
            df_cola=df_cola,
            df_paquetes=df_pkts,
            run_id=run_id,
            df_calendario=calendario,
        )
        partes.append(df_ventanas)

    df_total = pd.concat(partes, ignore_index=True)

    if ruta_salida_csv is not None:
        ruta_salida_csv = Path(ruta_salida_csv)
        ruta_salida_csv.parent.mkdir(parents=True, exist_ok=True)
        df_total.to_csv(ruta_salida_csv, index=False)

    return df_total
