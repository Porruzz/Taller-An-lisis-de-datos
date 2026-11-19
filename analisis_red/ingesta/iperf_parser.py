"""
Parser de salidas JSON de `iperf3 --get-server-output -J` y del calendario de
fases. Esta información se emplea exclusivamente para verificación experimental
e interpretación posterior, nunca como variable predictora del modelo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def leer_calendario_fases(ruta_calendario: Path) -> pd.DataFrame:
    """Lee el archivo `calendario_fases.csv` con tiempos reales y perfiles de carga."""
    ruta_calendario = Path(ruta_calendario)
    if not ruta_calendario.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "fase_idx",
                "fase_nombre",
                "tipo_transicion",
                "tasa_mbps",
                "duracion_s",
                "t_inicio_epoch",
                "t_fin_epoch",
                "pausa_previa_s",
                "archivo_iperf",
            ]
        )
    df = pd.read_csv(ruta_calendario)
    for col in ("tasa_mbps", "duracion_s", "t_inicio_epoch", "t_fin_epoch", "pausa_previa_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _extraer_metricas_iperf_json(ruta_json: Path) -> Dict[str, Any]:
    """Extrae métricas del cliente (emisor A) y servidor (receptor B) de un JSON de iperf3."""
    datos = json.loads(ruta_json.read_text(encoding="utf-8", errors="replace"))
    end_blk = datos.get("end", {})

    end_client = end_blk.get("sum_sent") or end_blk.get("sum") or {}

    # En iperf3 >= 3.12 UDP, las estadísticas del receptor llegan en end.sum_received
    server_sum = end_blk.get("sum_received") or {}
    if not server_sum:
        server_end = datos.get("server_output_json", {}).get("end", {})
        server_sum = server_end.get("sum_received") or server_end.get("sum") or {}
    if not server_sum:
        streams = end_blk.get("streams", [])
        if streams and isinstance(streams[0], dict):
            server_sum = streams[0].get("udp", {})

    return {
        "archivo": ruta_json.name,
        "duracion_real_s": float(end_client.get("seconds", 0.0)),
        "tasa_ofrecida_cliente_mbps": float(end_client.get("bits_per_second", 0.0)) / 1e6,
        "paquetes_enviados_cliente": int(end_client.get("packets", 0)),
        "tasa_recibida_servidor_mbps": float(server_sum.get("bits_per_second", 0.0)) / 1e6,
        "paquetes_servidor": int(server_sum.get("packets", 0)),
        "paquetes_perdidos_servidor": int(server_sum.get("lost_packets", 0)),
        "pct_perdida_servidor": float(server_sum.get("lost_percent", 0.0)),
        "jitter_ms_servidor": float(server_sum.get("jitter_ms", 0.0)),
    }


def leer_resumen_iperf_corrida(directorio_crudos: Path, run_id: str) -> pd.DataFrame:
    """Consolida todos los archivos `cXX_faseYY.json` correspondientes a una corrida."""
    directorio_crudos = Path(directorio_crudos)
    archivos = sorted(directorio_crudos.glob(f"{run_id}_fase*.json"))
    filas: List[Dict[str, Any]] = []
    for idx, ruta in enumerate(archivos, start=1):
        try:
            info = _extraer_metricas_iperf_json(ruta)
            info["run_id"] = run_id
            info["fase_idx"] = idx
            filas.append(info)
        except Exception as exc:
            filas.append(
                {
                    "run_id": run_id,
                    "fase_idx": idx,
                    "archivo": ruta.name,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(filas)


def leer_resumen_iperf_todas(
    directorio_crudos: Path,
    corridas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Consolida la comprobación de iperf3 (emisor vs receptor B) para todas las corridas."""
    directorio_crudos = Path(directorio_crudos)
    calendario = leer_calendario_fases(directorio_crudos / "calendario_fases.csv")

    if corridas is None:
        corridas = sorted({p.name.split("_")[0] for p in directorio_crudos.glob("c*_fase*.json")})

    partes = [leer_resumen_iperf_corrida(directorio_crudos, r) for r in corridas]
    partes = [p for p in partes if not p.empty]
    if not partes:
        return pd.DataFrame()

    df_iperf = pd.concat(partes, ignore_index=True)
    if not calendario.empty:
        df_iperf = df_iperf.merge(
            calendario[["run_id", "fase_idx", "fase_nombre", "tipo_transicion", "tasa_mbps", "pausa_previa_s"]],
            on=["run_id", "fase_idx"],
            how="left",
        )
    return df_iperf
