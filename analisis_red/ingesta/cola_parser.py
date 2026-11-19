"""
Parser de registros de muestreo de cola TBF generados con `tc -j -s qdisc show`.
Cada muestra válida consiste en un par de líneas:
  1) Marca temporal Unix en segundos (salida de `date +%s.%N`)
  2) Arreglo JSON de qdiscs reportado por `tc -j -s`
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _extraer_objeto_tbf(
    objetos_qdisc: List[Dict[str, Any]],
    handle_objetivo: str = "10:",
) -> Optional[Dict[str, Any]]:
    """Localiza el objeto qdisc con kind='tbf' y handle especificado."""
    for item in objetos_qdisc:
        if (
            isinstance(item, dict)
            and str(item.get("kind", "")).lower() == "tbf"
            and str(item.get("handle", "")) == handle_objetivo
        ):
            return item
    return None


def leer_registro_cola(
    ruta_log: Path,
    run_id: str,
    handle_objetivo: str = "10:",
) -> pd.DataFrame:
    """
    Lee el archivo `cXX_cola.log` y construye una tabla ordenada por tiempo
    con el contador acumulado de descartes (`drops_cum`) y banderas de integridad.

    Si alguna muestra tiene marca temporal inválida, JSON truncado o carece del
    objeto `tbf` con el handle esperado, se marca con `sample_ok=False` para que
    la etapa de alineación aísle ese tramo sin interpolar ni cruzar filas.
    """
    ruta_log = Path(ruta_log)
    if not ruta_log.exists():
        raise FileNotFoundError(f"No se encontró el archivo de cola: {ruta_log}")

    lineas = [ln.strip() for ln in ruta_log.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]

    registros: List[Dict[str, Any]] = []
    idx = 0
    muestra_num = 0

    while idx < len(lineas):
        linea_ts = lineas[idx]
        try:
            ts = float(linea_ts)
        except ValueError:
            # Línea fuera de sincronía (no es timestamp); se registra anomalía y avanza
            registros.append(
                {
                    "run_id": run_id,
                    "sample_idx": muestra_num,
                    "timestamp": float("nan"),
                    "drops_cum": float("nan"),
                    "bytes_qdisc": float("nan"),
                    "packets_qdisc": float("nan"),
                    "overlimits": float("nan"),
                    "sample_ok": False,
                    "error_msg": f"timestamp_invalido:{linea_ts[:30]}",
                }
            )
            muestra_num += 1
            idx += 1
            continue

        if idx + 1 >= len(lineas):
            registros.append(
                {
                    "run_id": run_id,
                    "sample_idx": muestra_num,
                    "timestamp": ts,
                    "drops_cum": float("nan"),
                    "bytes_qdisc": float("nan"),
                    "packets_qdisc": float("nan"),
                    "overlimits": float("nan"),
                    "sample_ok": False,
                    "error_msg": "muestra_sin_json",
                }
            )
            muestra_num += 1
            idx += 1
            continue

        linea_json = lineas[idx + 1]
        try:
            payload = json.loads(linea_json)
            if not isinstance(payload, list):
                raise ValueError("La salida JSON de tc no es una lista.")
            tbf = _extraer_objeto_tbf(payload, handle_objetivo=handle_objetivo)
            if tbf is None or "drops" not in tbf:
                registros.append(
                    {
                        "run_id": run_id,
                        "sample_idx": muestra_num,
                        "timestamp": ts,
                        "drops_cum": float("nan"),
                        "bytes_qdisc": float("nan"),
                        "packets_qdisc": float("nan"),
                        "overlimits": float("nan"),
                        "sample_ok": False,
                        "error_msg": f"sin_objeto_tbf_{handle_objetivo}",
                    }
                )
            else:
                registros.append(
                    {
                        "run_id": run_id,
                        "sample_idx": muestra_num,
                        "timestamp": ts,
                        "drops_cum": int(tbf["drops"]),
                        "bytes_qdisc": int(tbf.get("bytes", 0)),
                        "packets_qdisc": int(tbf.get("packets", 0)),
                        "overlimits": int(tbf.get("overlimits", 0)),
                        "sample_ok": True,
                        "error_msg": "",
                    }
                )
        except Exception as exc:
            registros.append(
                {
                    "run_id": run_id,
                    "sample_idx": muestra_num,
                    "timestamp": ts,
                    "drops_cum": float("nan"),
                    "bytes_qdisc": float("nan"),
                    "packets_qdisc": float("nan"),
                    "overlimits": float("nan"),
                    "sample_ok": False,
                    "error_msg": f"json_invalido:{type(exc).__name__}",
                }
            )

        muestra_num += 1
        idx += 2

    df = pd.DataFrame(registros)
    if df.empty:
        raise ValueError(f"El archivo de cola está vacío o no contiene muestras válidas: {ruta_log}")

    # Ordenar por tiempo preservando filas con timestamp NaN al final (marcadas como inválidas)
    df = df.sort_values(by=["timestamp", "sample_idx"], kind="mergesort", na_position="last").reset_index(drop=True)
    return df


def verificar_coincidencia_drops(salida_textual_tc: str, salida_json_tc: str, handle: str = "10:") -> Tuple[bool, int, int]:
    """
    Comprueba que el campo `drops` en la salida JSON de `tc -j -s` coincida con
    el valor `dropped` de la salida textual `tc -s` para el qdisc TBF.
    """
    patron = re.compile(rf"qdisc tbf\s+{re.escape(handle)}.*?dropped\s+(\d+)", re.DOTALL | re.IGNORECASE)
    coincidencia = patron.search(salida_textual_tc)
    if not coincidencia:
        raise ValueError(f"No se encontró 'dropped' para qdisc tbf {handle} en la salida textual.")
    dropped_txt = int(coincidencia.group(1))

    datos_json = json.loads(salida_json_tc)
    tbf = _extraer_objeto_tbf(datos_json, handle_objetivo=handle)
    if tbf is None or "drops" not in tbf:
        raise ValueError(f"No se encontró objeto tbf {handle} en la salida JSON.")
    drops_json = int(tbf["drops"])

    return (dropped_txt == drops_json), dropped_txt, drops_json
