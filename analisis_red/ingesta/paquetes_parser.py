"""
Parser de capturas de paquetes exportadas con `tshark` (`cXX_paquetes.csv`),
con soporte complementario de lectura directa de archivos PCAP estándar.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List

import pandas as pd


def leer_csv_paquetes(ruta_csv: Path) -> pd.DataFrame:
    """
    Lee el CSV exportado con:
      tshark -r cXX.pcap -T fields -E header=y -E separator=, \
        -e frame.time_epoch -e frame.len

    Retorna un DataFrame ordenado por `time_epoch` con columnas:
      - `time_epoch` (float64): tiempo Unix en segundos
      - `frame_len` (int64): longitud de trama original reportada en la captura
    """
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV de paquetes: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    renombre = {}
    for col in df.columns:
        limpia = col.strip()
        if limpia in ("frame.time_epoch", "time_epoch"):
            renombre[col] = "time_epoch"
        elif limpia in ("frame.len", "frame_len"):
            renombre[col] = "frame_len"
    df = df.rename(columns=renombre)

    if "time_epoch" not in df.columns or "frame_len" not in df.columns:
        raise ValueError(
            f"El archivo {ruta_csv.name} debe contener columnas frame.time_epoch y frame.len. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    df["time_epoch"] = pd.to_numeric(df["time_epoch"], errors="coerce")
    df["frame_len"] = pd.to_numeric(df["frame_len"], errors="coerce")
    df = df.dropna(subset=["time_epoch", "frame_len"]).copy()
    df["frame_len"] = df["frame_len"].astype("int64")
    df = df.sort_values("time_epoch", kind="mergesort").reset_index(drop=True)
    return df[["time_epoch", "frame_len"]]


def leer_pcap_binario(ruta_pcap: Path) -> pd.DataFrame:
    """
    Lector binario ligero de archivos PCAP clásicos (libpcap micro/nanosegundos)
    que recupera (time_epoch, orig_len) como verificación directa de `frame.len`.
    """
    ruta_pcap = Path(ruta_pcap)
    if not ruta_pcap.exists():
        raise FileNotFoundError(f"No se encontró la traza PCAP: {ruta_pcap}")

    datos = ruta_pcap.read_bytes()
    if len(datos) < 24:
        raise ValueError(f"Archivo PCAP truncado: {ruta_pcap}")

    magic = datos[:4]
    if magic == b"\xd4\xc3\xb2\xa1":
        endian, escala = "<", 1e-6
    elif magic == b"\xa1\xb2\xc3\xd4":
        endian, escala = ">", 1e-6
    elif magic == b"\x4d\x3c\xb2\xa1":
        endian, escala = "<", 1e-9
    elif magic == b"\xa1\xb2\x3c\x4d":
        endian, escala = ">", 1e-9
    else:
        raise ValueError(f"Formato PCAP no reconocido (magic={magic.hex()}). Use tshark CSV.")

    offset = 24
    registros: List[Dict[str, float]] = []
    hdr_fmt = f"{endian}IIII"
    tam = len(datos)

    while offset + 16 <= tam:
        ts_sec, ts_sub, incl_len, orig_len = struct.unpack_from(hdr_fmt, datos, offset)
        offset += 16
        if offset + incl_len > tam:
            break
        offset += incl_len
        registros.append({"time_epoch": ts_sec + ts_sub * escala, "frame_len": int(orig_len)})

    df = pd.DataFrame(registros)
    if df.empty:
        return pd.DataFrame(columns=["time_epoch", "frame_len"])
    df["frame_len"] = df["frame_len"].astype("int64")
    return df.sort_values("time_epoch", kind="mergesort").reset_index(drop=True)
