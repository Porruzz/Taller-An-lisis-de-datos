"""Módulo de ingesta de archivos crudos de captura, cola TBF e iperf3."""

from .cola_parser import leer_registro_cola, verificar_coincidencia_drops
from .paquetes_parser import leer_csv_paquetes, leer_pcap_binario
from .iperf_parser import leer_calendario_fases, leer_resumen_iperf_corrida, leer_resumen_iperf_todas

__all__ = [
    "leer_registro_cola",
    "verificar_coincidencia_drops",
    "leer_csv_paquetes",
    "leer_pcap_binario",
    "leer_calendario_fases",
    "leer_resumen_iperf_corrida",
    "leer_resumen_iperf_todas",
]
