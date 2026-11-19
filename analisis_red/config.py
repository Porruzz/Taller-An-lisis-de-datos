"""
Configuración central del laboratorio de predicción de congestión en enlace LAN.
Define rutas del proyecto, parámetros del montaje experimental, particiones de
corridas y constantes de modelado y evaluación.
"""

from pathlib import Path

# Directorios principales del repositorio
BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_DIR = BASE_DIR / "datos"
DATOS_CRUDOS_DIR = DATOS_DIR / "crudos"
DATOS_PROCESADOS_DIR = DATOS_DIR / "procesados"
RESULTADOS_DIR = BASE_DIR / "resultados"

# Garantizar existencia de carpetas de salida
DATOS_CRUDOS_DIR.mkdir(parents=True, exist_ok=True)
DATOS_PROCESADOS_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros físicos/lógicos del enlace experimental
IP_EMISOR = "192.168.50.10"
IP_RECEPTOR = "192.168.50.20"
SUBRED_LAN = "192.168.50.0/24"
PUERTO_UDP = 5201
TASA_LIMITE_MBPS = 10.0
BURST_TBF_KB = 16
LIMIT_TBF_KB = 32
TAMANO_DATAGRAMA_UDP = 1200

# Partición temporal estricta por corridas completas en orden de adquisición
CORRIDAS_ENTRENAMIENTO = ("c01", "c02", "c03")
CORRIDAS_VALIDACION = ("c04",)
CORRIDAS_PRUEBA = ("c05", "c06")
CORRIDAS_TODAS = CORRIDAS_ENTRENAMIENTO + CORRIDAS_VALIDACION + CORRIDAS_PRUEBA

# Variables del modelo de regresión lineal (estrictamente derivadas de la traza PCAP)
COLUMNAS_PREDICTORAS = ["r_k", "r_k_1", "r_k_2", "n_k"]
COLUMNA_OBJETIVO = "y_k"

# Límites de integridad del muestreo de cola (ventana esperada ~1 s)
DELTA_MIN_S = 0.20
DELTA_MAX_S = 2.00

# Umbral base de alerta (descartes pronosticados >= 1 paquete)
UMBRAL_ALERTA_ESTANDAR = 1.0

# Estilo visual consistente para gráficas
PALETA = {
    "azul_oscuro": "#1B365D",
    "azul_medio": "#2B6CB0",
    "verde": "#2F855A",
    "naranja": "#DD6B20",
    "rojo": "#C53030",
    "morado": "#6B46C1",
    "gris": "#718096",
    "gris_claro": "#E2E8F0",
}
