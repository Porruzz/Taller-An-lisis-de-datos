"""Módulo de evaluación cuantitativa, análisis de inicios y casos de estudio."""

from .metricas import (
    calcular_metricas_regresion,
    calcular_metricas_alertas,
    construir_tabla_comparativa,
)
from .inicios import (
    analizar_inicios_y_persistencia,
    analizar_rampas_vs_saltos,
)
from .casos_estudio import (
    extraer_tabla_inicios,
    extraer_casos_falsa_alarma,
    extraer_casos_inicio_omitido,
)

__all__ = [
    "calcular_metricas_regresion",
    "calcular_metricas_alertas",
    "construir_tabla_comparativa",
    "analizar_inicios_y_persistencia",
    "analizar_rampas_vs_saltos",
    "extraer_tabla_inicios",
    "extraer_casos_falsa_alarma",
    "extraer_casos_inicio_omitido",
]
