# Laboratorio de Análisis de Datos en Redes
**Predicción de congestión en un enlace LAN mediante captura de tráfico y regresión lineal**  
**Curso:** Inteligencia Artificial  
**Autor:** Santiago Parra Acuña  

Implementación modular y reproducible para la adquisición de trazas UDP bajo un cuello de botella controlado (cola TBF de 10 Mbit/s), construcción de series temporales alineadas por ventana, entrenamiento de regresión lineal y evaluación de alertas tempranas de congestión.

---

## Estructura del Repositorio

```text
Taller_Analisis_Datos_Redes/
├── analisis_red/                     # Paquete modular principal
│   ├── __init__.py
│   ├── config.py                     # Rutas, parámetros de red, particiones y constantes
│   ├── ingesta/                      # Lectura de fuentes crudas
│   │   ├── cola_parser.py            # Parser de cXX_cola.log (tc -j -s) y verificación
│   │   ├── paquetes_parser.py        # Parser de cXX_paquetes.csv (tshark) y respaldo PCAP
│   │   └── iperf_parser.py           # Parser de JSON de iperf3 (cliente/servidor) y calendario
│   ├── series/                       # Construcción de serie temporal y dataset
│   │   ├── alineacion.py             # Ventanas [t_{k-1}, t_k), r_k, d_k, n_k y control de anomalías
│   │   └── caracteristicas.py        # Rezagos (r_{k-1}, r_{k-2}), objetivo y_k=d_{k+1} y partición
│   ├── modelos/                      # Modelos predictivos y reglas de alerta
│   │   ├── regresion.py              # LinearRegression, recorte en cero y calibración de umbral
│   │   └── regla_base.py             # Regla simple de referencia basada en r_k >= tau
│   ├── evaluacion/                   # Métricas y diagnóstico de episodios
│   │   ├── metricas.py               # MAE, RMSE, R², matriz de confusión, precisión y sensibilidad
│   │   ├── inicios.py                # Distinción inicio vs. persistencia y rampas vs. saltos
│   │   └── casos_estudio.py          # Extracción de falsas alarmas e inicios omitidos
│   └── visualizacion/
│       └── graficas.py               # Series alineadas de 3 paneles, matrices y diagnósticos
├── scripts_captura/                  # Montaje y captura reproducible en Linux / Docker
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── configurar_cola_tbf.sh
│   ├── limpiar_cola_tbf.sh
│   └── ejecutar_experimento.sh
├── datos/
│   ├── crudos/                       # Trazas .pcap, _cola.log, _paquetes.csv, JSON de iperf3
│   └── procesados/                   # ventanas.csv, dataset_supervisado.csv, resultados
├── resultados/                       # Figuras generadas en alta resolución (PNG 300 DPI)
├── main.py                           # Script principal de ejecución reproducible
├── Taller_Analisis_Datos.ipynb       # Notebook interactivo con resultados ejecutados
├── Informe_Taller_Analisis_Datos.pdf # Informe técnico final
├── requirements.txt
└── .gitignore
```

---

## Instalación

Requiere Python 3.10 o superior:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

---

## Reproducibilidad de la Captura Experimental

### Opción A: Dos computadores Linux conectados por Ethernet
En el receptor **B** (`192.168.50.20/24`):
```bash
iperf3 -s
```
En el emisor **A** (`192.168.50.10/24`), reemplazando `enp3s0` por la interfaz dedicada:
```bash
sudo ./scripts_captura/ejecutar_experimento.sh enp3s0 192.168.50.10 192.168.50.20 ./datos/crudos
```

### Opción B: Dos nodos Linux aislados en red Ethernet virtualizada (`192.168.50.0/24`)
Con Docker instalado, el archivo `scripts_captura/docker-compose.yml` levanta ambos nodos (`lan_emisor_a` y `lan_receptor_b`) en una red bridge aislada con los comandos nativos de Linux (`tc`, `iperf3`, `tcpdump`, `tshark`):

```bash
docker compose -f scripts_captura/docker-compose.yml up -d --build
docker exec lan_emisor_a bash /lab/scripts/ejecutar_experimento.sh eth0 192.168.50.10 192.168.50.20 /lab/datos/crudos
```

---

## Ejecución del Análisis y Modelado

Para reconstruir las ventanas desde los archivos crudos, entrenar la regresión lineal sobre `c01`–`c03`, calibrar umbrales sobre `c04`, evaluar en `c05`–`c06` y exportar todas las gráficas:

```bash
python main.py
```

También puede abrirse y ejecutarse paso a paso `Taller_Analisis_Datos.ipynb`.

---

## Protocolo de Modelado y Evaluación

- **Criterio de congestión:** una ventana $k$ se considera congestionada cuando la cola TBF del emisor registra $d_k \ge 1$ descartes.
- **Variables de entrada (exclusivamente desde la traza PCAP):** tasa actual $r_k$, rezagos $r_{k-1}$ y $r_{k-2}$, y conteo de paquetes $n_k$. El calendario de tasas de `iperf3` nunca ingresa como predictor.
- **Variable objetivo:** descartes de la ventana siguiente, $y_k = d_{k+1}$.
- **Partición estricta por corridas completas en orden de adquisición:** entrenamiento (`c01`, `c02`, `c03`), validación (`c04`), prueba (`c05`, `c06`).
- **Pronóstico continuo:** $\hat{y}_k^{+} = \max(0, \hat{y}_k)$, conservando los valores positivos como estimaciones continuas.
- **Inicio vs. persistencia:** se define un inicio observado cuando $d_k = 0$ y $d_{k+1} > 0$, reportando la fracción de inicios anticipados al cierre de $k$ y comparándola con la detección durante episodios ya activos.
