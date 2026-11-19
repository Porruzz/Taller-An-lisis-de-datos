#!/usr/bin/env bash
# Orquestador reproducible del laboratorio de prediccion de congestion LAN.
# Ejecuta 6 corridas independientes con perfiles variados (rampas graduales,
# cargas sub-umbral y saltos bruscos), capturando trazas PCAP, registros de
# cola TBF (tc -j -s), salidas JSON de iperf3 y tablas de paquetes con tshark.
#
# Uso en emisor A (Linux o contenedor Docker):
#   ./ejecutar_experimento.sh [IFACE] [IP_EMISOR] [IP_RECEPTOR] [OUT_DIR]
set -euo pipefail

IFACE="${1:-eth0}"
SRC_IP="${2:-192.168.50.10}"
DST_IP="${3:-192.168.50.20}"
OUT_DIR="${4:-/lab/datos/crudos}"

mkdir -p "${OUT_DIR}"

VERIFY_LOG="${OUT_DIR}/verificacion_montaje.txt"
TCPDUMP_LOG="${OUT_DIR}/tcpdump_resumen.txt"
CALENDARIO_CSV="${OUT_DIR}/calendario_fases.csv"

echo "run_id,fase_idx,fase_nombre,tipo_transicion,tasa_mbps,duracion_s,t_inicio_epoch,t_fin_epoch,pausa_previa_s,archivo_iperf" > "${CALENDARIO_CSV}"
: > "${TCPDUMP_LOG}"

{
  echo "=== Verificacion inicial del montaje ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  echo "--- ip -br addr ---"
  ip -br addr
  echo "--- ping a receptor ${DST_IP} ---"
  ping -c 3 "${DST_IP}"
  echo "--- qdisc original en ${IFACE} ---"
  tc qdisc show dev "${IFACE}"
} | tee "${VERIFY_LOG}"

aplicar_tbf() {
  tc qdisc replace dev "${IFACE}" root handle 10: tbf \
    rate 10mbit burst 16kb limit 32kb
}

# Verificar coincidencia entre drops (JSON) y dropped (textual)
aplicar_tbf
{
  echo "--- Verificacion TBF textual (tc -s) ---"
  tc -s qdisc show dev "${IFACE}"
  echo "--- Verificacion TBF JSON (tc -j -s) ---"
  tc -j -s qdisc show dev "${IFACE}"
} >> "${VERIFY_LOG}"

ejecutar_corrida() {
  local run_id="$1"
  shift
  local fases=("$@")

  echo ""
  echo "============================================================"
  echo "  INICIANDO CORRIDA: ${run_id} (${#fases[@]} fases)"
  echo "============================================================"

  # Reiniciar contadores de la cola TBF antes de iniciar la corrida independiente
  aplicar_tbf
  sleep 1

  local pcap_file="${OUT_DIR}/${run_id}.pcap"
  local cola_file="${OUT_DIR}/${run_id}_cola.log"
  local csv_file="${OUT_DIR}/${run_id}_paquetes.csv"
  local tcpdump_err="${OUT_DIR}/.${run_id}_tcpdump.err"

  # 1. Iniciar captura tcpdump (-U asegura volcado inmediato por paquete)
  tcpdump -i "${IFACE}" -nn -s 96 -B 4096 -U -w "${pcap_file}" \
    "src host ${SRC_IP} and dst host ${DST_IP} and udp dst port 5201" \
    2> "${tcpdump_err}" &
  local tcpdump_pid=$!
  sleep 1.5

  # 2. Iniciar muestreo continuo de la cola TBF (par de lineas: epoch + JSON)
  (
    running=1
    trap 'running=0' INT TERM
    while [ "${running}" -eq 1 ]; do
      ts=$(date +%s.%N)
      out=$(tc -j -s qdisc show dev "${IFACE}") || break
      printf '%s\n%s\n' "${ts}" "${out}"
      sleep 1 &
      wait $! || true
    done
  ) > "${cola_file}" &
  local tc_pid=$!

  # Conservar muestras previas al trafico
  sleep 3

  # 3. Ejecutar fases consecutivas de iperf3
  local idx=1
  for spec in "${fases[@]}"; do
    # Formato spec: nombre|tipo_transicion|tasa_mbps|duracion_s|pausa_previa_s
    IFS='|' read -r fase_nombre tipo_trans tasa dur pausa <<< "${spec}"
    local fase_tag
    fase_tag=$(printf "fase%02d" "${idx}")
    local json_name="${run_id}_${fase_tag}.json"
    local json_path="${OUT_DIR}/${json_name}"

    if [ "${pausa}" != "0" ]; then
      echo "  [${run_id} ${fase_tag}] Pausa previa de ${pausa}s..."
      sleep "${pausa}"
    fi

    local t_ini
    t_ini=$(date +%s.%N)
    echo "  [${run_id} ${fase_tag}] ${fase_nombre} (${tipo_trans}): ${tasa}M durante ${dur}s -> ${json_name}"

    iperf3 -c "${DST_IP}" -u -b "${tasa}M" -l 1200 -t "${dur}" -i 1 \
      --get-server-output -J > "${json_path}"

    local t_fin
    t_fin=$(date +%s.%N)

    echo "${run_id},${idx},${fase_nombre},${tipo_trans},${tasa},${dur},${t_ini},${t_fin},${pausa},${json_name}" >> "${CALENDARIO_CSV}"
    idx=$((idx + 1))
  done

  # Conservar muestras posteriores al trafico
  sleep 4

  # 4. Detener muestreo de cola y captura limpiamente
  kill -INT "${tc_pid}" 2>/dev/null || true
  wait "${tc_pid}" 2>/dev/null || true
  sleep 1

  kill -INT "${tcpdump_pid}" 2>/dev/null || true
  wait "${tcpdump_pid}" 2>/dev/null || true

  # Registrar estadisticas de tcpdump (incluyendo posibles descartes del capturador)
  {
    echo "=== ${run_id} ==="
    cat "${tcpdump_err}"
    echo "--- Verificacion final tc -s (${run_id}) ---"
    tc -s qdisc show dev "${IFACE}"
    echo ""
  } | tee -a "${TCPDUMP_LOG}"
  rm -f "${tcpdump_err}"

  # 5. Exportar tiempos y longitudes de trama con tshark
  tshark -r "${pcap_file}" -T fields -E header=y -E separator=, \
    -e frame.time_epoch -e frame.len > "${csv_file}" 2>/dev/null

  echo "  [${run_id}] Exportado ${csv_file} ($(wc -l < "${csv_file}") lineas)."
}

# =====================================================================
# Diseno de 6 corridas independientes (orden de adquisicion fijo):
#   Entrenamiento: c01, c02, c03
#   Validacion:    c04
#   Prueba:        c05, c06
# Cada corrida tiene >= 2 episodios de congestion separados por recuperacion,
# variando orden, tasas, duraciones, rampas graduales y saltos bruscos.
# =====================================================================

# c01 (Entrenamiento): Rampa clasica + segundo episodio en rampa mas alta
ejecutar_corrida "c01" \
  "Base|base|4|15|0" \
  "Ascenso_1|rampa|8|15|0" \
  "Sobrecarga_1|rampa|14|15|0" \
  "Recuperacion_1|recuperacion|5|15|0" \
  "Ascenso_2|rampa|9|12|0" \
  "Sobrecarga_2|rampa|15|15|0" \
  "Recuperacion_2|recuperacion|4|12|0"

# c02 (Entrenamiento): Rampa escalonada suave + salto brusco desde base
ejecutar_corrida "c02" \
  "Base|base|3|12|0" \
  "Ascenso_1|rampa|7|12|0" \
  "Ascenso_2|rampa|9.2|12|0" \
  "Sobrecarga_1|rampa|13|15|0" \
  "Recuperacion_1|recuperacion|4|15|2" \
  "Sobrecarga_2|salto_brusco|16|15|0" \
  "Recuperacion_2|recuperacion|5|12|0"

# c03 (Entrenamiento): Tres episodios mixtos (salto brusco, rampa media, rampa fuerte)
ejecutar_corrida "c03" \
  "Base|base|5|12|0" \
  "Sobrecarga_1|salto_brusco|13.5|12|0" \
  "Recuperacion_1|recuperacion|4|12|0" \
  "Ascenso_1|rampa|8.5|12|0" \
  "Sobrecarga_2|rampa|12|14|0" \
  "Recuperacion_2|recuperacion|3.5|12|0" \
  "Ascenso_2|rampa|9|10|0" \
  "Sobrecarga_3|rampa|17|12|0" \
  "Recuperacion_3|recuperacion|4|10|0"

# c04 (Validacion): Rampa + carga alta sub-umbral + salto brusco
ejecutar_corrida "c04" \
  "Base|base|4|14|0" \
  "Ascenso_1|rampa|7.5|12|0" \
  "Ascenso_2|rampa|9.2|12|0" \
  "Sobrecarga_1|rampa|14.5|15|0" \
  "Recuperacion_1|recuperacion|4.5|15|0" \
  "Estres_subumbral|subumbral|8.8|10|0" \
  "Recuperacion_2|recuperacion|5|10|0" \
  "Sobrecarga_2|salto_brusco|15|14|0" \
  "Recuperacion_3|recuperacion|4|12|0"

# c05 (Prueba 1): Rampas con carga sub-umbral intermedia para evaluar falsas alarmas
ejecutar_corrida "c05" \
  "Base|base|4|14|0" \
  "Ascenso_1|rampa|7|12|0" \
  "Ascenso_2|rampa|9.1|12|0" \
  "Sobrecarga_1|rampa|13|16|0" \
  "Recuperacion_1|recuperacion|4|14|0" \
  "Estres_subumbral|subumbral|9.2|10|2" \
  "Recuperacion_2|recuperacion|5|12|0" \
  "Ascenso_3|rampa|8.5|10|0" \
  "Sobrecarga_2|rampa|15.5|15|0" \
  "Recuperacion_3|recuperacion|4|12|0"

# c06 (Prueba 2): Dos saltos bruscos directos vs una rampa gradual (3 episodios)
ejecutar_corrida "c06" \
  "Base|base|3.5|14|0" \
  "Sobrecarga_1|salto_brusco|14|15|0" \
  "Recuperacion_1|recuperacion|4.5|15|0" \
  "Ascenso_1|rampa|8.8|14|0" \
  "Sobrecarga_2|rampa|13.5|15|0" \
  "Recuperacion_2|recuperacion|4|14|0" \
  "Sobrecarga_3|salto_brusco|16.5|14|0" \
  "Recuperacion_3|recuperacion|4|12|0"

# Retirar cola experimental al finalizar
echo ""
echo "=== Retirando cola experimental TBF en ${IFACE} ==="
tc qdisc del dev "${IFACE}" root || true
{
  echo "--- Estado restaurado de qdisc al finalizar ---"
  tc qdisc show dev "${IFACE}"
} >> "${VERIFY_LOG}"

echo "=== Experimento completo finalizado con exito ==="
