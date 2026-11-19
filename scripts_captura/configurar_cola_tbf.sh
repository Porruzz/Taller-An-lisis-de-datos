#!/usr/bin/env bash
# Configura la cola experimental TBF (10 Mbit/s, burst 16kb, limit 32kb)
# en la interfaz de salida del computador emisor (A).
# Uso: sudo ./configurar_cola_tbf.sh [IFACE]
set -euo pipefail

IFACE="${1:-eth0}"

echo "=== Verificacion de interfaz: ${IFACE} ==="
ip -br addr show dev "${IFACE}"

echo "=== Configuracion original de qdisc ==="
tc qdisc show dev "${IFACE}"

echo "=== Aplicando cola TBF (rate 10mbit, burst 16kb, limit 32kb) ==="
tc qdisc replace dev "${IFACE}" root handle 10: tbf \
    rate 10mbit burst 16kb limit 32kb

echo "=== Verificacion textual (tc -s) ==="
tc -s qdisc show dev "${IFACE}"

echo "=== Verificacion JSON (tc -j -s) ==="
tc -j -s qdisc show dev "${IFACE}"
