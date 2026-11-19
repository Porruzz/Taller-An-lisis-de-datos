#!/usr/bin/env bash
# Retira la cola experimental TBF en la interfaz de salida del emisor (A).
# Uso: sudo ./limpiar_cola_tbf.sh [IFACE]
set -euo pipefail

IFACE="${1:-eth0}"

echo "=== Retirando cola experimental en ${IFACE} ==="
tc qdisc del dev "${IFACE}" root || true

echo "=== Estado restaurado de qdisc ==="
tc qdisc show dev "${IFACE}"
