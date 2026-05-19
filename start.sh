#!/bin/sh
set -e

echo "Starting Ryvion Node..."
echo "Hub URL: ${RYV_HUB_URL:-https://api.ryvion.ai}"
DEVICE="${RYV_DEVICE:-${RYV_DEVICE_TYPE:-cpu}}"
echo "Device: ${DEVICE}"
echo "GPUs: ${RYV_GPUS:-auto}"

exec /usr/local/bin/ryvion-node \
    -hub "${RYV_HUB_URL:-https://api.ryvion.ai}" \
    -device "${DEVICE}" \
    -gpus "${RYV_GPUS:-auto}"
