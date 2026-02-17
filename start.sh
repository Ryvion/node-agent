#!/bin/sh
set -e

echo "Starting Ryvion Node..."
echo "Hub URL: ${RYV_HUB_URL:-https://ryvion-hub.fly.dev}"
echo "Device Type: ${RYV_DEVICE_TYPE:-cpu}"
echo "GPUs: ${RYV_GPUS:-auto}"

exec /usr/local/bin/ryvion-node \
    -hub "${RYV_HUB_URL:-https://ryvion-hub.fly.dev}" \
    -type "${RYV_DEVICE_TYPE:-cpu}" \
    -gpus "${RYV_GPUS:-auto}"
