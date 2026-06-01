#!/usr/bin/env bash
# run_em_node.sh — start the EM-capable node pointed at production, with the
# gprMax bundle auto-pull wired and chat-model prewarm + auto-update disabled.
# One short command to run it (avoids long-paste mangling):
#     cd ~/ryvion-node && git pull && bash tools/run_em_node.sh
set -u
export PATH="$PATH:/usr/local/go/bin"
cd "$(dirname "$0")/.." || exit 1   # ryvion-node repo root

echo "[run_em_node] building EM node binary..."
go build -o /tmp/node ./cmd/ryvion-node || { echo "build failed"; exit 1; }

# reclaim any prewarmed GGUFs. Keep the EM bundle (gprmax-v1) so restarts don't
# re-download 2.8GB; set RYV_EM_REFRESH_BUNDLE=1 to force a fresh pull.
rm -rf "$HOME"/.ryvion/models/* 2>/dev/null || true
if [ "${RYV_EM_REFRESH_BUNDLE:-0}" = "1" ]; then
  rm -rf "$HOME"/.ryvion/runtimes/em/gprmax-v1 2>/dev/null || true
fi

export RYV_DISABLE_MODEL_PREWARM=1     # do NOT download chat models
export RYV_DISABLE_AUTO_UPDATE=1       # stay on this EM build (local build reports "dev")
export RYV_EM_ALLOW_UNSIGNED_BUNDLE=1  # accept the self-hosted bundle for this test
export RYV_EM_BUNDLE_BASE_URL=https://huggingface.co/datasets/Danfir/ryvion-em-bundle/resolve/main

echo "[run_em_node] starting node -> https://api.ryvion.ai (Ctrl+C to stop)"
exec /tmp/node --hub https://api.ryvion.ai --type gpu --ui-port 45899
