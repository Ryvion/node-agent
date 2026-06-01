#!/usr/bin/env bash
# run_em_node.sh — start an EM-capable node pointed at production, OS-aware:
#   * Linux  -> gprMax engine (gprmax-v1 bundle), GPU lane (NVIDIA CUDA).
#   * macOS  -> Meep  engine (meep-v1  bundle), opt-in CPU lane (Apple Silicon).
# The native EM bundle is SELF-CONTAINED (its own python + engine), so nothing
# needs to be preinstalled — no conda env, no Docker. One short command to run it:
#     cd ~/ryvion-node && git pull && bash tools/run_em_node.sh
set -u
export PATH="$PATH:/usr/local/go/bin"
cd "$(dirname "$0")/.." || exit 1   # ryvion-node repo root

OS=$(uname -s)
case "$OS" in
  Darwin)
    ENGINE=meep ; BUNDLE_KEY=meep-v1
    # Meep is a CPU FDTD solver on Apple Silicon (no CUDA) -> opt-in CPU lane.
    export RYV_EM_ALLOW_CPU=1
    # Use a DISTINCT node identity so the EM node coexists with an inference node
    # already running on this Mac (override by exporting RYV_KEY_PATH yourself).
    export RYV_KEY_PATH="${RYV_KEY_PATH:-$HOME/.ryvion/em-node-key}"
    ;;
  *)
    ENGINE=gprmax ; BUNDLE_KEY=gprmax-v1
    ;;
esac

echo "[run_em_node] $OS -> engine=$ENGINE bundle=$BUNDLE_KEY building node binary..."
go build -o /tmp/node ./cmd/ryvion-node || { echo "build failed"; exit 1; }

# reclaim any prewarmed GGUFs. Keep the EM bundle so restarts don't re-download;
# set RYV_EM_REFRESH_BUNDLE=1 to force a fresh pull.
rm -rf "$HOME"/.ryvion/models/* 2>/dev/null || true
if [ "${RYV_EM_REFRESH_BUNDLE:-0}" = "1" ]; then
  rm -rf "$HOME"/.ryvion/runtimes/em/"$BUNDLE_KEY" 2>/dev/null || true
fi

export RYV_DISABLE_MODEL_PREWARM=1     # do NOT download chat models
export RYV_DISABLE_AUTO_UPDATE=1       # stay on this EM build (local build reports "dev")
export RYV_EM_ALLOW_UNSIGNED_BUNDLE=1  # accept the self-hosted bundle for this test
export RYV_EM_BUNDLE_BASE_URL=https://huggingface.co/datasets/Danfir/ryvion-em-bundle/resolve/main

echo "[run_em_node] starting node -> https://api.ryvion.ai (Ctrl+C to stop)"
exec /tmp/node --hub https://api.ryvion.ai --type gpu --ui-port "${RYV_EM_UI_PORT:-45899}"
