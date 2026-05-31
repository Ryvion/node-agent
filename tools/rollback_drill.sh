#!/usr/bin/env bash
# rollback_drill.sh — PROVE the auto-update crash-loop rollback on THIS machine.
#
# What it does:
#   1. builds a GOOD node binary (v1.0.0) and a POISON one (v99.0.0, built with
#      `-tags faultinject` so it crashes right after startup).
#   2. boots GOOD and lets it commit itself as the known-good rollback target.
#   3. swaps the on-disk binary to POISON (simulating a bad auto-update landing).
#   4. runs a supervisor loop that mimics systemd/SCM restarting a crashing
#      service. Layer-2 must detect the crash loop and revert the on-disk binary
#      back to GOOD. PASS == the binary's --version is 1.0.0 again.
#
# Safe + self-contained: isolated RYV_DATA_DIR, unreachable hub, no root, no real
# service, no network, no GitHub. The faultinject crash is NEVER compiled into a
# production binary (build tag), so this cannot affect a real fleet.
#
# Usage:  ./tools/rollback_drill.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # ryvion-node repo root

DRILL="$(mktemp -d 2>/dev/null || mktemp -d -t ryv-rollback-drill)"
trap 'rm -rf "$DRILL"' EXIT
export RYV_DATA_DIR="$DRILL/data"
export RYV_HUB_URL="http://127.0.0.1:9"        # unreachable -> registration fails, grace commits
export RYV_BOOT_HEALTHY_GRACE_SECONDS=2        # commit fast for the drill (prod default 180s)
export RYV_FAULT_CRASH_VERSION=99.0.0          # only the poison build (v99) honors this
mkdir -p "$RYV_DATA_DIR"
BIN="$DRILL/ryvion-node"
STATE="$RYV_DATA_DIR/update_state.json"

run_capped() {  # run_capped <seconds> <cmd...> — portable timeout (no coreutils dep)
  local secs="$1"; shift
  "$@" >/dev/null 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1; waited=$((waited + 1))
    [ "$waited" -ge "$secs" ] && { kill "$pid" 2>/dev/null; break; }
  done
  wait "$pid" 2>/dev/null
}

echo "== building GOOD (1.0.0) and POISON (99.0.0, faultinject) =="
go build -o "$DRILL/good" -ldflags "-X main.version=1.0.0" ./cmd/ryvion-node || { echo "FAIL: build good"; exit 1; }
go build -tags faultinject -o "$DRILL/poison" -ldflags "-X main.version=99.0.0" ./cmd/ryvion-node || { echo "FAIL: build poison"; exit 1; }

echo "== boot GOOD; let it commit known-good (.prev) =="
cp "$DRILL/good" "$BIN"
"$BIN" --hub "$RYV_HUB_URL" --ui-port 0 >/dev/null 2>&1 &
GOOD_PID=$!
sleep 5
kill "$GOOD_PID" 2>/dev/null
wait "$GOOD_PID" 2>/dev/null   # fully dead before we overwrite the file (avoid corrupting it)
if ! grep -q '"committed":true' "$STATE" 2>/dev/null; then
  echo "FAIL: good binary never committed as known-good"
  echo "  state: $(cat "$STATE" 2>/dev/null || echo '<none>')"
  exit 1
fi
echo "  ledger: $(cat "$STATE")"

echo "== simulate a bad auto-update: swap POISON onto the on-disk binary =="
cp "$DRILL/poison" "$BIN"
echo "  on-disk version now: $("$BIN" --version 2>/dev/null)"

echo "== supervisor loop (mimics systemd) — Layer-2 should revert by boot 3 =="
for i in 1 2 3 4 5 6; do
  ver="$("$BIN" --version 2>/dev/null)"
  echo "  boot $i: on-disk = $ver"
  case "$ver" in
    *1.0.0*)
      echo
      echo "  PASS ✓  crash loop detected and the binary auto-rolled-back to the good version."
      echo "  final ledger: $(cat "$STATE" 2>/dev/null)"
      exit 0 ;;
  esac
  run_capped 15 "$BIN" --hub "$RYV_HUB_URL" --ui-port 0   # poison crashes; boot 3 reverts first
done

echo
echo "  FAIL ✗  binary did not roll back within 6 boots."
echo "  final ledger: $(cat "$STATE" 2>/dev/null)"
exit 1
