#!/bin/bash
# Rebuild the flm-parity probe tools after a reboot (/tmp = tmpfs, wiped).
# Usage: ./rebuild-probes.sh   (run from ~/hrx-ws/amd-hrx-graph)
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
export ROCMLIB=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
export LD_LIBRARY_PATH=$ROCMLIB:$ROOT/build/bin
set -e
echo "== rebuilding zgreedy =="
g++ -std=c++17 -O2 -I "$ROOT/ggml/include" -I "$ROOT/include" "$ROOT/research/zgreedy.cpp" \
    -L "$ROOT/build/bin" -Wl,-rpath,"$ROOT/build/bin" -lllama -lggml -lggml-base \
    -lggml-cpu -lpthread -o /tmp/zgreedy
echo "== done: /tmp/zgreedy =="
echo "Sanity (CPU oracle, expect 9079/236761/107/2717/108/1882/735/1156):"
env GGML_HRX_DISABLE=1 /tmp/zgreedy ~/zaya-q4nx-c43.gguf 0 2>&1 | grep -E "^tok[0-7]=" || true
