#!/bin/bash
# Round-16 zaya ngl99 isolation cells (post-round-15 buft fix), zgreedy token gate.
# Usage: ./round16.sh <cell>   cell in {B,C,D,E}
cd ~/hrx-ws/amd-hrx-graph
export ROCMLIB=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
export LD_LIBRARY_PATH=$ROCMLIB:build/bin
MODEL=~/zaya-q4nx-c43.gguf
CELL=$1
echo "=== [$CELL] $(date +%H:%M:%S) ==="
case $CELL in
  B) # ngl99, F16 device weights — current open state
     timeout 500 env GGML_ZAYA_DEQUANT_F16=1 /tmp/zgreedy $MODEL 99 2>&1 | grep -E "^(tok|top5|text|model load|.*FAIL)|sched_reserve: graph splits|sched_reserve: sched copies" ;;
  C) # ngl99 F16 + flash-attn -> CPU
     timeout 500 env GGML_ZAYA_DEQUANT_F16=1 GGML_HRX_CPU_OPS=FLASH_ATTN_EXT /tmp/zgreedy $MODEL 99 2>&1 | grep -E "^(tok|top5|text|model load|.*FAIL)|sched_reserve: graph splits|sched_reserve: sched copies" ;;
  D) # ngl99 F16 + NO_FLASH_ATTN (hybrid attn KV -> CPU)
     timeout 500 env GGML_ZAYA_DEQUANT_F16=1 GGML_HRX_NO_FLASH_ATTN=1 /tmp/zgreedy $MODEL 99 2>&1 | grep -E "^(tok|top5|text|model load|.*FAIL)|sched_reserve: graph splits|sched_reserve: sched copies" ;;
  E) # ngl99, F32 dequant device weights
     timeout 500 env /tmp/zgreedy $MODEL 99 2>&1 | grep -E "^(tok|top5|text|model load|.*FAIL)|sched_reserve: graph splits|sched_reserve: sched copies" ;;
esac
echo "=== [$CELL] done $(date +%H:%M:%S) rc=$? ==="
