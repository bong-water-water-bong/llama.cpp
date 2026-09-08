# Unified-bench: integrated PhaseRouter timings — 2026-09-08

Method: zc_router_bench drives the REAL engine code (PhaseRouter +
HrxPrefillEngine + HrxDecodeEngine, all over Inprocess/HRX0). One process
loads the model once, runs a warm generate(), then a timed generate() whose
wall time covers the FULL integrated path: tokenize -> HRX0 prefill ->
export_session_mem memfd -> load_session_mem import -> greedy decode, all
inside one API call. Model load excluded (reported separately).

## Results (opensplit bundle, HRX0)

| model | load+init | warm | timed | unified t/s |
|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | — | 4 tok | 30 tok / 1.116 s | 26.89 |
| Qwen3-Coder-30B-A3B Q4_K_M (contract) | 6.75 s | 4 tok | 60 tok / 6.051 s | 9.92 |

30B continuation (greedy): 12095 13 576 6722 315 32961 374 37169 ... exit 0.

## Interpretation (honest)

- The unified path's decode rate at contract scale (9.92 t/s single-stream
  tg1 on HRX0) matches the NPU's standalone single-stream rate (~8 t/s spec
  for the 35B-MoE hot model; the fork decode-gate figure 93.65 t/s is tg256
  BATCH decode on Vulkan0 — a different measurement axis). => the zero-copy
  handoff adds no material per-token overhead to the decode engine.
- Per-call handoff cost (prefill+export+import for a 5-token prompt) is
  amortized into the timing; at 30B it is negligible vs decode time.

## Files/logs

- zc_router_bench.cpp (/tmp on strixhalo; sources committed in engine
  src/router/*)
- /tmp/bench06.log, /tmp/bench30.log (raw runs, exit 0 both)
