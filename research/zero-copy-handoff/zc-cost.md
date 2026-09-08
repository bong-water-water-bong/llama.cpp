# zero-copy handoff — measured cost (rt_zc_time) — 2026-09-08

Goal mtsy05dx task zero-copy-handoff — cost measurement leg.

## Method

rt_zc_time.cpp: prefill N random tokens on CPU (q35 build, hermetic), then
time 3 reps of:
- F: llama_state_save_file + llama_state_load_file (today's D2 file path)
- M: llama_state_get_data -> memfd MAP_SHARED -> llama_state_set_data
     (zero-copy shared-memory path, no file I/O)
plus 5 decode steps for the per-token compute reference.

## Results

| model | state | F file total | M memfd total | decode/step | M vs decode |
|---|---|---|---|---|---|
| qwen3-0.6B Q4_K_M | 29.4 MB | 20.9 ms | 17.4 ms | 7.8 ms | 2.2x/step |
| Qwen3-Coder-30B-A3B Q4_K_M | 50.3 MB | 26.1 ms | 19.3 ms | 38.9 ms (CPU) | 0.50x/step |

## Reading

- M (shared memory) is faster than F (file) on both models (17% / 26%) and
  performs zero filesystem I/O (no write, no read, no fsync, no temp file).
- Handoff is a ONE-TIME per-sequence cost. Compute phase = the full
  continuation (>= 500 tokens in the target workload): at 30B HRX decode
  (~10 ms/tok, 93-95 t/s) that is >= 5 s vs a 19 ms handoff => handoff is
  ~0.4% of the compute phase, far under the "<<" bar. Even at CPU decode
  (38.9 ms/tok) the 19 ms handoff is half of one decode step.
- On the real NPU->HRX flow the prefill runs on HIP/NPU (~660 tok/s) and the
  state is 292 MB at 2962 tokens; extrapolating linearly the memfd path is
  ~110 ms vs the file path's ~150 ms + disk, still << a 500-token HRX decode
  phase (~5-13 s).

## Artifacts

- research/zero-copy-handoff/rt_zc_time.cpp
- logs /tmp/zc_time30.log
