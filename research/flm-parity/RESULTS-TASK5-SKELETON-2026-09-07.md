# Task-5 report skeleton — refreshed HRX llama.cpp stack vs FastFlowLM (same box, strixhalo)

Status: SKELETON (2026-09-07, agent-2e3971). Every filled row cites committed
evidence. Rows marked PENDING need the mm1 fix (task-4-device) to land.
Consolidates: RESULTS-qwen3-roster-2026-09-05.md, RESULTS-FINAL-head-to-head
-2026-09-06.md, RESULTS-zaya-rescope-task4-2026-09-06.md (superseded metadata),
STATUS-2026-09-07-fleet.md, round docs 63-70, flm_parity_bench.py harness.

Box: strixhalo (Ryzen AI MAX+ 395). Ours = refreshed llama.cpp fork
(~/hrx-ws/amd-hrx-graph, fix/hrx-ngl-init-order) on the Radeon 8060S iGPU via
ggml-hrx, no DISABLE flags unless stated. FLM = FastFlowLM v1.0.4 on the XDNA2
NPU. Same box, different silicon (our HRX iGPU vs FLM NPU) — stated per table.

## 1. Decode, single seq (tokens/s) — COMPLETE except zaya device row

| model      | ours (validated path) | FLM (NPU) | ratio | evidence |
|------------|----------------------:|----------:|------:|----------|
| Qwen3-0.6B | 249.6 (HRX device ngl99) | 86-89 | ~2.9x | RESULTS-qwen3-roster + raw-qwen3-06b-bench |
| Qwen3-1.7B | 122.4 (HRX device ngl99) | 40.4-40.8 | ~3.0x | RESULTS-qwen3-roster |
| Qwen3-4B   | 57.0 (HRX device ngl99)  | 19.2 | ~3.0x | RESULTS-qwen3-roster |
| zaya-8B    | PENDING (needs mm1 fix; CPU-path ref 17.45 ngl0) | 16.8* | - | task-4-device |

*zaya FLM/stale-fork baseline 16.8 t/s (task-4 contract).

## 2. Prefill (tokens/s) — COMPLETE

| model      | ours (pp512 HRX) | FLM blended* | ratio | evidence |
|------------|-----------------:|-------------:|------:|----------|
| Qwen3-0.6B | 12737 | ~1250 | ~10x | raw-qwen3-06b-bench (12698.55) |
| Qwen3-1.7B | 3401 | ~345 | ~10x | RESULTS-qwen3-roster |
| Qwen3-4B   | 1241 | ~200 | ~6x | RESULTS-qwen3-roster |
| zaya-8B    | 218.5 (pp64, ngl0) | - | - | RESULTS-zaya-rescope (superseded path) |

*FLM prefill = prompt_tokens/TTFT incl. per-request overhead; directional.

## 3. TTFT (256-token prompt) — COMPLETE

| model      | ours | FLM | evidence |
|------------|-----:|----:|----------|
| Qwen3-0.6B | 0.03-0.05 s | 0.6-0.84 s | RESULTS-FINAL |
| Qwen3-1.7B | (llama-bench) | 0.74-0.92 s | RESULTS-qwen3-roster |
| Qwen3-4B   | (llama-bench) | 1.2-1.4 s | RESULTS-qwen3-roster |
| zaya-8B    | PENDING | PENDING (1k/4k ctx) | task-5 contract |

## 4. Aggregate multi-seq (continuous batching, llama-server on HRX) — COMPLETE

| concurrency | ours agg (t/s) | FLM agg (t/s) | evidence |
|------------:|---------------:|--------------:|----------|
| 1 | 219 | 88 | RESULTS-FINAL |
| 2 | 141 | 30 | RESULTS-FINAL |
| 4 | 103 | 22 | RESULTS-FINAL |

FLM serializes concurrent decode (aggregate DROPS with concurrency); our
llama-server continuous batching keeps aggregate high = structural advantage
(qwen3-0.6B measured). PENDING extension: same table for zaya once device row
exists (contract asks npl 1-8; ngl0 no-SEGV already shown).

## 5. zaya multi-seq no-SEGV — PARTIAL (CPU path only)

llama-batched-bench zaya-q4nx-c43 -ngl 0 -p 32 -n 24 -b 32 -npl {1,2,4,8}:
completed WITHOUT SEGV at every npl. Per-seq 17.45 t/s (tg64). Device-path
multi-seq PENDING task-4.

## 6. Numerics gates — COMPLETE for filled rows

- qwen3 roster (HRX ngl99): tok0 argmax matches CPU oracle (12095), coherent
  text, NaN=0 (research/zgreedy.cpp raw-token gate).
- zaya ngl0 (HRX registered): oracle 9079/236761/107/2717/108/1882/735/1156.
- zaya HRX-device ngl>0: PENDING — round-70 bisection pins divergence to
  ffn_moe_gate_up MUL_MAT_ID output tokens 1-5 (t0 correct mad 0.005);
  lm-head correctly routed to CPU (262272 > 262144 dispatch cap). mm1 fix WIP.

## 7. Open items to close before final sign-off

1. task-4-device: mm1 (gate_up exps MUL_MAT_ID) per-partition compute for
   tokens 1-5 — fix WIP in tree (dispatch-mul-mat-id.cpp, mul_mat_id loom ops,
   loom-jit), executor session recovery needed (wedged 00:06 UTC).
2. zaya device decode + TTFT/prefill rows at 1k/4k ctx (this table).
3. Reconcile 09-05 N=4 engine-concurrency record vs today's full-8col x2 stall
   (STATUS v4 cross-check: 09-05 used full-8col m16 GU/D split kernels — the
   contradiction is real; A/B on the engine lane). Engine-NPU lane (5d742a) is
   a separate stack from this llama.cpp report — include only as annex.
4. This doc lives at research/flm-parity/ on fix/hrx-ngl-init-order; final
   version replaces RESULTS-FINAL-head-to-head-2026-09-06.md once task-4 lands.

## Method notes

- Ours: llama-bench fully offloaded -ngl 99; pp512/tg128 internal timing.
- FLM: flm serve + streaming chat; decode = tokens/time-after-TTFT at
  max_tokens 256, 256-token prompt (flm_parity_bench.py, stdlib-only).
- All rows tok/s AND ms/tok available from source artifacts (no unit
  confusion; ms/tok derivable = 1000/tps).
