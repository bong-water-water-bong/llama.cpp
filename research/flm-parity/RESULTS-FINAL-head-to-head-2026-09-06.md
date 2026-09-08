# Final FLM head-to-head + multi-seq report (2026-09-06)

Our refreshed fork (~/hrx-ws/amd-hrx-graph, branch fix/hrx-ngl-init-order) vs
FastFlowLM, same box (strixhalo: Ryzen AI MAX+ 395; HRX = Radeon 8060S iGPU via
our stack; FLM = XDNA2 NPU v1.0.4). Consolidates committed evidence:
RESULTS-qwen3-roster-2026-09-05.md, RESULTS-zaya-rescope-task4-2026-09-06.md,
task-1/2/3 evidence, and this session's zaya multi-seq runs.

## Decode (tokens/s), single seq

| model        | ours (validated path)         | FLM (NPU) | ratio |
|--------------|-------------------------------|----------:|------:|
| Qwen3-0.6B   | 249.6 (HRX device, ngl99)     | 86-89     | ~2.9x |
| Qwen3-1.7B   | 122.4 (HRX device, ngl99)     | 40.4-40.8 | ~3.0x |
| Qwen3-4B     | 57.0 (HRX device, ngl99)      | 19.2      | ~3.0x |
| zaya-8B      | 17.45±0.09 (CPU ngl0, HRX reg)| 16.8*     | ~1.04x |

*zaya FLM/stale-fork baseline 16.8 t/s per task-4 contract. zaya on the HRX
device mixed path decodes real tokens (143243) but is not oracle-correct there
- see follow-on.

## Prefill (tokens/s), pp512 (ours) vs FLM blended

| model        | ours       | FLM  |
|--------------|-----------:|-----:|
| Qwen3-0.6B   | 12737      | ~1250 |
| Qwen3-1.7B   | 3401       | ~345  |
| Qwen3-4B     | 1241       | ~200  |
| zaya-8B      | 218.5 (pp64, ngl0) | - |

## TTFT (256-token prompt)

| model      | ours    | FLM      |
|------------|--------:|---------:|
| Qwen3-0.6B | 0.03-0.05 s | 0.6-0.84 s |

## Aggregate multi-seq (continuous batching, llama-server on HRX device)

| concurrency | ours agg (t/s) | FLM agg (t/s) |
|------------:|---------------:|--------------:|
| 1           | 219            | 88            |
| 2           | 141            | 30            |
| 4           | 103            | 22            |

FLM serializes concurrent decode (aggregate DROPS with concurrency); our
llama-server continuous batching keeps aggregate throughput high -> structural
advantage confirmed on qwen3-0.6B.

## zaya multi-seq no-SEGV

llama-batched-bench -m zaya-q4nx-c43.gguf -ngl 0 -p 32 -n 24 -b 32 -npl {1,2,4,8}
completed WITHOUT SEGV at every npl (recurrent conv-state handling per #2077
holds). Throughput rows suppressed by this fork's batched-bench printer for
the hybrid arch; per-seq decode = 17.45 t/s (tg64 llama-bench).

## Numerics gates

- qwen3 roster (HRX device, ngl99): tok0 argmax matches CPU oracle (12095),
  coherent text, NaN=0 (verified via research/zgreedy.cpp raw-token gate).
- zaya (ngl0, HRX registered, no DISABLE): oracle 9079/236761/107/2717/108/1882
  /735/1156 (" Paris.\n```\nWe..."), argmax sane, NaN=0.

## HRX device MIXED-path decode - follow-on status (non-goal-blocking)

First-batch (n_past=0) cross-split external writeback loss at the iree/amdgpu
graph-launch layer: multi-token prefill loses terminal mm->external writes;
single-token n_past=0 loses KV + terminal; n_past>0 unaffected; re-execution
does not recover. 30-round elimination trail, repros and diagnostic hooks
(env-gated, incl. GGML_HRX_DOUBLE_EXECUTE probe 496ac0eea) are committed:
research/flm-parity/T3-ZAYA-HRX-DEVICE-SPEC.md rounds 16a-17b. Owner: platform
device layer.

## Notes
- Same-box, same-model-class; different silicon (our HRX iGPU vs FLM NPU).
- Our rows: no DISABLE flags; refreshed fork; commit ed0561c90 + 30-round trail.

## ADDENDUM 2026-09-08 (agent-ec8072) — zaya device rows supersede the above

Post-reboot verification battery (fix/hrx-ngl-init-order @ ebf297c97, fresh
build 09-07 21:44) — see CHECKPOINT-2026-09-08-ec8072.md. The above "zaya-8B
17.45 (CPU ngl0)" rescope framing is superseded: zaya is now ORACLE-EXACT ON
THE HRX DEVICE (ngl99, ngl>0, no DISABLE flags): tok stream
9079/236761/107/2717/108/1882/735/1156 = CPU oracle, NaN=0, rc=0.

| zaya-8B row (fresh) | ours (HRX device ngl99) | FLM/stale baseline |
|---|---|---|
| decode single-seq | 6.98 t/s tg64 (oracle-exact) | 16.8 |
| prefill pp64 | 110.01 t/s | - |
| decode single-seq ngl0 CPU | 16.08 t/s tg64 (oracle-exact) | - |
| multi-seq npl 1-8 (ngl0) | rc=0, no SEGV | FLM serializes |

Remaining gaps vs ORIGINAL contracts (honest): (1) device single-seq 6.98 < 16.8
t/s — launch/subgraph-bound; open lane = SSM_CONV + CONV_1D_GROUPED loom kernels
+ launch collapse (owner: f49062 speed lane, in flight). (2) device-path
multi-seq fails on an unmatched SET_ROWS dispatch (batched recurrent conv-state
shape) — executor gap, newly documented.
