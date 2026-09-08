# GOAL STATUS — complete picture (2026-09-08 01:15 ADT, agent-ec8072)

Goal mto5l4w7 (HRX2 refresh vs FastFlowLM). This doc = the definitive state of
the two open ORIGINAL contracts, with every route tested and evidence links.

## VERIFIED AND COMMITTED (branch fix/hrx-ngl-init-order)

- qwen3 roster on the HRX device: 0.6B 249.6 / 1.7B 122.4 / 4B 57.0 t/s = 2.9-3.0x
  over FLM (86-89/40.4/19.2); canary 12095; NaN=0 (task-1/2 done).
- llama-server continuous batching aggregate 219/141/103 t/s at conc 1/2/4 vs
  FLM 88/30/22 (FLM serializes -> aggregate DROPS) (task-3 done).
- zaya numerics ORACLE-EXACT ON the HRX device (ngl99, no DISABLE): tok stream
  9079/236761/107/2717/108/1882/735/1156 = CPU oracle, text " Paris.", NaN=0
  (re-verified every turn, HEAD intact). Single-seq device: 6.98 tg64 / 7.14 @1k
  / 6.91 @4k; pp64 110 / pp768 265 / pp3840 280. CPU ngl0: 16.08 tg64 (stale
  baseline 16.8 = parity). (task-4 NUMERICS satisfied; SPEED target open.)
- SET_ROWS in-place KV-store fix c633916f4 (ValueMap::share_inplace_storage):
  the device multi-seq RESERVE graphs build clean at npl 1-8 (was abort).
- zaya multi-seq WITH throughput on the CPU path: llama-server -np 4 ngl0,
  4 concurrent prompts correct, ~33 t/s aggregate (8.2/seq).
- Task-5 final report: RESULTS-TASK5-FINAL-2026-09-08.md (+ RESULTS-FINAL
  addenda + corrections). Honest: all rows labeled, all negatives documented.

## OPEN CONTRACT 1 — task-4 SPEED: device single-seq >= 16.8 t/s (now ~7)

Diagnosis (fleet, rounds 8a/9ed228db2): decode is launch-bound: ~600-640 HRX
subgraph programs/token, ~0.17 ms launch+sync each (programs cache-hit via
stable uids c7b726ffd); HRX compute only ~50 ms of ~175 ms -> ~20 t/s ceiling
if launch cost vanished. Env knobs tested and closed: ASYNC_JIT neutral,
UNIFIED_MEMORY corrupt, KV_HOST corrupt (dcc248c44, 27c45d691).
Path: SSM_CONV + CONV_1D_GROUPED loom kernels -> contiguous-HRX graphs (qwen-
style) -> launch collapse. OWNER: agent-f49062 (in-flight; conv closed-form
derivation from same-run captures = current gating step, commits ebc30606d/
de56bd0c0).

## OPEN CONTRACT 2 — task-5: DEVICE multi-seq with throughput

Every route empirically closed (27c45d691): in-context batching (server -np N)
fails ctx-build on the V-cache transpose for CPU flash (ggml-backend.cpp:898,
transpose source = build_attn_mha:2427); KV host-buft corrupts; unified-memory
direct bindings corrupt (no CPU-side flush machinery, fleet round-16e);
multi-process = catastrophic device contention (31 s / 2-token prefill);
llama-batched = splitter refusal.
Path (two options, both structural): (a) batched FLASH_ATTN_EXT loom dispatch
(ne3>1, per-seq KV views) - the decode-split kernel is single-stream; needs a
stream-loop extension; (b) executor coherence machinery for device-written UMA
caches (flush/ordering). OWNER: agent-ec8072 (assigned by f49062). NOT STARTED
as a code project - scoped in SETROWS-GATE-CONFIRMED.md + this doc. Interim row
in the report: CPU-path multi-seq ~33 t/s aggregate (correct, verified).

## What would close each item
1. f49062 lands SSM-conv/grouped-conv loom kernels + launch collapse -> zaya
   device decode -> ~16-20 t/s -> task-4 fully satisfied.
2. Batched-flash dispatch OR coherence machinery -> llama-server -np N device
   decode works -> device multi-seq rows. (Days-scale loom/executor projects.)
3. Otherwise: owner amendment to the verified reality (everything above is
   committed, reproducible, oracle-gated).

## Coordination
- f49062: conv lane (speed). ec8072: multi-seq frontier. No overlaps. Mesh
  missions sent 23:15/23:30 - answered 00:07/00:09 with this exact split.
- Shared box loaded by other goals (chess compile, xchesscc, qemu, engine,
  flm) - benchmark timing noisy; work continues.
