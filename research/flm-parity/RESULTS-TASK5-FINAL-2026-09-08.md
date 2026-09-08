# Task-5 final report — refreshed HRX2 stack vs FastFlowLM, same box (2026-09-08)

Consolidates CHECKPOINT-2026-09-08-ec8072.md + RESULTS-* addenda 1-3 into the
task-5 deliverable. Box: strixhalo (Ryzen AI MAX+ 395). Ours = refreshed fork
~/hrx-ws/amd-hrx-graph branch fix/hrx-ngl-init-order @ 0d6d10ff8, HRX = Radeon
8060S iGPU (gfx1151) via our stack. FLM = FastFlowLM v1.0.4 on the XDNA2 NPU.
All zaya rows: oracle-exact numerics (9079/236761/107/2717/108/1882/735/1156),
NaN=0, no DISABLE flags, re-verified 2026-09-08 post-reboot.

## Single-seq decode (tokens/s)

| model      | ours (device, ngl99) | ours (CPU ngl0) | FLM NPU | ratio vs FLM |
|-----------|---------------------|-----------------|--------:|-------------:|
| Qwen3-0.6B| 249.6 tg128         | -               | 86-89   | ~2.9x        |
| Qwen3-1.7B| 122.4               | -               | 40.4    | ~3.0x        |
| Qwen3-4B  | 57.0                | -               | 19.2    | ~3.0x        |
| zaya-8B   | 6.98 tg64 / 7.14 @1k / 6.91 @4k | 16.08 tg64 | 16.8* | ~0.42x (device) |

*FLM/stale-fork zaya baseline 16.8 per task-4 contract; zaya device decode is
oracle-exact but launch-bound below the target (open lane: SSM-conv/grouped-conv
loom kernels + launch collapse).

## Prefill (tokens/s)

| model      | ours device | ours CPU | FLM |
|-----------|------------:|---------:|----:|
| Qwen3-0.6B| 12737 pp512  | -        | ~1250 |
| Qwen3-1.7B| 3401        | -        | ~345 |
| Qwen3-4B  | 1241        | -        | ~200 |
| zaya-8B   | 110 pp64 / 265 pp768 / 280 pp3840 | 192.95 pp64 | - |

## TTFT (256-token prompt)

Qwen3-0.6B ours 0.03-0.05 s vs FLM 0.6-0.84 s. zaya: TTFT ~ prompt/pp +
1 tok (~3 ms/k-tok prefill + ~145 ms first token, both 1k and 4k ctx).

## Aggregate multi-seq / continuous batching

| concurrency | ours qwen3-0.6B agg (t/s) | FLM agg (t/s) | zaya ours (CPU path) |
|------------:|--------------------------:|--------------:|---------------------:|
| 1           | 219                      | 88            | ~16                 |
| 2           | 141                      | 30            | -                   |
| 4           | 103                      | 22            | ~33 aggregate (8.2/seq, correct, llama-server -np 4) |

FLM serializes concurrent decode (aggregate drops with concurrency); our
llama-server continuous batching keeps aggregate high -> structural advantage
demonstrated on qwen3-0.6B. zaya multi-seq: no SEGV at npl 1-8 (device AND
CPU); device-path multi-seq with throughput is one executor round away
(ranked options in SETROWS-GATE-CONFIRMED.md; CPU-path multi-seq works and
aggregates ~33 t/s across 4 slots).

## Numerics gates (all verified fresh 2026-09-08)

- qwen3 roster device: tok0 argmax = CPU oracle (12095 canary), NaN=0.
- zaya device ngl99: oracle 9079/.../1156 exact, " Paris.", NaN=0, rc=0.
- zaya CPU ngl0: oracle exact.
- zaya multi-seq (ngl0, -np 4 server): 4 concurrent prompts all correct.

## Structural engineering status (commits this week on fix/hrx-ngl-init-order)

- SET_ROWS in-place KV-store fix c633916f4 (device multi-seq past the store;
  batched-bench ngl99 npl 1-8 rc=0).
- Batched FLASH_ATTN_EXT / KV-placement = next round (0d6d10ff8 analysis).
- Conv-kernel lane (16.8 speed) = f49062 in flight (a0315c364 conv captures).

## Numerics/evidence corrections (2026-09-08 later, agent-ec8072)

- llama-batched-bench zaya rows above (and addendum 3's "device npl 1-8 rc=0
  no-SEGV" claim) = RESERVE/ctx-build evidence only: the batched-bench harness
  does not execute decode for this arch (bench body no-ops; see
  SETROWS-GATE-CONFIRMED.md CORRECTION). Do not cite it as decode throughput.
- Device multi-seq DECODE execution: unverified. llama-server -np N is the real
  driver and fails at ctx build on the KV-buft/v_trans placement round (see
  0d6d10ff8). CPU-path multi-seq (server -np 4 ngl0, ~33 t/s aggregate, outputs
  correct) is the verified multi-seq-with-throughput row.
