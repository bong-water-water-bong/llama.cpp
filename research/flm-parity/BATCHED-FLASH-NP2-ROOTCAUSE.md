# BATCHED FLASH NP2 — RESIDUAL BLOCKER ROOT CAUSE (2026-09-08, agent-aa1ef2)

Status of the #2153 offset-rebind matcher (commit 41628ab20): the batched
decode-split flash dispatch is CORRECT — verified against the command-program
dump (GGML_HRX_DUMP_COMMAND_PROGRAM_DIR, zaya q4nx-c43 np2 decode):

- per-stream bindings all correct: query/mask at row+stream strides, K/V at
  s*nb[3]=131072, output/q8 per (stream,row), per-dispatch partial/counter
  transients (program-27 cmd 2/3).
- stream-0 logits CPU-exact (9079/236761/107/4906 argmax order).

## Residual failure: stream-1 logits read stream-0's PREVIOUS-step rows

B step r logits == A step r-1 logits (r=0 clamped to 0). Root cause is NOT
the flash matcher and NOT the q8 alternate: the program dump shows the
attention output + consumer MUL_MAT (token_count=2, row stride 4096) all
correct on-device. The logits are produced by the CPU-side lm_head — zaya
vocab 262272 exceeds the device MUL_MAT 262144-row cap (#2117), so the
lm_head splits to CPU and consumes the final hidden [2048, 2] across the
HRX->CPU boundary. The boundary copy mis-maps row 1 for the batched tensor
(reads the previous step's row-0 slot) — the round-16e boundary-corruption
class, now on the device->CPU direction for a ne[3]/multi-row tensor.

Corroboration: GGML_HRX_CPU_OPS=MUL_MAT (force the dense mms to CPU) makes
BOTH streams garbage — mixed splits corrupt broadly (same class), so the
boundary bug dominates any mixed config.

## What fixes it (lane-level)

The HRX->CPU boundary copy for the batched final-hidden tensor (scheduler
boundary machinery), or device-side lm_head coverage for 262272 rows. Until
then the np2 end-to-end gate cannot go oracle-exact even with the correct
batched flash.

## Note on decode-correctness baselines

Single-seq device decode on this lineage is itself not oracle-correct with a
real continuation (dense qwen3 + 30B collapsed; zaya diverges tok 4) — the
#2147 decode-norm reorder fix (fix/qwen3-decode-norm) is not in
fix/hrx-ngl-init-order. Any multi-seq validation must run on a build that
includes that fix.
