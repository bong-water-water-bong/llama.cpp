# CORRECTION (agent-f49062, 2026-09-07): the "value-preserving permutation" reframe is RETRACTED

00-FINDING.md / 01-LOGITS.md / 03-ERROR-STRUCTURE.md claimed the gate_up mm outputs in
bad rows were "value-preserving deterministic permutations of the correct values"
(sorted-correlation ~0.99+). That claim is WRONG - a statistical artifact. This file
supersedes those conclusions. The round-63..70 verdict ("mm1 per-partition compute
genuinely wrong for t1-5") is VINDICATED, with new fragment-level proof and a live
instrument for the continuation.

## Why the reframe was wrong
Sorted(multiset) correlation is ~1.0 for ANY two independent draws from the same
marginal distribution once n is large. Control measurement (run1 dumps + oracle):
sorted-corr between DIFFERENT rows of the same block = 0.928-0.9999 (oracle rows vs
oracle rows!). The gate_up channel values are near-iid per row, so sorted-corr can
never discriminate "same values permuted" from "genuinely different values". Every
permutation-structure test that followed (per-row perms, tile maps, rank matching) was
matching noise.

## What the in-kernel publish trace proves (NEW, solid)
Instrument added to mul_mat_id_f32_f32_wmma_core.loom (this tree, working WIP + 2 small
blocks marked PUBLISH-TRACE f49062): lane-0 of each (subgroup, fragment 00/10) records
the %values[0] it is ABOUT TO publish, into row_debug at byte 2048 +
(a*64+t)*4+s*2+f (f32), sampled at channels ch = t*64+s*32+f*16. Run:
GGML_HRX_PROGRAM_DUMP=row_debug, dumps = <uid>_001_common.moe_routing.row_debug.bin
(uid = 10866+27*block for the gate_up mm program).

Results (all 40 blocks x 6 rows x 64 tiles x 4 samples):
- GOOD rows (block 0-2,4,5,8,9,11-18,... row a0): 251-256/256 samples == oracle[ch]
  within f16 noise. The mma accumulators, fragment store, stage read and publish
  addressing are ALL CORRECT for those rows - end to end.
- BAD rows (all other (block, row) cells): 0-19/256 aligned - the value at the publish
  point is already wrong, UNIFORMLY across every channel, tile, subgroup and fragment
  of the row (no per-(sg,frag) or per-tile structure; all-or-nothing per row).
- Therefore the corruption = in the per-partition compute/fetch (mma input path) for
  the affected (block, partition) cells, NOT in the write path and NOT in the readback
  (both read paths see the same genuinely-wrong buffer content).

## What remains true from the earlier docs
- The per-block x per-token cross-check data + scripts (permutation-finding/*.py) are
  still valid as raw measurements; only the "permutation" interpretation is wrong.
- The 38-39 logits collapse (9079: 17.47 -> -0.28) = real (zlogits.cpp).
- Both read paths (PROGRAM_DUMP d2h and strided buffer_get) agree at any instant.
- The downstream "dilution" (b3/b6 moe_out ~exact despite 77% bad gate_up) = explained
  by the small value scales at those blocks (b3 gate rms 0.24 -> silu/down errors
  diluted below 0.1 by matmul averaging), NOT by any clean-copy absorption.

## Where the bug now stands (sharpest formulation)
For the 6-token prefill, the gate_up mm computes CORRECT rows for partition ordinal 0
in ~25 blocks and WRONG rows elsewhere: wrong = every channel of the row, uniformly,
at magnitudes comparable to the correct values. The decode steps (1 token) produce
grammatical text, so the corruption is tied to the multi-token (partition > 0 or
specific data) configuration. Next probes:
(a) decode the WIP's existing in-kernel captures in row_debug (rd_fview weights at
    f32[32 + e*256 + ch*32 + pkt*4], rd_act activations at f32[8192 + token*32 +
    pkt*4] - see the WIP loom code) and compare the fetched weight/activation values
    against the model file + oracle inputs to isolate the wrong fetch;
(b) compare the wrong rows' values against W[E_p] x x[t] recomputed for WRONG expert
    pairings (needs the gguf weights + oracle activations in numpy - doable from
    ~/zaya-q4nx-c43.gguf).
