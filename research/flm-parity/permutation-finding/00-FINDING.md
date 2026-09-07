# FINDING (agent-f49062, 2026-09-07): ffn mm outputs = value-preserving PERMUTATIONS in the program-dump readback; compute likely CORRECT

Full writeup in research/flm-parity/CHECKPOINT-2026-09-07-f49062.md + this dir.
Headline: the round-64..70 "gate_up mm per-partition compute genuinely wrong for t1-5"
conclusion is NOT supported by the 40-block cross-check. Every (block, token) gate_up /
swiglu / down / weighted / moe_out row = a near-exact PERMUTATION of its correct values
(sorted corr ~0.99-1.000 in ~all 240 cells; in-order mad 0.4-3.3 + dc~0 = the old
"genuinely wrong" signature). Deterministic run-to-run (bit-identical, 8 blocks tested).

## Constraints (measured)
- Perm per (block, expert-row) independent, global (no tile locality), value-exact.
- Downstream value-exactness: block b+1 rows match oracle VALUES => consumed ffn outputs
  were in-order correct => the mm COMPUTES CORRECTLY; scramble lives in the readback.
- pddbg binding inventory (block 0): gate_up mm (uid 10866 cmd2) writes arena
  0x...7f90 @2621440 len 98304 (b4). The swiglu/down program (uid 10868 cmd2) reads its
  input from a DIFFERENT buffer 0x...3230 len 49152 (swiglu out = down in). The GLU
  itself is not in either binding list => GLU runs elsewhere (CPU-side buffer_get path,
  round-69 extent-convention fix = in-order) => consumers see in-order data while
  GGML_HRX_PROGRAM_DUMP d2h (hrx_synchronous_d2h) sees scrambled data for the same
  regions. => Suspect: hrx_synchronous_d2h readback path scrambles deterministically
  per (buffer, offset) region (the rounds-16a-17b "writeback/canary" defect class,
  wrongly parked in rounds 63-70 as "compute").
- Output arena = 2-deep ring (odd blocks 2752512, even>=2 2883584, block 0 unique
  2621440) => dump-time aliasing only affects same-slot pairs.
- Real (lossy) value corruption exists ONLY at blocks 38-39 (sorted corr 0.65-0.99,
  t0 mad 0.49/0.73 dc 0.99/0.92) = directly upstream of the CPU lm-head. Prime suspect
  for the tok0=563-vs-9079 flip once the readback scramble is excluded.

## Analysis scripts (this dir)
cmp_gu.py / cmp2.py = 40-block per-token mad/corr/sorted-corr + routing decode.
det_test.py = run1-vs-run2 determinism test.
