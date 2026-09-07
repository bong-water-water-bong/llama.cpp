# Writeback-loss revival: the prefill = first-execution corruption (agent-f49062)

Date: 2026-09-07. Follows addenda 1-6. This is the synthesis that ties the whole hunt
back to the ORIGINAL task-4 blocker wording ("execution-side first-batch cross-split
writeback loss (canary 456 vs 12095)... iree/amdgpu device-layer owner").

## The synthesis
The zaya prefill (the FIRST llama_decode = the FIRST execution of every freshly-built
HRX program) suffers deterministic writeback/copy corruption in specific
(buffer, command, region) slots, while the DECODE steps (the 2nd..9th executions of
the same programs) land correctly:
- The prefill's tok0 (lm-head over the corrupt first-execution outputs) = wrong (563).
- The decode text is coherent because steps 2+ land; it is the correct continuation
  of the wrong tok0 (no CPU oracle existed for that continuation - consistent).
- The corruption shows up at every CPU-side read of first-execution HRX data: the
  conv island's qk staging (Q-side; the K-side copy from an earlier command validates
  correct - mad 0.011), the QK_dw/QK_grp outputs (garbage state rows, wrong values),
  the x (post_attn_norm) rows for t>=1, and the gate_up rows.
- Region/command dependence explains the earlier confusion: some regions validate
  bit-exact (input_norm, Kraw, t0 rows), others are corrupt - exactly the
  "cross-split" pattern of the rounds-16a-17b canary tests.
- GGML_HRX_DOUBLE_EXECUTE=1 (the rounds-17a/17b probe, which re-launches each program
  once as a warm-up) catastrophically breaks zaya because the recurrent state
  double-applies - as its own comment warns - so it is not usable to confirm on zaya;
  but the decode-steps-already-land evidence supports the first-launch-loss model.

## What this means
- The mm kernels, norms, projections, and all loom kernels are exonerated (bit-exact
  numpy validation of the entire upstream chain).
- The defect = first-execution writeback loss / cross-backend copy races for specific
  HRX buffer regions - the iree/libhrx transfer or writeback layer
  (hrx_synchronous_d2h/h2d via iree_hal_device_transfer_*, host-visible buffer
  coherence), i.e. the "device-layer owner" direction from the ORIGINAL task-4 note.
- The rounds-63..70 mm1-compute detour and the later readback/permutation theories
  were all downstream symptoms or artifacts of this same first-execution corruption.

## Decisive next test (device-layer)
Run the SAME program twice back-to-back and compare the d2h readback after each
execution (the rounds-17 canary, on the zaya geometry): the first readback should show
the corruption, the second the correct values. This isolates first-launch writeback
loss from any semantic error. Then fix the writeback/sync in libhrx
(~/hrx-ws/hrx-rocm/libhrx/src/libhrx/transfer.c) or the ggml-hrx transfer usage.
