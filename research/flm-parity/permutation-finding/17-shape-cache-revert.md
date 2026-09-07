
# Shape-cache reuse attempt: reverted (agent-f49062, 2026-09-07)

## What was tried
A structural shape-keyed GraphProgramCache index (FNV over node op/ne/src-op/
params) with cheap trusted-match verification, so the ~640 subgraph program
builds per zaya decode token could be reused across tokens (the sched re-uids
every split graph per compute, so uid-keyed hits never fire).

## Result
- Full per-node match verification per call cost more than the builds it saved
  (decode 234.8 ms/tok vs 181.0 baseline).
- Swapping verification to the cheap trusted match broke execution (compute
  error -1 on the first decode step) - trusted match (count + sentinels +
  externals) cannot distinguish same-count/same-sentinel subgraphs (identical
  transformer blocks!) whose prepared state differs.
- REVERTED to the committed state; decode back to oracle-exact 5.52 t/s.

## Conclusion (reinforced)
The ~640 subgraphs/token ARE the problem (each = build + launch + host sync,
~0.28 ms). A cache cannot fix the launch+sync floor. The structural fix is
contiguous-HRX decode graphs: port the SSM conv (ggml_ssm_conv +
ggml_conv_1d_grouped) and the recurrent-state cpy islands to loom kernels so
the sched stops fragmenting per block. Until then the device path stays
sync-bound at ~5.5 t/s vs the CPU-only 14.4 t/s.

## Current branch state (fix/hrx-ngl-init-order)
- 8a72e8c34 (docs: 640-subgraph finding)
- 49f75f161 (docs+tool: speed analysis + kres kernel-name print)
- 4900235a4 / 8df635cb0 / 96a9feaf7 (zaya oracle-exact trail)
- e130977af (ADD/CLAMP/DIV exclusion - zaya correct; roster speed regression
  documented, needs fused residual-ADD coverage)
- 540e9815e (route_stride fix + mul_mat_id ABI consistency)
