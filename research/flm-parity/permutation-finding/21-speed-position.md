
# ZAYA SPEED: current position + decision (agent-f49062, 2026-09-07 ~19:40)

## Verified state (branch fix/hrx-ngl-init-order, HEAD 1d9e8fcfc)
- Zaya decode = ORACLE-EXACT on the HRX device: tok stream
  9079/236761/107/2717/108/1882/735/1156, text " Paris.", no DISABLE flags.
- Device-path decode: 5.6 t/s (zgreedy_t 179 ms/tok) vs the 16.8 target.
- CPU-only (ngl0): 14.4 t/s zgreedy_t / 17.45 t/s llama-bench tg64.
- qwen3-0.6B tg128 at HEAD: 48.9 t/s (the ADD exclusion cost); 242.9 with the
  GGML_HRX_ALLOW_ADD A/B gate; 30B/35B MoE + zaya require the exclusion.

## Key numbers this session
- ADD claims A/B: dense-qwen3 242.9 vs 48.9 (claims = the roster regression).
- Conditional claim (direct MUL_MAT src, no MUL_MAT_ID): dense 227.8 + zaya
  still correct, but MoE models alias-fail ("value alias target N is not
  transient" - fused chains claim the ADD and alias-relayout a subgraph
  external). Reverted.
- All mms forced to CPU on the zaya: 155 ms/tok (6.45 t/s) vs 179 with mms on
  the HRX => the HRX mm offload is NET-NEGATIVE at the current fragmentation
  (~640 subgraphs/token, ~0.28 ms launch+sync each).

## Why the device path loses to the CPU
The decode graph alternates CPU ops (SSM conv, grouped conv, zaya rope, GLU,
CONCATs, residual ADDs after the exclusion) with HRX ops at 1-3-node
granularity; ~640 subgraphs/token each pay a program build (uid cache never
hits) + launch + host sync. The flash-attn/mms/norms are all claimed already;
the CPU islands fragment everything and the launch floor dominates.

## The remaining work (multi-session, structural)
1. SSM conv + grouped-conv loom kernels (biggest CPU islands per block).
2. Zaya rope (partial-head) + GLU (ne2-fused gate/up) kernel coverage.
3. Residual-ADD ordering fix (round-16f host-binding flush) so standalone ADD
   claims are safe again - restores the roster AND merges zaya subgraphs.
4. Dispatch-coverage-driven ADD reclaim (registration order / alias fallback)
   once (3) lands - the MoE alias failure must be resolved in the dispatcher.
5. Optionally the graph-uid cache fix (structural match) AFTER the subgraph
   count drops (the full-match verification only pays off then).

## Decision
Keep the blanket ADD exclusion (correctness for zaya + MoE lanes); the roster
speed regression (48.9 vs 242.9) and the zaya 16.8-target both wait on the
structural items above. The zaya numerics milestone (the multi-week blocker)
is complete and committed; the speed gap is now a well-characterized
kernel/dispatch engineering program with the fix plan documented
(13-20 doc series on the branch).
