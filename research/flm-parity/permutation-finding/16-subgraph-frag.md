
# ZAYA DECODE = 640 SUBGRAPH-FRAGMENTATION PER TOKEN (agent-f49062, 2026-09-07)

## New measurement (GGML_HRX_CACHE_STATS instrumentation, decode 16 steps)
- 10,880 get_or_build calls, hits=0: EVERY call carries a fresh graph uid (the
  ggml sched stamps sched->splits[i].graph.uid = ggml_graph_next_uid() per
  compute, ggml-backend.cpp:1494) so the GraphProgramCache NEVER hits; the
  decode performs ~640 program builds per token.
- The 640 = the count of HRX-subgraphs the sched produces per token: the zaya
  graph alternates CPU-forced ops (SSM conv, grouped conv, recurrent-state
  cpy) with claimed HRX ops every few nodes, so the sched splits ~640 tiny
  subgraphs; each is a program build + launch + host sync (~0.28 ms each ->
  179 ms/token). The CPU-only path pays none of this and runs the same math in
  69 ms.
- Even with a perfect cache, ~640 launches/token at ~0.1-0.3 ms each would
  still land ~60-190 ms/token. The structural fix is fewer, larger subgraphs.

## Fix directions (updated)
1. Structural (the real fix): port the zaya CPU-forced ops to loom kernels
   (SSM conv 1d, grouped conv 1d, the recurrent-state cpy/set_rows pieces) so
   the decode graph becomes contiguous-HRX like qwen3 (which is 1-2 subgraphs
   per token and runs 249 t/s at 0.6B). The flash-attn + wmma mms + norms are
   already claimed; only the conv/state islands keep the graph fragmented.
2. Cache fix (cheap, helps while fragmented): make get_or_build match the last
   program structurally (node-count + op pattern + external slots) instead of
   by uid, so the ~640 per-token builds become prepared-program replays. The
   match_current_graph machinery already exists; the uid fast-path is what
   defeats it.
3. Sched-level: investigate ggml_backend_sched split control (fewer, larger
   HRX subgraphs by claiming the CPU ops' neighbors), or async subgraph
   pipelining in the HRX backend (submit all subgraphs, sync once per token).

## Status reminder
- zaya decode = oracle-exact on the HRX device (tok stream 9079/236761/107/
  2717/108/1882/735/1156, text " Paris.") with e130977af + 540e9815e.
- Device-path speed 5.3-5.9 t/s vs the 16.8 target; CPU-only 14.4-17.45 t/s.
- Dense-qwen3 roster decode regressed ~5x by the ADD exclusion (49 vs 249.6
  t/s): needs the residual-ADD chain fusion or ordered re-claim (round-16f).
