
# ZAYA DEVICE-PATH SPEED ANALYSIS (agent-f49062, 2026-09-07)

## Measurements (zgreedy_t: per-step timing, raw prompt, 16 decode steps)
- CPU-only (ngl0):  69.2 ms/tok (14.44 t/s); prefill 133.7 ms (44.9 t/s)
- HRX mixed (ngl99): 187.6 ms/tok (5.33 t/s); prefill 689 ms (8.7 t/s)
- llama-bench ngl99 -ub 1024: tg128 5.14, pp64+tg64 9.88 (bench aborts with the
  default -ub 512: GGML_ASSERT(n_ubatch > n_keep_tail) in split_equal)
- tok0=9079 + oracle stream on BOTH paths (correctness confirmed on the mixed
  path with the ADD-exclusion build).

## Composition (kres profile per run, ~17 executions: 1 prefill + 16 decode)
Per-execution kernel families (all HRX): mul_mat (21), mul_mat_decode (21),
binary_bc (21 = chain-fused bias/residual adds), flash_attn_decode_split_next_q8
(16), rmsnorm_binary (14), binary (12), unary (9), set_rows (9 = state cpy),
mul_mat_id (9), moe_build_expert_table (9), moe_build_expert_partition (9).
=> the decode is ALREADY mostly on-device; ~140 kernel launches per token.

## Conclusion
The mixed path is ~2.7x SLOWER than all-CPU: the device work (wmma mms,
flash-attn, norms) is fast but the per-subgraph launch+sync overhead dominates
(~1.3 ms x ~140 launches/token). The zaya graph alternates CPU ops (SSM conv,
grouped conv 1d, recurrent state handling) with HRX subgraphs 40x per layer
chain, so the backend-sched executes ~140 small HRX subgraphs per token, each
with a host-side round trip.

## Fix directions (in descending leverage)
1. Collapse the per-token launches: the recorded-graph replay already exists
   (graph-program-cache); ensure one program per contiguous HRX run (the graph
   executor's program formation currently fragments per op-chain) and reduce
   host syncs to once per token (async submission + event-based wait).
2. Port the SSM conv + grouped conv to loom kernels (missing piece for an
   all-HRX zaya graph; qwen3 is all-HRX and hits 249 t/s at 0.6B).
3. Re-claim standalone residual ADDs ONLY when their producers are HRX-side
   (fixes the roster regression: dense qwen3 0.6B tg128 49.1 vs 249.6 t/s with
   the blanket ADD exclusion from e130977af) - the zaya needs the CPU-produced
   ordering fix (round-16e round-16f) before its ADDs can be re-claimed.
