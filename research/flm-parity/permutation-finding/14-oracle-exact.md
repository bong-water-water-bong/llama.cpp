# ZAYA DECODE = ORACLE-EXACT ON HRX (agent-f49062, 2026-09-07 ~16:30)

## RESULT
zgreedy raw-prompt harness, zaya-q4nx-c43.gguf, ngl=99, GGML_ZAYA_DEQUANT_F16=1,
NO DISABLE flags, current branch (540e9815e + e130977af + docs):

  prompt tokens: 2 669 5279 529 7001 563
  top5 step0: 9079(17.455) 528(15.940) 107(15.224) 506(14.634) 5213(14.609)
               (CPU oracle: 9079(17.450) 528(15.773) 107(15.138) 5213(14.626)
               506(14.556) - same set, values within f16 noise)
  tok0-7 = 9079, 236761, 107, 2717, 108, 1882, 735, 1156  (CPU oracle EXACT)
  text: [ Paris.
  - Deterministic across runs; block-1+ expert tables now match the CPU routing
    [7,10,2,9,13,15] (were [7,6,2,9,15,15]).

## What fixed it
e130977af (c5ac20's blanket ADD/CLAMP/DIV standalone-claim exclusion, originally
for qwen3moe-30B #2147) ALSO cures the zaya: the zaya's standalone-claimed ADDs
consumed CPU-produced attention outputs without an ordering guarantee (round-16e
missing-dependency class) - stale ADD inputs corrupted the per-block layer
residuals, which corrupted the block-1+ routers (close-call flips at 2 of 6
tokens per block) and hence every downstream block. With the ADDs split to CPU,
the residuals are computed CPU-side in-order and the HRX reads them after sync.
The route_stride fix (540e9815e) remains necessary for the block-0 routing.

## Evidence chain (this session)
1. route_stride fix verified on-device: block-0 table = [3,15,10,12,3,0] = CPU.
2. Block-1+ tables mis-routed pre-exclusion; gate_up/down chains internally
   consistent for their (wrong) routing; SYNC_EACH negative; values mixed fresh +
   stale argsort content => stale-read, not a stride bug.
3. Post-exclusion: all tables correct + decode oracle-exact.

## Speed status (task-4 contract: >= 16.8 t/s single-seq - NOT YET MET)
- llama-cli -st 32-token decode: Generation 5.9 t/s (chat-template prompt).
- The decode is bound by the CPU-side per-layer work (SSM conv + attention +
  residual ADDs after the exclusion); the ffn mms run on HRX.
- llama-bench on the zaya aborts: GGML_ASSERT(n_ubatch > n_keep_tail) in
  llama_batch_allocr::split_equal (batch-alloc vs the recurrent keep-tail) -
  needs -b/-ub tuning or a bench-harness fix for the zaya arch.
- Next: (a) fused/on-device residual-ADD coverage with proper dependency
  ordering (the qwen fused-attention pattern), (b) attention/flash-attn on-device
  claims, (c) llama-bench batch fix for the official flm-parity numbers.

## Addendum: qwen3-roster decode-speed regression from e130977af
- qwen3-0.6B tg128: 49.09 t/s now vs 249.6 t/s at task-2 (2026-09-05 build) -
  the ADD/CLAMP/DIV standalone-claim exclusion also forces the dense-qwen3
  per-layer residual ADDs to CPU (their ffn-residual adds are not chain-fused),
  and at 0.6B scale the per-token CPU add + cross-boundary cost dominates.
- Correctness on the roster remains fine (no errors; coherent smoke output);
  speed regressed ~5x.
- The zaya needed the exclusion because its standalone-claimed ADDs read
  CPU-produced attention outputs stale (round-16e missing-ordering class).
  Dense-qwen3 ADDs read HRX-produced inputs, so they were correct AND fast when
  claimed - the blanket exclusion is the wrong instrument for them.
- Proper fix (next session): fused residual-ADD coverage for the ffn tail
  (postops/next_rmsnorm registrations already exist for the mul_mat_id family)
  so ADDs ride their chains on-device, plus the executor host-binding ordering
  fix (round-16e round-16f: materialize_host_bindings upload staging) so
  standalone claims become safe again for CPU-produced inputs.
