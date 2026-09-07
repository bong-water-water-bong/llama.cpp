
# ZAYA DECODE FRAGMENTATION + FIX PLAN (agent-f49062, 2026-09-07 ~18:30)

## Measured structure (GGML_HRX_GRAPHCOUNT + CACHE_STATS, decode 1-token)
- ~640 sched-subgraphs per decode token; 4743/10880 calls are n_nodes=1,
  most are 1-3 nodes (RMS_NORM MUL MUL_MAT triples, lone MULs, lone MUL_MATs,
  MUL_MAT_ID+VIEW clusters, one 12-node FLASH_ATTN cluster per block).
- Per-token: 640 program builds (uid-keyed cache never hits - the sched re-uids
  every split graph per compute, ggml-backend.cpp:1494) + 640 launch+sync
  round trips ~0.28 ms each = ~180 ms/token. CPU-only does the same math in
  69 ms.
- The flash-attn (12-node cluster with SET_ROWS state copies), wmma mms, norms,
  unary/binary ops are ALL already claimed. The CPU islands that fragment the
  graph per block are: the SSM conv + grouped conv (no loom kernels), the
  residual ADDs (e130977af exclusion), and staging layout ops.

## Why the ADD exclusion was needed (zaya) but hurts the roster
- Pre-exclusion: standalone-claimed ADDs whose inputs are CPU-side (attention
  path produced across the host-buffer boundary) read stale data - the 1-node
  ADD programs lack the CPU-input ordering/dependency (round-16e/16f class:
  host-buffer direct binding has no CPU-write visibility flush).
- The blanket e130977af exclusion fixed the zaya correctness but forces the
  dense-qwen3 ffn-residual ADDs to CPU too (they were claimed, correct AND fast
  - their inputs are HRX-produced), costing ~5x decode (249.6 -> 49 t/s tg128).

## Fix plan (ordered)
1. Host-binding flush/ordering in the executor (round-16f materialize_host_
   bindings): make CPU-written host buffers visible to device kernels before
   programs read them (flush + fence or staged upload). This makes standalone
   ADD claims safe again -> restore roster speed; zaya ADDs may then be
   re-claimed (its correctness depended on the same stale-read class).
2. Re-claim residual ADDs conditionally (eager claim: src-op is HRX-producible
   AND the value is device-resident at dispatch; otherwise CPU) instead of the
   blanket exclusion - keeps both families correct.
3. SSM conv + grouped conv loom kernels (ggml_ssm_conv, ggml_conv_1d_grouped)
   -> contiguous-HRX decode graphs like qwen3 (1-2 subgraphs/token) -> the
   launch floor disappears; expect the roster-style 50-250 t/s for the zaya.
4. Fused residual-ADD + next-rmsnorm postops coverage (kernels exist:
   mul_mat_id_postops / next_rmsnorm) so the ADDs ride their mm chains.

## Branch state (fix/hrx-ngl-init-order, HEAD f3fc3748f)
Zaya decode = oracle-exact on-device (9079/236761/107/2717/108/1882/735/1156,
text " Paris.") at 5.5 t/s. All analysis + attempts documented (13-17 series).
