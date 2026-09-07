# ZAYA ROUTER FIX STATUS (agent-f49062, 2026-09-07 ~14:40)

## What landed (commit 540e9815e)
1. **route_ids stride fix** (dispatch-mul-mat-id-common.h): the expert-table
   builder now derives route_stride from the ids value's row pitch when the ids
   are a non-contiguous view (zaya top-1 = strided slice of the [16, tok]
   argsort, pitch 64B). Before: route_stride = route_count = 1 made EVERY token
   read token-0's rank-r expert (observed routing [3,5,12,7,2,4] = token-0's
   top-6, exactly the wrong-stride read). After: each token reads its own
   top-1 ([3,15,10,12,3,0] = CPU oracle routing). Contiguous ids (qwen3 top-k)
   are untouched.
2. **mul_mat_id ABI consistency**: b30173 landed the row_debug binding in the
   dispatch C++ AND the manifest but never the kernel half (kernel ABI = 5
   bindings). Removed row_debug from both to make the committed kernels load.
3. **libhrx aql_ring PM4 probe** (EXTERNAL repo ~/hrx-ws/hrx-kind, not
   committed here): copied hrx-rocm's tolerant version (ignore INVALID_ARGUMENT
   on HSA_AMD_AGENT_INFO_PM4_EMULATION for gfx1151). The gfx1151 intermittently
   rejects the probe; the old hard-fail made hrx_gpu_initialize fail code=3
   (device create) with no retry. Rebuilt the deps libhrx in-tree (13:38).

## Current state
- zaya -ngl 99 runs END-TO-END on HRX (init OK, kernels compile, no ABI errors)
  in SERVER/-st mode. Decode text "We need to understand the problem from"
  (still != CPU oracle " Paris.").
- **CRITICAL COMPARISON CAVEAT**: -st/server mode applies the chat template ->
  the prefill mm is 20 ROWS (templated prompt!), while the CPU oracle files
  (ffn_oracle r03, 6 rows) come from a raw-prompt CLI run. 20-row dumps cannot
  be compared row-wise to the 6-row oracle with the existing numpy stack.
  The 10866-era 6-row dumps came from plain-CLI runs (cap-rp style).
- Plain-CLI mode NOW FLOODS stderr with "[zaya] dbg:" (1B+ lines in ~5 min =
  graph built in a tight loop, no visible error). This is NEW (cap-rp CLI runs
  at 13:11 were clean). Suspect: a decode-path graph/planning loop triggered by
  the current dispatch set for the CLI batch shape. UNRESOLVED.

## Next steps (next session)
1. Fix or sidestep the plain-CLI flood to get a comparable 6-row capture:
   try -b 128 (cap-rp used n_batch=128!), or env-diff against the 13:11
   working run (cap-rp.log in /tmp), or run with GGML_HRX_SYNC_EACH-style
   instrumentation that was active in the 10866-era runs.
2. Verify routing on a 6-row dump: rows should now match
   W[CPU-route]x[t] with mad ~0.006 (f16).
3. If routing is fixed and text is still wrong, audit the swiglu/down/
   weighted path + the SSM conv island under the fixed routing.
4. b30173's uncommitted kernel WIP (row_debug arg + round-37 publish fix +
   postops tweaks) is recoverable from dangling stash commit a99bf0f5 if
   needed; do NOT re-apply as-is - it fails the loom SUBRANGE proof for zaya
   token_count 20/25 (the committed kernels compile all shapes).

## Environment notes
- HRX init is flaky when the box is busy; retry-loop script /tmp/zaya_retry.sh
  pattern (init-fail -> kill -> wait 60s -> retry). libhrx aql fix makes init
  succeed regardless of the PM4 probe state.
- The flm FastFlowLM NPU server holds /dev/accel/accel0 while serving (fleet
  chat traffic); observed correlation with init failures is NOT causal (init
  failed with accel0 closed too). The aql_ring fix is the real cure.
