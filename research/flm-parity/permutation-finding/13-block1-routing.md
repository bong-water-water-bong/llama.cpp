# ZAYA PREFILL BLOCK-1+ ROUTING = STALE/MIS-BOUND IDS (agent-f49062, 2026-09-07 16:1x)

## State of play (all on-device, zgreedy raw-prompt harness, post-540e9815e)
- tok0 = 563 (oracle 9079); top5: 563(12.254) 236743(11.691) 108(11.564) 9079(11.400) —
  the HRX logits are systematically off (oracle top5 9079(17.45)...); 9079 falls to 4th.
- tok1-4 = oracle-matching (236743~236761, 107, 2717, 108) then a repetition loop — the
  decode path itself is nearly right; the failure is the PREFILL final logits.

## Validated chain (block 0 = EXACT vs CPU oracle, mads ~0.002-0.008)
gate_up-0 routing = [3,15,10,12,3,0] = CPU EXACT (the route_stride fix WORKS on-device);
down-0, moe_out-0, layer_out-0, attn-0 = all oracle-exact.

## The divergence: per-block expert TABLES (captured via GGML_HRX_PROGRAM_DUMP=expert_table)
Block  0 table -> routing [3, 15, 10, 12, 3, 0]  = CPU EXACT
Block  1 table -> routing [7, 6, 2, 9, 15, 15]   vs CPU biased-argsort top-1s
                                                  [7, 10, 2, 9, 13, 15]  (r04_014 probs +
                                                  blk.1.zaya_router_biases; derivation
                                                  validated on block 0 against r03_002)
- Wrong positions: token 1 (10->6) and token 4 (13->15). The gate_up/down chain is
  INTERNALLY consistent for the mis-routing (down-1 matches W_down[e]silu(gate_up) at
  mad ~0.001) - so the mm computes correctly for the experts its table assigns.
- All 40 blocks' layer_out diverge from block 1 onward (worst mad grows 0.22 -> 7.8 with
  depth; block 39 rows 2-5 are back to ~0.015-0.02 near-exact while rows 0-1 diverge).
- The decode steps (1-token mms) route CORRECTLY (tok1-4 match the oracle) - only the
  6-token prefill block-1+ mms mis-route.

## Implied mechanism
Same kernel (ggml_mul_mat_id_f32_f32_wmma), same dispatch, same table-build for all
blocks; block 0 correct + blocks 1+ wrong at 2 of 6 token positions => the table-build
READ A STALE/MIS-BOUND ids value for blocks 1+ (the ids = CPU-written argsort view; the
HRX must bind it after the CPU finishes). This is the round-16e class ("HRX programs
missing CPU-produced activation inputs from binding lists") applied to the per-block
argsort/route_ids of the REPLAYED prefill programs. The block-0 ids bind correctly (first
program) while the replayed per-block programs bind a stale buffer (partially-updated
argsort or a reused allocator buffer - observed values mix block-0 argsort ranks).

## Next steps
1. Trace the value binding of route_ids for the replayed prefill programs in
   graph-program-cache.cpp / command-program-executor.cpp (the recorded-graph replay
   path, GGML_HRX_* instrumentation already in tree): does the replay rebind externals
   per execution, or reuse the first-execution addresses?
2. Check whether forcing GGML_HRX_SYNC_EACH (per-op sync) fixes the block-1+ routing -
   that distinguishes a sync/ordering issue from a replay-binding bug.
3. Candidate minimal fix: add the route_ids (and all CPU-produced externals) to the
   program's dependency list so each replay binds fresh addresses; or disable program
   replay for programs with CPU-produced external inputs.

## Update (same session): SYNC_EACH test + race conclusion
- GGML_HRX_SYNC_EACH=1 does NOT fix the block-1+ mis-routing (tables unchanged:
  [7,6,2,9,15,15] vs CPU [7,10,2,9,13,15]) => not an HRX-stream ordering issue.
- The mis-read values mix CORRECT block-1 top-1s (tokens 0/2/3/5) with stale values
  at tokens 1/4 => the table-build read the argsort-1 buffer MID-WRITE or from a
  reused arena slot whose tail was not yet written - a CPU-write vs HRX-read race on
  the shared ids buffer. Block 0 is always correct because the first argsort write
  precedes the first HRX program by construction.
- The ggml sched cannot order a dependency it does not know: the table-build is an
  HRX-generated command, and if the per-block route_ids value is not recognized as an
  input dependency of the mm program (round-16e class: CPU-produced activation inputs
  missing from binding lists), the table-build can start while the CPU argsort write
  is still in flight (arena slots are reused across the 40 blocks).
- Next: audit the mm dispatch graph-value dependency list for route_ids (dispatch-
  mul-mat-id-common.h match + graph-program-cache external binding); candidate fixes:
  (a) declare the ids value as a graph input dependency of the mm program,
  (b) force an explicit h2d of the ids before the table-build command, or
  (c) allocate the argsort outside the reused arena (ggml_cont the ids before the mm).
