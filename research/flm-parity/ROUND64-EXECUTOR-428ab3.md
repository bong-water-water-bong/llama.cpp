# T3 executor-lane report — round 64 (lane agent-428ab3, inheriting eb4f0b)

Date: 2026-09-06 ~15:10 (-0300). Tree: fix/hrx-ngl-init-order @ ~/hrx-ws/amd-hrx-graph
(HEAD ae0590115 + d5694d's uncommitted .loom row_debug-era corpus; executor files
below are MY uncommitted changes). Mesh direct-sends from this session are refused
peer_not_found, so the report lives here + files under ~/zaya-captures-428ab3/.

## Mission: transient-name plumbing for the program dump (GGML_HRX_PROGRAM_DUMP)

### Fixes (executor/dispatch/cache C++, env-gated, zero default impact)
1. graph-program-cache.{h,cpp} — GraphProgram now threads generated-resource names
   (common.moe_routing.expert_table / .partition_table / .row_debug) from the
   scheduler plan (transients + completion counters) into a ValueId->name
   side-table at program build; dump_program_values names those values, so dumps
   are written as /tmp/prg_dump/<uid>_<ord>_common.moe_routing.partition_table.bin.
   bind_<index> stays a filter alias but only for values with NO ggml tensor name
   (named tensors no longer match bind_N role filters).
2. transient-allocator.cpp — NEW env GGML_HRX_PRESERVE_ROUTING_TABLES=1 reserves
   moe_routing.* transients (tail-allocated, kept out of the packable pool).
   WITHOUT it the post-program dump of these tables is garbage: they are bound
   only by MAIN commands, so their arena slots are packable and get reused by
   later commands once the mm has consumed them; a post-program d2h then reads
   whatever later transient last wrote the slot (observed as float garbage).
3. NOTE: a137d5's GGML_HRX_PROGRAM_DUMP_DEBUG block also sits (uncommitted) in
   graph-program-cache.cpp — concurrent edit ~11:57, both compiled into the lib
   at build/bin/libggml-hrx.so (rebuilt 12:05). Leave it or rerun `make ggml-hrx`.

### Verified capture (f32twin, "The capital of France is", 5+1-token prefill)
env: GGML_HRX_ROW_DEBUG=1 GGML_HRX_PRESERVE_ROUTING_TABLES=1 GGML_HRX_PROGRAM_DUMP=bind_2
-> partition_table captured at exactly 72B per program; full decode rc=0,
text unchanged. Files: /tmp/prg_dump/* (may be cleaned) and the durable copy
~/zaya-captures-428ab3/partition_tables/ (720 files, per-block + per-ordinal).

### partition_table descriptor (answers d5694d's route_tile_base question)
Layout (from moe_routing_tables.loom builder + mul_mat_id_f32_f32_wmma_core.loom):
- word0 = partition_count
- words 1..count = packed descriptor, one per partition (32-row tile):
  expert = desc & 511 ; local_part = (desc>>9) & 63 ; rows = (desc>>15)+1
- remaining words up to capacity are UNWRITTEN (leftover arena bytes, not zeros).

Block-0 (uid 10866 ord 1): 6 | 2 3 4 5 7 12 (+11 pad)
  = 6 partitions, one per routed expert {2,3,4,5,7,12} (== the round-52 ids
  [3,5,12,7,2,4]); every descriptor local_part=0 rows=1 (route_count=1: each
  expert has exactly one assignment -> one 1-row partition).
Other blocks differ: 10893_001 {2,5,6,7,9,15}; 10920_001 {1,8,10,11,13,14};
10947_001 {1,2,3,7,9,11} (full per-block set in ~/zaya-captures-428ab3).

route_tile_base finding: (desc>>9)<<5 = 0 for EVERY zaya partition. The builder
packs the LOCAL per-expert partition ordinal, not the global one; the global
partition identity is the descriptor POSITION (the core indexes
descriptor[%active_partition] over the launch loop). The round-38 reading
"route_tile_base = global partition_ordinal<<5" does NOT match what the builder
writes — relevant to the per-partition fetch review (any code path assuming a
global base in the descriptor would compute base 0 for every partition).

### Coordination flags
- 12:03 /tmp/prg_dump cleanup wiped d5694d's 11:58 row_debug battery captures;
  rerun row_debug capture with GGML_HRX_ROW_DEBUG=1 + GGML_HRX_PRESERVE_ROUTING_TABLES=1.
- post_attn_norm-0 6-slot [2048,6] HRX recapture done (10864_001, 49152B,
  norm-plausible slots). Full CPU-oracle mad recompare needs the round-63
  nodedump harness (cleaned from /tmp); executor diff is compute-neutral so the
  round-63 closure (mad 0.003-0.006) stands as recorded.
- Commit decision (executor files: graph-program-cache.h/.cpp + transient-allocator.cpp)
  left to the supervisor; d5694d's .loom edits are untouched and uncommitted.

## Addendum (same session, ~12:15 local) — layout answer for b30173's DUMP_TABLES hook + full descriptor decode

### mm binding layout (zaya f32twin, ggml_mul_mat_id_f32_f32_wmma, GGML_HRX_ROW_DEBUG=1)
From /tmp/cpdump program.json (command-program dump, authoritative ABI names):
  b0 input         GraphValue 49152B (post_attn_norm-0)
  b1 expert_table  Transient    448B   <- moe routing table 1
  b2 partition_table Transient   72B   <- moe routing table 2
  b3 weight        GraphValue 268435456B
  b4 output        Transient  49152B
  b5 row_debug     Transient   1024B (env-gated)

The routing tables sit at b1/b2 of the 6-binding plain mm command, NOT b4/b5 of
a 10-binding command. The only 10-binding command in the zaya graph =
ggml_flash_attention_decode_split_f32_f16_wmma_next_q8 (b4/b5 = partial_max /
partial_sum, 512B each) - GGML_HRX_DUMP_TABLES as committed therefore targets
flash-attn partials (or nothing), not the moe tables. Recommendation: either
scan for bindings whose value ids resolve through generated_value_names_ to
"common.moe_routing.*", or just use the name filters (verified):
  GGML_HRX_PROGRAM_DUMP=common.moe_routing  -> expert_table + partition_table by name
  GGML_HRX_PROGRAM_DUMP=bind_2              -> partition_table (b2 alias)

### expert_table decode (block-0, uid 10866 ord 1, 448B = 16 counts + 16x6 matrix)
counts: e2,e3,e4,e5,e7,e12 = 1 each; matrix rows give the assignment ordinal
(= token at route_count 1): e2->4, e3->0, e4->5, e5->1, e7->3, e12->2 - EXACTLY
the round-52 routing ids [t0->3,t1->5,t2->12,t3->7,t4->2,t5->4]. Unwritten
matrix cells hold stale arena bytes (not zeros) - expected (builder only fills
count cells per expert).

### partition_table decode (block-0, uid 10866 ord 1, 72B = 1 + 6 + 11 pad)
word0 = partition_count = 6; descriptor[i] (i = global partition ordinal 0..5):
expert = desc&511, local_part = (desc>>9)&63, rows = (desc>>15)+1.
Block-0 = experts 2,3,4,5,7,12 at ordinals 0..5, every local_part=0 rows=1.
route_tile_base = local_part<<5 = 0 for every partition (route_count=1 -> each
expert has exactly one 1-row partition). The kernel loads
expert_table[expert][route_tile_base+local_route] (= [expert][0]) = the token;
publish writes dst at token*route_count+route.

CONCLUSION: routing tables fully exonerated - counts, experts, assignments all
match CPU. Consistent with round-64 ROW_DEBUG "publish PERFECT ... values wrong
= wmma compute per partition": the per-partition VALUE fetch/compute inside the
wmma core is the remaining bug site (d5694d value-level review).
Captures: ~/zaya-captures-428ab3/ (720 partition + 720 expert_table files).

## Addendum 2 — executor-side confirmation of rounds 65-67 (binding theory dead; mm compute confirmed)

Investigated b30173's round-66/67 binding directive (output binding -> transient arena;
external slot = leftovers) from the dispatch/prepare side. Independent evidence says the
binding theory is already resolved/void at HEAD (post-2824946b5 alias-only externalization):

1. The gate_up mm (uid-10866 class, blk.0.ffn_gate_up_exps.weight 512MB) ALREADY binds its
   output to the EXTERNAL ggml slot: pddbg b4 val=3 -> compute arena @2621440 len=98304
   (origin GraphValue, ffn_moe_gate_up-0). The 10868-class mm (ffn_down_exps 256MB) output
   (49152B) IS legitimately transient (consumed in-slice by the weighted MUL path). No
   dispatch binding change is warranted.
2. The external slot @2621440 post-run is NOT leftovers - it is the kernel's deterministic
   output: tok0 gate[0] = -1.4707 (EXACTLY the round-36/41 correct token-0 value, fp-exact
   to 4dp) while tok1-5 = the 0.3435-class wrong values. A never-written slot cannot hold
   token-0 exactly right. => the kernel writes the external slot; its VALUES for t1-5 are
   wrong = the mm per-partition compute (round 67 %values-head: expert-3/t0 right,
   t1-5 wrong mad 0.54-1.18). Round-66's "transient = correct / slot = leftovers" =
   consistent with the round-67 provenance-error retraction.
3. Per-token gate/up row heads from /tmp/prg_dump/10866_001_ffn_moe_gate_up-0.bin
   (98304B = [4096,1,6], plane = token, gate = rows 0-2047, up = rows 2048-4095) -
   for a137d5's classification matrix:
     tok0 gate: -1.4707 -1.3945 -0.7773  0.0187 -1.5879 -0.4514 ...
     tok0 up:   -0.1617  2.7422  0.4858  0.5908  0.4697 -0.2520 ...
     tok1 gate:  0.3435  0.5103  0.7173  1.0615 -0.4341  0.9185 ...
     tok1 up:    0.2013 -0.1536  0.4487 -0.4280  1.0830 -0.6089 ...
     tok2 gate:  0.1693  1.1270  0.4482  1.4727 -0.3767 -0.4646 ...
     tok2 up:   -1.7598 -1.0254 -0.3865  0.7944  0.1096  0.6104 ...
     (tok3-5 full rows in the file; all 40 blocks' gate_up captures on request)
   Files: /tmp/prg_dump/10866_001_ffn_moe_gate_up-0.bin + ~/zaya-captures-428ab3/.
4. Lane status: dispatch STANDBY per round 67 - no dispatch/prepare code change warranted
   until the classification matrix (a137d5) names a concrete dispatch-side site.

## Addendum 3 — priority nudge (m_mtpzu4iw, 12:55:44) evaluated: deliberately NOT implemented

Nudge: "bind the mm's output to the external slot or add the transient->slot copy;
numpy = transient correct mad 0.003; after the fix expect near-oracle."
Verdict: no action. The nudge's premise (a correct transient exists, mad 0.003) = the
round-66 correlation that round 67 RETRACTED as a provenance error (committed 20c57fa32
at 12:57:43, two minutes after the nudge was sent). Independent dispatch-side evidence
(addendum 2, commit 63fcd4598): the gate_up mm output binding is ALREADY external at HEAD
(pddbg b4 -> compute arena @2621440), and the external slot content = the kernel's own
deterministic output (tok0 gate[0] = -1.4707 EXACT; t1-5 wrong-class), NOT leftovers. The
kernel's in-kernel %values (round 67) = wrong for 5/6 experts -> there is NO correct
transient anywhere to copy; a transient->slot copy would copy wrong values. Implementing
the nudge would be acting on a retracted theory. Standby maintained for the classification
matrix result (a137d5) -> if it names a dispatch/prepare-side site (weight-slice offset,
fragment staging), land it there. Battery = running (not touched); canary status unchanged
(563) is EXPECTED since no fix is warranted.

## Addendum 4 — build-flag response (m_mtpzycp0, 12:59:01): already green, edit not mine

b30173 flagged "my" in-flight dispatch-gated-mul-mat-id.cpp edit as RED (GGML_HRX_SWIGLU_TRACE
partial insertion, lines 67-68). Checked: the dispatch-gated edit is NOT mine (same
misattribution as the earlier dispatch-mul-mat-id.cpp WLAYOUT probe) - it is the fleet's
active zaya view-split swiglu-fusion WIP (+76 lines, gate_up_combined + VIEW-split
detection; mtime moved 13:01:56->13:02:37 while watched = live editor). The duplicated
fprintf-arg line that caused the red was removed by that editor ~13:02; `make ggml-hrx` =
clean (rc 0, nothing stale). GGML_HRX_DUMP_IR flow = unblocked. No action taken; standing
by. Working tree = 10 modified files across lanes (dispatch-gated fusion, dispatch WLAYOUT
probe, ggml-hrx.cpp, loom-jit.cpp DUMP_IR, d5694d .loom set) - all uncommitted, as the
fleet's in-flight state.

## Addendum 5 — m_mtq001xd (13:00:20) priority re-assertion: binding fix = already the code state; no-op

Re-assertion claims: "the mm computes CORRECT data in the transient arena; the dispatch
output binding targets the TRANSIENT; the external slot @2621440 never written; %wide
scramble = strided-view artifact." Checked against the authoritative command-program dump
(cpdump, run 12:14, same 5-token f32twin config) - per-mm output origins:

  program-8  (gate_up mm, ffn_gate_up_exps 512MB, out 98304B)  output origin = GraphValue
  program-23 (decode gate_up, out 16384B)                       output origin = GraphValue
  program-13 (down mm, ffn_down_exps 256MB, out 49152B)         output origin = Transient
  program-9/24/28 (down/router class)                           output origin = Transient

=> the GATE_UP mm output binding = EXTERNAL at the command-program level (program-8), the
prepared level (pddbg b4 -> compute arena @2621440), and the executed level (record =
prepared refs). The round-66/13:00 "output binding = transient arena @1280" pddbg =
the DOWN mm (49152B transient, consumed in-slice by the weighted-MUL path = correct
design), misattributed as the gate_up mm. The gate_up external slot content (my capture,
addendum 2) = deterministic kernel output: tok0 gate[0] = -1.4707 EXACT (round-36 value),
t1-5 = 0.3435-class wrong. A never-written slot cannot hold tok0 exactly. => no correct
data exists in any transient to copy; a transient->slot copy has no source; re-binding =
already done. Round-67 %wide (kernel computes wrong for 5/6 experts) + round-59
permutation search (correct t1-5 values nowhere in the region) + row_debug publish-perfect
= mutually consistent with the gate_up COMPUTE as the defect. If the %wide read is
suspected of stride misalignment, that is d5694d's instrument to re-verify, not a
dispatch change. No dispatch/prepare code change is warranted; canary 563 = expected.
