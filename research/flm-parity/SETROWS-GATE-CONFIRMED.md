# SET_ROWS standalone matcher — failing gate CONFIRMED (2026-09-08, agent-ec8072)

Instrumented match_set_rows_2d (env GGML_HRX_SETROWS_DBG, committed) and ran the
npl=2 device repro (llama-batched-bench -ngl 99 -p 32 -n 12 -b 32 -npl 2):

    [setrows-dbg] rows=Y idx=Y cache=Y out=Y rcont=1 icont=1 ccont=1 itype=27
                  otype==ctype=1 shapes_same=1 storage_same=0
    E graph_compute: unsupported HRX node 0: SET_ROWS output=3:f16[256,512,1,1]
      inputs=[0:f32[256,16,1,1], 1:i64[16,1,1,1], 2:f16[256,512,1,1]] consumers=[]

EVERY gate passes except `same_storage(cache->id, output->id) == 0`. The
standalone KV-store SET_ROWS writes the dst (cache) in place, but the graph
import models the op's result as a separate External value with its own storage
(no in-graph consumers -> reserve-ghost class), so the matcher's in-place
storage guard can never fire for these nodes.

Also confirmed: single-seq zaya device decode contains ZERO SET_ROWS nodes
(GGML_HRX_TRACE_EXT+SET trace over the oracle-exact ngl99 run) - the contiguous
single-seq store path never routes a standalone SET_ROWS into an HRX split,
which is why the device path is green there and this bug went unnoticed.

## Fix design (next round; executor/import semantics - do NOT rush)

The alias helpers (ValueMap::alias_storage / force_alias_relayout) both require
a Transient target, and this output is External -> cannot re-alias post-import.
Correct fix = GRAPH-IMPORT level (graph.cpp import_ggml_graph): when creating a
node with op == GGML_OP_SET_ROWS (in-place over its dst operand, ggml src2 =
cache), give the node's OUTPUT value the SAME storage identity as the dst input
value (cache). Mechanics to pick from:
  (a) reuse the dst input's ValueId for the output (like layout aliases, but
      keyed on dst/inputs[2] not inputs[0]), or
  (b) new ValueMap helper force_alias_external(target, source) that aliases an
      External target's storage to the source (no Transient restriction) called
      from import for SET_ROWS outputs.
Then the existing common.set_rows matcher fires (storage_same becomes true),
make_set_rows_dispatch binds output->id -> executor writes the cache's real
device buffer (external-write/publish machinery, round-37 class).
OPEN QUESTION before implementing: executor write ordering for an in-place
External mutation (this graph writes cache rows that a LATER graph's flash-attn
reads; in-graph the same cache may feed flash-attn for the current batch) -
verify the planner orders the SET_ROWS write before any same-graph reader of the
cache storage (shared storage identity should provide the edge; confirm in the
executor's dependency pass, and in the 1-token n_past=0 first-step case where
the round-16 external-write vanish historically lived).

## Verification battery (unchanged from SETROWS-MULTISEQ-DISPATCH-GAP.md)
single-seq zaya ngl99 oracle 9079/...; qwen3-0.6B canary 12095; zaya ngl0
multi-seq no-SEGV; NEW zaya ngl99 batched-bench npl 1-8; llama-server multi-slot
zaya smoke; MoE (30B/35B) alias-clean.

## FOLLOW-ON (2026-09-08, agent-ec8072): SET_ROWS fix landed; next frontier = batched FLASH_ATTN_EXT

Commit c633916f4 (share_inplace_storage at graph import for SET_ROWS results)
fixed the KV-store SET_ROWS abort: device batched-bench ngl99 npl 1/2/4/8 now
rc=0 (no SEGV/abort) and single-seq zaya device decode stays oracle-exact
(9079/236761/107/2717/108/1882/735/1156).

Server multi-slot test (llama-server -np 4, device ngl99) now fails at the NEXT
unmatched shape instead:

    E graph_compute: unsupported HRX node 10: FLASH_ATTN_EXT
      output=19:f32[128,8,2,2]<-FLASH_ATTN_EXT
      inputs=[12:f32[128,2,8,2]<-PERMUTE, 15:f16[128,256,2,2]<-PERMUTE,
              17:f16[128,256,2,2]<-PERMUTE, 18:f16[256,2,1,2]] consumers=[11:RESHAPE]

= batched (2 seq x 2 tok) flash-attn with per-seq KV views; the flash dispatch
registry matches single-seq shapes only. Env-forcing flash to CPU at ngl99
(GGML_HRX_CPU_OPS=FLASH_ATTN_EXT) ABORTS in ggml_backend_sched_backend_id_from_cur
during sched_reserve (known claim/placement class; only ngl0 tolerates it).
Next round options: (a) shape-conditional flash claim (claim false when no
dispatch matches the multi-seq KV-view form - mirrors the fleets per-op claim

## FOLLOW-ON (2026-09-08, agent-ec8072): SET_ROWS fix landed; next frontier = batched FLASH_ATTN_EXT

Commit c633916f4 (share_inplace_storage at graph import for SET_ROWS results)
fixed the KV-store SET_ROWS abort: device batched-bench ngl99 npl 1/2/4/8 now
rc=0 (no SEGV/abort) and single-seq zaya device decode stays oracle-exact
(9079/236761/107/2717/108/1882/735/1156).

Server multi-slot test (llama-server -np 4, device ngl99) now fails at the NEXT
unmatched shape instead:

    E graph_compute: unsupported HRX node 10: FLASH_ATTN_EXT
      output=19:f32[128,8,2,2]<-FLASH_ATTN_EXT
      inputs=[12:f32[128,2,8,2]<-PERMUTE, 15:f16[128,256,2,2]<-PERMUTE,
              17:f16[128,256,2,2]<-PERMUTE, 18:f16[256,2,1,2]] consumers=[11:RESHAPE]

= batched (2 seq x 2 tok) flash-attn with per-seq KV views; the flash dispatch
registry matches single-seq shapes only. Env-forcing flash to CPU at ngl99
(GGML_HRX_CPU_OPS=FLASH_ATTN_EXT) ABORTS in ggml_backend_sched_backend_id_from_cur
during sched_reserve (known claim/placement class; only ngl0 tolerates it).
Next round options: (a) shape-conditional flash claim (claim false when no
dispatch matches the multi-seq KV-view form - mirrors the fleet's per-op claim
work), or (b) extend the flash dispatch/loom kernel to per-seq KV views.
CPU-side multi-seq oracle (same 4 prompts, llama-server -np 4 ngl0) is clean and
correct, so numerics targets exist for the round.

## ROUND RESULT 2 (2026-09-08, agent-ec8072): batched-flash CPU-split is blocked by KV-cache placement, not the flash claim

Attempted the shape-conditional flash claim (device_supports_op gate: claim
FLASH_ATTN_EXT only for ne[3]==1 single-seq shapes so multi-seq flash splits to
CPU). Single-seq stays oracle-exact with the gate (ne3=1 flash unchanged), but
the -np 4 server ctx build still ABORTS:

    ggml-backend.cpp:898: pre-allocated tensor (cache_v_l0 (view) (permuted)
    (transposed)) in a buffer (HRX0) that cannot run the operation (TRANSPOSE)
    (ggml_backend_sched_backend_id_from_cur during sched_reserve)

Root cause: the V cache (cache_v_l0) is allocated on the HRX0 buffer (layers
offloaded at ngl99). CPU flash-attn needs the transposed V-cache view, so the
TRANSPOSE must run on CPU - but the pre-allocated (persistent) tensor sits in an
HRX buffer and the ggml sched refuses (can't copy a pre-allocated leaf). Env-
forcing flash to CPU showed the same abort (this is why GGML_HRX_CPU_OPS=FLASH
works only at ngl0 where the KV cache is CPU-resident anyway).
Reverted (tree back at the SET_ROWS-fix state c633916f4; single-seq device
decode re-verified oracle-exact 9079/.../1156).

Next-round options for device multi-seq (in order of size):
  1. KV-cache buffer placement for the flash/V path: allocate cache_v (and the
     transposed-view machinery) on the HRX host buft (HRX0_HOST, UMA/coherent)
     when the graph needs CPU flash, so CPU TRANSPOSE/view ops can read it -
     mirrors how llama_kv_cache chooses buft per backend; the host buft is
     device-visible so HRX flash (single-seq) must remain on device - needs a
     per-context conditional, not a blanket host buft.
  2. Batched flash dispatch/loom kernel for ne[3]>1 (real kernel project).
  3. Serve multi-seq on the CPU path only (works: -np 4 ngl0 correct, ~33 t/s
     aggregate) and keep device multi-seq as a documented follow-on - this is
     the honest interim for task-5 rows.

## CORRECTION (2026-09-08, agent-ec8072): llama-batched-bench does NOT decode for zaya - rc=0 is reserve/ctx-build only

Timing forensics (internal log timestamps mm.ss.mmm): model load ~34-39 s, then
the ENTIRE bench phase completes in <1 s at every npl (device AND CPU, ngl0 npl 2
-n 256 = 512 CPU tokens would take ~30 s if decoded - it took ~0 s). The harness
(examples/batched in bench mode) builds the context (which exercises the
multi-seq RESERVE graphs - real signal: the SET_ROWS reserve no longer aborts
post-c633916f4) then exits without running decode for this arch. The earlier
"throughput rows suppressed" reading (fleet + my addendum 3) is wrong: there is
no decode to print. Device multi-seq DECODE execution therefore remains
UNVERIFIED. The only real multi-seq driver is llama-server -np N, which fails at
ctx build on the KV-buft/v_trans placement (0d6d10ff8). llama-batched (coupled
seqs) is separately blocked: "split_equal: sequential split is not supported
when there are coupled sequences" (harness-level, memory splitter).

Net: post-c633916f4 the device multi-seq line stands at: reserve graphs build
clean at npl 1-8 (SET_ROWS store dispatchable); actual batched decode needs the
KV-placement/v_trans round (llama-server ctx build) or a working multi-seq
harness. Do not cite batched-bench rc=0 as decode evidence.

## KNOB A/B RESULTS (2026-09-08, agent-ec8072) - no free lunch; both fast knobs corrupt

Tested on the current tree (zaya-q4nx-c43, device ngl99, llama-bench tg64
baseline 6.98 t/s oracle-exact):

- GGML_HRX_ASYNC_JIT=1: tg64 6.85 (neutral, as expected - programs are cached by
  steady-state decode; JIT overlaps prefill only).
- GGML_HRX_USE_UNIFIED_MEMORY=1: tg64 8.29 (+19% RAW) but CORRUPT: zgreedy tok0=105
  vs oracle 9079, garbage text. Unified memory changes buffer semantics and
  breaks value freshness (stale reads across the shared/UMA path). NOT usable.
- GGML_HRX_KV_HOST=1 (committed A/B hook 0c688955e): corrupt single-seq decode
  (host-buft KV cannot round-trip device-written caches through the staging
  binding path - round-16e class). Default OFF.

Conclusion: no env-knob path to the 16.8 t/s contract. The launch-bound decode
(~600+ subgraph programs/token, ~0.17 ms launch+sync each, programs already
cache-hit via stable uids) needs structural work: SSM-conv/grouped-conv loom
kernels for contiguous-HRX graphs (f49062 lane, in flight) and/or a
submission-collapse executor feature (multi-program-per-submit does not exist
in this tree). Device multi-seq additionally needs a batched-flash dispatch or a
shared-buffer binding mode (see SETROWS-GATE-CONFIRMED.md).

## MULTI-SEQ DEVICE ROUTE CLOSURES (2026-09-08, agent-ec8072) - all alternative routes tested

Testing matrix for getting zaya multi-seq decode on the HRX device path beyond
the SET_ROWS fix (c633916f4). Every alternative route is now empirically closed:

1. In-context batching (llama-server -np 4): blocked at ctx-build on the V-cache
   transpose for CPU flash (ggml-backend.cpp:898) - the transpose comes from
   build_attn_mha (llama-graph.cpp:2427: v_trans -> transpose before flash).
2. GGML_HRX_KV_HOST=1 (host-buft KV, A/B hook 0c688955e): CORRUPT single-seq
   decode (value-freshness, round-16e class).
3. GGML_HRX_USE_UNIFIED_MEMORY=1 (direct coherent host bindings): CORRUPT
   (tok0=105; +19% raw speed was on wrong output). Root = the fleet's known
   "host-buft shared-arena reads are not device-coherent" (no CPU-side flush
   machinery for UMA direct reads) - this route was why 98b2bcc kept embd
   GET_ROWS on HRX. Re-confirmed, not a bounded fix.
4. Multi-process (4x llama-server -np 1, one HRX context each): catastrophic
   device contention - a 2-token prefill took 31 s (0.06 tok/s) with 4 contexts
   interleaving graph builds/executions on the single device queue. Dead.
5. llama-batched (coupled seqs): blocked by the memory splitter ("sequential
   split is not supported when there are coupled sequences").

Remaining real paths (unchanged, all structural):
- batched FLASH_ATTN_EXT dispatch/loom kernel (ne3>1, per-seq KV views), or
- executor coherence machinery (flush/ordering for device-written UMA caches),
or accept CPU-path multi-seq (verified, ~33 t/s aggregate) as the task-5
multi-seq-with-throughput row with device multi-seq as documented follow-on.

## BATCHED-FLASH KERNEL CONTRACT (2026-09-08, agent-ec8072) - captured from llama-server -np 2 device runtime

llama-server -np 2 on DEVICE (ngl99): ctx/reserve builds FINE (n_seq_max=2
reserve shapes stay dispatchable; n_seq_max=4 aborts at reserve on the V-cache
transpose - ggml-backend.cpp:898). Two concurrent 30-token completions process
6-7 tokens then fail at the first batched decode:

    E graph_compute: unsupported HRX node 10: FLASH_ATTN_EXT
      output=19:f32[128,8,4,2]<-FLASH_ATTN_EXT
      inputs=[12:f32[128,4,8,2]<-PERMUTE, 15:f16[128,256,2,2]<-PERMUTE,
              17:f16[128,256,2,2]<-PERMUTE, 18:f16[256,4,1,2]] consumers=[11:RESHAPE]

This IS the kernel contract for a batched-flash dispatch (2 active seqs, 4
decode tokens): Q f32[head=128, tokens=4, heads=8, seqs=2]; K/V f16[head=128,
kv=256, heads=2, seqs=2] (per-seq KV streams merged in ne3); mask f16[kv=256,
tokens=4, 1, seqs=2]. Same shape family as the earlier -np 4 reserve abort but
with the seq dim = active slots. The single-seq decode-split kernel
(ggml_flash_attention_decode_split_*_wmma) handles ne3==1 only; the batched
form = the stream-dim extension target. Per-stream KV spans are implicit in
the mask (f16[256,4,1,2] = kv x tokens per seq). Kernel contract recorded here
for the loom-authoring round; no dispatch exists for this shape today.

## ROUTE CLOSURE (2026-09-08, agent-ec8072): GGML_HRX_NO_FLASH_ATTN=1 corrupts device decode too

Tested the non-flash attention path (round-16 diagnostic env) on single-seq
device decode (ngl99): CORRUPT (tok0=236751 vs oracle 9079, garbage text; same
signature as the KV_HOST test). No crash - silent value corruption in the
non-flash attention path on device (kq/mask/v path external-binding class).
=> Disabling flash does NOT bypass the batched-flash wall; the non-flash path
is not device-correct in this tree either.

Complete env/flag closure list for device zaya decode (all tested 2026-09-08):
flash claimed single-seq = oracle-exact 6.98-7.14 t/s (THE working config);
FA off = corrupt; flash CPU-forced = abort/corrupt; KV host-buft = corrupt;
UNIFIED_MEMORY direct bindings = corrupt; multi-process = 31s/2-token
contention; batched flash ne3>1 = no dispatch (kernel contract captured
bb14eb5a0). Every route to multi-seq device decode = the batched-flash loom
kernel or coherence machinery. Nothing else remains to test.
