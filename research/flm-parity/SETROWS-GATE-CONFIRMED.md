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
