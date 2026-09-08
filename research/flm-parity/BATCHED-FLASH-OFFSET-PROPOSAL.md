# BATCHED FLASH_ATTN_EXT — OFFSET-REBIND PROPOSAL (grounded, 2026-09-08, agent-aa1ef2)

Follow-up to BATCHED-FLASH-IMPL-PLAN.md (efe770683). This proposes a SMALLER
route than Option A of that plan: per-stream **binding-offset rebinding** of
the existing decode-split kernel. If the batched tensors are contiguous, NO
new copy kernel and NO kernel rewrite are needed — only a matcher extension.

## Grounding (read from current code, adc03bd2b)

- `DispatchBinding { ValueId value; size_t offset; size_t length; }`
  (dispatch/dispatch.h:27). Bindings are value + byte range — NOT
  node-bound. Transients already appear as bindings (partial_max/sum/output
  in dispatch-flash-attention.cpp:385-388), so binding a dispatch input to a
  buffer slice is established machinery.
- The decode-split matcher emits one `Dispatch` per query ROW
  (dispatch-flash-attention.cpp:419-440) with per-row offsets:
  q = {id, row*nb[1], query_row_bytes}, mask = {id, row*nb[1], mask_row_bytes},
  output = {id, row*nb[2], output_row_bytes}; K/V = {id, 0, full_byte_count}.
- The matcher hard-requires ne[3]==1 everywhere (lines 274-277) and the
  kernel contract treats buffers as flat [head, tokens, heads, 1].

## The batched shape (from the np2 runtime capture)

q f32[128,4,8,2] · K f16[128,256,2,2] · V f16[128,256,2,2] · mask f16[256,4,1,2] ·
out f32[128,8,4,2]. ne[3]=N = active sequences. For a CONTIGUOUS tensor the
stream-s slice is exactly bytes [s*nb[3], s*nb[3]+byte_count_per_stream) with
the SAME intra-stream strides (nb[0..2]) the single-seq kernel already
compiles against.

## Proposed change (matcher-only)

1. New matcher `match_flash_attention_ext_decode_split_batched` registered at
   priority > 75 (or gate the existing matcher) accepting ne[3] = N > 1 with
   the same type/head-size/token-count gates, plus:
   - per-stream contiguity gate: nb[3] == ne[0]*ne[1]*ne[2] * type_size for
     q/K/V/out and mask nb[3] == mask stream stride (identical per-stream
     layout: nb[0..2] equal to the single-seq expectations of
     has_query_layout/has_key_value_layout/has_mask_layout/has_output_layout);
   - mask per-stream column stride consistent (mask ne[2]==1 already).
2. Emission: for stream s in [0,N), for row r in [0,query_token_count):
   one Dispatch with
     q      = {q->id,     s*q->nb[3] + r*q->nb[1], query_row_bytes}
     K      = {key->id,   s*key->nb[3],           per_stream_key_bytes}
     V      = {value->id, s*value->nb[3],         per_stream_value_bytes}
     mask   = {mask->id,  s*mask->nb[3] + r*mask->nb[1], mask_row_bytes}
     output = {out->id,   s*out->nb[3] + r*out->nb[2],   output_row_bytes}
   plus the SAME partial/completion/q8 transients per (row,stream) dispatch.
   Kernel + loom + compile params UNCHANGED (token_capacity per stream is the
   per-stream kv token count, already the compile-param contract).
3. If the contiguity gate fails for ANY input (view-wrapped stream dim), the
   matcher returns false → node splits to CPU → the known kv-placement abort
   class (#2153's current failure at n_seq_max=4). That abort is the
   SEPARATE cache_v_l0-view item, unchanged by this proposal.

## Open questions the implementation must resolve (lane-level)

1. **Transient sharing**: the existing per-row dispatches share
   partial_max/sum/output at offset 0 (they are serialized by the executor).
   With rows×streams dispatches the same sharing holds ONLY if the executor
   serializes within a match — verify, else per-(row,stream) transients.
2. **Completion counter**: the counter request bumps per dispatch and the
   reduce waits for key_value_head_count. With N×rows dispatches the expected
   count must scale (N×rows×kv_heads) or the reduce fires early on stale
   partials — the one semantic change beyond bindings.
3. **q8_output alternate**: next_q8_output per stream (or rebind per stream
   slice) so the follow-up q8 consumer of stream s reads stream s.
4. **Prefill mask width**: batched mask columns are per-stream KV rows
   (ne[0]=256 covers BOTH streams' KV? or per-stream 128?). The captured
   shape mask f16[256,4,1,2] has ne[0]=256 with ne[3]=2 — reconcile whether
   the kernel's kv-tile walk over the bound region sees the right rows.

## Validation (from the plan, unchanged)

- np2: device decode vs CPU ngl0 multi-seq oracle, token-exact.
- np4: ctx build must no longer abort at ggml-backend.cpp:898 (the
  cache_v_l0 view item — see below).
- Single-seq regression: zaya 9079/... stream byte-exact; qwen roster canary
  12095; 30B/35B alias-clean.

## Relationship to the remaining np4 abort

The n_seq_max=4 ctx-build abort (cache_v_l0 transposed view on HRX0 vs CPU
TRANSPOSE) is NOT fixed by this matcher — it is the V-cache view placement
item and must be solved in the same pass (either a TRANSPOSE/view claim for
that shape or kv-buft placement fix) for -np 4 acceptance.

## Effort

Matcher extension + 3 semantic checks (counter, transients, q8 alternate) +
device validation: SMALL-MEDIUM (hours, not days) vs Option B's kernel
rewrite. This is the bounded intermediate the plan's Option A wanted, minus
the copy dispatch.
