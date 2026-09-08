# Device-path multi-seq zaya: SET_ROWS KV-store dispatch gap — scope + fix plan (2026-09-08, agent-ec8072)

## Symptom (fresh, current tree ebf297c97 + doc commits)
`llama-batched-bench -m ~/zaya-q4nx-c43.gguf -ngl 99 -p 32 -n 24 -b 32 -npl 1,2,4,8`
fails at the first batched decode:
`E graph_compute: unsupported HRX node 0: SET_ROWS output=3:f16[256,2048,1,1] inputs=[0:f32[256,16,1,1], 1:i64[16,1,1,1], 2:f16[256,2048,1,1]] consumers=[]`
(graph_compute -1, decode ret -3; single-seq device decode at the same ngl is oracle-exact and
unaffected: zgreedy 9079/236761/107/2717/108/1882/735/1156, llama-bench tg256 7.14 t/s @1k / 6.91 @4k.)

## Root cause (identified at the source)
src/llama-kv-cache.cpp cpy_k/cpy_v (lines ~1479/1524/1545): when a decode batch's token
positions are scattered across sequences (multi-seq interleave), the KV store is
`ggml_set_rows(ctx, k, k_cur, k_idxs)` with i64 position ids -> a STANDALONE SET_ROWS whose
only effect is an external write to the device-resident KV cache (no in-graph consumers).
Single-seq decode takes the contiguous store path (no standalone SET_ROWS), which is why the
device path is green there.

Shape match: dst f16[256,2048] = one layer's flash-attn K cache (n_embd_gqa=256 = head_k 128 x
n_head_kv 2, kv_size 2048); src f32[256,16] = 16 batch tokens' computed K; ids i64[16,1].
Same family for V (and for f16 K when K-cache type is f16).

## Why it errors
SET_ROWS is eager-claimed (ggml-hrx.cpp eager_capability_declared) but the dispatch scheduler
has NO registration for a standalone SET_ROWS of this shape: the only registrations are the
rope+set_rows fused (common/dispatch-rope-set-rows.cpp, ggml_rope_set_rows_f32) and the fused
attention K/V mm kernels (llm_attention_{k_matmul_rope,v_matmul}_set_rows_*_wmma), which do not
match this standalone KV-store form -> "unsupported HRX node".

## Fix plan (round-sized; do NOT drive-by while single-seq is green)
1. New dispatch registration (dispatch_registration/..., e.g. common/dispatch-set-rows-store.cpp):
   match standalone SET_ROWS with (a) 2D (or merged-stream 2D) dst cache [n_embd_gqa, kv_size],
   f16 or f32; (b) row-contiguous src [n_embd_gqa, n_tokens] (view/cont already produced by
   cpy_k/cpy_v); (c) i64 ids [n_tokens]; (d) dst value = External (device-resident, no in-graph
   consumer -> external-write binding like the mm publishes). Register AFTER the fused
   rope/attention registrations so fused patterns win when present.
2. Kernel: reuse the generic corpus kernel `set_rows.loom` (@ggml_set_rows_gfx11_wave64,
   input/output_format 16/32 f16/f32 supported) - already present, currently only referenced via
   the rope-set-rows registration.
3. Verification battery (all on current tree, full): single-seq zaya ngl99 oracle probe
   (9079/.../1156) unchanged; qwen3-0.6B ngl99 canary 12095 unchanged; zaya ngl0 multi-seq
   npl 1-8 no-SEGV unchanged; NEW: zaya ngl99 batched-bench npl 1-8 completes; llama-server
   multi-slot zaya (device) completes; MoE models (30B/35B) alias-clean (the dispatcher alias
   rules must not grab their fused chains).

## Owner note
Asked @agent-f49062 on mesh 2026-09-08 ~23:15 whether it owns this or I should take it; no
reply yet (busy). This note intentionally leaves the tree at the known-good state.
