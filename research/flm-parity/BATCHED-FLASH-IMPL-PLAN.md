# BATCHED-FLASH IMPLEMENTATION PLAN (grounded, 2026-09-08, agent-ec8072)

Target: dispatch + kernel for FLASH_ATTN_EXT with ne3>1 (multi-seq decode,
llama-server slots). Contract captured in bb14eb5a0. This plan is grounded in
the actual kernel/dispatch code (read 2026-09-08).

## Kernel structure today (flash_attention_decode_split_f32_f16_wmma.loom, 1225 L)
Single-stream decode-split FA: produce_partials (wmma over KV tiles, kDecodeKvTileSize)
-> partial_max/sum/output -> reduce_fused (direct<=256 / cooperative 257-2048 KV)
-> pack q8 outputs. NO stream/ne3 parameter anywhere - buffers are flat
[head, tokens(<=kDecodeRowCapacity), heads, 1]. Dispatch matcher (dispatch-flash-
attention.cpp match_decode_split_flash_attention_f32_f16) hard-requires
ne[3]==1 on q/k/v/o + mask ne[2]==ne[3]==1, plus layout gates (has_query_layout /
has_key_value_layout / has_mask_layout) and per-shape kernel compile params.

## The batched shape (from np2 runtime capture)
q f32[128,4,8,2], K/V f16[128,256,2,2], mask f16[256,4,1,2], out f32[128,8,4,2].
Two streams (ne3=2); in llama.cpp decode each active slot contributes its token(s)
with per-stream KV rows + per-stream mask columns.

## Extension design options (in increasing size)
A. Executor/import decomposition: the dispatch scheduler cannot add graph ops, but
   a matcher CAN emit transients + initialization dispatches. For each stream s:
   cont-copy q/K/V/mask stream slices into per-stream transient buffers, then run
   the EXISTING single-stream decode-split dispatch per stream (N dispatches), then
   interleave outputs back. Requires a device-side strided->cont copy dispatch
   (round-68 class: strided views are NOT contiguously bindable; the executor's
   copy path today is host staging, not device-to-device strided). Cost: copies of
   K/V per stream per step (~128x256x2x2 f16 = 128 KB/stream - acceptable at
   decode scale) but adds N-1 copy launches -> decode gets MORE launch-bound.
   Size: medium (copy dispatch + matcher + output interleave); the copy kernel is
   the missing primitive.
B. Stream-loop kernel extension: add a stream (ne3) dimension to produce_partials/
   reduce: each wave handles one stream's KV tiles; partials per stream; mask
   indexed per (stream, token). Q/K/V/mask buffers carry the stream stride (nb3).
   Keeps 1 launch/step. Size: LARGE (rewrite of the 1225-L kernel + matcher +
   layout gates + compile params + validation vs the CPU multi-seq oracle).
C. Coherence machinery (device-writable UMA caches) -> lets CPU flash handle the
   batched shapes (host-side reference FA) while device mms stay claimed. Size:
   executor memory-ordering project (round-16e class), unknown risk.

## Recommendation
B is the correct long-term shape (no extra launches, matches the single-seq
perf profile); A is a bounded intermediate that PROVES multi-seq numerics on
device with the existing kernel (throughput secondary). C only if the CPU-flash
route becomes preferred.

## Validation
- Single-seq device oracle must stay exact (9079/...) on every change.
- Batched decode vs CPU ngl0 llama-server -np 2 oracle (correct outputs known).
- llama-server -np 4 device ctx build + concurrent completions = the acceptance.
- Regressions: qwen roster canary 12095, 30B/35B alias-clean, zaya ngl0.

## Effort reality
Kernel project sized LARGE (multi-day for one lane). Grounded alternative for
immediate task-5 evidence = CPU-path multi-seq (verified ~33 t/s aggregate) as
the multi-seq-with-throughput row + device single-seq rows; device multi-seq
stays a documented follow-on until A/B/C lands. All routes tested and closed;
this plan + contract = the complete spec.
