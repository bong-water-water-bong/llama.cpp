# Pre-attn chain: input path VALIDATED; conv stage = first wrong device output; state cache suspect (agent-f49062)

Follows 06/07 docs. All numpy, bit-validated against device dumps (6-token prefill, block 0
unless noted; same prompt).

## 1. The full input chain is CORRECT on device (mad 0.0)
- zaya.cpp applies input_hidden_states_scale.weight[2048] + bias to the token embeddings
  BEFORE the first norm (inpL = (embd + bias) * scale) - the missing term that broke my
  earlier reconstruction.
- With it: cur = rmsnorm((embd+bias)*scale, attn_norm.w) reproduces the device input_norm-0
  for all 6 tokens EXACTLY (mad 0.000000). Token embedding rows are contiguous
  (data[t*2048..]); embedding values are sparse/coarse (exact zeros - model property).
- Q/K projections: W_q (1024x2048) / W_k (256x2048) Q4NX dequant x cur reproduces the
  device Qraw/Kraw to f16 noise (mad 0.010/0.031).

## 2. The FIRST WRONG device output = the SSM conv stage (QK_dw / QK_grp)
- QK_dw-0 (7 x 1280, dw conv of [state2 + tokens6] -> valid 7): rows 0-4 have sane rms
  (1.30, 1.29, 1.04, 0.60, 0.52) but do NOT match ANY numpy reconstruction of w-conv
  applied to the (validated) QKraw rows (best mad ~4 across all tested tap orders and
  row mappings) => values wrong on all rows, not just the tail.
- Rows 5-6: rms 84.9 / 80.8 = GARBAGE (vs ~1 for rows 0-4) => the conv-state rows
  (cca_state recurrent cache: conv_state = 2 x n_qk + prev_hs = 2048) are GARBAGE at the
  prefill start, i.e. the recurrent state cache is NOT zero-initialized / mis-viewed.
- QK_grp-0 (grouped conv out, 6 x 1280, feeds the attention Q/K): rows 4-5 = rms
  4449 / 9575 in block 0 (rows 0-3 sane 1.7-7.7). Across the 40 blocks the row-rms
  garbage (rms > 50) appears only in blocks 0 (4449/9575), 37 (39/45), 38 (453/983);
  other blocks' rows all rms 0-4 (but untested for correctness, and given the dw rows are
  wrong everywhere in b0, per-block value checks are pending).
- conv weights in-file: ssm_conv1d.weight = [2, 1280] ne0=2 (tap-fastest!); the earlier
  (2,1280) reshape reads were transposed; corrected read still does not match => the
  defect is not my reconstruction.

## 3. Interpretation (sharpest formulation)
- The gate_up mm, its input norm, the attention-input norm, and the Q/K projections are
  all provably correct on device. The conv stage output (QK_dw/QK_grp) is the first
  tensor whose device values are wrong (all rows) with outright garbage on the
  state-derived rows 5-6 (dw) / 4-5 (grp) in b0/b37/b38.
- The conv input includes the recurrent conv_state from the layer's cca state cache; its
  garbage implies the state cache content (or its view/offset into the s_l buffer) is
  wrong at prefill start - the likely root of the whole multi-token corruption
  (state -> conv rows -> Q/K -> attention -> residual -> x -> gate_up rows).
- Why decode (1 token) stays grammatical: single-token conv uses the state updated by
  the previous step; the prefill's 6-token conv state path (or its initial content)
  differs.

## Next probes
a. Dump the conv INPUT (cca_conv_input = concat(state, QKraw)) via the conv kernel's
   input binding (pddbg uid 10846 shows b2 = 35840B in/out - identity names needed) and
   inspect the state rows directly (expect zeros; garbage confirms the cache/offset bug).
b. Compare the DECODE (1-token) QK_dw (2 rows = 10240B) against numpy from the decode
   cur + state-after-prefill: clean => multi-token conv path is the discriminator.
c. Inspect the cca state buffer allocation/zeroing (llama-memory-hybrid / build_rs /
   get_s_l) for the zaya recr cache at n_past=0.

## Addendum (same session): the conv_state garbage is UNIVERSAL (decode too)
- Decode-path QK_dw (1-token steps, uid 11930 zone, all 8 steps): row 0 rms 1.20-1.20
  (the token row = sane) but row 1 rms 75.4-76.0 = GARBAGE in EVERY step. The conv
  state-derived output row is garbage in the decode path as well as the prefill
  (rows 5-6, rms 81-85).
- The garbage rms ~75-85 is far above the qk scale (~1-2) => the recurrent conv_state
  cache content (or its view/offset into the s_l buffer) is garbage at ALL times, not
  just at n_past=0. The prev_hs half of the same state row appears fine (token-0 chains
  are exact and V2 uses prev_hs).
- Consequence: the conv token rows that mix in the state are corrupted; the state
  update loop (last_conv_states = the conv tail rows) re-writes garbage into the cache,
  sustaining the corruption.
- The decode text (" is used to hide the problem. The") is coherent BUT that only proves
  the token-row q/k of the decode conv is usable; the decode may still be subtly wrong
  vs the CPU continuation of tok0=563 (no CPU oracle for the 563-continuation exists -
  the r30 series vanished). Do not treat "grammatical text" as decode-correctness.
## Addendum 2 (same session): SSM_CONV = CPU fallback inside the HRX graph
- The loom corpus has NO conv kernel (ops/ list + manifest: zero conv entries) and no
  SSM_CONV op-params/dispatch handling, yet QK_dw/QK_grp are bound+dumped as
  3-binding kernel commands (out/weight/in). => the conv-family ops execute as CPU
  fallbacks (graph_replay_should_fallback machinery) with host-visible buffers.
- The conv weight binding = 5120B @ the model tensor offset = the F16-converted
  ssm_conv1d.weight [2,1280] (2560 f16 = 5120B) - confirms the F16 device-weight path.
- The conv input binding (b2) = 35840B = 7 rows x 1280 f32 - one row short of the
  8-row [conv_state(2) + QKraw(6)] concat; the fallback's input assembly (cross-backend
  concat of the CPU-pinned zeroed state cache + the HRX QKraw via d2h) is the suspect
  for the corrupted rows (output rows 0-4 value-wrong vs every numpy reconstruction;
  rows 5-6 = rms 81-85 garbage; decode row 1 = rms 75).
- The recurrent cache (cache_s_l0) = CPU buffer, zeroed at alloc (buffer_clear 0) with
  the explicit comment pinning it to CPU because HRX lacks SCALE/conv kernels.

=> Next probe (decisive): capture the conv fallback input region (the value bound as
   b2 of uid 10846 / 11930, likely the materialized cca_conv_input) and compare against
   [zeros(2,1280) ; QKraw] - distinguishes "state upload corrupt" vs "QKraw d2h corrupt"
   vs "fallback assembly wrong". The 10843_001_Qraw/Kraw dumps prove the HRX-side
   projections are correct; the copy path is unverified.
