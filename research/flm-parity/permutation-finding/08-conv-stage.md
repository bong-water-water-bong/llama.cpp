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
