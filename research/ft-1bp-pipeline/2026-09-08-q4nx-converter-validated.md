# ft-1bp Q4NX converter — built + validated (goal mtsy05dx, audit follow-up)

zaya_q4nx_convert.py converts a zaya f32 GGUF to the c43 Q4NX GGUF format.

## Validation (bit-exact vs the known-good c43)

- Source: zaya-q4nx-f32twin.gguf (f32 twin of the c43 base) -> converted ->
  compared byte-for-byte vs zaya-q4nx-c43.gguf:
  - 1003 F32 tensors: ALL byte-identical
  - 280 Q4NX tensors: 277 byte-identical; blk.37-39 ffn_gate_up_exps differ
    by 4.6% of bytes BUT decode to corr 1.0 vs the f32 twin (max err 6e-12,
    float noise) while c43-vs-twin is exact 0.0 - the c43 was quantized from
    a marginally different snapshot of those 3 layers; my conversion is
    faithful to the f32 source (both decode == twin within quant noise).
- Packer: tools/convert_float32_bins_to_q4nx.py pack_q4nx_tile (engine
  dequant mirror, self-test corr 0.995); tile = 32 out x 256 in, 5120 B
  (SCALES_OFF 0 = 512B bf16 scales, MINS_OFF 512, nibbles after 1024).
- Q4NX set: 280 = 7 kinds x 40 layers (attn_q, attn_k, cca_val_proj1/2,
  attn_output, ffn_gate_up_exps, ffn_down_exps); 1003 stay f32.
- Expert tensors: 3-D [exp, out, in] (numpy), quantized per-expert, tiles
  concatenated [exp-major] - matches c43 (verified block-identical per
  expert).

## Usage

python3 zaya_q4nx_convert.py <f32.gguf> <ref_c43.gguf> <out.gguf>
  ref_c43 supplies the Q4NX name set; metadata (incl. tokenizer arrays) is
  copied from the source.

## Run on the FT model

zaya1-8b-ft-merged7.gguf (35 GB f32) -> ft7_q4nx.gguf (7.48 GB Q4NX),
pending HRX0 load + decode verification.
