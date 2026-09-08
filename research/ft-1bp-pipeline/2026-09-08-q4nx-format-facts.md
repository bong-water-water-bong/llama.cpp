# ft-1bp Q4NX conversion — exact path + format facts (2026-09-08, goal mtsy05dx)

Audit gap: FT model delivered as f32 GGUF (works on HRX, CPU==HRX0 verified);
the 1BP/Q4NX moat form is the remaining quant step. This note pins the exact
format facts so the conversion can be completed correctly.

## Q4NX format facts (measured from zaya-q4nx-c43.gguf, the known-good file)

- 280 tensors are Q4NX (type 42/43): per layer (40) x {attn_q, attn_k,
  cca_val_proj1, cca_val_proj2, attn_output, ffn_gate_up_exps, ffn_down_exps}
- 1003 tensors stay f32 (token_embd, output_norm, attn_norm, ffn_norm,
  post_attn_norm, router ffn_gate/biases/eda/mlp2/mlp4, res_scales, conv,
  embeddings, etc.)
- Q4NX tensor data in the GGUF: uint8 array shaped (n_tiles, 4608);
  attn_q: 256 tiles = (1024 out /32) x (2048 in /256); tile = 32 out-rows x
  256 in-cols; 4608 B/tile = 4096 data (4-bit, 32x256/2) + 512 scales.
- Tile0 layout probe: bytes 0..127 = 32 f32 row scales (dense, nonzero);
  data region begins ~byte 128 (nibble-packed); bytes 4224+ = second scale
  region (per-16-row?). 87% nonzero density.
- Logical f32 twin (zaya-q4nx-f32twin.gguf) ne = [2048, 1024] = ne0 2048
  (in, fastest), ne1 1024 (out). Q4NX tile (r,c) covers f32[in c*256.., out r*32..].

## Authoritative packer

- tools/convert_float32_bins_to_q4nx.py: pack_q4nx_tile() reconstructed from
  the engine's dequant_i8_signed_to_float_ex, self-test asserts corr 0.995,
  verified end-to-end for zaya1-8b.q4nx (issue #1763). Uses symmetric int4,
  scale = absmax/7, tc(0..7)=0..7 / tc(8..15)=-8..-1.
- Its layout: [out, in], tile rows 32 / cols 256; nibble packing per ws12
  packer (2 rows per 256B lane, bitwise-or at col*8 + rowhalf*4...). The
  4608-vs-5120 difference: c43 tiles are 4608B (this engine's Q4NX), the
  ws12 loom TILE_BYTES=5120 is the OTHER (loom) format - do NOT mix.

## Completion path (when budget allows)

1. Read f32 tensor from the FT GGUF (ne0=in fastest) -> logical [in, out]
2. Transpose to packer [out, in]; tile (r,c) = rows r*32.., cols c*256..
3. pack_q4nx_tile per tensor -> (n_tiles, 4608) uint8
4. Write GGUF: copy FT7 metadata + 1003 f32 tensors as-is; add 280 Q4NX
   tensors via GGUFWriter add_tensor(name, bytes, raw_shape=[n_tiles,4608],
   raw_dtype=42) - matching c43's file encoding (verify n_bytes matches
   c43 per tensor)
5. Validate: load on HRX0 (q35 build), decode vs CPU oracle - the zgreedy_dev
   gate; also byte-compare tile0 vs c43 for a shared (un-FT) tensor to
   confirm the packer is bit-exact before trusting FT values.
