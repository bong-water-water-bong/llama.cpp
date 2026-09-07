# Probe (a) result: rd_act activation-fetch decode (agent-f49062, 2026-09-07)

Follow-up to 04-CORRECTION.md. Data: the publish-trace run's row_debug dumps
(/tmp/prg_dump/<uid>_001_common.moe_routing.row_debug.bin, uid = 10866+27*block).
Regions (f32 index from buffer start; buffer = 100 KB = 25600 f32):
- rd_act activations: f32[8192 + token*32 .. +32] (first 32 k-values of the token
  each partition fetched; written at quant_block==0/quant_group==0).
- rd_fview weights: f32[32 + expert*256 + channel*32 + packet*4] (first 8 output
  channels x first 32 k-values of the expert's dequantized rows).
- Publish trace (mine): f32[2048 + (a*64+t)*4 + s*2 + f] = lane-0 fragment samples.

Decoded rd_act (all 40 blocks x 6 rows): no near-duplicate rows in any block; the six
fetched activation rows are distinct everywhere (max pairwise corr <= 0.49) and their
rms scales smoothly (row a0 grows 0.02 -> 2.0 across blocks; rows a1-5 ~0.4-2.4) -
consistent with genuine per-token activations, NOT a duplicated/misrouted fetch at the
32-value sample resolution. Caveat: samples only k 0..31 of the first quant block; a
wrong-row fetch of LATER k-windows, or a wrong-EXPERT weight fetch, would not show here.

Next probes (unchanged priority, in 04-CORRECTION.md):
(b) compare wrong rows' values against W[E_p] x x[t] recomputed for wrong expert
    pairings (needs gguf weights + oracle activations in numpy);
(c) extend the in-kernel capture to later quant_blocks (k-window offset theory);
(d) decode rd_fview weight captures vs the model-file dequant (Q4NX -> f16) to rule
    the weight fetch in/out.
