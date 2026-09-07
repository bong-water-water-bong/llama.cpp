# Prefill logits: 9079 collapses 17.47 -> -0.28 on device (agent-f49062, 00:5x UTC)

zlogits.cpp (this dir) prints top-12 prefill logits + scores of 9079/563.

CPU (GGML_HRX_DISABLE=1): top0=9079 @ 17.4690, 563 @ 12.9783
Device (ngl99, F16 dequant):  top0=563 @ 10.9678, 9079 @ **-0.2826**, top12 spread 9.0-11.0

Interpretation: a uniform ~6pt shift would put 9079 at ~11; it sits at -0.28 => the final
activation (norm(layer_out-39)) is DIRECTIONALLY damaged (orthogonal-ish to W[9079]),
not merely scaled. This localizes the real consumed-path corruption to the LAST blocks'
outputs (38-39 = the only blocks whose dumps show genuine value loss, sorted corr
0.65-0.99) - directly upstream of the CPU lm-head. All earlier blocks' consumed outputs
are in-order correct (block b+1 value-exactness argument).

=> Real bug candidate: arena/slot overlap or end-of-graph damage affecting the final
blocks' ffn outputs (moe_out-39/layer_out-39), NOT the gate_up mm compute. The mm1
per-partition fix direction (rounds 63-70) addressed a readback artifact.
