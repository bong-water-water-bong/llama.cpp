# SSM_CONV closed form LOCKED (2026-09-08, agent-ec8072) - weight-layout artifact resolved

The dw-conv semantics are now verified against same-execution graph captures at
corr 0.99999 (prefill) / 0.9998 (decode). The earlier "QK_dw rms ~27 vs
conv-input ~7 rules out the direct conv reading" (26-conv-prereqs addendum 3,
f49062) was a WEIGHT-LOADING ARTIFACT, not a layout op: the gguf
ssm_conv1d.weight [2, 1280] is ggml-contiguous (ne0=tap fastest) = the two taps
are INTERLEAVED per channel (w[tap][c] = file[2c + tap]). Reading it as
row-major (2,1280) scrambles the taps into near-uncorrelated garbage.

## Verified closed form (matches ggml-cpu ops.cpp:9605 ggml_compute_forward_ssm_conv_f32)

    conv_input x = [n_state=2 rows | token rows] x [d_inner=1280 channels]
                   (state rows zero on first execution, carried across steps;
                    d_inner = 2*n_qk region of the CCA q||k concat)
    y[t][c] = w[0][c]*x[t][c] + w[1][c]*x[t+1][c] + bias[c]     (d_conv = 2 taps)
    output rows = ncs - 1   (prefill 8->7 rows, decode 3->2 rows)
    row 0 = state-window output (~bias-only when state zeroed)

Weight layout for numpy/gguf readers: raw = load_f32(name); W = raw.reshape(1280, 2).T
-> W[tap][channel]. DO NOT reshape(2,1280).

## Capture recipe (repro)
    env GGML_HRX_DISABLE=1 GGML_ZAYA_DEQUANT_F16=1 GGML_DUMP_NODE=1 \
        GGML_DUMP_CONVSRC=1 GGML_DUMP_FILTER=QK_dw /tmp/zgreedy ~/zaya-q4nx-c43.gguf 0
    convsrc = /tmp/nodedump/convsrc_8x1280x1x1.bin (prefill) / 3x1280 (decode, last step)
    QK_dw = first 7x1280 / 2x1280 r0X file in /tmp/nodedump (r03_000 = prefill layer 0)
    reads: cs = np.fromfile(f,'<f4').reshape(1280,N).T  -> cs[pos][channel]

## Implication for the loom kernel
The SSM_CONV kernel contract is now fully specified (formula above + captured
shapes/layouts). This was the gating step for writing the SSM_CONV loom kernel
(contiguous-HRX zaya graphs -> launch collapse -> the 16.8 t/s path). Conv layer
0 blk.0 weights used; per-layer same structure.

## Grouped conv (CONV_1D_GROUPED) - structure + weight-layout warning (agent-ec8072)

Not yet fully derived (f49062 is authoring SSM_CONV first; this is for when the
grouped-conv kernel follows). Captured pair available: /tmp/nodedump r02_000 =
QK_dw (7x1280, input), r02_001 = QK_grp ADD (6x1280, output; QK_grp = grouped
conv + cca_conv_grp.bias), from GGML_DUMP_FILTER=QK_dw,QK_grp.

Structure (src/models/zaya.cpp ggml_conv_1d_grouped, line 27): the op splits
the 1280-channel QK into 10 groups x 128 IC/OC; per group: standard
ggml_conv_1d(weight_g [2,128,128], input_g = QK channels [g*128,(g+1)*128)),
outputs concat along dim 1. ggml_conv_1d = ggml_im2col + ggml_mul_mat
(ggml.c:4596): weight reshaped [IC*K=256, OC=128], im2col [OL=6, IC*K], result
reshaped/permuted -> [OL=6, OC=128] per group.

WEIGHT-LAYOUT WARNING (same trap as ssm_conv1d): cca_conv_grp.weight [2,128,
1280] ggml-contiguous = tap k fastest (memory index = k + 2*ic + 256*oc).
Raw flat read: W[oc][kk] = file[256*oc + kk] with kk = k + 2*ic -> use
raw.reshape(1280, 256)[oc_range] per group (NOT reshape(2,128,1280) row-major).
Verify against r02_000/r02_001 before writing the kernel.

## CONV_1D_GROUPED closed form VERIFIED (2026-09-08, agent-ec8072) - corr 1.0 vs graph capture

    QK_grp[ol][g*128 + oc] = SUM_{k=0..1} SUM_{ic=0..127} QK_dw[ol+k][g*128 + ic]
                                 * W[k][ic][g*128 + oc] + grp_bias[g*128 + oc]
    ol = 0..L-2 (L=7 input tokens -> 6 outputs, valid conv, s=1), g = 0..9 groups
    (10 groups x 128 channels over the 1280-wide q||k QK)

Verified against same-run captures (r02_000 QK_dw input 7x1280, r02_001 QK_grp
output 6x1280, blk.0.cca_conv_grp weights): corr 1.000000, mad ~0.16 = f16
accumulation (ggml_conv_1d emits im2col as F16 -> f16 mm - confirms the mm path).

Weight memory (tap-fastest, [ne0=2 taps, ne1=128 ic, ne2=1280 oc]):
    W[k][ic][oc] = flat[k + 2*ic + 256*oc]  ==  raw.reshape(1280, 128, 2)[oc][ic][k]
Group g reads QK_dw channels [g*128,(g+1)*128) and writes output channels
[g*128,(g+1)*128). Bias per output channel. Derivation from zaya.cpp:27 helper +
ggml.c:4596 conv_1d (im2col col = ic*2+k, mm contracts over 256, output permute
-> [ol, oc] after concat).

Verification script: /tmp/conv_grp_check.py (captures in /tmp/nodedump).
Kernel contract for CONV_1D_GROUPED now fully specified - ready for the second
loom kernel.

## SSM_CONV kernel validation (2026-09-08, agent-ec8072) - kernel correct but CANNOT fire: CPU-bound concat region

Preserved f49062's kernel integration (a0ee433aa) after its session died and
validated the device path with the kernel compiled in (binary built 01:49:18
predates all WIP edits - the committed state == the built state):

- Device single-seq decode: ORACLE-EXACT (9079/236761/107/2717/108/1882/735/1156,
  " Paris.") - no regression, no abort (the eager claim + gate fix are coherent).
- Speed: tg64 6.65 t/s vs 6.98 baseline - UNCHANGED.
- Forced-CPU conv (GGML_HRX_CPU_OPS=SSM_CONV): 6.67 - identical.
- GGML_HRX_EXECTIME: no ssm_conv kernel execution (80 log mentions are all
  tensor-load lines).

CONCLUSION: the kernel + dispatch + gate are correct but NEVER FIRE. The conv
region is CPU-bound because cca_conv_input is produced by ggml_concat(conv_state
[CPU-pinned recurrent cache], QKraw_t [HRX]) and GGML_OP_CONCAT has no HRX
kernel/claim - the concat lands on CPU, dragging the conv input + the SSM_CONV
with it (the eager claim never sees the op; the fleet's known CONCAT gap,
52ea3aa04). The 640-subgraph fragmentation persists from the CPU concat+conv
islands, so no speed change.

NEXT STEP (for the conv-speed lane): a GGML_OP_CONCAT loom kernel + dispatch
(contiguous channel-concat over the [state|tokens] x 1280 case), OR restructure
so the state concat happens device-side. Until then the SSM_CONV kernel stays
inert-but-validated. Kernel contract: CONV-CLOSED-FORM.md (both convs verified).
