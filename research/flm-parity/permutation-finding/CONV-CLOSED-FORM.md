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
