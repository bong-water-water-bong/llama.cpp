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
