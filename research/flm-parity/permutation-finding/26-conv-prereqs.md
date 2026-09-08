
# Zaya conv kernel pre-reqs (agent-f49062, measured shapes)

## Time split (GGML_HRX_EXECTIME, decode 175 ms/tok)
- HRX backend graph_compute: ~640 calls at ~78 us avg = ~50 ms/token.
- The rest (~125 ms) = the CPU-side islands + the sched mixed-mode overhead
  (the pure-CPU decode is 69 ms; the CPU islands pay per-subgraph thread-pool
  launches in the mixed mode).
- 16.8 t/s (60 ms/tok) needs the mm-work offloaded at device speed with the
  subgraph count collapsed (~<100) and the CPU-island overhead gone.

## Conv shapes (GGML_DUMP_NODE captures, zaya-q4nx-c43)
- cca_conv_input (CONCAT of conv_state + QKraw_t): [2 + n_t, 1280, 1, 1]
  (prefill n_t=6 -> 8x1280; decode n_t=1 -> 3x1280).
- ssm_conv1d.weight: [2, 1280] (2-tap per-channel conv, f32).
- ssm_conv1d.bias: [1280].
- cca_conv_grp.weight: [2, 128, 1280] (grouped 2-tap conv, 128 groups).
- cca_conv_grp.bias: [1280].
- QK_dw (conv out + bias): [n_t+1?, 1280]-shaped in the dumps (7x1280 prefill,
  2x1280 decode) - the exact output layout needs the CPU ssm_conv reference
  (ggml-cpu ops.cpp ggml_compute_forward_ssm_conv_f32: dst = [d_inner,
  n_t, n_s]) + the round-28 branch verified the dw-conv math (md 3e-5).

## Next session (kernel program)
Write loom kernels for GGML_OP_SSM_CONV + GGML_OP_CONV_1D_GROUPED in
kernel-corpus/kernels/loom-libs/ops/, add the manifest exports, register the
dispatches (dispatch_registration/common), and claim them eagerly. Once the
convs run on-device the per-block attention chain becomes contiguous HRX and
the subgraph count collapses toward the qwen3-like structure.
