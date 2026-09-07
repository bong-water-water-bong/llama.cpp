## Addendum 8: correction - conv = CPU-side (no mis-dispatch); my conv-input layout assumption was likely transposed

- The recnode wg=35x1 (256 lanes) with 35840B bindings that I read as "the conv
  mis-dispatched to ggml_binary_bc_f32" is actually the BIAS-ADD command (a real
  GGML_OP_ADD = binary_bc over the 35840B post-conv tensor) in the same program. The
  SSM_CONV itself has no dispatch registration anywhere and no loom kernel; it executes
  CPU-side (ggml), with its output uploaded via host staging (the "CPU#QK_dw-0 (view)"
  uploads) and the bias-add done on HRX. No mis-dispatch.
- My numpy reconstructions of the conv used x[i] = the i-th logical row = physical
  i*1280+c. But the ggml conv_input tensor = ne [8, 1280] with ne0 = 8 (the time dim
  FASTEST): element (t, c) at physical t + c*8 (ops.cpp reads s[k + c*ncs] =
  layout-aware). The dump reshape (7,1280) of the [1280,7] output = correct, but my
  INPUT-side row assumption = transposed vs the actual staged layout. ALL conv-stage
  comparisons from addenda 3-7 are therefore unreliable (mad/corr vs a mis-laid input).
- What remains airtight: the entire upstream chain validated bit-exact; the gate_up mm
  input x (post_attn_norm output) = wrong on-device for t>=1 (via the W*x == dump
  identity + the dump != oracle); the norm is consistent with its input; the residual
  (residual_post_attn) is wrong for t>=1; the attention stage output feeds it.

## Next (corrected)
Rebuild the conv-input numpy with the ggml layout ((t,c) at t + c*8 for the [8,1280]
concat; state = zeros; tokens at t = 2..7 = QKraw rows in the (t,c) layout) and
re-compare against the QK_dw/QK_grp dumps. If the conv now matches, the corruption is
DOWNSTREAM of the conv (flash-attention or its Q/K/V staging) - which also matches the
qwen3moe #2147 finding that the attention-class ops are the fragile ones on this stack.
