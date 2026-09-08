// Dispatch for the SSM (Mamba-style) 2-tap depthwise conv, f32.
// Contract (CONV-CLOSED-FORM.md): input [ncs = n_t+2, d_inner] contiguous,
// weight [2, d_inner] tap-interleaved, output [d_inner, n_t].
#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_ssm_conv_dispatches(DispatchRegistryBuilder & builder);

}  // namespace ggml::hrx
