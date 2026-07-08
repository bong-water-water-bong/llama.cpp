// hip_zaya_ops.hip — Zaya CCA attention HIP kernels for AMD Strix Halo (gfx1151)
//
// Implements:
//   ggml_ssm_conv        — depthwise 1D convolution used by Mamba/Zaya CCA
//   ggml_conv_1d_grouped — grouped 1D convolution used by Zaya CCA
//
// Metal reference: ggml/src/ggml-metal/ggml-metal.metal

#include <hip/hip_runtime.h>
#include <cstdio>

// ─── ggml_ssm_conv ─────────────────────────────────────────────────────
//
// CPU reference: ggml_compute_forward_ssm_conv_f32 in ggml-cpu/ops.cpp
//
// Inputs:
//   src0: [d_conv-1+n_t, d_inner, n_seqs]  (convolution input, sliding window)
//   src1: [d_conv, d_inner]                 (convolution weight)
// Output:
//   dst:  [d_inner, n_t, n_seqs]            (convolution output)
//
// Each thread handles one (d_inner, token, seq) — computes dot product
// of conv weight [d_conv] with sliding window input [d_conv].

extern "C" __global__ void kernel_ssm_conv_f32(
    const float* __restrict__ src0,  // [ncs, nr, n_s]
    const float* __restrict__ src1,  // [nc, nr]
    float* __restrict__ dst,         // [nr, n_t, n_s]
    int64_t nc,   // d_conv
    int64_t ncs,  // d_conv - 1 + n_t
    int64_t nr,   // d_inner
    int64_t n_t,  // tokens per sequence
    int64_t n_s,  // number of sequences
    int64_t nb01, int64_t nb00, int64_t nb02,  // src0 strides
    int64_t nb11,                                // src1 stride
    int64_t nb0,  int64_t nb1,  int64_t nb2     // dst strides
) {
    int64_t ir = blockIdx.x * blockDim.x + threadIdx.x;  // d_inner index
    int64_t i2 = blockIdx.y;                               // token index
    int64_t i3 = blockIdx.z;                               // seq index

    if (ir >= nr || i2 >= n_t || i3 >= n_s) return;

    // Pointer arithmetic matching the CPU implementation
    const float* s = (const float*)((const char*)src0 + ir * nb01 + i2 * nb00 + i3 * nb02);
    const float* c = (const float*)((const char*)src1 + ir * nb11);
    float*       x = (float*)((char*)dst + ir * nb0 + i2 * nb1 + i3 * nb2);

    float sumf = 0.0f;
    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }
    x[0] = sumf;
}

// Vectorized version — processes 4 floats at once for nc % 4 == 0
extern "C" __global__ void kernel_ssm_conv_f32_4(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    int64_t nc,
    int64_t ncs,
    int64_t nr,
    int64_t n_t,
    int64_t n_s,
    int64_t nb01, int64_t nb00, int64_t nb02,
    int64_t nb11,
    int64_t nb0,  int64_t nb1,  int64_t nb2
) {
    int64_t ir = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i2 = blockIdx.y;
    int64_t i3 = blockIdx.z;

    if (ir >= nr || i2 >= n_t || i3 >= n_s) return;

    const float4* s = (const float4*)((const char*)src0 + ir * nb01 + i2 * nb00 + i3 * nb02);
    const float4* c = (const float4*)((const char*)src1 + ir * nb11);
    float*        x = (float*)((char*)dst + ir * nb0 + i2 * nb1 + i3 * nb2);

    float sumf = 0.0f;
    int64_t nv = nc / 4;  // number of float4 vectors
    for (int64_t i0 = 0; i0 < nv; ++i0) {
        sumf += s[i0].x * c[i0].x + s[i0].y * c[i0].y + s[i0].z * c[i0].z + s[i0].w * c[i0].w;
    }
    // Handle remaining elements
    for (int64_t i0 = nv * 4; i0 < nc; ++i0) {
        sumf += ((const float*)s)[i0] * ((const float*)c)[i0];
    }
    x[0] = sumf;
}

// ─── ggml_conv_1d_grouped ──────────────────────────────────────────────
//
// Standard grouped 1D convolution (NOT depthwise — groups > 1, groups != IC).
// For Zaya CCA: groups = n_head + n_head_kv,
//   kernel: [kernel_size, IC_G, OC]  (IC_G = IC / groups, OC = total output channels)
//   input:  [L, IC, N]
//   output: [L_out, OC, N]   where L_out = (L - kernel_size) / stride + 1
//
// We implement it as a single fused kernel (not decomposed into per-group
// conv_1d calls) for efficiency.

extern "C" __global__ void kernel_conv_1d_grouped_f32(
    const float* __restrict__ kernel,  // [K, IC_G, OC]
    const float* __restrict__ input,   // [L, IC, N]
    float* __restrict__ output,        // [L_out, OC, N]
    int64_t K,     // kernel size
    int64_t IC,    // total input channels
    int64_t IC_G,  // input channels per group
    int64_t OC,    // total output channels
    int64_t OC_G,  // output channels per group
    int64_t groups,
    int64_t L,     // input length
    int64_t L_out, // output length
    int64_t N,     // batch size
    int64_t stride,
    int64_t i_stride, // input stride per channel (L)
    int64_t o_stride  // output stride per channel (L_out)
) {
    int64_t oc = blockIdx.x * blockDim.x + threadIdx.x;   // output channel
    int64_t l  = blockIdx.y;                                // output position
    int64_t n  = blockIdx.z;                                // batch index

    if (oc >= OC || l >= L_out || n >= N) return;

    // Determine which group this output channel belongs to
    int64_t g = oc / OC_G;
    if (g >= groups) return;

    // Input channel offset for this group
    int64_t ic_offset = g * IC_G;

    float sumf = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        int64_t input_pos = l * stride + k;
        if (input_pos >= L) break;
        for (int64_t ic = 0; ic < IC_G; ++ic) {
            float k_val = kernel[k * IC_G * OC + ic * OC + oc];
            float i_val = input[input_pos + (ic_offset + ic) * i_stride + n * L * IC];
            sumf += k_val * i_val;
        }
    }
    output[l + oc * o_stride + n * L_out * OC] = sumf;
}

// ─── Launcher wrappers ──────────────────────────────────────────────────

extern "C" {

void launch_ssm_conv_f32(
    const float* src0, const float* src1, float* dst,
    int64_t nc, int64_t ncs, int64_t nr, int64_t n_t, int64_t n_s,
    int64_t nb01, int64_t nb00, int64_t nb02,
    int64_t nb11,
    int64_t nb0, int64_t nb1, int64_t nb2,
    hipStream_t stream
) {
    // Use vectorized version if nc % 4 == 0
    if (nc % 4 == 0) {
        dim3 block(256, 1, 1);
        dim3 grid((nr + 255) / 256, n_t, n_s);
        hipLaunchKernelGGL(kernel_ssm_conv_f32_4, grid, block, 0, stream,
            src0, src1, dst, nc, ncs, nr, n_t, n_s,
            nb01, nb00, nb02, nb11, nb0, nb1, nb2);
    } else {
        dim3 block(256, 1, 1);
        dim3 grid((nr + 255) / 256, n_t, n_s);
        hipLaunchKernelGGL(kernel_ssm_conv_f32, grid, block, 0, stream,
            src0, src1, dst, nc, ncs, nr, n_t, n_s,
            nb01, nb00, nb02, nb11, nb0, nb1, nb2);
    }
}

void launch_conv_1d_grouped_f32(
    const float* kernel, const float* input, float* output,
    int64_t K, int64_t IC, int64_t IC_G, int64_t OC, int64_t OC_G,
    int64_t groups, int64_t L, int64_t L_out, int64_t N,
    int64_t stride, int64_t i_stride, int64_t o_stride,
    hipStream_t stream
) {
    dim3 block(256, 1, 1);
    dim3 grid((OC + 255) / 256, L_out, N);
    hipLaunchKernelGGL(kernel_conv_1d_grouped_f32, grid, block, 0, stream,
        kernel, input, output,
        K, IC, IC_G, OC, OC_G, groups, L, L_out, N,
        stride, i_stride, o_stride);
}

} // extern "C"
