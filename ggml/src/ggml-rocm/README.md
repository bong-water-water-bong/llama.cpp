# ggml-rocm — ROCm Custom Kernel Backend

Hand-tuned HIP kernels for 1-bit/ternary inference on AMD Strix Halo (gfx1151).
Sources folded from the [1bit project](https://github.com/1bit-systems/1bit).

## Kernels

| Domain | Functions | Description |
|--------|-----------|-------------|
| **Ternary GEMV** | `rcpp_ternary_gemv`, `rcpp_ternary_gemv_halo`, `rcpp_ternary_gemv_halo_f16`, `rcpp_ternary_gemv_sherry_f16`, `rcpp_ternary_gemv_tq1_halo_f16`, `rcpp_ternary_pack_pk_i4` | Phase 5 decode kernels for ternary {-1,0,+1} weights packed to pk_i4 format |
| **Bonsai GEMV** | `bonsai_q1_gemv_launch`, `bonsai_tq2_gemv_launch` | Q1 (1-bit) and TQ2 (ternary 2-bit) weight formats for Bonsai-1.7B |
| **Sherry GEMV** | `sherry_ternary_gemv_launch`, `sherry_ternary_gemv_with_scales_launch`, `sherry_ternary_gemv_scalar_ref_launch` | Sherry 1.25-bpw ternary GEMV |
| **KV Cache** | `rcpp_kv_cache_attn_prefill`, `rcpp_kv_cache_attn_decode`, `rcpp_kv_cache_attn_decode_fd`, `rcpp_kv_cache_attn_decode_i8`, `rcpp_kv_cache_attn_prefill_i8` | Flash-Decode attention on AMD WMMA |
| **Prefill GEMM** | `rcpp_standalone_gemm`, `rcpp_standalone_launch_wmma_4x4_vec`, +12 more variants | WMMA-based standalone GEMM for prefill (28.4 TFlops) |
| **KV Quantization** | `rcpp_kv_requantize_pq3`, `rcpp_kv_requantize_pq3_v`, `rcpp_pq3_requantize_launch`, `rcpp_pq3_fd_decode_launch` | PlanarQuant-3 KV cache compression |
| **Medusa** | `rcpp_medusa_tree_attn_decode_fd`, `rcpp_medusa_small_m_gemv` | Medusa speculative decoding tree attention |
| **Model Loader** | `rcpp_bitnet_load_h1b`, `rcpp_bitnet_free`, `rcpp_tokenizer_load`, `rcpp_tokenizer_encode`, `rcpp_tokenizer_decode` | .h1b model format loader + tokenizer |
| **Runtime** | `rcpp_rmsnorm_fp16`, `rcpp_rope_fp16`, `rcpp_silu_glu_fp16`, `rcpp_softmax_fp32`, `rcpp_residual_add_fp16`, `rcpp_relu2_glu_fp16`, `rcpp_fp16_gemv`, `rcpp_embedding_lookup_fp16` | Model runtime ops (norm, RoPE, activation, GLU, embedding) |
| **Quantization** | `rcpp_quantize_fp16_to_i8`, `rcpp_quantize_fp16_to_i8_rowscale`, `rcpp_pk_i4_to_fp16`, `rcpp_decode_pk_i4_to_fp16_launch`, `rcpp_hadamard_rotate_fp16_butterfly_launch` | Weight quantization helpers |
| **Sampling** | `rcpp_argmax_fp32`, `rcpp_top_k_fp32`, `rcpp_sample_multinomial_fp32` | Token sampling helpers |

## Build

```bash
cmake -B build -DGGML_ROCM=ON -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build ggml-rocm
```

Requires ROCm 6.1+ with HIP compiler (amdclang++). Tested on AMD Strix Halo (gfx1151).

## Usage

The library can be loaded via `dlopen` and symbols resolved via `dlsym`, or
linked directly. All kernel functions accept a `void* stream` parameter for
HIP stream-based async execution (pass `nullptr` for default stream).

## Source Fold

Sources are folded from the 1bit project at `$PROJECT_ROOT/1bit/` via relative
CMake paths. The live sources remain in the 1bit project; ggml-rocm consumes
them in-place.
