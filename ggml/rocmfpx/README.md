# ROCmFPx Experimental Formats

This folder contains the reference layer for the proposed ROCmFP3, ROCmFP6, and
ROCmFP8 quantization family. It is intentionally separate from `ggml/rocmfp4/`
so the promoted ROCmFP4 GGUF formats and kernels are not affected while the new
layouts are evaluated.

## Current Status (July 15, 2026)

- CPU reference quantize/dequantize exists for all three formats.
- `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, and `Q8_0_ROCMFPX` are registered as
  experimental GGUF tensor types.
- ROCm/HIP and Vulkan kernels support `CPY`, `GET_ROWS`, `SET_ROWS`, and
  `MUL_MAT`/`MUL_MAT_ID` for all three formats.
- **All backend-ops tests pass** on ROCm0 (Radeon 8060S, gfx1151).
- Qwen3-1.7B real model benchmarks completed (see below).
- Default quant presets include lean coherency routing (see Layouts section).

## Kernel Optimizations (July 2026)

The following optimizations have been applied to the HIP/CUDA backend kernels:

### FP3 Codebook Table Decode (+75% on FFN-up)

Replaced per-bit extraction loop + conditional sign branch with direct
`__constant__` codebook lookup table (`rocmfpx_fp3_codebook[8]` in
`rocmfpx_hip_codebook.cuh`). The 12-bit window (4 × 3-bit codes) is loaded
via register shift rather than per-bit iteration.

### FP3 VDR 2→4

Doubled values-per-thread in the MMVQ (token-generation) path from 2 to 4,
improving thread occupancy.

### FP6 Fast Sign-Magnitude Pack (+15%)

Replaced conditional branch decode with XOR-based sign-magnitude unpacking;
converted bit extraction to register-shifted 24-bit window loading with
codebook table lookup.

### FP8 VDR 2→8 (+8%)

Increased values-per-thread to match standard q8_0, doubling thread occupancy
in the MMVQ path.

### Precomputed Scale LUT

Replaced branch + bit-manipulation float reconstruction in
`rocmfpx_ue4m3_to_fp32_finite` with a 128-entry `__constant__` float table.

### FP6 Expanded Device Layout (Off by Default)

When `GGML_ROCMFP6_EXPANDED_DEVICE=1` (in `common.cuh`), FP6 blocks are
expanded from packed 26-byte to 34-byte int8 format on the GPU. This
eliminates all bit-unpacking at the cost of +31% memory bandwidth. Beneficial
on compute-bound GPUs (CDNA3 MI300X) but harmful on bandwidth-bound unified
memory GPUs (gfx1151). Guarded by `#if` and fully functional.

### FP3 MMQ Correctness Fix

Fixed out-of-bounds read in `rocmfpx_pack4_fp3_vec_cuda` where start_bit
was incorrectly calculated as `12 * base` instead of `base * 3`. This caused
MUL_MAT failures at n≥9 output columns.

## Performance

### Microbenchmarks — MUL_MAT (Radeon 8060S / gfx1151)

| TYPE | BPW | Shape | us/run | GB/s | GFLOPS |
|------|:---:|:-----:|:------:|:----:|:------:|
| q3_0_rocmfpx | 3.50 | 4096x14336 | 117 | 220 | **1004** |
| q6_0_rocmfpx | 6.50 | 4096x14336 | 244 | 196 | **480** |
| q8_0_rocmfpx | 8.25 | 4096x14336 | 284 | 214 | **414** |
| q4_0 (ref) | 4.50 | 4096x14336 | 81 | 411 | **1458** |
| q4_0_rocmfp4_fast | 4.50 | 4096x14336 | 74 | 425 | **1597** |

### Real Model Inference — Qwen3-1.7B (Radeon 8060S / gfx1151)

| Format | Size | BPW | PP128 (t/s) | TG64 (t/s) |
|--------|:---:|:---:|:-----------:|:----------:|
| Q4_0 | 999 MiB | 4.87 | 1384 | 155 |
| Q3_0_ROCMFPX (LEAN) | 940 MiB | 4.58 | **3344** | 140 |
| Q6_0_ROCMFPX (LEAN) | 1.32 GiB | 6.75 | 2734 | 120 |
| Q8_0_ROCMFPX (pure) | 1.65 GiB | 8.44 | 2823 | 103 |

Q3_0_ROCMFPX is **2.4× faster** at prompt processing than Q4_0 (3344 vs 1384
t/s) due to the LEAN preset routing bulk FFN-up to Q3_0.

## Validation Script Index

```text
scripts/check-rocmfpx-reference.sh        # CPU reference math
scripts/check-rocmfpx-qwen-all.sh         # core Qwen gates
scripts/check-rocmfpx-all.sh              # qwen-all + optional smokes
scripts/check-rocmfpx-summary.sh          # full JSON summary runner
scripts/sweep-rocmfpx-backend-ops.sh      # test-backend-ops per backend
scripts/sweep-rocmfpx-agent-size-table.sh # LEAN vs AGENT MiB/BPW
scripts/sweep-rocmfpx-perplexity.sh       # calibration PPL sweep
scripts/sweep-rocmfpx-decode-tune.sh      # decode-tune matrix
scripts/build-rocmfpx-agent-fixtures.sh   # proxy Hermes/OpenClaw AGENT GGUFs
```

## Layouts

All formats use 32-weight blocks.

| Format | Payload | Scale bytes | Block bytes | BPW | Purpose |
|---|---:|---:|---:|---:|---|
| `Q3_0_ROCMFPX` | 32 packed 3-bit codes | 2, one per 16 weights | 14 | 3.50 | Experimental low-bit candidate |
| `Q6_0_ROCMFPX` | 32 packed 6-bit codes | 2, one per 16 weights | 26 | 6.50 | Experimental quality candidate |
| `Q8_0_ROCMFPX` | 32 signed 8-bit codes | 1, one per 32 weights | 33 | 8.25 | Experimental high-quality reference |

`ROCmFP3` uses a tiny signed codebook: `0, +/-1, +/-2, +/-4`.
`ROCmFP6` uses signed-magnitude integer levels up to `31`.
`ROCmFP8` uses signed int8 levels clamped to `[-127, 127]`.

### Quant Presets

Default LEAN presets:
- `Q3_0_ROCMFPX`: selective `Q5_K` on attention Q/O and early K/V, boosted
  FFN-down at `Q5_K`, selective FFN-gate at `Q6_0_ROCMFPX`, bulk FFN-up on
  `Q3_0_ROCMFPX`, embeddings/output at `Q4_0_ROCMFP4_FAST`.
- `Q6_0_ROCMFPX`: early attention and boosted FFN-down at `Q8_0_ROCMFPX`,
  embeddings/output at `Q6_0_ROCMFPX`, bulk gate/up on `Q6_0_ROCMFPX`.
- `Q8_0_ROCMFPX`: pure FP8-family preset.

Opt-in `*_AGENT` presets boost attention/FFN routing for tool-call /
Hermes / OpenClaw style workloads.

## Files

| File | Purpose |
|------|---------|
| `rocmfpx.h` | Public API: quantize/dequantize/validate, block structs |
| `rocmfpx.c` | CPU reference: quantize, dequantize, MSE scale selection |
| `rocmfpx_hip_codebook.cuh` | `__constant__` codebook tables (FP3 8-entry, FP6 64-entry) |
| `test_rocmfpx.c` | CPU reference test (validation + imatrix weighted MSE) |

## Key Kernel Files (ggml/src/ggml-cuda/)

| File | Lines | What |
|------|-------|------|
| `mmq.cuh` | 1057–1252 | MUL_MAT quantized tile loaders for FP3/FP6/FP8 |
| `vecdotq.cuh` | ~440–680 | Vec-dot (MMVQ) and codebook-optimized pack4 functions |
| `dequantize.cuh` | 114–150 | Per-element dequantization for GET_ROWS/CPY |
| `cpy-utils.cuh` | 384–834 | On-device quantize for CPY/SET_ROWS |
| `cpy.cu` | 615–680, 773–800 | FP6 expanded stride handling in CPY |
| `ggml-cuda.cu` | 697–3549 | FP6 block expand/pack, tensor size calc |
| `rocmfp4/rocmfp4_hip_scale.cuh` | 111+ | `rocmfpx_ue4m3_to_fp32_finite` + scale LUT |

## AMD WMMA (Wave Matrix Multiply-Accumulate)

ROCmFPX now supports AMD WMMA for int8 quantized MMQ on RDNA3+ GPUs (gfx11+).
The implementation uses rocwmma library's 16×16×16 int8 WMMA intrinsics:

1. Tile registers are written to a per-warp `__shared__` staging buffer in
   row-major int8 format
2. rocwmma `load_matrix_sync` loads into fragments (2 iterations: K=0..15,
   K=16..31)
3. `mma_sync` accumulates results in int32
4. `store_matrix_sync` stores back to the tile

Enabled via `add_compile_definitions(AMD_WMMA_AVAILABLE)` in the HIP build.
Controls the dispatch between DP4A and WMMA paths in MMQ kernels.

## FP6 Expanded Device Layout

The `GGML_ROCMFP6_EXPANDED_DEVICE` flag (in `common.cuh`) controls whether
FP6 blocks are expanded from packed 26-byte to 34-byte int8 format on GPU.
Default: 0 (off). Set to 1 for compute-bound GPUs (CDNA3 MI300X series)
where eliminating bit-unpacking outweighs the +31% memory bandwidth cost.
The stride fixes in `getrows.cu`, `set-rows.cu`, and `cpy.cu` handle the
different block sizes correctly.

## Future Work

- **WMMA staging optimization**: The current implementation uses shared memory
  staging for format conversion. Loading directly from the original shared
  memory tile (when layout matches) would eliminate 2 round-trips.
- **RDNA4 int8 WMMA intrinsics**: RDNA4 (gfx12) supports native
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8` instructions, avoiding rocwmma
  library overhead. Add a compile-time path for gfx12+.
- **Real perplexity benchmarks**: End-to-end quality evaluation comparing
  ROCmFPX formats against standard q4_0/q8_0 at various BPW.
