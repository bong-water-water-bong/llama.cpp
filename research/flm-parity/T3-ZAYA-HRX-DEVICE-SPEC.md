# T3-A — Zaya Q4NX decode on the HRX device: dequant-at-load route (spec, 2026-09-05)

Status: SCOPED. Tile->logical mapping PROVEN from the CPU op source. Not yet
implemented (multi-session task). Alternative to the full loom-kernel-corpus
port (T3-B).

## The mapping (proven from ggml-cpu.c `ggml_compute_forward_mul_mat_q4nx`)

Q4NX weights are stored tile-framed: src0 = [8192, n_tiles] (each tile = a
32x256 submatrix, 5120 B block, 8192 logical elements). For a logical weight
[rows=R, cols=K] with n_tc = K/256:

    tile index t = tr*n_tc + tc        (tr = t/n_tc, tc = t % n_tc)
    tile (tr,tc) holds logical rows [tr*32, tr*32+32) x cols [tc*256, tc*256+256)
    logical[tr*32+r][tc*256+c] = dequant_tile[r*256+c]     (row-major tile)

R is not stored in the GGUF dims (they are [8192, n_tiles]); R = (n_tiles/n_tc)*32,
and K = the activation input width the tensor multiplies (arch-known per tensor
name, or n_tc = ne10/256 at compute time). NOT a permutation - a plain tile grid.

## Route A: dequant type42 -> f32/bf16 at load (host), logical dims

1. zaya loader (src/models/zaya.cpp): when a GGUF tensor is GGML_TYPE_Q4NX,
   read K (activation width feeding this weight, from arch/tensor-name), compute
   n_tc = K/256, n_tr = n_tiles/n_tc; allocate a F32 tensor of logical shape
   [R, K]; dequantize every tile (dequantize_row_q4nx, already in-tree) into the
   logical layout. Memory: 7.5 GB q4nx -> ~30 GB f32 (UMA 120 GB, fine); bf16
   halves it.
2. The custom GGML_OP_MUL_MAT_Q4NX / MUL_MAT_ID_Q4NX then vanish from the graph
   (weights are F32) -> standard HRX MUL_MAT / MUL_MAT_ID path.
3. Numeric gate: decode zaya-q4nx-c43 must reproduce the CPU oracle tokens
   9079/236761/107/2717/108/1882 (" Paris . ``` We") and match the f32 path.

## Op coverage gap (measured: ggml-hrx supports_op list)

HRX supports: NONE ARGSORT CLAMP FLASH_ATTN_EXT GET_ROWS GLU MUL_MAT MUL_MAT_ID
PERMUTE RESHAPE RMS_NORM ROPE SET_ROWS SOFT_MAX SUM_ROWS VIEW.

Zaya graph additionally uses (per the round-28 bisection): SCALE (recurrent-state
scaling), ggml_ssm_conv / conv_1d / grouped conv (im2col path), standalone
SILU/MUL/ADD (may already be covered by GLU fusion or CPU splits). Observed
failure with -ngl 99 today:

    ggml-backend.cpp:898 pre-allocated tensor (cache_s_l0 ...) in a buffer (HRX0)
    that cannot run the operation (SCALE)

Fix directions:
- add SCALE (+ SILU/MUL/ADD/CONV as needed) to the HRX supports_op table
  (small kernels; or reuse existing op plumbing), and/or
- force the recurrent-state region (cache_s_*, conv state) to CPU buffers via
  tensor-buffer overrides so only GEMM/RMS/ROPE/FLASH_ATTN run on HRX0.

## Route B (the original T3, bigger): 1bit decode kernel corpus onto C's loom registry

A's 44 .loom kernels (hrx-v2-src/ggml/src/ggml-hrx2/) + route JSON -> C's
compile-time kernel corpus + dispatch registration. Keeps weights in Q4NX
(4x less memory BW) and enables fused decode kernels; requires loom kernel
porting + registry glue + per-tensor numeric gates. Multi-session.

## Recommendation

Route A first (bounded, reuses proven CPU dequant + standard HRX mm), then
Route B later for bandwidth. Either way the zaya recurrent/conv region needs
the SCALE/conv op-coverage or buffer-override work above.

## Repro / env

    cd ~/hrx-ws/amd-hrx-graph
    export ROCMLIB=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
    export LD_LIBRARY_PATH=$ROCMLIB:build/bin
    ./build/bin/llama-cli -m ~/zaya-q4nx-c43.gguf -ngl 99 -p "The capital of France is" -n 4 -st
    # today: abort on SCALE-on-HRX0 (see above). ngl 0: CPU decode correct (slow).
