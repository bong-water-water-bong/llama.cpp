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
# T3-A implementation notes (2026-09-05, round 2) — loader hook analysis

Follows the mapping proof in this file's first half. Concrete integration
points found by reading the refreshed-fork loader:

## Why a simple type conversion is NOT enough
- The tile framing lives in the ggml SHAPE: type42 tensors are created as
  [ne0=8192, ne1=n_tiles(, ne2=n_expert)] where ne0=8192 is "one tile's element
  count", not the logical K. llama.cpp's type-conversion machinery (f16->f32
  etc.) preserves ne, so it cannot produce a valid logical [K, R] mm operand.
- The dequant must SCATTER tiles into logical positions (tile t=tr*n_tc+tc ->
  logical rows [tr*32,+32) x cols [tc*256,+256)), it is not a reshape.

## Hook points (llama-model-loader.cpp)
1. `llama_model_loader::create_tensor(...)` (~line 1054) receives the EXPECTED
   logical dims `ne` (e.g. zaya.cpp calls create_tensor(tn(...), {n_embd,
   n_embd_q}, 0)). At the point where the tensor ggml node is made from the
   file meta (`ggml_new_tensor` with cur->ne / cur->type), instead:
   - allocate GGML_TYPE_F32 with logical ne (ne = the expected list),
   - mark the tensor "needs q4nx scatter" (store the tile dims from cur),
   - register a custom data-fill in `load_data`/`load_data_for` that reads each
     5120-B tile block and scatters via dequantize_row_q4nx into the logical
     F32 layout. Experts (ne2>0): per-expert tiles are contiguous
     (expert e owns tiles [e*tpe, (e+1)*tpe)); logical per-expert dims follow
     from the expected ne (e.g. {n_embd, n_ff_x*2, n_expert}).
2. The graph then needs NO custom ops: ggml_mul_mat dispatches on a->type, and
   F32 weights take the standard MUL_MAT / MUL_MAT_ID path -> HRX0.

## After dequant: remaining HRX op-coverage for the zaya graph
Observed abort with -ngl 99 (before any dequant work):
  ggml-backend.cpp:898 pre-allocated tensor (cache_s_l0 ...) in buffer HRX0
  that cannot run the operation (SCALE)
- ggml-hrx supports ADD/MUL/... via supported_binary_f32_tensor (F32 contiguous
  + import_binary_kind + binary_kind_supported) and unary (SILU etc.) via
  supported_unary_f32_tensor; plus qwen pattern helpers + eager list (NONE
  ARGSORT CLAMP FLASH_ATTN_EXT GET_ROWS GLU MUL_MAT MUL_MAT_ID PERMUTE RESHAPE
  RMS_NORM ROPE SET_ROWS SOFT_MAX SUM_ROWS VIEW).
- Missing for zaya: SCALE (recurrent-state scaling on pinned cache_s_l* tensors
  -> the abort). Possibly also ggml_ssm_conv/conv_1d/grouped-conv (verify by
  running after SCALE is handled). Fixes:
  (a) add SCALE (+ conv ops if needed) to the HRX capability list with small
      loom kernels, or
  (b) pin the zaya recurrent/hybrid memory (llama_memory_hybrid recr part) to
      CPU buffers so only GEMM/RMS/ROPE/FLASH_ATTN/ARGSORT land on HRX0.
  Option (b) is smaller and matches how CPU-only ops are normally handled;
  the blocker is that llama's memory hybrid places recr state on the layer
  backend. Investigate the recurrent-memory buft selection (llama-kv-cache /
  llama-memory-hybrid) for a per-backend CPU pin, mirroring upstream patterns.

## Suggested execution order (next session)
1. Loader dequant for the DENSE 2D tensors only (wq/wk/wo/cca_val/ffn_gate/...,
   not the 3D experts); validate on CPU (ngl 0) that decode still reproduces the
   oracle tokens 9079/236761/107/2717/108/1882 (mapping correctness), then
   extend to experts.
2. Fix the SCALE/recurrent pinning so -ngl 99 no longer aborts; check which ops
   then fall to CPU splits; iterate until decode runs with weights on HRX0.
3. Numeric gate on device + tok/s >= 16.8 (stale-fork bar).
4. Then zaya multi-seq (task 5) via llama-server slots.

## Env (runs)
    export ROCMLIB=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
    export LD_LIBRARY_PATH=$ROCMLIB:$PWD/build/bin
# Round 3 findings (2026-09-05): zaya-on-refreshed-GPU bring-up progress

Landed:
- llama-memory-recurrent.cpp: recurrent state (cache_r/cache_s) pinned to CPU
  when the layer device is HRX (ggml-hrx cannot run SCALE/conv on that state).
  Fixes the "-ngl 99 abort: pre-allocated tensor cache_s_l0 in buffer HRX0
  cannot run SCALE". (uncommitted; part of the zaya port branch work)

Next blockers found (each fix reveals the next backend limit - ggml-hrx is
tuned for qwen/llama-class shapes):
1. Embed GET_ROWS: ggml-hrx eager-declares GET_ROWS but the compute kernel
   only covers qwen-pattern shapes. zaya embedding f32[2048, 262272]
   (vocab 262272) fails: "unsupported HRX node 0: GET_ROWS". qwen3-0.6B
   (vocab ~151936) works on HRX, so the limit is shape/vocab dependent.
   -ot token_embd.weight=CPU overrides do NOT help: the scheduler still
   assigns the GET_ROWS op to HRX (which over-claims support) and compute
   fails. Fix options: (a) HRX get_rows kernel for large vocab tables, or
   (b) tighten eager_capability_declared so GET_ROWS is only claimed for the
   qwen-pattern shapes (supported_qwen_attention_projection_get_rows_tensor),
   letting the scheduler split the embed lookup to CPU.
2. After embed: expect conv/SSM/grouped-conv ops + the type42->F32 loader
   dequant (T3-A spec) to be needed in sequence.

Bar for zaya decode (single-seq): stale fork today = pp64 152 t/s, tg32
15.2 t/s (DISABLE_FUSION=1). Refreshed fork target >= that.

Recommendation: treat the refreshed-fork zaya GPU path as a dedicated port
project (this spec + T3-A). The qwen3-roster FLM-parity goal is already met
on the refreshed fork (see RESULTS-qwen3-roster-2026-09-05.md).

# Round 4 (2026-09-05): two backend fixes landed; next blocker = HRX program-cache validation

Landed (this session):
1. llama-memory-recurrent.cpp — recurrent state pinned to CPU when the layer
   device is HRX (HRX can't run SCALE/conv-family ops on cache_r/cache_s).
   Removes the "-ngl 99 abort: pre-allocated tensor cache_s_l0 ... cannot run
   SCALE".
2. ggml-hrx eager_capability_declared — GET_ROWS no longer eager-claimed
   (loom kernels only cover the qwen attention-projection GET_ROWS, claimed
   via supported_qwen_attention_projection_get_rows_tensor). Large-vocab
   embedding lookups (zaya f32[2048, 262272]) previously routed to HRX and
   failed at compute ("unsupported HRX node GET_ROWS"); now split to CPU.
   Regression check: qwen3-0.6B still decodes on HRX (pp256 10868, tg64 239).

Next blocker (precise): after both fixes, zaya -ngl 99 fails with
  ggml-backend sched graph compute: "node 0 input value 0 metadata does not
  match current tensor"
Source: ggml/src/ggml-hrx/runtime/graph-program-cache.cpp bind_current_value()
— the HRX GraphProgramCache validated a cached program's expected input
metadata against the current tensors and it differs. Working hypothesis: the
mixed CPU/HRX split (embed + recr on CPU) makes the HRX subgraph's input copy
tensors (CPU->HRX edges) fresh per graph with metadata the program-cache key
did not anticipate (program key = command structure, not full tensor metadata),
so prefill->decode or per-ubatch rebuilds collide on the key with different
input shapes. Next steps: (a) reproduce with a tiny CPU-split graph to confirm,
(b) extend the program shape key to include input tensor ne/type metadata, or
(c) bypass: make the first HRX op's input a stable-shape buffer.

Then remaining for the zaya port: conv/SSM/grouped-conv op coverage on HRX +
the type42->F32 loader dequant (T3-A) + numeric gate (oracle 9079...) +
>= 15-17 t/s single-seq.

Round 5 control result (2026-09-05): qwen3-0.6B with -ot "token_embd.weight=CPU"
(-ngl 99, mixed CPU-embed/HRX split) DECODES FINE (generation ~77 t/s with the
per-token embed copy; correct output). So the CPU/HRX mixed split mechanism
itself works; the zaya "node 0 input value 0 metadata does not match current
tensor" (graph-program-cache bind_current_value) is ZAYA-SPECIFIC. Differentiator
vs qwen = the recurrent-state machinery (cache_s/cache_r now CPU-pinned, with
per-step reshape/roll views feeding the HRX subgraph) and/or the CCA conv/SSM
path. Next debug: minimal zaya graph slice with recr state -> HRX op to find the
first node whose input metadata drifts between prefill and decode; likely fix in
ggml-hrx graph-program-cache shape keying (include input ne/type) or by keeping
the recr-origin tensors off the HRX subgraph boundary.
Round 6 (2026-09-05): three safe fixes landed (commit 6de1dc7)
1. tensor_metadata_matches: zero-offset same-layout views now match cached
   non-view values (real bug - llama graph rebuilds alternate plain/reshape
   forms of the same logical input; HRX DDR_PATCH rebinds per dispatch).
2. GET_ROWS eager-claim removed (qwen-pattern only) - zaya 262272-vocab embed
   lookup splits to CPU.
3. ROPE claimed only for full-head NORMAL/NEOX (supported_rope_tensor) - zaya
   partial-head CCA rotary (n_rot=64, head=128) splits to CPU.
All qwen-regression-checked (0.6B pp256 ~10900, tg64 ~240). zaya -ngl 99 now
fails at: VIEW [128,2,2]->[256,2] (contiguous reshape into HRX KV SET_ROWS;
cross-boundary view of a CPU-produced value not elidable in the HRX subgraph).
Attempted layout-op un-claim (VIEW/RESHAPE/PERMUTE) REGRESSED qwen - reverted.

Port map remaining (each = scheduler/kernel gap on the zaya graph):
a. cross-boundary VIEW into HRX SET_ROWS (investigate dispatch-scheduler
   elision for subgraph-external view inputs; or make the CPU-rope output feed
   a copy before the view via graph shape, or extend ggml rope kernel to
   partial-head).
b. conv/SSM ops (ggml_ssm_conv / conv_1d / grouped conv) on the CCA path.
c. MoE expert path details (MUL_MAT_ID with the type42 custom CPU op today;
   needs T3-A dequant for HRX experts).
d. T3-A type42->F32 loader dequant (llama-model-loader create_tensor +
   load_data scatter) - then standard HRX MUL_MAT on bf16/f32 weights.
e. numeric gate (oracle 9079...) + >=15-17 t/s single-seq + multi-seq (task 5).
Round 7 (2026-09-05): VIEW blocker characterized precisely (instrumented, then
stripped - tree clean, qwen still pp256 10856 / tg64 241).

The failing node in the zaya -ngl 99 decode is the Vcur flash-attention feed:
  VIEW "Vcur-0 (cont) (reshaped) (view)" [128,2,2]->[256,2], input from a MUL
  (in the HRX subgraph), consumer SET_ROWS/KV.
Import diagnostics: the view output value is NOT aliased (alias_source=-1) and
same_storage(input,output)=0 because the ggml view tensor's view_src POINTS
PAST its immediate src0 - the tensor is a (cont)->(reshape)->(view) chain where
view_src != src0 (cont materialized a separate buffer). The HRX import only
aliases when output.view_src == the in-graph input tensor; a view over a
cont/reshape chain with a dangling view_src becomes an opaque separate-storage
value, so the dispatch scheduler cannot elide the VIEW and no VIEW dispatch
exists -> "unsupported HRX node".

Fix candidates (NOT attempted - risk to the working qwen path, which also has
cont->reshape->view chains that alias correctly):
  a. import: for layout-alias ops with equal element count + contiguous in/out,
     alias the output to the input value (record view_offs when nonzero). Must
     verify against qwen (its chains alias via view_src already; the change
     should be a no-op for them) and confirm the zaya cont's real buffer is the
     one bound.
  b. llama graph side: avoid the (cont)->reshape->view chain over a CPU-derived
     tensor (make the zaya attention build feed flash-attn a directly-shaped
     contiguous value).
Then the remaining port map (from round 6): conv/SSM ops, MoE expert path, T3-A
type42->F32 dequant, numeric + perf gate.
Round 8 (2026-09-05): zaya -ngl 99 now EXECUTES end-to-end on the refreshed fork (commit 437b7ec)
Five more gaps closed (all qwen-regression-clean): (1) import relayout alias for
dangling-view_src chains (graph.cpp + value-map force_alias_relayout) - VIEWs
now elide; (2) SOFT_MAX not eager; (3) ARGSORT not eager; (4) MUL_MAT output
>262144 rows -> CPU (lm_head vocab 262272); (5) empty-tensor guard -> CPU.

NEXT FRONTIER (two issues):
A. MIXED-MODE NUMERICS: -ngl 99 runs but output != CPU oracle (ngl 0 gives
   "Paris..."; ngl 99 gives "informative Financial Financial..."). Some HRX-split
   op produces wrong data vs the all-CPU path (or a cross-device data-flow bug).
   Bisect with the round-28 per-op dump tooling: run one layer with ngl 99 vs
   the f32 CPU reference, find the first divergent node. NOTE: even ngl 0 via
   this llama-cli now reports "Compute error" (the earlier correct CPU decode
   was the standalone zgreedy driver on an older build) - re-verify the ngl 0
   baseline first.
B. SPEED: ~1 t/s because the type42 expert GEMMs (the per-layer cost) run the
   CPU custom op; decode only becomes FLM-class after T3-A (type42->F32 dequant
   at load -> standard HRX MUL_MAT) + whatever RMS/rope/attn remain HRX-clean.
   Even CPU-only decode needs the custom ops multithreaded/optimized to be a
   meaningful interim (llama-cli ngl 0 was ~0 t/s = effectively serialized).

State: branch fix/hrx-ngl-init-order (15 commits), spec rounds 1-8, all qwen
wins intact. Stale fork zaya bar: 15.2-16.8 t/s.
