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
Round 9 (2026-09-05): GGML_HRX_DISABLE knob landed (3de16b1); pure-CPU zaya = oracle
- create_registry_context honors GGML_HRX_DISABLE=1 (0 devices). Use for pure-CPU
  reference runs: env GGML_HRX_DISABLE=1 <binary> -ngl 0.
- RESULT: current-tree zaya-q4nx-c43 CPU decode reproduces oracle 9079/"Paris".
  Round 6-8 changes (relayout alias, SOFT_MAX/ARGSORT/GET_ROWS/ROPE claims,
  MUL_MAT carve-out, empty guard) are numerically EXONERATED on the CPU path.
- The ngl0-with-HRX-live divergence ("Consultation Financial") = type42 weights
  pinned on HRX0_HOST + claimable f32 ops (RMS_NORM/ROPE/...) routed to HRX
  host kernels, whose numerics differ from CPU kernels. Expected mixed-kernel
  artifact of the half-ported state; resolves when zaya runs coherently on one
  path.
- PATH FORWARD (unlocked): T3-A type42->F32 dequant-at-load (llama-model-loader
  create_tensor + load_data scatter, logical dims from the expected ne) so the
  expert GEMMs become standard HRX MUL_MAT like qwen; then validate whole-graph
  HRX decode vs the GGML_HRX_DISABLE=1 CPU oracle per-op (ZAYA_1LAYER), then
  multi-seq + perf gate. The refreshed-fork zaya GPU port spec is complete
  (rounds 1-9): the remaining work is T3-A + validation, both well-scoped.
Rounds 10-12 (2026-09-05): T3-A IMPLEMENTED + CPU-VALIDATED (loader dequant)
- llama-model-loader.cpp: Q4NX tensors now dequantize at load into F32 (or F16
  with GGML_ZAYA_DEQUANT_F16=1) LOGICAL-layout tensors (create_tensor uses the
  arch's expected ne; load_all_data scatter-fills from the tile-framed Q4NX
  bytes; mmap auto-disabled for Q4NX models). Custom GGML_OP_MUL_MAT_Q4NX no
  longer used - standard MUL_MAT/MUL_MAT_ID on any backend.
- VALIDATED: CPU decode (GGML_HRX_DISABLE=1, F32) = oracle 9079 "Paris" -
  the tile->logical mapping is bit-correct. qwen3 unchanged (pp ~10630, tg
  ~242). ggml-hrx.cpp round-11: MUL_MAT requires F32 activations (zaya f16
  mms -> CPU); GLU not eager-claimed.
- INCIDENT: an -ngl 99 F16 zaya run OOM'd the loaded box (88/122 GB) at 07:38
  -> reboot at 07:42 (interrupted an in-flight link: llama-cli + libllama .431
  were 0-byte; rebuilt clean). Lesson: dequant models need ~15-30 GB extra;
  run GPU zaya validation on a quiet box or with servers stopped.
- NEXT FRONTIER (GPU): zaya -ngl 99 F16 loads but graph compute fails:
  "external value 9 has an empty binding" (HRX subgraph input without a bound
  buffer) - a device/host buffer-binding issue in the mixed F16-device split.
  Debug: find which external value (index 9) lacks a binding; likely a weight
  or activation the scheduler expects on HRX0 but that resolved to a host
  buffer (or the reverse). Then: perf gate + multi-seq (task 5).
- Commits: 62db4e9 (T3-A + round-11 claims), assert fix. Branch
  fix/hrx-ngl-init-order. NOTE /tmp is tmpfs - wiped on reboot (zgreedy4 +
  harness must be rebuilt from research/ sources after any reboot).
Round 13 (2026-09-05): ZAYA GPU EXECUTION ACHIEVED on the refreshed fork
- Extended empty-guard (all srcs) + dropped zero-length external bindings in
  command-program-bindings (0-width recr slices are no-ops).
- zaya-q4nx-c43 -ngl 99 GGML_ZAYA_DEQUANT_F16=1 now EXECUTES end-to-end on
  HRX0: Prompt ~25 t/s, Generation ~7 t/s (F16 weights on device; type42 no
  longer custom-op).
- Correctness NOT yet oracle ("restrictionrapra..." vs "Paris..."). CPU+F16 =
  oracle 9079 -> F16 rounding exonerated; divergence is HRX-side in the mixed
  split (expert F16 MUL_MAT/MUL_MAT_ID + claimed f32 ops vs CPU numerics, or a
  CPU<->HRX boundary data-flow bug).
- NEXT (dedicated session): per-op bisection with ZAYA_1LAYER + dumps vs the
  GGML_HRX_DISABLE=1 CPU oracle; likely fixes: HRX mm accum-order validation
  (compare one F16 expert mm vs CPU), boundary-copy audit, or forcing the
  recr-adjacent subgraph fully CPU while keeping the expert mms on HRX.
  Then: perf (reduce CPU/GPU round trips; target >=16.8) + multi-seq (task 5).
Round 14 (2026-09-05): divergence isolation matrix - SYSTEMIC, not a single op

1-layer + full-decode tests (CPU F16 oracle = 9079/"Paris"; ZAYA_1LAYER CPU
token = 88048):
- HRX default (-ngl 99, F16 dev weights): 1-layer token 28453 != 88048.
- Force RMS_NORM->CPU: 52617 (still !=).
- Force MUL_MAT_ID->CPU: 1926 (still !=).
- Force MUL_MAT->CPU: 131213 (still !=).
- Force RMS+MM+ID+SUM+CLAMP->CPU (only flash-attn HRX): 53440 (still !=).
- Full decode, flash-attn->CPU (+ hybrid attn-KV pinned CPU): "LatestConf..." garbage.
- Full decode, weights->CPU (-ot .*weight=CPU) + HRX live: garbage.
- Full decode, weights CPU + flash CPU + all compute CPU + HRX STILL LIVE: garbage.
- Full decode, GGML_HRX_DISABLE=1 (0 devices): ORACLE.

CONCLUSION: with the HRX backend REGISTERED as a live device, the zaya decode
diverges even when every weight and every compute op is forced to CPU. The
corruption is SYSTEMIC to the scheduler/placement/buffer layer when HRX0 is a
candidate backend (not a single op's numerics). Candidates: (a) some tensor is
still placed/computed via HRX0/HRX0_HOST and misread (stale/alias across the
CPU<->HRX boundary - possibly interacting with the round-8 force_alias_relayout
or no-mmap loading), (b) the scheduler's split/claim interacts badly with the
zaya recurrent-state views. The CPU-only path (HRX_DISABLE) is bit-correct.

Diagnostic knobs added (env-gated, default-off, safe): GGML_HRX_NO_FLASH_ATTN
(flash-attn->CPU + hybrid attn KV->CPU), GGML_HRX_CPU_OPS=<ops> (force listed
ops to CPU), GGML_HRX_DISABLE (0 devices), GGML_ZAYA_DEQUANT_F16 (F16 dequant).

RECOMMENDATION: this needs the ggml-hrx/round-28 owner (scheduler placement +
boundary data-flow audit with a debugger), not more op-level toggling. All
qwen3 wins + the CPU oracle path are intact and committed.
Round 15 (2026-09-05): two more fixes + issues filed
- llama.cpp llama_prepare_model_devices: n_gpu_layers==0 now clears GPU/IGPU
  devices -> pure-CPU runs (qwen3 ngl0 "command buffer not recording" error and
  the zaya HRX-live CPU corruption FIXED; zaya -ngl 0 now coherent ~25 t/s).
- llama-model-loader create_tensor: Q4NX buft probed from the CONVERTED F16/F32
  logical meta (was probing the Q4NX tile meta -> host-only buffer -> weights
  stayed host at ngl>0 and ran through the broken host path). Weights now land
  on the HRX0 device buffer at ngl99.
- zaya -ngl 99 with device weights still corrupt (mixed-split bounces of the
  CPU-forced zaya ops through the HRX host path) - tracked as issue #2116.
- Issues filed on 1bit-MONSTER: #2115 (HRX host path corrupts CPU-mixed graphs),
  #2116 (zaya ngl99 mixed-split corrupt, task-4 blocker), #2117 (ggml-hrx
  over-claims ops -> hard errors no CPU fallback).
- Commits: 7ac6a8e (ngl0 gate + buft probe). Branch fix/hrx-ngl-init-order.

Round 16 (2026-09-06): CPU oracle re-verified via RAW-TOKEN probe; GPU matrix staged; NPU contended by zaya-m1 lane
- METHODOLOGY FIX: llama-cli text output is NOT a valid oracle signal for this
  model - llama-cli wraps -p in the zaya chat template, so CPU decode prints
  "<think>We need to respond..." while the raw continuation is " Paris.\n```\nWe".
  Earlier round text comparisons (llama-cli "Paris" vs "restrictionrapra") are
  confounded; the token-level gate must use a raw-completion probe.
- Added research/zgreedy.cpp (raw tokenize, greedy 8 steps, prints prompt tokens,
  per-step top-1 ids + step0 top-5 argmax sanity). Build:
  g++ -std=c++17 -O2 -I ggml/include -I include research/zgreedy.cpp \
      -L build/bin -Wl,-rpath,$PWD/build/bin -lllama -lggml -lggml-base \
      -lggml-cpu -lpthread -o /tmp/zgreedy   (/tmp = tmpfs; rebuild after reboot)
- CPU ORACLE RE-VERIFIED (current tree, GGML_HRX_DISABLE=1, F32 dequant, raw
  prompt "The capital of France is"): tok0=9079 (top5 9079/528/107/5213/506,
  argmax sane), 236761, 107, 2717, 108, 1882, 735, 1156; text " Paris.\n```\nWe
  have two" - matches the stale-fork oracle token-for-token. CPU path intact.
- GPU matrix staged as research/flm-parity/round16.sh (cells B-E: ngl99 F16 dev
  weights; +FLASH_ATTN_EXT->CPU; +GGML_HRX_NO_FLASH_ATTN; F32 dev weights) with
  the zgreedy token gate. NOT RUN: NPU is single-tenant and the zaya-m1 lane
  (npu_engine_zr1 from ~/wt/zaya-m1, engine/npu/build) holds the device with
  recurring jobs (PIDs 611764/617112/624546, 22:07-22:4x); flm serve (35b-a3b)
  idle since 22:10. Run cells in a quiet gap; each is ~1-2 min + load.
- Branch fix/hrx-ngl-init-order. Commits: aab33c1..HEAD (this doc + tools).

Round 16b (2026-09-06): layer-0 origin confirmed; all per-op forcing fails; executor staging audited
- Tooling: GGML_HRX_CPU_OPS now gates ALL claim paths (binary/unary included), not
  just the eager list (ggml-hrx.cpp device_supports_op). Enables forcing ADD/MUL
  (zaya res_scale) to CPU for bisection. Build: make -C build -j llama-cli.
- Post-round-15 isolation (zgreedy raw-token gate, ZAYA_1LAYER where noted):
  * ngl0 + HRX registered (round-15 device-list fix): tok0=9079 ORACLE (verified
    raw-token; llama-cli text is template-confounded, see round 16).
  * ngl99 full: tok0=16745 "gregregre..." (F16=F32=unified: byte-identical logits)
  * ngl99 + ZAYA_1LAYER: tok0=28453 vs CPU-1layer 88048 -> CORRUPTION STARTS IN
    LAYER 0 (32 graph splits). Partial ngl (1/2/4/8) hard-aborts (over-claim).
  * Per-op forcing at 1-layer, all STILL wrong: RMS_NORM->CPU 52617, MUL_MAT->CPU
    131213, ADD+MUL->CPU 1125 (res_scale), SUM_ROWS/CLAMP->CPU 28453 (unchanged).
    Matches round-14 tokens exactly -> not a single op kernel; NOT fixable by
    op-class routing.
- Code audit (for the executor owner): cross-backend host bindings are materialized
  in command-program-executor.cpp materialize_host_bindings (weights -> HostWeightCache
  device buffers; others -> HostStagingBuffer device staging). Ordering in
  execute_prepared_command_program: upload_async (chunked hrx_stream_update_buffer)
  -> command lists -> download_synchronous (stream sync + d2h) - looks correct.
  Suspects remaining: (a) graph-program-cache staleness across reused host_data
  pointers (bindings hash includes pointers/generations; allocator reuse across
  graphs), (b) constant images baked into cached programs going stale, (c) device
  kernel bug on a shape only zaya hits (e.g. n_groups=10 CCA conv/dw, ne=6 batch
  traces), (d) dmesg -ENOMEM "amdxdna_gem_shmem mmap Failed to insert pages" seen
  ~11:25 (round-15-era runs) - page-insertion failures could silently zero device
  reads of some host buffers.
- NEXT (needs undisturbed device + instrumentation): dump layer-0 subgraph boundary
  values (first HRX subgraph input vs CPU source) to localize the first corrupt
  byte; check dmesg during run; verify program-cache identity/generation handling.

Round 16c (2026-09-06): split trace + shared-arena coherence finding (best root-cause candidate yet)
- GGML_SCHED_DEBUG=1 trace of layer-0 decode (32 splits) shows the graph alternates
  CPU/HRX0 nearly every split. EVERY HRX split reports "0 inputs" - HRX subgraphs
  receive no explicit cross-backend input copies. Cross-backend data flows through
  the shared HRX0_HOST compute arena (259 MB) that CPU and NPU both access
  directly, relying on CPU<->NPU coherence of that arena.
- ggml-hrx.cpp buffer_alloc (host buft): host_visible -> memory_type =
  HOST_LOCAL|DEVICE_VISIBLE, HOST_COHERENT ONLY when use_direct_host_bindings
  (GGML_HRX_USE_UNIFIED_MEMORY). Code comment: "DEVICE_VISIBLE permits handle-
  based stream copies without implying direct device access." Yet the executor
  direct-binds host-buft buffers into command programs (no materialization; they
  have buffer handles). => Default host buft is NOT device-coherent while the
  executor relies on direct device access to it. FITS the corruption: wrong-but-
  deterministic at every CPU<->HRX boundary; dtype/memory-mode independent;
  clean at ngl0 (nothing on device).
- WHY cell G (USE_UNIFIED_MEMORY=1, HOST_COHERENT) did NOT fix: flag likely not
  honored for this NPU/driver memory (or read at device init only, before buft
  creation). Needs verification with alloc success + a coherence micro-test.
- FIX CANDIDATES (next session, in order):
  (a) Do not direct-bind non-coherent host-buft buffers: in materialize_host_bindings
      treat host-buft bindings like host_data (staging upload/download) unless
      the buffer is truly HOST_COHERENT. command-program-executor.cpp.
  (b) Force HOST_COHERENT on host buft by default and verify allocs succeed +
      corruption goes away (then keep or gate it).
  (c) Micro-test coherence: CPU-write a host-buft tensor, run a trivial device
      kernel reading it, compare; ditto device->CPU. Establishes the mechanism.
- zaya-m1 lane still takes the NPU in bursts; cells run in gaps (see round16.sh).
  Commits: 4705403..HEAD.

Round 16d (2026-09-06): qwen GPU decode is ALSO broken (zero logits) - shared device-path bug, not zaya-specific
- qwen3-0.6B Q4_K_M ngl99 via zgreedy: top5 step0 = all 0.000, tok0=0 repeated,
  llama-cli prints NOTHING. The RESULTS-qwen3-roster doc (task-1/2 evidence) used
  llama-bench pp512/tg128 = THROUGHPUT ONLY; "coherent NaN-free output" was never
  validated on the GPU path with a token gate. Tasks 1-3 evidence predates the
  round-15 commits (09:45 vs 10:08) - GPU-output correctness at that time is
  UNVERIFIED either way (llama-cli text is template-confounded, see round 16).
- qwen ngl99 graph = 4 splits: CPU embd -> HRX0 layers (0 inputs) -> CPU output
  (output_norm + lm_head) consuming l_out-26 via ONE cross-backend input copy.
  All-zero logits => the HRX->CPU handoff of l_out-26 is zeros: either the device
  kernel never wrote the tensor the sched reads, or the d2h copy/staging zeroes.
- Signature contrast: zaya logits = FINITE garbage (16745@19.0), qwen = all ZERO.
  zayas last CPU ops read device outputs via the shared HRX0_HOST arena (stale
  garbage), qwens last CPU op reads via a staged input copy (zeros). Unifying
  hypothesis: device kernels write into different/stale buffer locations than
  the CPU/sched reads - a buffer generation/identity or arena-reuse defect in
  the executor/program-cache path (bindings carry identity+generation; cache
  keys on them; stale generation across sched buffer reuse => kernels bind stale
  handles => current buffers stay zero; arena reads see stale contents).
- NEXT (executor owner): (1) minimal d2h test: compute one device tensor, read
  it back via backend get vs buffer get - are they equal? (2) audit generation
  bumps vs ggml buffer reuse in graph-program-cache/prepared-program paths;
  (3) check whether device kernels write where the sched expects (bind l_out-26
  after a 1-layer qwen decode, dump device buffer bytes).

Round 16e (2026-09-06): qwen GPU decode FIXED - GET_ROWS claim restored with shape cap; CPU->HRX boundary proven corrupt
- BISECT: qwen3-0.6B ngl99 zero-logits first appears at 98b2bcc (removed the
  eager GET_ROWS claim). Base tree hrx-graph-develop-v2 (and c80f41f) decode
  qwen correctly. Root commit identified by checkout+rebuild+test per commit.
- BASE GRAPH (working, GGML_SCHED_DEBUG): qwen ngl99 = ONE all-HRX0 split -
  token-embd GET_ROWS, layers, output_norm, lm_head ALL on device. Zero CPU
  ops in the graph. llama logits readback (HRX->CPU) works.
- MECHANISM (proven on demand): any CPU op producing data consumed by an HRX
  subgraph corrupts decode. Reproduce: GGML_HRX_CPU_OPS=GET_ROWS on qwen ngl99
  (embd -> CPU) => all-zero logits. Base qwen worked because its graph was
  single-split all-HRX; zaya cannot be single-split (conv/SCALE/recr ops are
  CPU-forced mid-graph), hence zaya stays corrupt while qwen is clean.
- FIX (committed): device_supports_op claims GGML_OP_GET_ROWS for 2D embd-style
  tables with rows <= 262144 (qwen vocab 151936; base-era kernel worked) and
  keeps zaya table slices (ne0==1) + huge vocab (262272) on CPU.
- AFTER FIX: qwen Q4_K_M ngl99 tok0=12095 = CPU argmax, coherent text
  ("Paris. The capital of France is also"). zaya 1-layer no longer aborts on
  GET_ROWS but still corrupt (28453->101018 routing shift; mid-graph CPU ops).
- NEXT for zaya (task-4): fix the CPU->HRX boundary data path in the executor
  (materialize_host_bindings upload_async staging vs direct host-buft binding)
  OR port zaya CPU-forced ops (conv/SCALE/partial-ROPE) to HRX kernels to reach
  an all-HRX zaya graph like qwen.

Round 16f (2026-09-06): boundary sub-bugs discriminated; device-buft-only experiment reverted
- Experiment: device_supports_buffer_type restricted to context->buft only (drop
  host_buft + arbitrary host bufts). Results:
  * qwen ngl99: still correct (12095); splits 4->2 (cleaner all-HRX).
  * DISCRIMINATOR (qwen ngl99 + GGML_HRX_CPU_OPS=GET_ROWS -> embd on CPU):
    went from ALL-ZERO logits to FINITE-WRONG logits (tok0=456). => two
    distinct defects: (i) direct host-buft/CPU-buffer binding into device
    kernels returns ZEROS (device never sees the CPU-written data - not
    device-coherent); (ii) even when the sched stages a real upload, the
    staged value is WRONG (finite, tok0=456 vs oracle 12095) - the
    upload_async / staging path has a value defect of its own.
  * zaya 1-layer: regressed to abort (VIEW f32[256,6,1,1]<-f32[128,2,6,1]
    claimed, no dispatch kernel - over-claim #2117). New splits changed claim
    patterns. => device-buft-only is NOT sufficient; REVERTED (tree clean at
    2fb0376a0, qwen re-verified 12095).
- Remaining work for the CPU->HRX boundary (either defect (i) or (ii) fixed):
  defect (i) fix = never direct-bind non-HOST_COHERENT host-visible buffers in
  materialize_host_bindings (stage them like host_data); defect (ii) fix =
  audit upload_async chunked hrx_stream_update_buffer vs kernel-read ordering
  and value semantics (kMaxInlineUploadBytes=63K; test a single large upload).
  Then re-test qwen+GET_ROWS->CPU canary (expect 12095) and zaya.

Round 16g (2026-09-06): staging upload flavor does not matter; defect is in staging buffer binding/cache identity
- Experiment: upload_prepared_host_staging switched to upload_synchronous
  (blocking ordered h2d, identical to the WORKING HostWeightCache weight path).
  Result: canary (qwen ngl99 + GGML_HRX_CPU_OPS=GET_ROWS) STILL broken (tok0=0),
  zaya unchanged. Reverted (tree clean at 5416f922d).
- Key contrast: HostWeightCache (device buffer + upload_synchronous ONCE, then
  cached; kernels bind entry->buffer) WORKS - weights always correct. HostStaging
  (per-exec device buffer + upload + bind) is broken in BOTH async and sync
  flavors. => The upload call itself is not the defect. Remaining candidates:
  (a) prepared-program kernel bindings for staging buffers resolve to a stale/
      colliding buffer (identity/generation bookkeeping; staging buffers are
      per-value-id in the cached prepared program - value-id reuse across
      graphs with different tensors could bind the wrong staging buffer),
  (b) the staging buffer is not what the kernel reads (bind offset/length),
  (c) device-side: iree command buffer update/dispatch ordering across two
      submissions is not actually FIFO on this backend.
- token_embd.weight is CPU-resident even in the WORKING qwen case (loader:
  "token_embd.weight ... cannot be used with preferred buffer type HRX0_HOST,
  using CPU instead") and the HRX GET_ROWS kernel reads it via the weight-cache
  path successfully. So table lookups work; FRESH activation uploads do not.
- HANDOFF (needs on-device instrumentation by the executor owner): after
  materialize_host_bindings, d2h-read the staging buffer back and compare to
  host_data; dump prepared.kernel.bindings vs staging.buffer identity; check
  command_program_bindings_hash collisions across ubatch frames.

Round 16h (2026-09-06): exhaustive negative set + final handoff state
- Tested and REJECTED this round: GGML_HRX_DISABLE_GRAPH_UID_FAST_PATH=1 (no
  change), GGML_HRX_VALIDATE_GRAPH_UID_CACHE=1 (no mismatch logs), sync staging
  uploads (round 16g), device-buft-only buffer claims (round 16f). All leave
  the qwen canary (ngl99 + GGML_HRX_CPU_OPS=GET_ROWS -> embd CPU) at tok0=0 and
  zaya corrupt. Tree clean; qwen all-HRX decode correct (12095) = the ONE
  working configuration.
- Binding analysis (backend-buffer-binding.cpp resolve_value_buffer):
  * HRX device buft / coherent host buft -> direct (buffer handle + offset);
    non-coherent host allocations -> host_data (context->base + offset) ->
    HostStagingBuffer per-exec upload. HostWeightCache (read-only weights)
    uploads ONCE blocking and WORKS.
  * Every individual path reads correct; the defect does not reproduce in any
    pure-static model. Remaining suspects (device-side, need IREE tracing):
    iree command-buffer submission ordering between per-exec upload buffers and
    cached dispatch buffers (HRX_TRACE_ZONE machinery exists in libhrx);
    device-side DDR_PATCH/buffer write targeting; NaN poisoning of a value.
- HANDOFF STATE (executor owner, on-device): enable libhrx tracing
  (HRX_TRACE_ZONE / iree tracing) and verify: (1) staging upload buffer content
  on device (d2h readback after materialize) == host bytes; (2) submission
  order upload-before-dispatch on the queue; (3) which external value first
  diverges in the qwen canary (embd in vs l_out out). All canaries, repro
  commands, and this log are in-tree (branch fix/hrx-ngl-init-order, commits
  through 8a297e58f + this round).

Round 16i (2026-09-06): instrumented proof - mixed-graph HRX dispatches leave writes at EXACT ZERO
- Added env-gated (GGML_HRX_DUMP_EXT) post-sync d2h readback in GraphExecutor::
  execute (graph-executor.cpp; reverted after the measurement, tree clean).
- RESULTS (qwen3-0.6B Q4_K_M ngl99):
  * WORKING all-HRX (1 split, 542 externals): cache_k_l0 reads REAL values
    (0.44, 0.26, -0.14, ...) after sync; tok0=12095.
  * BROKEN canary (+GGML_HRX_CPU_OPS=GET_ROWS, splits 533-ext + 7-ext + 4-ext):
    after hrx_stream_synchronize, cache_k_l0/v_l0 (28 layers, external write
    targets of the main split) AND result_output (in the 7-ext split) ALL READ
    EXACTLY ZERO. Kernels "succeed" (no error) but no bytes land anywhere the
    CPU reads - while the very same external weights (read-only inputs) read
    back fine (real values present).
- CONCLUSION: in mixed (multi-split / host-staged-input) graphs the HRX dispatch
  submission silently no-ops or writes to unbound locations - NOT a numerics or
  per-op kernel bug (identical kernels write correctly in the single-split
  program). Prime suspect: iree command-buffer submission when a program mixes
  host-staged uploads (update_buffer) with dispatches, or a per-frame prepared-
  program/binding mismatch in the multi-split path. The 542-ext working program
  vs 533-ext broken program differ by: (a) a host-staged external input (embd),
  (b) result_output living in a SECOND split (output_norm/lm_head split off).
- NEXT (executor owner, on-device): libhrx IREE tracing (HRX_TRACE_ZONE) to see
  whether the canary programs are submitted and whether iree_hal_command_buffer
  batch dispatch with update_buffer commands actually executes; or minimal
  repro: build the 533-ext program shape with vs without one host-staged input.

Round 16j (2026-09-06): recorded-replay path exonerated; invariant = staged-input programs write NOTHING
- Experiment: can_use_prepared_fast_path forced false (disable recorded-replay /
  reserve-binding theory: fast path keys on NODE COUNT only). Result: canary
  still tok0=0, zaya unchanged, and WORKING qwen still 12095 -> both the replay
  path AND the fresh path work for single-split and both fail for mixed. Not
  the reserve-binding/replay bug. Reverted (tree clean).
- SCHED_DEBUG=2 of the canary shows the exact topology: split#1 HRX0 = all
  layers 0-26 + layer-27 attn + KV SET_ROWS (nodes #16/#18 write cache_k/v);
  split#2 CPU = residual GET_ROWS (forced); split#3 HRX0 = layer-27 tail +
  output_norm + lm_head -> result_output. KV and result_output both exact-zero
  post-sync => EVERY HRX program that contains a host-staged external input
  (embd in split#1, node_973/974 in split#3) silently writes nothing, while the
  identical kernels in the no-staging single-split program write correctly.
- INVARIANT (all 19+ rounds): mixed-graph HRX programs with host-staged
  external inputs -> dispatches no-op (no error, zero writes). Upload flavor
  (async/sync, 16g), buffer claims (16f), UID cache + validation (16h), replay
  path (16j) all exonerated. Remaining suspect: the kernel binding table when a
  program has staged bindings (host_data entries must be replaced by their
  device staging buffers in the iree dispatch table; a host pointer in a device
  binding slot would silently no-op every dispatch in the program).
- NEXT (executor owner, needs iree-level debug): inspect the iree binding table
  of a program WITH vs WITHOUT host staging (prepare_command_program +
  bind_prepared_command_list kernel bindings); verify staged bindings resolve
  to device buffers in the dispatch table; add iree validation (the iree
  dispatch may be failing validation silently - enable IREE_DEVICE debug).

Round 16k (2026-09-06): prepared-program caching exonerated - bug survives ALWAYS-FRESH prepare+execute
- Experiment: execute() bypassed BOTH prepared-program caches and ran a fresh
  prepare_command_program + bind + execute per call (execute_command_program).
  Canary STILL tok0=0 (exact zeros), working qwen unchanged. => not stale
  external bindings, not replay, not cache reuse of any kind. The freshly-built
  canary split#1 program (host-staged embd in, KV SET_ROWS write targets out)
  writes NOTHING; the freshly-built working program writes correctly.
- Note: leaf/position and attn-mask tensors ARE host-staged in the WORKING
  program and it works => "staged inputs" per se are not fatal; the canary
  program differs structurally (its first op consumes the staged embd; output
  head is a second program).
- Remaining unknowns (all need interactive instrumentation by the executor
  owner): (1) does the canary split#1 graph->program match actually CONTAIN the
  SET_ROWS/FLASH_ATTN KV-write dispatches, or does the dispatch registry
  silently drop them for this graph shape? (2) iree dispatch validation/
  binding table for the canary program. (3) why the working programs staged
  embd works. (4) whether the matcher drops nodes for graphs whose first
  consumer input is host-staged. All cache/replay/staging-upload/buffer-claim
  mechanisms eliminated (rounds 16f-16k). The defect sits inside graph->program
  matching or the iree dispatch of the mixed-shape program.

Round 16l (2026-09-06): matcher exonerated - programs are COMPLETE; defect is per-execution binding/constant values
- Used the built-in command-program dump (GGML_HRX_DUMP_COMMAND_PROGRAM_DIR ->
  kernels.txt per unique shape): the CANARY split#1 program contains the FULL
  pipeline (program-2 decode: 387 lines, main 0 rmsnorm, mms, attention
  postprocess, flash_attention_decode_split x N, binary/rmsnorm tail = all 28
  layers + KV-writing flash kernels present). Working program-0 (401 lines) is
  the same pipeline PLUS ggml_get_rows_f32 at the head (embd on HRX). => the
  graph->program matcher does NOT drop nodes for mixed graphs.
- CONCLUSION after rounds 16f-16l: dispatches complete + full execution speed +
  exact-zero writes => kernels execute but receive WRONG per-execution
  binding/constant values for their write targets (KV cache rows / result
  buffer) in mixed/multi-split programs. Suspects narrowed to: (a) per-frame
  position/n_past constants or leaf values bound into the program (the flash-
  attn KV write index math depends on them), (b) the external write-binding
  offset for cache views resolved from a stale tensor->data at bind time,
  (c) iree command buffer / binding table slot mismatch when the program has
  staged externals (the canary differs from working only by: embd staged vs
  on-device get_rows + mid-graph CPU GET_ROWS split).
- NEXT: instrument the ACTUAL iree dispatch of one KV-writing flash kernel in
  the canary program: dump its binding buffer refs + constants at launch vs
  the working program; verify the cache view offset + n_past value on device.
  Dumps: /tmp/hrxdump (working) and /tmp/hrxdump2 (canary) - program.json +
  kernels.txt per shape.
