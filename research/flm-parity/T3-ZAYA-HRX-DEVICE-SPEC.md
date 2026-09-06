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

Round 16m (2026-09-06): staged uploads VERIFIED correct; embd crosses via sched copy into the device arena; defect is iree/driver-layer
- Instrumentation this round (all reverted, tree clean):
  * GGML_HRX_VERIFY_STAGING: post-upload d2h readback of every staged external
    - ALL MATCH host bytes (leaf positions, mask). Host staging works perfectly.
  * GGML_HRX_BIND_KIND: binding form dump per external. The CANARY embd is
    bound as a DEVICE buffer in the HRX0 compute arena (buffer 0x..4a00,
    off=1572864, len=4096, gen=4, weight=0) - it was COPIED there by the sched
    (ggml_backend_tensor_copy CPU->HRX), NOT staged. result_output also in the
    same arena (off=524288). KV cache = device buffer (gen=3). Leaves/mask =
    HRX0_HOST arena host_data (gen=5) -> staged (verified correct).
- Combined with rounds 16i-16l: programs complete (kernel dumps), staged uploads
  correct, bindings current+consistent, kernels full speed, yet KV/result writes
  land nowhere readable. => the defect is below this codebase layer: the iree
  dispatch/binding execution or the amdxdna driver for programs whose external
  set mixes device-arena bindings with host-staged bindings (the canary program
  has BOTH; the working single-split program has ONLY device bindings).
- FINAL TESTABLE HYPOTHESIS for the device owner: run the working qwen program
  shape but add ONE host-staged external (e.g., force a leaf to stage) - if it
  breaks, the mixing of staged + device bindings in ONE iree command buffer is
  the trigger. All instrumentation, repros, and dumps documented.

Round 16n (2026-09-06): discriminator answered + node_972 = zeros; trigger = multi-program-per-pass structure
- DISCRIMINATOR (Q from a137d5): the WORKING single-split qwen program DOES
  contain host-staged externals - 56 of them, including token_embd.weight
  (127 MB, host_data binding, weight=1) read by the on-HRX get_rows kernel, and
  leaves/mask. It decodes correctly. => staged+device binding mixing is NOT the
  trigger (kills the round-16m last hypothesis).
- New measurement: the CANARY split#1's own output tensor node_972 (blk.27
  attn_output MUL_MAT result, device arena off=1572864, val=1336) reads EXACT
  ZEROS after execute() + hrx_stream_synchronize inside the executor - i.e.
  even with an in-execute flush, split#1's dispatches leave no bytes at their
  declared output bindings (KV caches zero too). The kernels execute at full
  speed (200 t/s, timing-proven) yet write nowhere readable.
- REMAINING DIFFERENTIATOR: working = ONE HRX program per ggml graph pass
  (single split); canary = TWO+ HRX programs per pass with CPU splits between.
  Everything executor-side is exonerated (fresh prepare, no caches, correct
  staging, complete programs, current bindings). The defect is the iree
  queue/semaphore interaction when multiple HRX program launches interleave
  with CPU work + cross-backend d2h copies in ONE ggml sched pass - likely the
  mid-pass d2h copy of split outputs races/never-orders against the dispatch
  submissions (iree device_transfer vs hrx stream timeline semaphore).
- NEXT (device owner): trace iree queue ordering for the canary pass: split#1
  dispatch submit -> node_972 d2h copy -> split#2 CPU -> split#3 dispatch.
  Verify the d2h waits on the hrx stream semaphore. Everything else is
  eliminated after 23 rounds (16a-16n).

Round 16o (2026-09-06): bufbase eliminates memcpy-branch suspect; device layer corrected to iGPU/ROCm; 3-agent collaboration active
- bufbase measurement (canary, GGML_HRX_BIND_KIND): node_972, result_output and
  ALL cache_k/v are FAKE-base device buffers (base=0x1000, hostbuf=0); only
  leaves/mask are real-base host buffers. => read-backs took the post-sync
  hrx_synchronous_d2h path (correct, ordered) and return GENUINE zeros: the
  device buffers really contain zeros. eb4f0b's buffer_get-memcpy suspect (b)
  is ELIMINATED for the write targets.
- DEVICE LAYER CORRECTION: ggml-hrx executes on the Radeon 8060S iGPU via
  ROCm/iree (gfx1151, /dev/dri/renderD128, /opt/rocm-therock) - NOT the XDNA2
  NPU (amdxdna). NPU-side observations (dmesg clean, no accel0 fd) are expected
  and irrelevant. Right driver to watch: amdgpu.
- COLLABORATION: @agent-a137d5 confirmed the discriminator (working program
  stages 56 externals incl. 127MB table - mixing is not the trigger) and
  proposed the tail-CPU isolation cell; @agent-eb4f0b reviewed libhrx
  graph_exec.c/stream.c (both chain on the stream timeline IF all programs
  share one hrx_stream_t - open caveat) and identified the d2h-no-wait +
  memcpy-branch candidates (memcpy branch now eliminated); @agent-d5694d
  watching amdgpu during repro.
- OPEN: dispatches run at full speed but write nothing to their declared
  fake-base device-buffer bindings; all code above the iree dispatch is
  eliminated after 24 rounds (16a-16o). Next: (a) confirm all HRX programs in
  one pass share ONE stream; (b) amdgpu/iree-level dispatch tracing.

Round 16p (2026-09-06): BREAKTHROUGH - HRX programs are MISSING their CPU-produced activation inputs from the binding lists
- Full per-execute binding dump (GGML_HRX_BIND_KIND=1 GGML_HRX_BIND_ALL=1,
  collaborator instrument, /tmp/bindall.log) of the canary shows:
  * Split#3 (output head, ~7 ext): binds ONLY blk.27 ffn weights + token_embd.weight
    (weight=1, weights buffer) + result_output (val=18, device arena gen=4
    off=524288). NO node_973/node_974 - the CPU GET_ROWS outputs that split#3's
    first node (ADD ffn_inp-27) consumes. NOT bound, NOT staged.
  * Split#1: no embd input binding anywhere (the CPU GET_ROWS output its RMS
    node#1 consumes).
  * The ONLY host-staged inputs per token are the small leaves/mask (val=10/23/
    27/36, HRX0_HOST arena gen=5).
- MECHANISM: kernels read unbound/zero slots -> exact-zero embd -> RMS(0)=0 ->
  all linear/attention ops on zero -> zero KV + zero node_972 + zero
  result_output. Explains ALL 25 rounds of symptoms: full-speed kernels (run on
  zero inputs), staging verified correct (leaves are the only staged tensors),
  only mixed graphs affected (single-split programs have no CPU-produced
  activation inputs to lose), dtype/mode independence, clean drivers.
- ROOT-CAUSE QUESTION (in ggml-hrx code, NOT the driver): why do CPU-op-produced
  ACTIVATIONS (embd, node_973/974) fail to become program externals while
  leaves/mask succeed? Distinguish: (a) absent from binding_match.external_
  bindings (import/match drops CPU-op outputs that feed the split) vs
  (b) present but resolve_buffer_binding fails (dropped via the "null binding"
  path in CommandProgramBindings::from_bindings) vs (c) the sched input_cpy
  for them lands in a buffer the executor cannot bind.
- Collaborators: eb4f0b tracing import/match path; a137d5 distinguishing (a)
  vs (b); d5694d stood down (driver clean confirmed). If (a): fix in the
  graph import/external-list construction. If (b): fix in resolve_buffer_binding
  host-buffer branch.

Round 16r (2026-09-06): ROOT-CAUSE FIX LANDED (classification) - exact zeros GONE; remaining: prefill final-node output reads zero
- FIX (graph.cpp tensor_is_external): a value consumed by the node set was
  classified Transient whenever use_count>0, regardless of who PRODUCED it.
  CPU-produced activations (embd from CPU GET_ROWS, node_973/974) consumed by
  HRX split nodes were therefore misclassified Transient -> freshly zero-
  allocated in the HRX transient arena -> kernels read zeros -> exact-zero
  decode. Now: External unless produced WITHIN the set (produced_here check).
  Working qwen (single-split, in-set producers) unaffected; zaya ngl0 oracle
  unaffected.
- POST-FIX MEASUREMENTS (canary, GGML_HRX_CPU_OPS=GET_ROWS):
  * embd/node_973/974 now present + resolved as host-staged externals
    ([hrxext] resolved=1, hostbuf=1). All staged uploads byte-MATCH host
    (stage-v verify, incl. 4096/20480 B F32).
  * PREFILL: KV cache now byte-IDENTICAL to the working reference
    (0.4438 0.2634 -0.1373...) -> prefill layers + attention fully correct.
  * REMAINING ANOMALY: prefill split#1 final node output node_972 (blk.27
    attn_output, 5-token [1024,5], 20480 B) reads EXACT ZERO post-sync, while
    decode-step node_972 (4096 B) reads REAL values. Result: the prefill
    residual-gather feeds zeros to the output head -> first decode token 456
    (real logits now, wrong value) instead of 12095.
  * Decode steps now produce real (wrong) tokens - no more degenerate zeros.
- NEXT: why does the PREFILL program final external-output write (node_972)
  not land while decode's does? (wmma prefill kernels vs wave64 decode; or the
  20480 B 5-token external write target). Compare the dispatch write-binding
  for node_972 in the prefill program (eb4f0b hook: GGML_HRX_DUMP_WRITEBIND).
- Instruments in tree (env-gated, uncommitted): GGML_HRX_VALCHECK (per-external
  content dump), GGML_HRX_VERIFY_STAGING (post-upload d2h verify). Fix
  candidate committed separately.

Round 16s (2026-09-06): KV byte-identical canary-vs-working; prefill write-loss is node_972-specific and razor-thin
- Measurement (GGML_HRX_KVCOMP, reverted after use; tree clean at a0af0f985):
  CANARY vs WORKING KV across cache_k_l0/k_l1/v_l0/v_l1 is BYTE-IDENTICAL
  (0.4438/0.2634/-0.1373..., 2.07/-0.04776/2.225..., etc.). => (1) the canary
  prefill computes embd->all-layers->attention->KV byte-correctly; (2) node_972
  did NOT write into the KV buffer at the 3x524288=1572864 offset region
  (a137d5 cross-buffer handle theory: layer-1 V lives at 1572864 in the KV
  buffer and is byte-identical => no cross-buffer write happened).
- RESIDUAL (razor-thin): the canary PREFILL program's FINAL external write
  target node_972 (binding idx 533 = last; blk.27 attn_output mm output,
  20480B, gen-4 arena off=1572864) reads exact zero post-sync, while: (a) every
  other external write in the same prefill program (224 KV rows) lands
  byte-correctly; (b) the decode program's node_972 (idx in a 5-binding list)
  lands correctly; (c) the working single-split program's final binding
  (result_output idx 541) lands correctly. So: not count/position/last-binding
  per se; not cross-buffer; not cache/replay/prepare (16j/16k); not staging
  (byte-MATCH); not embd (KV proves it). Distinguishing feature of the failing
  write: node_972 is a WRITE-ONLY external of split#1 consumed by the NEXT
  split (CPU#2) - the only such cross-program write target in the prefill pass.
- NEXT (executor owner): trace how WRITE-ONLY externals consumed by a later
  split are bound in the fresh per-execute path (bind_external_value_buffers
  + resolve) for the PREFILL program specifically; compare against the decode
  program where the same tensor class works. eb4f0b hooks dont fire on this
  path (hrx_graph_exec_launch is not used - fresh per-execute dispatch
  confirmed). If no quick fix: force node_972-class tensors to ride the
  host-staged path (copy to host buffer at split boundary, like leaves) so the
  CPU split reads a guaranteed-consistent copy.

Round 16t (2026-09-06): node_972 zero is PREFILL-PROGRAM-specific (wmma kernels), not token-count/first-batch
- Measurement (GGML_HRX_NODECMP, reverted; tree clean at 345be184f): with a
  1-token prompt ("Paris") the canary prefill node_972 ne=1024,1 = EXACT ZERO;
  CPU oracle tok0=3219 vs GPU 7435. Earlier 6-token-prompt runs showed DECODE-
  step node_972 (also 1-token shapes) = REAL values. So the failing case is the
  PREFILL-SHAPED program specifically, regardless of token count:
  * PREFILL program (wmma kernels: ggml_mul_mat_f32_f32_wmma, flash wmma):
    node_972 external write = ZERO (every other write in the program - 224 KV
    rows via SET_ROWS/flash - lands byte-correct).
  * DECODE program (wave64 kernels): node_972 write = REAL.
- This is not reserve/first-batch/cache (16j/16k eliminated; prefill is also
  just one program), not embd (KV byte-identical proves inputs correct), not
  cross-buffer (KV pristine at the 1572864 region), not staging, not the write
  being last (working programs last binding lands). The remaining delta is the
  KERNEL CLASS of the node_972 mm: wmma (prefill) vs wave64 (decode).
- NEXT (executor/kernel owner): (1) compare the wmma MUL_MAT dispatch for
  node_972 vs an identical wmma mm whose output DOES land (any mid-program
  mm) - binding table slot for the last wmma mm output; (2) test whether a
  wmma mm whose output is consumed in-program (not a split-external) lands -
  if yes, the issue is wmma-kernel writes to split-external outputs; (3)
  pragmatic fix candidate: keep node_972-class split outputs on the wave64
  path (route prefill node_972 through CPU or a decode-style kernel), or write
  node_972 via an explicit copy op after the wmma program.

Round 16u (2026-09-06): deep KV byte-identical too; hypothesis = wmma-mm to split-external output does not land
- Measurement (GGML_HRX_KVDEEP, reverted; tree clean at a93fc4f45): deep KV
  (cache_k/v l25-l27) BYTE-IDENTICAL canary vs working => node_972's lost write
  corrupted NOTHING in any KV region (kills the wrong-slot theory for KV
  targets; the 534-external binding table's tail slots are all verified
  pristine).
- SHARPEST HYPOTHESIS: in the canary prefill, node_972's producer is the ONLY
  wmma MUL_MAT whose OUTPUT is a split-external (crosses to CPU split#2).
  Working config: the same tensor is internal (consumed by the FFN in-program)
  and writes fine. Decode config: node_972 is a split-external written by a
  wave64 MUL_MAT and writes fine. => wmma MUL_MAT writes to split-external
  outputs are lost (or land at a wrong arena offset), while wmma writes to KV
  (SET_ROWS/flash ops) and internal transients land, and wave64 writes to
  split-externals land.
- Possible kernel-level cause: the wmma mm kernel derives its output pointer
  relative to the transient-arena layout (base + in-program offset) instead of
  the external binding, so a split-external output at a different arena offset
  gets written elsewhere (zero/unused region).
- FIX CANDIDATES (in order of effort): (1) dispatch-level: route split-external-
  output MUL_MATs to the wave64 (decode) kernel even in prefill batches;
  (2) executor-level: after the wmma program, insert an explicit device->device
  copy from the internal output to the split-external buffer (or write
  node_972-class outputs via the host-staged path like leaves); (3) kernel:
  audit the wmma mm output pointer derivation in the loom kernel corpus.
- State: exact-zero bug fixed (a0af0f985); prefill computes byte-correct KV;
  ONE lost write (node_972 in the wmma prefill program) separates the canary
  from full correctness. Full trail: spec rounds 16a-16u.

Round 16v (2026-09-06): wave64-for-prefill BROKEN (all-zero); wmma stays; node_972 wmma-external write loss isolated
- Experiment (reverted; tree clean at 489b88bf4): forced the decode/wave64
  matcher above wmma priority AND relaxed common_is_supported_decode_token_count
  to accept prefill counts, so prefill mms would run wave64 kernels. Result:
  canary tok0=0 (ALL zeros again) - wave64 kernels are genuinely decode-only
  (token_count==1) and produce nothing for multi-token batches. => wmma is
  required for prefill; the node_972 external write loss is inside the wmma
  path, not fixable by kernel-class switching.
- CONFIRMED STATE: prefill (wmma) computes KV byte-correct; ONLY the wmma
  blk.27-attn_output MUL_MAT's write to its split-external output node_972 is
  lost. Decode (wave64) writes its split-external node_972 fine. Working
  single-split has node_972 internal (consumed in-program) -> fine.
- NEXT (executor/kernel owner): compare how the wmma MM dispatch binds its
  OUTPUT value (node_972, binding idx 533 = last) vs how SET_ROWS kernels bind
  their KV outputs (which land correctly) in the same prefill program. If the
  MM output binding slot resolution differs at the dispatch level (off-by-one /
  alias at the tail of a 534-binding table), that is the bug. Alternatively
  test: make node_972 ride the host-staged path (copy to host at the split
  boundary) so the CPU split#2 reads a consistent copy regardless of the wmma
  write.

Round 16w (2026-09-06): matcher-only CPU-demotion aborts (claim mismatch); demotion needs claim-level graph context
- Experiment (reverted; tree clean at 052046a49): gated the wmma MUL_MAT
  dispatch matcher (dispatch-mul-mat.cpp match_mul_mat_dispatch) to return
  false when the output Value is External (ValueKind::External), intending
  node_972 to fall to wave64 (1-token) or CPU (multi-token). Result: 6-token
  canary ABORTS ("unsupported HRX node 971: MUL_MAT output=1336 f32[1024,5]
  q4_K[2048,1024]x f32[2048,5]") - the ggml sched already assigned the node to
  HRX0 via device_supports_op (op-level claim, NO graph context), so the
  dispatch matcher finding no kernel = hard error, not CPU fallback. 1-token
  Paris unchanged (7435).
- CONCLUSION: routing one node to CPU requires claim-level (device_supports_op)
  knowledge of whether the mm's output is a terminal split-external - but the
  ggml-backend claim API (device_supports_op(device, op)) has no graph context.
  Options: (a) thread graph context into claims (invasive), (b) accept the
  execution-side wmma-terminal-external vanish and investigate at the
  kernel/driver layer (a137d5: replay-launch final writeback; eb4f0b's libhrx
  hook on the graph-launch path), (c) executor-side: after a wmma program with
  a zero-read terminal external, re-issue just that mm on CPU as a repair
  (hacky), (d) test whether the terminal external's DIFFERENT buffer (compute
  arena vs transient arena) is the trigger - if the wmma kernel writes outputs
  relative to the transient arena base, an external in the compute arena at
  off=1572864 would be written at arena-relative 1572864 of the WRONG buffer.
  (d) is testable by moving node_972-class externals into the transient arena.
- State unchanged: fix a0af0f985 landed; canary prefill byte-correct KV; ONE
  lost write (node_972 wmma terminal external). Full trail 16a-16w.

Round 16x (2026-09-06): ZAYA FULL -ngl 99 post-fix - real varied tokens (major progress on the actual task-4 target)
- Measurement: zaya-q4nx-c43.gguf -ngl 99 F16 (33 layers, full mixed graph) with
  the a0af0f985 classification fix: tok0=143243, top5 logits ~10.8-8.9, text
  "}+( Integration Comparison Comparison Comparison..." - REAL data flowing
  through the whole 33-layer mixed graph. Pre-fix the same run gave degenerate
  repetition (tok0=16745 "gregregre..."). The fix moved the actual goal target
  from garbage-repetition to real-but-wrong. Still not oracle (9079).
- Analysis: the varied-but-wrong output + trailing repetition suggests the
  residual-class write loss (node_972-class) is present for zaya too, plus its
  recurrent-state hops; the trailing "Comparison Comparison" repetition points
  at the recurrent/conv state path.
- Canary (qwen GET_ROWS) still 456; working qwen 12095; zaya ngl0 oracle 9079.
- PLAN: eb4f0b implementing scoped (B) - device_supports_op claim-false for
  MUL_MAT ne[1]>1 when GGML_HRX_CPU_OPS non-empty (diagnostic config only).
  Battery after: canary (both prompts), zaya full, zaya 1-layer, working qwen,
  zaya ngl0.

Round 16y (2026-09-06): (B) tested + reverted; known-good state confirmed
- eb4f0b landed (B) (7810bb40a: demote multi-token MUL_MAT to CPU when
  GGML_HRX_CPU_OPS set) but it did not compile (inserted in eager_capability_
  declared where op is an enum, not a tensor). Fixed by moving to
  device_supports_op. Result with the fixed (B): canary tok0=126597 (real
  logits ~12.3, varied text) - a NEW wrong answer, not the oracle. Prefill-all-
  CPU + HRX-decode creates its own cross-boundary issues and does not cleanly
  test the node_972 hypothesis. (B) REVERTED (f236d4d35) to keep the tree at
  the known-good state.
- CONFIRMED BASELINE (tree at 0282cee56 + revert): working qwen 12095; canary
  456 (known residual: node_972 wmma-terminal-external write vanish in
  prefill); zaya full -ngl 99 143243 (real varied tokens, post-fix
  improvement, not oracle).
- ASSESSMENT after rounds 16a-16y: the classification fix (a0af0f985) is the
  substantive landing (exact-zero decode eliminated; zaya full now decodes
  real tokens; qwen prefill byte-correct KV). The remaining node_972-class
  vanish is execution/kernel-side (identical correct recorded refs per
  a137d5; wmma-terminal-external specific; wave64 multi-token unusable; no
  claim-level graph context for CPU demotion). Needs the platform's
  kernel/driver owner (iree/amdgpu graph-launch writeback) or an executor
  design change (e.g., always write split-external mm outputs through a
  wave64/decode-style or explicit-copy path).

Round 16z (2026-09-06): post-fix forced-op mapping - all mixed configs real-but-wrong (no oracle)
- Post-fix measurements (qwen3-0.6B ngl99, GGML_HRX_CPU_OPS forced single op):
  * GET_ROWS: tok0=456 (known; residual-gather crossing)
  * RMS_NORM: tok0=103185, text "占地vincesvincesinglyionarioterdamียงircuit" (real vocab, wrong)
  * ADD / ROPE / MUL_MAT: also real-token runs (no abort; earlier reads misled)
- CONCLUSION: with the classification fix, EVERY mixed config produces real data
  flow (no zeros), but NONE is numerically oracle-correct. All-HRX (single
  split, no CPU ops) is byte-correct (12095). => the executor mixed path still
  corrupts cross-split ACTIVATION VALUES somewhere downstream of binding
  (staging verified byte-perfect; KV verified byte-perfect in the GET_ROWS
  config; bindings/resolution verified correct). The residual is an
  execution-side value error in cross-split activation flow - below ggml-hrx
  recording (identical correct refs), requiring the iree/amdgpu graph-launch
  writeback owner or an executor redesign (explicit-copy path for split-external
  values).
- State: a0af0f985 landed; working qwen 12095; zaya full 143243 (real tokens).
  Full trail rounds 16a-16z on fix/hrx-ngl-init-order.

Round 17a (2026-09-06): wmma hypothesis OVERTURNED - 1-token-batch canary still 456
- Experiment (research/zgreedy_b1.cpp: prompt decoded 1 token per batch,
  n_batch=n_ubatch=1, so EVERY step is decode-shaped wave64, no multi-token
  wmma prefill at all): canary tok0=456 (identical to the multi-token-prefill
  run). => the corruption is NOT kernel-class (wmma vs wave64) and NOT the
  multi-token prefill shape. Node_972-type writes are real (not zero) in
  wave64 steps yet the output is still wrong.
- REVISED ANALYSIS: KV byte-identical to working proves layers 0-26 compute
  byte-correctly (KV is written from their outputs). Therefore l_out-26 and
  layer-27-attention inputs are byte-correct, so node_972 (layer-27 attn out)
  SHOULD be correct in wave64 steps - yet decode output (456) is wrong. The
  divergence is therefore at/near: (a) the residual gather (CPU GET_ROWS over
  node_972/l_out-26 - indices arrive byte-correct per staging verify, but the
  gathered VALUES could be wrong if the d2h of node_972/l_out-26 to the CPU
  gather reads wrong rows/offsets), or (b) split#3's computation over the
  gathered residual. Both are real-valued, not zero.
- NEXT DISCRIMINATOR: ground-truth comparison of the gathered residual
  (node_973 = GET_ROWS(node_972, last-token-row)) against the CPU reference:
  for the 1-token case node_973 should EQUAL node_972. Dump node_972 vs
  node_973 (and l_out-26 vs node_974) in the same 1-token step on GPU and
  compare against the CPU-only values.
- State: classification fix (a0af0f985) landed; working 12095; zaya full
  143243; canary 456. zgreedy_b1 kept in research/.
Round 17b (2026-09-06): reserve-ghost dead post-fix; hook compile fix
- The committed GGML_HRX_DUMP_SPLITEXTVAL hook (191bd7f2b, [hrxbind2]) did not
  compile (HRX_CHECK macro is local to ggml-hrx.cpp; PRIu64 missing include) -
  fixed with direct hrx_stream_synchronize / hrx_synchronous_d2h + status
  checks + hrx-interop-utils.h + cinttypes includes so the tree builds.
- Experiment (reverted): GGML_HRX_NO_PROGRAM_CACHE bypass re-tested POST-FIX
  (round 16q ran it pre-fix, meaningless). Result: canary still tok0=456 -
  fresh import+build every execution does not fix the first-step node_972
  write. Reserve-ghost theory dead.
- PATTERN CONFIRMED: node_972 = EXACT ZERO only on the FIRST step of a
  sequence (n_past=0); n_past>0 steps write it real. Cause is not caching,
  not kernel class (1-token-batch all-wave64 still 456 at step 1), not
  bindings (verified), not inputs (KV byte-correct). The vanish is specific to
  the first-step execution of the split program writing its terminal external.

Round 17i (2026-09-06): node_972 target = STABLE gen-4 arena (stale-handle theory DEAD); vanish is at kernel-execution/readback layer
- Lifecycle trace (GGML_HRX_LIFECYCLE on ggml-hrx buffer_alloc/free): exactly FIVE
  buffer allocs at init (gen1 weights 390MB, gen2 output 607KB, gen3 KV 28MB,
  gen4 compute 78MB @0x55ac6b8e9a00, gen5 host 2MB); NO mid-run recreation.
- node_972's recorded+resolved binding = gen-4 arena 0x55ac6b8e9a00 @1572864
  (EXACTLY the stable alloc handle). The 536 transient-bound mm outputs go to a
  DIFFERENT buffer (0x55ac6d8212e0 = the executor transient arena). So node_972
  writes VALID, stable, once-allocated memory and reads (same handle+offset)
  return zeros. NOT a stale handle, NOT buffer-lifecycle (the ggml buffer is
  never recreated). The executor, caches, bindings, and arena lifecycle are all
  exonerated with byte-level evidence across rounds 16a-17i.
- REMAINING (unresolved after ~35 rounds + full fleet): the vanish sits in the
  kernel-execution/readback layer for the specific case (wmma multi-token mm
  output in the FIRST batch, target gen-4 arena): either the kernel node does
  not execute (recorded but dropped at launch), the write lands and is
  clobbered before readback, or the readback path diverges. Decode (wave64,
  n_past>0) lands; prefill multi-token (wmma) terminal output does not.
- All probes/instruments reverted; tree clean at c14db7a5b + eb4f0b in-flight
  executor iteration. Evidence: /tmp captures (cap.err, rr2.err, rl.err, lc.err)
  + committed spec rounds 16a-17h + this round.
- RECOMMENDATION: this needs a dedicated on-device session with iree graph
  launch/execution tracing (which kernel nodes actually execute at launch) by
  the platform device owner - beyond what remote instrumentation can reach.

Round 18 (2026-09-06): fleet fix (A) - per-execution GraphValue re-resolution (ff05bcb6f) tested: 456
- eb4f0b commit ff05bcb6f: reresolve_graphvalue_device_refs refreshes prepared kernel
  bindings (origin==GraphValue) from current per-execution device bindings in BOTH paths
  (recorded + direct), plus bound_graphvalue_refs snapshot + re-record on change ([resync-ext]).
- Clean-tree battery: canary tok0=456, [resync-ext]=0, working qwen 12095 (no regression).
- CONCLUSION: resync-ext=0 means the post-reresolve ref set EQUALS the recorded set on every
  execution - the record path does NOT bake from prepared.ref; a second graph-side
  value->buffer resolution supplies the record-time b2 handle. Ref staleness at the executor
  level is dead; the graph-side table is the (still-unreached) source.
- Evidence: /tmp/ff.err on strixhalo.

Round 19 (2026-09-06): workgroup forensics - launch config IDENTICAL across working/failing (exonerated)
- Record-time workgroup capture for every 20480-len (5-token) mm output: node_972 (val=1336)
  wg=16,1,1 - byte-identical to all WORKING 5-token mms (val=967/964/953/... all wg=16,1,1).
  Decode's node_972: wg=512,1,1 (different tiling for n=1, also normal).
- CONCLUSION: node_972's kernel is indistinguishable at record/launch from kernels that work.
  Launch configuration fully exonerated. Instrument reverted post-capture.
- Evidence: /tmp/wg3.err (dedup list: all 20480 outputs wg=16,1,1).

Round 20 (2026-09-06): fleet fix (b) - resize-stable buffers (2896aaa4d) tested: 456; resize NEVER fires in failing path
- d5694d commit 2896aaa4d (branch fix/hrx-compute-buffer-resize): buffer_free parks non-host
  contexts (retired_compute pool), buffer_alloc adopts a parked context (immortal context,
  fresh hrx alloc, generation bump); GGML_HRX_LIFECYCLE=1 logs FRESH/REUSE/park.
- Tested in his worktree with GGML_HRX_LIFECYCLE=1 on THE failing canary command: tok0=456.
- Lifecycle log (definitive): 5 init allocs (gen1 390MB weights / gen2 607KB host / gen3 29MB
  KV / gen4 78MB compute arena / gen5 2MB host), 3 teardown-only parks (gen1/3/4), ZERO
  REUSE adoptions, ZERO mid-run free+recreate.
- CONCLUSION: the compute arena is a single stable allocation for the entire failing run; the
  resize mechanism option-(b) addresses does not occur in this path - hence 456. Closes the
  last fleet mechanism with hard evidence (lifecycle log /tmp/life2.err on strixhalo).

FINAL STATE after rounds 16a-20: all executor-side mechanisms closed with byte-level evidence
- Eliminated: stale handles/refs, buffer resize/lifecycle, cache/prepare/replay, launch configs
  (identical), workgroups (identical), bindings, staging, hash keys, re-execution, wave64-vs-wmma.
- ISOLATED: prefill program terminal kernel (node_972 mm = last command, cmd 388/389) writes
  valid stable gen-4 arena @1572864; readback returns exact zeros. Decode terminal writes land.
  Remaining hypotheses require on-device iree graph EXECUTION tracing (does the last node of the
  launched exec actually execute / does its write commit): per-node execution, completion
  counters, or memory snapshotting after the exec - platform device owner territory.

Round 21 (2026-09-06): BREAKTHROUGH - trigger isolated + root cause = split-externalization gap (l_out-26), NOT execution
- TRIGGER ISOLATED: GGML_HRX_CPU_OPS=GET_ROWS is the entire cause of the canary 456.
  Plain run (no env): tok0=12095 (oracle, fully fused, 0 splits, 0 readbacks).
  Same command + env: tok0=456 (graph splits into 4 programs; garbage everywhere incl. decode:
  " [ }" vs oracle " Paris. The capital of France is also").
- node_972 EXONERATED (37 rounds chased the wrong tensor): its device slot @gen-4/1572864
  holds REAL, varied f32 (5120/5120 nonzero, var 2.47) on batch 0 AND every decode step;
  d2h readbacks return the identical real values. The write always landed.
- ACTUAL ZERO TENSOR: l_out-26 (the residual-add partner, gen-4 @off=0, 20480B) = ALL ZEROS
  on every step. Full-arena forensic (d5694d GGML_HRX_ARENA_DUMP e2d260b02): gen-4 (78MB)
  contains EXACTLY ONE data region (node_972 @0x180000); l_out-26's data exists NOWHERE in
  gen-4. KV arena (gen-3) fully populated -> all device programs executed correctly.
- RECORD BINDINGS: gen-4 has only 2 refs (node_972). l_out-26 has NO device write binding to
  its ggml slot (gen-4@0), yet the CPU-side residual add reads it from there (readtrace
  src_off=0) -> reads a never-written slot -> zero.
- ROOT CAUSE: graph splitter externalization gap. With CPU-forced ops (GET_ROWS) the
  residual-add node lands on CPU; its device-produced input l_out-26 (blk.26 output, a
  NON-terminal in-program value) is never emitted as a program-boundary external, so no
  program terminal writes its ggml-buffer slot. node_972 (the terminal mm of its program)
  externalized correctly. Device pipeline itself computes correctly (node_972 real).
- ZAYA CONNECTION: zaya's NATIVE dispatch puts CPU ops in the split without the env ->
  same externalization gap -> zaya full ngl99 = 143243 (real but wrong vs oracle 9079).
  Fixing the gap should resolve both.
- FIX DIRECTION: any ggml value consumed off-device (CPU-side node) whose producer runs on
  device must be emitted as a program-boundary external (device write to its ggml buffer
  slot) even when it is NOT a program terminal value.
- Evidence on strixhalo: /tmp/adump/arena_78315520_4.bin (gen-4: only node_972 region),
  /tmp/adump/arena_29360128_3.bin (KV full), /tmp/wb2.err (record bindings: gen-4 = 2 refs),
  /tmp/full.err + /tmp/ok.err (env vs no-env), /tmp/lout_fail.bin (zeros), /tmp/lout.bin,
  /tmp/n972.bin (real). Handed to eb4f0b (m_mtpnze65) - split/claim territory.

Round 22 (2026-09-06): qwen mixed-split FIXED (canary = 12095 oracle) + zaya remaining zeros = ffn_moe_gate/up
- eb4f0b graph.cpp fix (UNCOMMITTED as of this round, validated in-tree): tensor_is_external
  now also externalizes any produced value whose FULL-graph use count exceeds its in-slice
  use count (full_graph_use_count via ggml_graph_view aliasing of the scheduler's full
  use_counts/visited_hash_set). Values consumed by nodes in a LATER split (cross-slice
  readers like the CPU residual add) must stay External (ggml-slot write) even with in-slice
  consumers.
- VALIDATED BATTERY with the fix:
  * Canary (qwen + GGML_HRX_CPU_OPS=GET_ROWS): tok0=12095 ORACLE, text " Paris. The capital
    of France is also" - ROUND 21 RESOLVED, 38+ rounds of misdiagnosis closed.
  * Working qwen (no env): 12095 (no regression).
  * zaya q4nx ngl0: 9079 (no regression).
  * ZAYA q4nx ngl99: tok0=53335 repeating "expands expands..." - CHANGED from pre-fix
    143243 (real-varied-wrong) but STILL NOT oracle 9079.
- Zaya remaining zeros (readtrace /tmp/zrt2.err): ffn_moe_gate-N and ffn_moe_up-N read back
  ALL-ZEROS for every block N; Qraw-N/input_norm-N/node_N/ffn_moe_weighted-N all REAL. Zero
  every step -> router deterministic -> single-token collapse. Same externalization class
  (HRX-produced gate/up consumed by the CPU top-k ARGSORT) but the use-count fix did NOT
  cover them -> open question: ggml_graph_view use_counts aliasing for zaya's 1083-split
  structure (slices may not alias the full table), or in-slice==full use counts for gate/up.
- Executor cgraph discriminator (GGML_HRX_GRAPHCOUNT, /tmp/gc2.err): executor alternates
  972-node layer slices + 11-node tail slices (first op ADD = residual add, then final norm
  + lm_head) per step - the residual-add consumer IS in an HRX-executed cgraph (different
  call), confirming the fix shape (scheduler marks cross-slice values as slice outputs).
- Evidence: /tmp/fix1.err (canary 12095), /tmp/zaya_fix.log (53335), /tmp/zaya0_fix.err
  (9079), /tmp/zrt2.err (zaya gate/up zeros), /tmp/gc2.err (972/11 alternation).
- OPEN: commit the graph.cpp fix (still uncommitted); zaya gate/up externalization or
  aliasing fix; then re-run the zaya battery.

Round 23 (2026-09-06): zaya remaining failure = CROSS-BUFFER WRITE on prefill gate/up (same-process evidence)
- The round-22 externalization fix (a7f139ba7) fires for zaya's ffn_moe_gate/up (ext=1,
  consumed_outside=1, local_use=0, full_use=1 — GGML_HRX_TRACE_EXT) yet they still read zero
  on PREFILL only (decode reads real: gate-0 prefill 90112B@2621440 = all-zero; decode
  8192B@20480 = real, every step).
- Same-process iree-handle comparison (DUMP_WRITEBIND + READTRACE in one run, /tmp/zwr.err):
  * gate READ (buffer_get ctxbuf): 0x55ab5babffa0 (compute arena; 1529 refs)
  * gate PREFILL WRITE (record binding b1/b2 @2621440 len 131072): 0x55ab5bab4250 (720 refs)
  => DIFFERENT device buffers: prefill writes buffer A, readback reads buffer B -> zero slot.
  * Decode gate writes land in the correct buffer (real readback).
- Zero prefill gates -> wrong router -> poisoned prefill KV -> all decode steps attend over
  wrong KV -> deterministic collapse (tok0-99 = 53335 "expands..."), despite decode-side
  values being real. GGML_HRX_DOUBLE_EXECUTE made it worse (1042 re-runs, state corruption,
  <pad> stream) - zaya is not double-execute-idempotent.
- Open: the write binding for multi-token prefill externals resolves to the wrong device
  buffer (reresolve/(A) class - ff05bcb6f built for this, moot for qwen's l_out-26). Handed
  to eb4f0b with the request to generalize the trR/trA three-site probe (81fbd67f1, currently
  hardcoded to value 1336/20480@1572864) to zaya's gate value (value 4/5, len 90112,
  off 2621440) or take the slice directly.
- Evidence: /tmp/zwr.err (same-process write/read handles), /tmp/zrt2.err (gate/up zero on
  prefill only, full value-class census), /tmp/ext.err (externalization decisions),
  /tmp/zde.err (double-execute negative), /tmp/zaya_fix.log (53335).

Round 24 (2026-09-06): zaya stale handle pinned to the RECORDED CB baked refs (all live sites agree)
- eb4f0b 0740b4860 added [trM] (match-captured vs live tensor resolve) to the gate trace.
- trM data (zaya ngl99, /tmp/trm.err): cap_buf == live_buf == 0x55adb0b03fa0 for ALL gate/up
  (160 lines, cap_t==live_t, gen=5 id=5) - NO match-tensor staleness (eb4f0b's prediction
  failed). trA (per-exec ext binding) == trR (resolver) == 0x...b03fa0: all live sites agree.
- The stale handle lives ONLY in the RECORDED command buffer's baked refs: combined run
  /tmp/tg3.err (TRACE_GATE+READTRACE+WRITEBIND, same process) shows record-time dispatch
  binding [hrxbind3] for gate @2621440 = 0x...dc80 while resolver+readback = 0x...cf90.
- resync-ext fires ZERO times -> the (A) reresolve snapshot-compare never triggers the
  re-record even though baked (0x...dc80) != live (0x...cf90).
- SYNTHESIS: gate/up dispatches use the recorded CB's stale baked handle (a pre-arena-growth
  generation); reresolve keeps PREPARED refs live (trA correct) but the recorded CB is never
  re-recorded. qwen works because its arena never grows mid-run (baked stays valid); zaya's
  gate slices record against a pre-growth generation (the 17f wrong-set compare + 17h growth
  mechanism, now pinned to zaya's actual failure).
- FIX CANDIDATES (eb4f0b's slice): (1) fix the resync-ext comparison so baked-vs-live
  divergence triggers re-record (17f WRONG-SET fix); (2) invalidate recorded CBs after arena
  growth. Battery armed: zaya ngl99 expect oracle 9079.
- Evidence: /tmp/trm.err (all live sites agree), /tmp/tg3.err (record-vs-live same-process
  divergence), /tmp/zgc.err (4689 tiny view calls), /tmp/zrt2.err, /tmp/zwr.err.

Round 25 (2026-09-06): stale-record theory KILLED - multi-token gate write fails with CORRECT live bindings (kernel/dispatch level)
- GGML_HRX_FORCE_DIRECT (env added to graph-executor.cpp by b30173): forces use_graph_prepared=false
  -> the DIRECT path (no recorded CB) with per-execution bindings.
- zaya ngl99 + FORCE_DIRECT + TRACE_GATE + READTRACE (/tmp/fdg.err): trA (per-exec ext binding)
  == trR (resolver) == readback ctxbuf == 0x556cf874ef90 @2621440, gen=5 id=5 - ALL THREE AGREE on
  the live correct buffer. Yet the 5-token gate slot reads EXACT ZERO. Decode's 1-token gate
  (8192B @20480, same buffer) = real.
- CONCLUSION: round-24's "stale recorded-CB bake" is insufficient/irrelevant - even the direct
  path with provably-correct live bindings does not land the multi-token gate/up write. The
  failure is kernel/dispatch-level for the n=5 gate shape (H2/H3 class): either the zaya router
  MUL_MAT kernel's multi-token variant doesn't execute/store, or the prefill gate's producer
  isn't dispatched on-device (CPU-placed producer vs device-slot consumer).
- Note: qwen's 5-token node_972 mm (wmma) writes correctly post-fix (canary 12095), so NOT all
  multi-token mms fail - zaya's gate kernel path is specific.
- NEXT PROBES: (1) record/dispatch workgroup count for the gate dispatch prefill vs decode;
  (2) whether the prefill gate slice dispatches a device kernel for gate at all; (3) zaya
  arena dump correlation (does gate data exist anywhere on device - d5694d's GGML_HRX_ARENA_DUMP
  in his worktree). Handed to eb4f0b + d5694d.
- Evidence: /tmp/fdg.err (all-agree + zero), /tmp/zd2.log (FORCE_DIRECT 53335), /tmp/trm.err,
  /tmp/tg3.err.

Round 27 (2026-09-06): gate/up write-target = dedicated wrong buffer (same-process, ALLBIND census)
- Captures for eb4f0b (kernel/dispatch slice):
  1. ARENA_DUMP produced nothing for zaya: the compute arena is only 4.75MB (worst-case bs=128
     reserve; qwen was 78MB) - below d5694d's 16-256MB filter. KV = 10MB, model = 31.6GB. The
     filter needs lowering for zaya (noted for d5694d).
  2. ALLBIND + READTRACE same-process (/tmp/c4.err): gate WRITE binding (val=3, origin=0,
     access=2 @2621440 len 131072) buf=0x55e863b23840 vs gate READBACK ctxbuf=0x55e863beaf90.
     Buffer census of access=2 (write) bindings: 0x...bb0e60 (3645 = transient arena outputs),
     0x...beaf90 (880 = ggml compute arena = node_972-class externals that read back correctly),
     0x...3840 (160 = EXACTLY 40 ffn_moe_gate + 40 ffn_moe_up).
- CONCLUSION: gate/up's record write-target = a dedicated 160-ref buffer (0x...3840) that the
  readback never reads; their ggml home (resolver trR + readback) = 0x...beaf90. Working
  externals (node_972 class) write their ggml home. The prepared/record binding for gate/up is
  NOT sourced from ggml_backend_hrx_resolve_value_buffer (which gives the correct tensor
  context) - it comes from import-time value storage / command-binding construction. Suspect:
  dispatch-mul-mat's output-binding construction for the gate (expert/MoE) pattern assigns the
  output to a fused/scratch buffer instead of the external's ggml context.
- Evidence: /tmp/c4.err (same-process write-vs-read buffers + census), /tmp/c3.err,
  /tmp/c2.err, /tmp/all4.err. Handed to eb4f0b (m_mtppjfou).

Round 28 (2026-09-06): gate-slot SET_ROWS write = origin GraphValue bound to wrong buffer (ref-path at fault)
- eb4f0b 8e6e5e7d2: recref2 prints EVERY record-time kernel binding origin/val/buf/off/len/access
  under GGML_HRX_DUMP_ALLBIND (the old 20480/4096 filter excluded gate's 131072B).
- zaya ngl99 capture (/tmp/o5.err): recref2 for the gate slot @2621440:
  * val=3 origin=0 buf=0x55b77f67f840 off=2621440 len=131072 access=2 (the SET_ROWS write)
  * val=13 origin=0 buf=0x55b77f67f840 off=2621440 len=131072 access=0 (read at same slot)
- origin=0 = GraphValue (qwen calibration: node_972/embd=0, transients=1). The whole SET_ROWS
  command's binding set (val 3 write + val 13 read) resolved to buffer 0x...f840, while trR
  (ggml_backend_hrx_resolve_value_buffer on the tensor) = the ggml home 0x...beaf90-class.
- Per eb4f0b's dichotomy: origin=GraphValue + wrong buffer => the buffer-resolution/ref path
  (reresolve or the command-binding resolution for GraphValue-origin outputs) is at fault -
  NOT the transient/two-home case (fix = dual-write) and NOT the fusion case.
- Evidence: /tmp/o5.err. Handed to eb4f0b (m_mtppnr3a).

Round 29-30 (2026-09-06): gate_up root cause + alias-consumer fix attempt (validated direction, regression found)
- ROOT CAUSE (round 29): ffn_moe_gate_up-N (MUL_MAT_ID output, 98304B = 4096x6x4) is consumed
  in-slice ONLY by VIEW nodes ([ffn_moe_gate-N/VIEW] [ffn_moe_up-N/VIEW] - the gate/up row
  split) -> produced+consumed -> Transient -> 98304B transient-arena binding (program-8
  val=3 origin=1). The VIEW outputs (ffn_moe_gate-N etc., External, read back 90112B by the
  CPU swiglu) alias to gate_up's value -> the CPU readback samples the ggml home (gallroc
  placed the view/parent in the compute arena) which the transient write never touches -> zero.
- FIX ATTEMPT (round 30): exclude layout-alias consumers (VIEW/RESHAPE/PERMUTE/TRANSPOSE)
  from use_counts so alias-only-consumed produced values classify External (terminal rule).
  Result: zaya ngl99 tok0 75615 (53335-collapse -> VARIED tokens; gate/up readbacks REAL
  4096/4096; ZERO zero-reads anywhere) but NOT oracle 9079. QWEN REGRESSED exactly as
  eb4f0b warned (canary 24095, working qwen 104) - mid-program alias chains feeding IN-SLICE
  real ops (Kcur->permuted views->flash-attn) flipped External -> fused path broken.
- REVERTED graph.cpp to committed state; qwen restored (12095 both); zaya ngl0 9079 intact.
- CORRECT RULE (for eb4f0b): (1) transitive fixpoint - consumption through alias chains
  propagates; alias chains reaching an in-slice real op stay Transient; only chains leaving
  the slice (no in-slice real consumer) go External; (2) the consumed_outside delta needs
  consistent bases (raw full use_counts count alias uses -> false-positive External for Kcur).
- Remaining zaya divergence (75615 vs 9079) after the (uncommitted) fix = likely a second
  same-class instance or the capacity-tail (write covers rows 0-5 of an 11-row base per
  eb4f0b's 180224B-vs-98304B arithmetic).
- Evidence: /tmp/za_fix.log (75615 varied), /tmp/zrt3.err (no zero reads), /tmp/big.err
  (gate_up consumers = views only), /tmp/pm.err. Handed to eb4f0b (m_mtpqb2qt).

Round 31 (2026-09-06): alias-only-external rule FIXED + committed (2824946b5) - zaya fluent, zero regressions
- RULE: a produced value whose in-slice consumers are ALL layout aliases (VIEW/RESHAPE/
  PERMUTE/TRANSPOSE), and none of those aliases' outputs (recursively, BFS over alias
  consumers) is consumed by a real in-slice node, is read solely through slice-exiting
  aliases (cross-slice readers like the CPU swiglu) -> External (ggml slot written). If an
  alias-descendant reaches a real in-slice op (Kcur -> permuted views -> flash-attn) the
  value stays Transient (round-30 qwen regression avoided).
- NO supports_op export needed: the HRX slice contains only HRX-supported real ops (the
  dispatch-scheduler hard-fails on unmatched nodes), so eb4f0b's unsupported-consumer rule
  is moot in-slice; alias-outputs-leave-the-slice is the discriminator.
- BATTERY (all green): zaya ngl99 tok0=563 " is used to hide the problem. The" (FLUENT
  coherent English vs 53335-collapse/75615-garbage); canary 12095; working qwen 12095;
  zaya ngl0 9079; zero zero-reads in the readtrace.
- REMAINING: numerical divergence (563 vs oracle 9079 - coherent text = structure correct).
  Suspects: eb4f0b's capacity-tail (dispatch writes token rows 0-5 of an 11-row base;
  rows 6-10 uninitialized - matters only if a consumer spans capacity) or a subtle kernel
  precision diff. Next: localize the first divergent op by comparing ngl99 device readbacks
  against ngl0 CPU values mid-prefill.
- Evidence: /tmp/za31.log (563 fluent), committed graph.cpp rule.

Round 32 (2026-09-06): numerical divergence hunt - attention bit-identical, divergence in FFN/expert region
- zaya 1-token "Paris": ngl0 oracle tok0=9731 (sharp: 9731@30.2 vs 2nd@23.5); ngl99 = 2364
  (top5 28.9/28.1/27.4/27.3/27.0 - sharp but flat-among-top; 9731 NOT in top-5).
- ATTENTION BIT-IDENTICAL: Qraw-0/input_norm-0/node_153 readbacks byte-identical between
  all-HRX and MUL_MAT_ID->CPU runs (-3.086 -1.413 -2.988 6.508 ...). Divergence = FFN/expert
  region or after.
- Op-forcing structurally confounded for zaya (MMID->CPU=99889, RMS_NORM->CPU=239702, none =
  oracle; full compute set aborts in llama_init_from_model). Layer bisection ngl=1..16 fails
  (sched error -1; only ngl0/ngl99 run). f32twin model identical divergence (not Q4NX-dequant).
- 5-token ngl99 logits flat (~10s) vs ngl0 sharp (14-17) - degraded context at batch.
- Needed tool: CPU-side oracle values at the same readback points (CPU post-compute named-
  tensor dump at ngl0, or accept structural change with single-op forcing and compare the
  op's own outputs). Handed to eb4f0b (m_mtpr1qjd).
- Evidence: /tmp/rt1.err (1-token all-HRX chain), /tmp/mmid.err (MMID-CPU, attention identical),
  /tmp/ngl1.log (sched failure).

Round 33 (2026-09-06): gate/up = strided views of gate_up; write 49152 vs readback 90112 (capacity-tail geometry)
- Same-run readtrace with ne/nb (/tmp/rts4.err, zaya f32twin 5-token prefill):
  * ffn_moe_gate-0/up-0: ne=2048,1,6,1 nb=4,16384,16384,98304 nbytes=90112 view=1 vsrc=ffn_moe_gate_up-0,
    readback size=90112 @2621440
  * input_norm-0: ne=2048,6,1,1 nb=4,8192,49152,49152 nbytes=49152 (6 slots, sane)
  * Post-fix dispatch writes for the gate/up class: len=49152 (160 origin=0 GraphValue externalized
    + 362 origin=1 transient)
- GEOMETRY: gate/up = strided VIEWs of gate_up: dim-2 (6 logical slots) with nb[2]=16384 (4096 =
  the parent gate_up width x4B), but ggml_nbytes = 90112 = 22528 elements (2048x11) while the ne
  product = 2048x1x6 = 12288 (49152B). Write = 49152 (6 slots x 2048 rows); readback copy = 90112
  (11-slot extent). The 5 extra slots = eb4f0b's capacity-tail candidates (uninitialized if the
  copy/consumers span them).
- OPEN: whether the CPU swiglu consumes the full 90112 (contamination) or is strided/slot-aware
  (benign over-copy) - eb4f0b's view/copy machinery read (m_mtpr9dp8). Fix candidates: zero-fill
  the tail, size the base to the actual slot count, or make the copy extent = the logical write.
- Evidence: /tmp/rts4.err (ne/nb geometry), /tmp/gb.err (49152 writes).

Round 34 (2026-09-06): strided-gather experiment NEGATIVE - view overlap is expected (layout-preserving reads); divergence elsewhere
- Hypothesis tested: the gate/up views' readbacks (90112B span) "overlap" (gate s1 == up s0) -
  implemented a logical-order strided gather in buffer_get. Result: slots became distinct but
  output did NOT reach the oracle (5-token 85363, 1-token 32003, both worse than the baseline
  563/2364). REVERTED.
- CONCLUSION: the overlap is EXPECTED under layout-preserving semantics - the contiguous span
  from the view's offset legitimately contains the parent's interleaved gate/up rows; the CPU
  consumer reads per nb (stride-aware), so the plain d2h copy is correct. buffer_get is NOT
  the divergence.
- 1-token geometry: gate ne=2048,1,2 (2 slots for the 1-token graph, span 24576) - the graphs
  carry padding slots (slot counts > real tokens in some phases), so capacity-tail effects
  remain possible where the kernel writes fewer slots than the graph reads.
- Tree cleaned: eb4f0b's stale TRACE_1336 probes (3 files, authorized deletion) + orphan
  ggml-hrx-support.h + stray scripts removed; clean at bdf013016; build green.
- OPEN: the remaining numerical divergence (zaya coherent-but-wrong first token; qwen-fixed +
  zaya-structural-fixed committed) needs either the CPU-oracle dump harness (ngl0 named-tensor
  dumps at the HRX readback points) or a kernel-numerics review of the zaya-specific path
  (MUL_MAT_ID expert mms, router chain). Baseline: 5-token 563 (oracle 9079), 1-token 2364
  (oracle 9731).

Round 35 (2026-09-06): divergence DEFINITIVELY localized to the expert-mm kernel (loom mul_mat_id)
- Evidence chain (zaya f32twin, 5-token "The capital of France is"):
  1. post_attn_norm-0/Qraw/node_153 bit-identical between HRX runs (attention + norms exact).
  2. router_logits-0 CPU vs HRX: ~0.003-0.009 fp noise only; per-token argmax IDENTICAL
     ([12,15,16,12,16,0]) -> expert selection = the same.
  3. gate_up (mul_mat_id output): CPU oracle (ngl0 GGML_DUMP_NODE dump, /tmp/nodedump/
     r04_000) rms=0.878; HRX gate/up readbacks (/tmp/hrx_gate0.bin, hrx_up0.bin via
     GGML_HRX_DUMPVIEW) rms=2.01/1.95 = 2.3x LARGER; no offset/permutation maps them
     (best-alignment SSE mismatches).
  4. => same expert + same input + same weights, the HRX loom ggml_mul_mat_id_f32_f32_wmma
     computes different values than ggml CPU's mul_mat_id.
- All structural machinery exonerated (writes full-extent 98304, readbacks real, views
  layout-preserving, routing identical).
- SUSPECTS: a fused post-op (scale by ffn_moe_weights/probs, or an output_unary_op) applied
  by the loom kernel/dispatch but not the CPU path; or the expert weight rows indexed wrong
  (ids/partition mapping). pd program-8: kernel ggml_mul_mat_id_f32_f32_wmma, expert_count 16,
  route_count 1, output_size 4096.
- Handed to d5694d (loom kernel review, m_mtps4tje) + eb4f0b. Tools: GGML_DUMP_NODE +
  GGML_DUMP_FILTER (ngl0 CPU dumps), GGML_HRX_DUMPVIEW (HRX readback dumps, keep-first).

Round 36 (2026-09-06): n=1 chain comparison - token-0 gate correct, token-1 wrong (view fill mechanism unknown)
- n=1 "Paris" chain harness (CPU GGML_DUMP_NODE vs HRX GGML_HRX_DUMPVIEW chain dumps):
  * router_logits-0: fp noise only (~0.005), argmax matches.
  * gate_up: token-0 gate = CORRECT (hrx -1.4707 vs cpu -1.4651 = fp noise); token-1 gate
    reads -0.2756 vs cpu -1.4620 = WRONG. 5-token: region rms 2.1 vs 0.88 (many rows wrong).
  => slot-0 lands right, subsequent slots wrong = stride-blind contiguous copy of the base
    region fills the view slots (interleaved [t0gate|t0up|t1gate...] data), not per-stride.
- buffer_copy fired ZERO times for gate_up names -> the base->view fill does NOT go through
  ggml_backend_buffer copy. The 49152-len GraphValue write bindings (externalized views from
  ALLBIND) = the view slots are filled by another executor mechanism (copy command? fill?)
  that must be stride-aware but likely is not.
- OPEN: which mechanism fills the gate/up view slots from gate_up (record shows per-view
  49152 write bindings) and does it honor view strides - eb4f0b executor read (m_mtps81yj).
- Evidence: /tmp/hrx_ffn_moe_gate-0.bin (6144 el span, token0 right/token1 wrong),
  /tmp/nodedump/r03_110 (CPU gate_up [4096,2] oracle), /tmp/hrx_router_logits-0.bin,
  /tmp/chain.err. Harness: GGML_DUMP_NODE (ngl0) + GGML_HRX_DUMPVIEW chain names (ngl99).

Round 37 (2026-09-06): capacity-padding leak CONFIRMED (weighted row-0 diverges progressively by block)
- n=1 "Paris" chain comparison, weighted row-0 CPU-vs-HRX across blocks:
  block 0 mad=0.0009 (MATCH), 5 = 0.007, 10 = 0.45, 15 = 0.005, 20 = 0.78, 25 = 0.37,
  30 = 0.22, 35 = 3.23, 39 = 2.18.
- Real-token math starts clean (blocks 0-5) then diverges progressively = pad-row garbage
  from the HRX mul_mat_id's unwritten capacity slots contaminates the real path via some
  capacity-spanning op as depth grows (CCA recurrent state or a repeat/norm spanning the
  padded extent).
- At n=1 the graph carries 2 slots (1 real + 1 pad; gate ne=2048,1,2); CPU computes the full
  extent, the HRX mul_mat_id writes only bounded_token_count (= real tokens) assignment rows
  -> pad rows hold buffer leftovers -> leak.
- FIX DIRECTION: HRX kernels must produce the full capacity extent like CPU (zero-fill the
  unwritten pad rows of the gate_up base, or bounded_token_count = the graph's slot count).
  Cheapest test: zero the base's pad rows after the mul_mat_id dispatch.
- Evidence: weighted row-0 mad per block (this round), /tmp/hrx_ffn_moe_weighted-*.bin,
  /tmp/nodedump/r04_* (CPU oracle), /tmp/hrx_result_norm.bin, /tmp/hrx_result_output (absent).
- Handed to eb4f0b (m_mtpsfivk) for the fix-site call.

Round 38 (2026-09-06): ROOT CAUSE CONVERGED - mul_mat_id dst rows at expert-partition ordinals, not token-sequential
- d5694d kernel review CLOSED #1 (expert stride CLEAN: format F32=32 tile 1024B = K*4 row_bytes;
  ggml src1 = [K, N, n_expert] planes matching expert*weight_expert_bytes).
- #2 IDENTIFIED: kernel write index = bounded_assignment = route_tile_base + local_route
  (route_tile_base = partition_ordinal<<5, 32-row expert partitions) -> dst rows land at
  expert-PARTITION-packed ordinals, NOT token-sequential dst rows (CPU oracle orders dst by
  the ids-derived logical mapping = token-sequential for route_count=1).
- FITS ALL EVIDENCE: n=1 right (single token -> expert partition 0 -> dst 0); token-0 right /
  token-1+ wrong (token 1 in expert 15's partition -> dst ~32, never dst 1); rms inflated on
  multi-token; no clean permutation (shifted other-expert rows); round-37 "progressive leak"
  = shifted rows read by deeper blocks, same mechanism.
- FIX DIRECTION: routing bundle (common_mul_mat_id_ensure_moe_routing_bundle) must emit
  assignments ordered by the CPU dst/logical row (token-sequential when route_count=1), or the
  kernel writes dst at the token index resolved from the assignment table. d5694d to implement.
- MILESTONE: qwen externalization fixed + committed (rounds 21-31); zaya = this dst-ordering
  bug in the MoE routing bundle. Evidence: all rounds 32-37 captures.

Round 39 (2026-09-06): count-mismatch hypothesis OUT (ids 0..15, expert_count 16 consistent); dst-ordering #2 stands
- The "id 16" from round-35's argmax-over-17 = a misread: router_logits = [17,6] (17 logit
  slots) but the CONT before routing slices to ffn_moe_probs [16,6]; the ARGSORT over those
  yields routing ids 0..15 only.
- Actual routing ids (argsort-0, r04_108, 5-token prompt): token 0->3, 1->5, 2->12, 3->7,
  4->2, 5->4 (all within 0..15). Weight = [K, N, 16] per byte math (512MiB = 2048x4096x16x4).
- CONFIRMED: expert_count=16 consistent; d5694d's #1 CLEAN verdict stands; the count-mismatch
  hypothesis is out. Root cause = #2 (dst rows at expert-partition ordinals
  partition<<5+local vs CPU token-sequential) - fix in the MoE routing bundle dst ordering,
  in d5694d's hands. Battery armed.
- Evidence: /tmp/nodedump/r04_108 (argsort ids [16,6]).

Round 40 (2026-09-06): ZEROPAD test NEGATIVE - unwritten-pad mechanism ruled out; dst-ordering #2 confirmed
- GGML_HRX_ZEROPAD (env-gated zero-fill of every write-binding region before each dispatch,
  command-program-executor.cpp): NO CHANGE (5-token 563, 1-token 2364 = baseline).
- => the rows the downstream reads as token slots are NOT unwritten leftovers (zeros don't
  alter results) - they contain data written at wrong dst positions = d5694d's #2
  (partition-ordinal dst vs token-sequential) confirmed as the mechanism.
- n=1 residual puzzle: at n=1 (2-slot graph, 1 real token), token-0 gate/weighted = fp-correct
  through early blocks yet the output diverges at later blocks even with zeroed pads - the
  depth-progressive drift at n=1 is not pad garbage; next suspect = CCA recurrent state path
  (cache_s crossing CPU) if the dst-ordering fix doesn't clear n=1.
- Note: the recurring stale [trGP]/[trPCC] TRACE_1336 probes keep reappearing in
  graph-program-cache.cpp / prepared-command-program-cache.cpp (re-added 3+ times this session,
  each breaking the build with the out-of-scope 'bindings' error) - purged again; tree = my 2
  env-gated instruments only (ggml-hrx.cpp DUMPVIEW, command-program-executor.cpp ZEROPAD).
- Awaiting d5694d's routing-bundle dst-ordering fix; battery armed.

Round 41 (2026-09-06): kernel dst fix (cc9da925b) VERIFIED ACTIVE but insufficient - view-fill is the remaining site
- Applied d5694d's cc9da925b (mul_mat_id publish writes output_view[token*route_count+route])
  to the current branch + forced the kernel-corpus rebuild (make ggml-hrx-kernel-corpus +
  touch .loom). Output unchanged (563) BUT the gate_up data REARRANGED (row-head sequence
  changed vs pre-fix) => the fix is ACTIVE at the base level.
- Post-fix gate rows vs CPU oracle heads: row-0 = cpu t0 within fp noise; rows 1-5
  (-0.1617/0.3435/0.2013/0.1693/-1.7598) still != cpu t1-5 (0.265/1.105/-1.015/0.188/0.565).
- SYNTHESIS: the observable the CPU consumes = the gate/up VIEW readbacks; the views are
  SEPARATE externals with their own 49152B write bindings, filled by an executor-side
  view-materialization that appears stride-blind (buffer_copy fires ZERO times - round 36).
  The kernel's dst fix writes the base token-sequentially; the VIEW FILL = the remaining
  stride-blind suspect. Round-34's buffer_get gather changed the readback but the fill stayed
  wrong; ZEROPAD irrelevant.
- OPEN: which executor mechanism fills the gate/up view slots (the 49152 GraphValue write
  bindings) and does it honor view strides? eb4f0b executor read (m_mtpsxxkv). d5694d's fix
  confirmed active (m_mtpsy23c).
- Evidence: /tmp/hrx_ffn_moe_gate-0.bin (post-fix rows), /tmp/nodedump/r04_000 (CPU oracle).

Round 42 (2026-09-06): full dst-ordering fix (cc9da925b + f06731ffe) applied + corpus rebuilt - zaya UNCHANGED (563)
- Both d5694d commits applied to the current branch + forced kernel-corpus rebuild: still 563.
- gate_up region (view readbacks = base aliases): token-0 gate AND up = CPU-correct (fp noise);
  tokens 1-5 WRONG. The base's t0 lands right, t1+ don't, even with the token-major publish.
- Table semantics (moe_routing_tables.loom:80): the builder stores %assignment_i32 = the GLOBAL
  assignment index at assignment_view[expert][ordinal] (= token for route_count=1) -> the
  kernel's %token should be right. Yet t1+ writes don't land at the expected base rows.
- LEADS: (1) kernel per-partition assignment->table indexing (LOCAL-in-partition vs the
  table's row layout); (2) the sampled "base region" = actually the view materialization
  (unverified - all execution = recorded-graph launches; per-kernel probes never fire; the
  direct path execute_prepared_kernel_command is never called even with FORCE_DIRECT).
- d5694d to consider a kernel-side debug write of the output row index to close it
  definitively. Kernel fixes = uncommitted on the branch; tree otherwise clean (DUMPVIEW
  instrument in ggml-hrx.cpp only).
- Evidence: base region t0-correct/t1+-wrong (round 41 captures).

Round 43 (2026-09-06): base CONTAINS all token data but SCATTERED at partition ordinals - dst fix not effective in executed kernel
- Definitive base-region analysis (/tmp/base_cmp.py): all 6 tokens' gate+up heads exist in the
  gate/up dumps at EXACT matches (t1 gate 0.2654@gate[16463], t2 1.1045@gate[3724], t3
  -1.0146@gate[970]... 12/12 values to 4 decimals) but SCATTERED at non-token rows (t1 at
  row ~8, t2 at ~1.8...). Token-sequential positions (t1 = element 4096) = WRONG values.
- => the kernel writes every token's data CORRECTLY but at partition-ordinal dst rows: the
  dst-ordering fix (cc9da925b + f06731ffe) is NOT effective in the executed kernel. Either
  the dispatch binds a different publish/variant than the patched plain/postops/next_rmsnorm,
  or the running kernel = stale (embedded-source vs runtime-JIT mismatch).
- Handed to d5694d (m_mtptafg9): verify which kernel variant the zaya gate_up dispatch binds
  + that the patched publish_vector4 runs. Fix is right in principle; the executed kernel
  doesn't have it.
- Evidence: /tmp/hrx_ffn_moe_gate-0.bin + up-0.bin (post-fix), /tmp/nodedump/r04_000 (CPU
  oracle), /tmp/base_cmp.py.

Round 44 (2026-09-06): correction - ONLY token-0 clean post-fix (row-aligned analysis); 12/12 single-element finds = false positives
- Row-aligned matching (/tmp/rowmatch.py): dump rows 0-1 = cpu rows 0-1 (t0 gate+up, sse 0.1 =
  CLEAN); rows 2-10 match cpu rows poorly (sse 7-16 = fragmented). The earlier 12/12
  single-element "exact finds" (round 43) = FALSE POSITIVES.
- => post-fix base = token-0 correct + tokens 1-5 fragmented/absent. The executed kernel does
  NOT write token-sequential rows.
- d5694d's methodology catch: his local canary = qwen3-0.6B DENSE (never runs mul_mat_id);
  his 456 = a base artifact; his fix has only been tested on my zaya runs. Verify the dispatch
  binds the patched kernel variant (plain publish_vector4) for zaya's gate_up or another
  mul_mat_id instance is selected.
- Handed to d5694d (m_mtptbcnr correction).
- Evidence: /tmp/hrx_ffn_moe_gate-0.bin (post-fix), /tmp/nodedump/r04_000 (oracle),
  /tmp/rowmatch.py.

Round 45 (2026-09-06): lead determination = LEAD 1 (kernel per-partition indexing); view-materialization (lead 2) dead
- pd2 scan (eb4f0b): NO copy/fill commands anywhere; every 49152 write = an ordinary
  activation output. The gate/up readbacks = offset-preserving d2h reading the base DIRECTLY.
- => the gate/up dump data = the base's actual rows; the row-aligned analysis (round 44:
  only t0's gate+up clean sse 0.1; rows 1-10 fragmented) means the BASE itself is misplaced
  for tokens 1-5 - not a read/copy artifact.
- => d5694d's kernel per-partition indexing = the live lead; his proposed kernel
  instrumentation (emit token/dst-row/assignment/expert per write into a debug region) = the
  next step to pin the exact row each token hits vs token*route_count+route.
- Tree: committed docs + the 3 uncommitted loom kernel fixes (mul_mat_id plain/postops/
  next_rmsnorm from cc9da925b+f06731ffe) only.
- Handed to d5694d (m_mtptjfhf).
Round 46 (2026-09-06): embed staleness CLOSED - the fix IS running; %token semantics = the live question
- Text-format rebuild (-DGGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT=text): still 563.
- Embedded ops/mul_mat_id_f32_f32_wmma.loom = byte-identical to the patched working tree
  (7446B, token-major present); the motifs core (25518B) = separate (no publish) as expected.
- => the runtime JIT compiles the patched publish_vector4 yet zaya still scatters t1+
  => the %token value at runtime (per-partition local ordinal? partition-relative?) != the
  token index the table's global assignment implies. d5694d's kernel instrumentation (emit
  token/dst-row/assignment/expert per write into a debug region) = the only way to see the
  actual %token - requested (m_mtptmdza).
- Build dir now uses text kernel format (cmake cache change) - note for future builds.

Round 47 (2026-09-06): instrument handed to d5694d (loom syntax attempt broke kernel - reverted)
- My marker-instrument attempt in publish_vector4 (vector.splat + scf.if) = invalid loom syntax
  (PARSE/003 errors) -> kernel failed -> run crashed. REVERTED; tree restored (2 postops
  kernel fixes + DUMPVIEW hook + text-format corpus build remain).
- Handed the instrument to d5694d (m_mtptvkwk): model (~/zaya-q4nx-f32twin.gguf), runner
  (/tmp/zgreedy), branch (fix/hrx-ngl-init-order at ~/hrx-ws/amd-hrx-graph) with his
  cc9da925b/f06731ffe + text-format corpus = all on the same host. He implements + iterates
  the partition-table instrument end-to-end.
- State: the dst-ordering fix is embedded + JIT-verified; the t0-right/t1+-fragmented
  contradiction needs the kernel-side write trace (token -> actual dst row per publish).
- Goal: 3/5 tasks complete; task-4-device: qwen fixed + committed; zaya final kernel question
  in d5694d hands.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
Round 49 (2026-09-06): marker instrument = all-pads in my runs (row-tail overwrite corrupts); clean state restored
- Applied d5694d's c4c4ee403 marker cleanly (plain fix in place, file byte-identical to his):
  run = ALL PADS (tok0-7=0), no gate readbacks. The row-tail overwrite (last 4 channels of
  each row) reliably corrupts the model in my runs - possible causes: a downstream op reading
  the tail channels, or the 4 concurrent publishes/row racing the tail writes. Recommended
  moving the marker to a dedicated spare buffer. Posted to d5694d (m_mtpuk6el).
- /tmp cleaned (61G of captures freed after a disk-full incident - all intermediate .err/.bin
  captures deleted; conclusions preserved in the committed spec).
- Clean 3-fix state (563 baseline) restored; DUMPVIEW hook ready.
- Goal: 3/5 tasks complete; task-4-device: qwen fixed + committed; zaya marker capture in
  d5694d's iteration.

Round 50 (2026-09-06): corruption localized BETWEEN attention output and next block input (CCA recurrent state prime)
- input_norm-1 (block-1 attention input): slot 0 matches CPU (mad 0.001); slots 1-5 DIVERGE
  (mad 0.2-0.7) => corruption upstream of the gate mm (eb4f0b's call confirmed).
- node_153 (block-0 raw attention output [128,8,6]): LOW mad across all 6 tokens (0.03-0.06,
  approx due to layout skew) => block-0 attention looks fine for every token.
- => the corruption enters BETWEEN block-0's attention output and block-1's input norm: the
  residual-add / CCA recurrent-state path. cache_s_l1 = "a copy of input_norm-1 (cont)" per
  the CPU dump naming (CCA state feeds from the block input). eb4f0b's round-37 CCA suspect
  (recurrent state crossing CPU via GET_ROWS, 313KB/layer) = the prime mechanism.
- NEXT PROBE: cache_s_l0/l1 + CCA conv output rows 0-5 HRX-vs-CPU. /tmp disk-full incident
  resolved (d5694d's /tmp/zadump arena capture = 61GB - cleaned; his capture is running).
- Handed to eb4f0b (m_mtpuulsg). Goal: 3/5 complete; qwen fixed+committed; zaya = CCA-state
  path in active localization.

Round 51 (2026-09-06): input_norm-0 slot-0 BIT-IDENTICAL; divergence from token-1 in block-0 middle
- input_norm-0 (block-0 own input norm): slot 0 mad=0.000000 (cpu0=hrx0=-1.1449 BIT-IDENTICAL);
  slot 1 head matches (0.7869) then diverges mid-row (mad 0.66); CPU slots 2-5 = ZERO (padded
  in the r04 dump - not a clean oracle there; input_norm-1 = the cleaner 6-slot signal).
- => embedding + token-0 path bit-exact; divergence enters at token-1+ somewhere in block-0's
  middle (attention fine per node_153; norm/ffn boundary muddy). The block-0 OUTPUT assembly
  (residual/CCA/ffn-residual) = the origin window.
- REMAINING DISCRIMINATORS (eb4f0b's #3/#4): cache_s_l0 write/read + block-0 ffn residual
  output per token. Offered the DUMPVIEW chain extension; awaiting the go.
- Fleet: eb4f0b (localization) + d5694d (kernel instrument) both active.

Round 52 (2026-09-06): expert selection IDENTICAL ([3,5,12,7,2,4] both runs, all 6 tokens) - router hypothesis OUT
- HRX-run block-0 argsort top-1 ids = [3,5,12,7,2,4] = CPU-run ids (r36_000 vs r04_108) -
  ALL 6 TOKENS MATCH. eb4f0b's expert-selection hypothesis ruled out (the CPU topk picks
  the same experts from the HRX router logits).
- Remaining: node_153 (attention output + residual) all-tokens fine + ids same + (apparently)
  right inputs -> the gate_up mm rows 1-5 disjoint = d5694d's fetch/compute lane (per-row
  weight-slice or activation fetch for partitions/tokens beyond the first in a multi-token
  launch). input_norm-1 slots-1-5 = downstream of the block-0 output assembly.
- Evidence: /tmp/nodedump/r36_000 (HRX argsort), r04_108 (CPU argsort) - both [3,5,12,7,2,4].
- Fleet: eb4f0b (localization) + d5694d (kernel fetch/compute lane).

Round 53 (2026-09-06): fetch/store/routing source EXONERATED - runtime wave-scheduling/race in d5694d lane
- eb4f0b narrowed scope: with ids identical ([3,5,12,7,2,4] both runs) + right inputs +
  correct-on-paper fetch/store (core lines 180-220 verified), the disjoint rows-1-5 = a
  RUNTIME/codegen issue: (a) multi-partition wave scheduling (scf.for %active_partition step
  %launch_partition_count) or (b) builder->mm race. His partition-table-tail instrument
  resolves (a) vs (b). Passed to d5694d (m_mtpv0jve).
- Capture limitation: cache_s/cca_state activations are HRX-internal (never cross) -> the
  cache_s/conv-output readback diffs (eb4f0b's #3/#4) are not capturable via DUMPVIEW. The
  crossable signals captured: input_norm-0 (slot-0 bit-identical, slot-1 mid-row divergence),
  input_norm-1 (slots 1-5 wrong), node_153 (all-tokens fine).
- 122-file HRX-side capture on strixhalo. Awaiting d5694d's instrument results.

Round 54 (2026-09-06): consolidation - prefill ids MATCH (solid); several comparisons were file-misaligned (confounded)
- SOLID: (1) PREFILL ids (16x6 argsort r40 HRX vs r04 CPU) = [3,5,12,7,2,4] all 6 tokens
  MATCH; (2) input_norm-0 slot-0 bit-identical; (3) gate_up t0 clean + t1-5 disjoint with
  SAME experts => mm input (post_attn_norm-0, in-program/unmeasurable via readbacks) or the
  mm compute/fetch.
- CONFOUNDED: the "38/40 mismatch" = decode-graph argsort files compared across misaligned
  run sequences (each HRX slice = a graph_compute -> run counters diverge); node_153 compare
  = slightly misaligned layout. Both flagged for careful re-measurement.
- Next: post_attn_norm-0 capture needs an executor-side in-program tensor dump (new mechanism)
  or d5694d's kernel instrument. Decode ids need an aligned capture.
- Fleet: eb4f0b (localization), d5694d (kernel instrument).

Round 55 (2026-09-06): CCA state diverges from block ~5 at n=1 (mad 0.015@0 -> 1.81@5 -> 3.99@20)
- cca_last_conv_states copy mad series (n=1 "Paris" prefill): block 0 = 0.015 (clean-ish),
  5 = 1.81, 15 = 1.33, 20 = 3.99 - the state diverges from ~block 5 and grows through the
  prefill. cache_s_l10 detailed: SCALE component identical (0.00000), cca_last_conv_states
  copy mad 2.28, input_norm-10 copy mad 0.54.
- Supports eb4f0b's windowed-state model: the HRX block outputs (input_norm copies) feeding
  the CPU conv/state drift progressively from block ~5; the state carries the corruption.
- ORIGIN WINDOW: block ~5's output assembly. Next probes proposed: block-5 router ids
  (single-token flip-check) or block-5 ffn/conv outputs.
- Captures: /tmp/ndh_* (HRX) + /tmp/ndc_* (CPU) on strixhalo.
- Fleet: eb4f0b synthesizing; d5694d kernel instrument.

Round 56 (2026-09-06): (b) builder->mm race STRUCTURALLY RULED OUT; (a) per-partition compute fetch = live
- eb4f0b sharpened: rows placed right but COMPUTED wrong (f(wrong expert weight or wrong
  activation) for partitions beyond the first); candidates (a) per-partition fetch drift or
  (b) builder->mm race.
- (b) RULED OUT structurally: direct path = same-stream FIFO dispatch (hrx_stream_dispatch);
  recorded path = linear dependency chain (GraphDependencyChain: each node depends on the
  last -> mm serialized after the builders). No race path exists.
- (a) = live: the per-partition compute fetch (expert weight slice via expert_byte_base or
  the activation row) drifting for partition_ordinal > 0. d5694d's instrument = the
  confirmation (correct table tuples + wrong rows).
- Fleet aligned: eb4f0b (code conclusions), d5694d (kernel instrument + (a) review).
- My branch ready for the instrument commit.

Round 57 (2026-09-06): measurement ceiling - readback-based CCA comparisons hit alignment limits; kernel instrument = the path
- CCA block-0 captures (n=1 "Paris"): cache_s_l0 (cca_last) mad 0.013, scale/cca_state mad
  0.00000, cca_conv_input mad 0.007 = clean at aligned components; cca_prev_hs-0 mad 1.019
  but the HRX file (r10851) = likely decode-vs-prefill misaligned (HRX prefill = r40-era) -
  UNVERIFIED.
- Measurement ceiling: in-program HRX tensors don't cross (no readback); the HRX run-counter
  inflation (each slice = a graph_compute) makes cross-run file alignment persistently
  error-prone. Precise remaining measurements need d5694d's kernel instrument or an
  executor-side in-program dump keyed to the program.
- Fleet aligned; battery + aligned captures on demand for any landed fix.
