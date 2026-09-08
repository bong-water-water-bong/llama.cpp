# 30B MoE decode regression — root cause (2026-09-08)

Goal mtsjhonv (unified two-engine stack), decode-rebuild task. Converges with
the launch-collapse profile (research/launch-collapse/2026-09-08-task2-profile.md):
same per-token dispatch overhead, more severe on MoE.

## Measured (strixhalo, llama-bench tg256, Qwen3-Coder-30B-A3B Q4_K_M)

| build | 30B decode | 0.6B dense decode |
|---|---|---|
| old llama-build bundle (ggml-hrx 0.9.11) | 40.0 tok/s | 174 tok/s |
| modern fork (15ff48549 / ab6103666) | 11.96 tok/s | 225 tok/s |

Dense decode is FINE on the modern fork (225 > 174). The 3.4x regression is
MoE-specific: llama.cpp fragments each 30B decode token into ~98 executor
graph_compute calls (13-31 node fragments, ~2/layer x 48, 5-10 kernels each)
vs the dense 975-node whole-layer programs (2 calls/token).

## Why the MoE graph fragments

The fused MoE decode matchers in dispatch-routed-ffn.cpp bail on the live
graph (GGML_HRX_DECODE_DBG instrumentation, dispatch-routed-ffn.cpp):

- match_decode_routed_ffn_gate_up_swiglu requires routed input shape
  [768(expert_hidden),1,1,1] but the live decode input is [768,8,1,1] — one
  token routed to top-8 experts, batched in ne[1]. The generic kernels that
  then run (ggml_mul_mat_id_swiglu_f32_f32_wmma, input_route_count=1) handle
  it per-expert.
- The gate/up MUL_MAT_ID root arrives with the DOWN-weight layout
  [768,2048,128,1] (is_routed_ffn_gate_up_weight wants [2048,768,128,1]);
  output shape gate also fails (live [2048,8] vs expected [768,8]).

Net: per-expert generic mul_mat_id kernels run instead of the fused
qwen3_moe_routed_gate_up_swiglu_q4k_q8 / routed_down_q4k decode kernels from
the tg8 plan (benchmarks/loom/qwen3_30b_a3b_q4_k_m.tg8.json — 14 fused
dispatches, zero standalone ADDs). Graph-layout drift: the matchers were
written against the Aug-28 llama.cpp MoE graph; the current llama.cpp emits a
different layout.

## Secondary: long-context decode-split FA unwired

decode-split FA (flash_attention_decode_split_*_next_q8) only wired for
KV<=2048 (dispatch-flash-attention.cpp is_supported_decode_key_value_token_count
= 2048). At the 30B benchmark KV (2962+) decode falls back to the prefill FA
kernel per token. The loom corpus defines the long-context path
(produce_partials + reduce_f32 exports, 32768-capable, reduce via execution
barrier) but C++ only emits the short fused kernel; loom reduce_fused template
has bands [64,256] and [257,2048] only — raising the C++ cap makes KV>2048
decode fail at loom JIT link. Needs the missing >2048 band or the two-dispatch
produce+reduce wiring.

## Fix direction (per owner: complete new build, each model to the engine)

1. Adapt the fused MoE decode matchers to the live llama.cpp layout (routed
   input [768,8], down-layout weights) so the tg8 fused decode plan engages —
   or align llama.cpp graph to the kernels. Collapses ~98 calls/token toward
   ~1-2/layer.
2. Wire long-context decode-split FA (KV>2048) via the corpus produce+reduce
   exports or a >2048 reduce_fused band.
3. Target: modern fork tg256 >= 40 tok/s at KV 2.9-3.2k, fused chains
   confirmed in program dumps.

Coordinate with launch-collapse (same dispatch-fusion fix class, 0.6B side).
Instrumentation: GGML_HRX_DECODE_DBG bail logging on dispatch-routed-ffn.cpp
(local, uncommitted in ~/wt/q35-hrx-fix feat/decode-fusion-30b).
