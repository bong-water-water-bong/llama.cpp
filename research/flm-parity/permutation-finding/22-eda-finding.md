
# Zaya ADD-claim corruption = the recurrent router_eda ADD (agent-f49062)

## Precise localization (ALLOW_ADD=1 = standalone ADDs claimed, corruption on)
- router_down-0 (the block-0 router_h = the prev_router source for block 1's
  EDA): row mads 0.005-0.009 = CORRECT vs the CPU oracle.
- router_eda-1 (the EDA output = ADD(router_h_1, prev_router * eda_scale)):
  row mads 0.56-0.81 = WRONG.
- Block-0 routing table [3,15,10,12,3,0] = correct; block-1+ tables
  [7,6,2,9,15,15] = wrong => the corrupted EDA feeds the router logits =>
  wrong argsort => wrong experts => wrong decode (tok0=563).
- With the ADD exclusion (default): EDA-1 mads 0.009-0.014 = correct,
  tok0=9079.
- The residual_post_attn ADDs (also standalone-claimed in the ALLOW state) are
  CORRECT when claimed - the EDA ADD is the corrupt one (it consumes the
  cross-block-live prev_router value and/or its broadcast MUL(prev, scale)).

## Consequence for the ADD-claim policy
- The conditional claim (direct MUL_MAT src, f799b66f1) excludes the zaya's
  EDA ADD (MUL-wrapped input) - that is why the zaya stayed oracle-exact
  under it. It also restored the dense-qwen3 roster (227.8 t/s).
- The only blocker to landing the conditional claim is the qwen3moe MoE
  "alias target not transient" failure (fused chains claim the ADD and
  alias-relayout a subgraph external). Fix = dispatch-scheduler alias-failure
  fallback (drop the failing chain match and retry the next registration,
  e.g. the standalone binary) - c5ac20 lane coordination item.

## Current state
Branch fix/hrx-ngl-init-order HEAD eb004cc20: blanket ADD exclusion; zaya
oracle-exact 5.6 t/s; roster 48.9 (242.9 behind GGML_HRX_ALLOW_ADD).
