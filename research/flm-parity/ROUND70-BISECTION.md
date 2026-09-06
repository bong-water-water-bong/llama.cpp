# Round 70 executor note (428ab3) — bisection vs the full CPU oracle set: divergence = gate_up mm t1-5 (round-67 re-confirmed); lm-head routing answered

## Bisection (post-strided-fix captures vs ~/zaya-decode/ffn_oracle/ + out_oracle/)
The full per-block CPU oracle set exists (r03_* = ffn internals incl. r03_004
MUL_MAT_ID ffn_moe_gate_up-0, r03_006 ffn_moe_down-0, r03_007 weighted; r04_* = layer
outputs). Comparison of the gate_up mm slot capture (fix-era) vs r03_004:
  tok0 gate mad 0.0051 / up 0.0052 = f16-correct
  tok1-5 gate mad 0.85-1.11 / up 0.88-1.01 = GENUINELY WRONG
  (slot row0 t1-5 = 0.3435/0.1693/-1.7227/-0.8857/-0.5151 vs oracle 0.2654/1.1045/
  -1.0146/0.1876/0.5645 = the round-41 oracle values confirmed)
moe_out (10870) vs r04_006: t0 mad 0.0011 correct, t1-5 mad 0.08-0.23 = downstream
propagation of the mm1 wrongness. => The round-68 "chain correct through residual /
result_output mm alone" = TOKEN-0-ONLY evidence (their weighted-0/layer_out captures =
tok0 ch0-3; token 0 = the only correct token). The divergence = the gate_up mm
per-partition compute for t1-5 (round-63 closure + round-67 %wide RE-CONFIRMED with the
real oracle). The round-64 "mm output correct mad 0.003 all 6" = definitively false
(gather-era corrupted correlation).

## result_output (lm-head) routing — why no hang on the current tree
dispatch-mul-mat-common.h:53 caps the dense mul_mat claim at output_size <= 262144.
Zaya's lm-head rows = 262272 > 262144 = NOT claimed = CPU (observed: zero result_output
HRX bindings, no >500MB program beyond the exps weights; run rc=0, no hang). Qwen
151936 < 262144 = claimed. The fleet's hung canary = likely an earlier build/config where
the claim differed. If/when the lm-head should go to HRX (speed), the 262272 = needs a
2-part row split above the 262144 cap (zaya-specific) - but that is downstream of the
mm1 t1-5 bug and premature until mm1 = fixed.

## Files
/tmp/fix_gu.bin (post-fix gate_up slot), oracle: ~/zaya-decode/ffn_oracle/r03_004*.bin,
~/zaya-captures-428ab3/. Fix target: mm1 (gate_up exps MUL_MAT_ID) per-partition compute
for tokens 1-5 (d5694d wmma lane; oracle + captures ready).
