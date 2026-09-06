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
## Silu-consistency appendix (round 70, for b30173's cross-check of m_mtq1iqj7)
Cascade = monotonic, consistent with the gate_up mm = the FIRST divergence:

per-token mad (HRX post-fix captures vs ~/zaya-decode/ffn_oracle/):
  token  | gate_up (r03_004) | swiglu (r03_005) | moe_out (r04_006)
  t0     | 0.0051            | 0.0027            | 0.0011
  t1     | 1.0105            | 0.4463            | ~0.15
  t2     | 0.8715            | 0.3178            | ~0.10
  t3     | 1.0575            | 0.4013            | ~0.23
  t4     | 0.9161            | 0.3615            | ~0.08
  t5     | 0.8963            | 0.3277            | ~0.11
The silu squashes the gate/up error ~2.4-2.8x (1.0 -> 0.4), the down mm + expert
weight attenuate further (0.4 -> 0.15): exactly the shape of a single upstream
error source at the gate_up compute.

Row-0 heads (HRX | oracle):
  gate t1 0.3435|0.2654   t2 0.1693|1.1045   t3 -1.7227|-1.0146   t4 -0.8857|0.1876  t5 -0.5151|0.5645
  up   t1 0.2013|-0.4716  (full rows in the files)
  swiglu t2 -0.1616|0.6216  t3 0.5129|-0.1126  t4 0.3215|0.1329  t5 0.0728|-0.2400

Expert table (block-0, gate+up share the bundle): e2->t4, e3->t0, e4->t5, e5->t1,
e7->t3, e12->t2; partition descriptors = experts at ordinals 0-5, local_part 0,
rows 1 (route_count=1). Captures: /tmp/fix_gu.bin, /tmp/fix_sw.bin, /tmp/prg_dump/
10866_001_common.moe_routing.{expert,partition}_table.bin; copies in
~/zaya-captures-428ab3/. Oracle: ~/zaya-decode/ffn_oracle/r03_00{4,5,7}_*.
