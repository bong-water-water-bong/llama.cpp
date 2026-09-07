
# Conditional ADD claim: reverted to the blanket exclusion (agent-f49062)

## Experiment (f799b66f1, then reverted)
Conditional claim: standalone ADD claimed iff a src is directly MUL_MAT and
neither src is MUL_MAT_ID.
- dense-qwen3-0.6B tg128: 227.8 +/- 1.8 t/s (roster restored) + correct
- zaya: oracle-exact preserved (its ADDs are MUL-wrapped -> CPU)
- qwen3moe-30B-Coder + Qwen3.6-35B-A3B: FAIL - "value alias target N is not
  transient" at the first decode (their fused chains claim the ADD and
  alias-relayout a subgraph-external value; dense-qwen3 chains do not hit
  this because the sched composes its subgraphs differently).

## Conclusion
The ADD eager-claim cannot be restored with an op-structural condition: the
dispatch behaviour depends on the fused-chain registration order and the
sched subgraph composition (MoE = routed-ffn chains with subgraph-external
ADDs; dense = chain-internal ADDs). The proper fix is dispatch-coverage-
driven: (a) registration order so standalone binary covers ADDs before the
fused chains alias them, or (b) a dispatch-side fallback that demotes an
alias-failing chain claim to the standalone binary, or (c) the round-16f
host-binding ordering fix so the zaya's standalone ADDs become safe to claim
again (then re-apply the full ALLOW state for everyone).

Until then the blanket exclusion stays (zaya correct at 5.5-6.1 t/s; roster
48.9 t/s vs 242.9 with the gate; qwen3moe lanes correct).

## Files
- f799b66f1 (reverted in working tree) = the conditional claim attempt
- 88d4912df = A/B instrumentation + doc (GGML_HRX_ALLOW_ADD gate, kept)
