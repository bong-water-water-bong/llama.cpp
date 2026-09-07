
# ADD-claim A/B verified (agent-f49062, 2026-09-07 ~19:00)

## Measurement (qwen3-0.6B tg128, r=5, same build + GGML_HRX_ALLOW_ADD env gate)
- GGML_HRX_ALLOW_ADD=1 (standalone ADD claims re-enabled): 242.91 +/- 0.76 t/s
- default (e130977af exclusion):                          48.89 +/- 0.44 t/s
=> The blanket ADD exclusion IS the ~5x dense-qwen3 roster regression (earlier
   39-t/s reading was a concurrent-run artifact).
- zaya with ALLOW_ADD=1: tok0=563 (corruption returns) + 212.8 ms/tok =>
  the exclusion is required for zaya correctness.

## Mechanism note
The roster's residual ADDs merge the per-layer subgraphs (fewer syncs) and run
on-device; their inputs are HRX-produced, so they were correct AND fast when
claimed. The zaya's ADDs sit in a graph with CPU conv/state islands and read
values that cross the CPU boundary - claimed standalone they corrupt (the
deterministic 4-of-6-rows pattern suggests a view/staging read issue, not a
timing race). With ADDs on CPU the zaya is oracle-exact.

## Next fix (planned)
Conditional ADD reclaim: allow the standalone ADD claim only when its
producing chain is HRX-producible (transitive src walk over the eager-claim
set, memoized per graph); zaya residual ADDs trace back to CPU roots
(huge-vocab GET_ROWS embd, ssm_conv) and stay CPU; roster ADDs trace to
HRX mms/norms and get claimed -> both correctness and speed. Falls back to
the current exclusion if the walk is inconclusive. This restores task-2/3
roster numbers at branch HEAD without re-breaking task-4 zaya.

## Branch state
HEAD 9ed228db2 + working-tree: GGML_HRX_ALLOW_ADD A/B gate (ggml-hrx.cpp).
Zaya: oracle-exact on-device at 5.5 t/s. Roster: 48.9 t/s with the exclusion
(default), 242.9 with ADDs claimed.
