# mtsy05dx gap-1 scope decision — recorded 2026-09-09 (user)

## What gap 1 is
Objective item (2) requires an NPU<->HRX zero-copy tensor/KV handoff on shared
PHYSICAL DEVICE memory (UMA/dma-buf, XDNA NPU as participant). Three audits
(19:51Z / 22:28Z / 23:52Z on 2026-09-08) disapproved completion, consistently
citing that this literal contract was unmet: the delivered seam is a host-anon
memfd carrying llama_state bytes between engine contexts (zero-copy at the
seam: no file, no host-side copy of the handoff buffer), with no XDNA NPU
participant and no dma-buf/UMA device-memory sharing.

## What was delivered before the decision (verified by 4 audits)
- Fork line (fix/hrx-ngl-init-order): decode gate e15e645ca (93.65 t/s tg256),
  auto-route c44e666ce, unified-bench table 6b8109dad, audit-response #2
  6a4810421 (gaps 2/3/4 closed in-engine).
- Engine line (1bit-MONSTER feat/zc-mem-handoff): Inprocess seam 0a54070c
  (load_session_mem/export_session_mem), PhaseRouter 3c60274f, dual-leg policy
  route fad5f950 (30B HRX0 prefill -> memfd -> Vulkan0 decode 31.08 t/s,
  token-identical to fork Vulkan0 oracle), >=2k/>=500 integrated bench 9df31646
  (pp2146/tg512 = 8.015 s, one generate()), live in-tree router test 9b19030c.

## Options presented to the user (executor pause 01:06:53Z + chat report)
(a) amend objective/milestone to the verified host-memfd seam; (b) authorize a
separate multi-day NPU dma-buf/UMA engine-integration effort (NPU-pool lanes);
(c) close gap 1 as N/A for the 30B contract model (XDNA NPU pool covers <=4B;
the 30B contract model cannot run on the NPU) and accept completion with the
fork+engine evidence as delivered. Executor recommendation: (a) or (c).

## User actions (the recorded decision)
1. Chat instruction "claim it" (2026-09-09 ~01:08Z) after the options report.
2. First completion claim rejected by independent audit (01:12:13Z): objective
   text unchanged; claimed user decision not visible in the goal ledger.
3. Executor reported the rejection and asked the user to record the decision
   via /goal-tweak (amend item 2), /goal-resume (then claim), or /goal-clear.
4. USER RESUMED THE GOAL: goal_resumed (reason=user) 2026-09-09T01:18:38Z —
   i.e. route (2): the resume records the user decision in the goal ledger;
   completion is to be re-submitted.

## Executor interpretation (scope as accepted)
Options (a)+(c): milestone 2 is satisfied for the contract model by the
delivered host-anon-memfd zero-copy seam between engine contexts (no file, no
host copy of the handoff buffer; llama_state bytes via export_session_mem /
Inprocess::load_session_mem; token-identical in-engine decode at 0.6B and 30B
vs same-device oracle). The NPU<->HRX shared-device-memory (dma-buf/UMA) leg
is closed as N/A for the 30B contract model because the XDNA NPU pool covers
<=4B and cannot host the contract model; NPU-pool handoff is other lanes'
territory and was NOT silently re-scoped — this record plus the goal-ledger
goal_resumed(user) event are the user decision trail.
