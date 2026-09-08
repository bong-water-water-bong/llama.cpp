# mtsy05dx — audit-response status (2026-09-08, post-rejection)

The goal completion was audit-rejected because engine-level integration was
proxy-only. This documents the real code delivered since, per finding.

## Finding-by-finding

1. decode-gate-routed / auto-route-fork — accepted by audit (fork-level, real).

2. zero-copy-handoff — "no engine integration, memfd host-only":
   DELIVERED: engine Inprocess::load_session_mem(fd) + HRX_STATE_MEMFD env
   (0a54070c on 1bit-MONSTER feat/zc-mem-handoff, full engine build green);
   cross-process SCM_RIGHTS fd handoff proven P==C token-identical
   (203926057); the REAL engine Inprocess code executed: state imported from
   memfd (14 tokens, pos=14, no file) (c4644bbd5). The decode leg then hits
   the b66 bundle GET_ROWS unsupported-node defect (q6_K token_embd lookup) —
   a pre-existing bundle limitation (#1982/#2145), not the handoff.

3. single-api-router — "no router code, caller sees the split":
   DELIVERED: PhaseRouter (one generate(prompt) call = policy prefill ->
   shared-memory fd -> policy decode) + HrxDecodeEngine over the real
   Inprocess seam (3c60274f, engine build green, in-tree test).

4. ft-1bp-pipeline — "f32 stand-in, not 1BP":
   DELIVERED: genuine Q4NX (type-43) conversion — converter validated
   bit-exact vs the known-good c43 (d038a2524); FT zaya -> ft7_q4nx.gguf
   (7.49GB, 280 Q4NX + 1003 f32) loads + decodes on HRX0, CPU==HRX0
   (0fe318681). Pipeline repeatable.

5. unified-bench — "component composition, no integrated run":
   PARTIAL: full measured table + thresholds + idle-HRX caveat (9af349ba6,
   6b8109dad). The integrated engine run of the unified path needs the b66
   GET_ROWS decode fix (other lanes' issue) or the engine ABI migration to
   the GET_ROWS-capable fork bundle - both external to this lane.

## Remaining external blockers (not my code)

- b66 bundle GET_ROWS decode gap (#1982/#2145) - other lanes' territory
- Engine Inprocess ABI pin to the b59/b66 header (0.0.10320) vs the newer
  GET_ROWS-capable fork lineage - an engine migration decision
