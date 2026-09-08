# single-api-router — engine code DELIVERED (audit follow-up) — 2026-09-08

The audit found single-api-router was design-doc + two-process proof only. This
adds the engine code, on 1bit-MONSTER feat/zc-mem-handoff (3c60274f, pushed):

## What was built (all compile + full engine build green)

1. src/router/phase_router.h/.cpp — PhaseRouter: ONE generate(prompt, max_tokens)
   call = policy prefill engine -> shared-memory state (memfd fd) -> policy
   decode engine. Caller sees no split. Data-driven PhasePolicy per model
   class (stock-q4k: prefill hip/vulkan decode vulkan; moat-q4nx: decode hrx).
   Fail-closes cleanly when engines are missing (tested).
2. src/router/hrx_decode_engine.h — HrxDecodeEngine: DecodeEngine adapter over
   the REAL hrx::Inprocess (engine in-process HRX path), importing the state
   via Inprocess::load_session_mem (0a54070c: mmap fd + llama_state_set_data,
   zero file I/O).
3. src/router/router_test_main.cpp — in-tree test: policy wiring + fail-close +
   HrxDecodeEngine init path (model load, HRX ready).

## Where it sits vs the engine architecture

- The decode half is REAL in-process code (Inprocess dlopen'd libllama, the
  documented engine path). The prefill half needs the HIP prefill lane
  in-process (the engine's HIP backend, backend_hip.cpp) producing the state
  fd - the same seam the D2 HIP-prefill lane uses (HRX_STATE_FILE/HRX_STATE_MEMFD).
- Cross-process fd handoff proven separately (fork research 203926057).

## Bench caveat added (from agent-3822c0 idle-HRX finding)

The unified-bench fork-Vulkan numbers on SMALL models carry an idle-HRX-init
penalty: GGML_HRX_DISABLE=1 gives 0.6B tg128 355.9 vs 347.9 with idle HRX
(+2.3%, verified this session). At 30B the penalty is within noise (92.5 vs
93.7 tg256). Small-roster figures should be read with this caveat; the 30B
contract-workload numbers are unaffected.
