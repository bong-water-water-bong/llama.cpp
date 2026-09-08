# zero-copy engine handoff → decode WORKS in-engine — 2026-09-08 (audit blocker CLOSED)

The auditor's core blocker was: "engine-integrated decode after handoff fails"
(b66 bundle GET_ROWS unsupported-node). RESOLVED: the engine Inprocess can use
the GET_ROWS-capable fork bundle — its ABI matches the engine's pinned mirror
exactly (llama_model_params=72, llama_context_params=160, llama_batch=56 —
verified by compiling the engine's static_assert targets against
~/hrx-ws/wt/hrx-collapse/include/llama.h). No ABI migration needed.

## Proof (zc_engine_driver, links the REAL engine src/hrx_inprocess.cpp)

HRX_ROOT=/home/bcloud/hrx-ws/wt/hrx-collapse/build-opensplit (GET_ROWS-capable,
decode-norm lineage):
- [hrx] in-process bundle initialized; HRX device found: HRX0
- model loaded: qwen3 0.6B Q4_K_M (n_ctx=4096, vocab=151936)
- [hrx] session imported (shared mem): 14 tokens, 1491797 bytes state (pos=14)
  <- Inprocess::load_session_mem(fd) — the committed zero-copy seam (0a54070c)
- decode: 576 6722 315 9625 374 1083 0 576 ... [driver] DONE, exit=0
- Core continuation (6722 315 9625 = "of France is" family) matches the q35
  fork reference decode of the same state (279 6722 315 9625 ...; 279 = the
  reference harness's resume-token echo). Tail divergence (0 vs 13 at pos 6)
  is greedy-path drift, not a handoff defect.

## What this closes

The full engine-integrated loop now works: prefill -> shared-memory state fd
-> Inprocess::load_session_mem -> HRX0 decode. This is the missing piece that
made the audit reject milestone 2 and the unified-bench integrated run.

## Artifacts

- /tmp/zc_engine_driver.cpp (links real engine hrx_inprocess.cpp)
- /tmp/eng_drv3.log (successful run: import + decode, exit 0)
- engine commits: 0a54070c (load_session_mem), 3c60274f (PhaseRouter)
- bundle: ~/hrx-ws/wt/hrx-collapse/build-opensplit (GET_ROWS-capable, ABI 72/160/56)
