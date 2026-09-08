# zero-copy engine seam — REAL Inprocess::load_session_mem verified in-engine — 2026-09-08

Audit follow-up: the zero-copy handoff was proven via standalone harnesses; this
runs the ACTUAL engine code path (hrx::Inprocess from 1bit-MONSTER, committed
0a54070c) linked directly.

## What was executed (zc_engine_driver, links the real src/hrx_inprocess.cpp)

1. Prefill 0.6B Q4_K_M (q35 fork) -> /tmp/eng_state.bin (1,491,865 B, 14 tok)
2. Driver memfd-loads the state bytes
3. hrx::Inprocess::init() + load_model() -> device HRX0 (the /opt/hrx bundle)
4. **Inprocess::load_session_mem(fd)** (my committed seam) ->
   "[hrx] session imported (shared mem): 14 tokens, 1491797 bytes state (pos=14)"
   -> returned 14 (SUCCESS)
5. hrx.generate() decode -> graph_compute error -1

## Result

- The zero-copy engine seam WORKS in the real engine code: the shared-memory
  state fd is mmap'd and imported via llama_state_set_data with the correct
  token count and position — no file, no copy. This is the audit gap closed.
- The decode step fails with graph_compute -1 on the /opt/hrx bundle
  (libllama 0.0.10224 = b66). This is the DOCUMENTED pre-existing bundle
  limitation: "b66 release bundle and the amd-hrx-graph fork both FAIL
  llama_decode at token 2 on HRX0 after llama_state_load_file" (issue #2145,
  D2 lane). The same state decodes fine on the q35-hrx-fix fork build (the
  GET_ROWS-capable lineage). NOT a defect in load_session_mem.

## Artifacts

- /tmp/zc_engine_driver.cpp (links real engine hrx_inprocess.cpp + router)
- /tmp/eng_drv2.log (full trace), /tmp/eng_state.bin
- engine commits: 0a54070c (load_session_mem), 3c60274f (PhaseRouter)
