# zero-copy handoff — CROSS-PROCESS SCM_RIGHTS fd passing PROVEN — 2026-09-08

Goal mtsy05dx zero-copy-handoff (audit follow-up). The earlier proof
(ec7610180) was single-process memfd. This adds the REAL cross-process
mechanism the engine integration needs: producer exports state to a memfd and
passes the fd over a unix socket (SCM_RIGHTS); a separate consumer process
imports from the received fd and decodes.

## Method (zc_fdpass.cpp, fork/exec, socketpair)

- producer (child): prefill 256 tokens, llama_state_get_data -> memfd
  (29,363,897 B), send_fd(SCM_RIGHTS) + resume token (raw state has no token
  bookkeeping), then decode its own reference stream "P".
- consumer (parent): recv fd, mmap MAP_SHARED, llama_state_set_data (full
  29,363,897 B consumed), decode from the received resume token -> "C".
- Model qwen3-0.6B Q4_K_M, q35 CPU build (hermetic), greedy argmax.

## Result

P: 164 36677 198 1447 2005 30 1106 1284 867 118
C: identical (TOKEN-IDENTICAL; consumer ran a few extra tokens - same stream)

State import consumed the full 29.4 MB from the received fd. Zero file I/O on
the handoff path.

## Engine integration status (audit follow-up, feat/zc-mem-handoff on engine)

- Inprocess::load_session_mem(fd): engine consumer seam - mmap the fd, strip
  the session-file header, llama_state_set_data on the raw state. Committed
  0a54070c on ~/1bit-MONSTER feat/zc-mem-handoff (auto-pushed), syntax-clean.
- HRX_STATE_MEMFD=<fd> env in backend_hrx.cpp alongside HRX_STATE_FILE
  (memfd wins; file fallback preserved).
- This proof supplies the cross-process fd-passing pattern the engine caller
  uses to deliver the fd (SCM_RIGHTS over its ipc socket).

## Artifacts

- /tmp/zc_fdpass.cpp (harness; both = fork+SCM_RIGHTS demo)
- /tmp/fdpass2.log (P==C token streams), /tmp/fdpass2_err.log
- engine commits: 0a54070c (load_session_mem + HRX_STATE_MEMFD)
