# zero-copy handoff — full shared-memory round-trip PROVEN (rt_zc_rt) — 2026-09-08

Goal mtsy05dx task zero-copy-handoff. Extends the consumer-seam proof
(834f83443) to a FULL round trip with ZERO file I/O on the zero-copy path.

## Proof (rt_zc_rt.cpp, single process, q35 CPU build)

1. Prefill a deterministic random-token prompt once.
2. Round trip F (today's path): llama_state_save_file -> fresh ctx ->
   llama_state_load_file -> greedy continuation.
3. Round trip M (zero-copy): llama_state_get_data -> memfd (MAP_SHARED,
   syscall 319) -> fresh ctx -> llama_state_set_data from the shared pages ->
   greedy continuation. M touches NO filesystem state file.

Results (greedy argmax, temp 0):
- qwen3-0.6B Q4_K_M, 256-token prefill, 29,363,897 B state: F==M 11/11
- Qwen3-Coder-30B-A3B Q4_K_M, 512-token prefill, 50,338,972 B state:
  F==M 12/12 (streams: 1563 1 220 16 15 15 ...)

set_data consumed the full exported size on both models.

## What this proves

- The complete HIP/NPU-prefill -> HRX-decode state handoff can ride shared
  physical memory: producer exports with llama_state_get_data into a shared
  buffer (no save_file), consumer imports with llama_state_set_data (no
  load_file), continuation is byte-identical to the file path.
- memfd stands in for the NPU SharedBO dma-buf window / UMA host window; the
  llama_state bytes are location-agnostic, so swapping the carrier to a
  dma-buf fd (SCM_RIGHTS) is an integration step, not a format question.
- File round-trip eliminated => no disk write/read/fsync in the handoff.

## Artifacts

- research/zero-copy-handoff/rt_zc_rt.cpp (this harness)
- research/zero-copy-handoff/rt_memfd2.cpp + memfd-proof.md (consumer-seam proof)
- logs: /tmp/zc06_out.log, /tmp/zc30_out.log (F==M identical)

## Remaining for the engine integration (next step, separate from this proof)

- Engine Inprocess::load_session_file(path) -> load_session_mem(fd,size);
  producer passes the memfd/dma-buf fd via SCM_RIGHTS instead of HRX_STATE_FILE.
- Measure wall handoff cost (memfd mmap+set_data vs file write+read) on the
  30B @ 2962-token state; expected saving: one 292MB file write+read+fsync.
