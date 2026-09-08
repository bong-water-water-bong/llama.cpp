# zero-copy handoff — shared-memory state import PROVEN (rt_memfd2) — 2026-09-08

Goal mtsy05dx task zero-copy-handoff. Question: can the llama_state KV blob
cross the HIP/NPU-prefill -> HRX-decode boundary through SHARED MEMORY (no
file, no host copy) and produce an identical continuation?

## Proof

Harness: rt_memfd2.cpp. Loads a real 30B HIP-prefill session blob
(/tmp/m2/short_blob.bin, 126,539,036 B, session v9, 1288 stored tokens,
Qwen3-Coder-30B-A3B Q4_K_M) into a memfd-backed MAP_SHARED region (the
stand-in for an NPU SharedBO dma-buf window), then imports the state TWO ways
into identical fresh contexts (q35-hrx-fix build, CPU):

- F: llama_state_load_file(ctx, path)  — today's D2 file round-trip
- M: llama_state_set_data(ctx, state_ptr, state_sz) — in-memory import from
  the shared pages (state region = blob minus 12B header minus token array)

Continuation (greedy argmax, temp 0) after the 1288 stored tokens:

F: 16 26095 320 16 26095 8 11397 220 16 26095 320 16 26095 8 11397 220 16 26095 320 16
M: identical — TOKEN-IDENTICAL 20/20
(set_data consumed the full 126,533,872-byte state region)

## What this proves

1. llama_state_set_data (in-memory) is a drop-in for llama_state_load_file:
   the 126MB state imports from shared pages with zero file I/O and yields a
   byte-identical decode. The D2 HRX_STATE_FILE blob path is removable.
2. The transfer vehicle can be any shared physical memory (memfd here; NPU
   SharedBO dma-buf / UMA host window next) — the llama_state bytes are
   layout-identical, so the KV never needs a file or a host-side copy.
3. This is the consumer-side seam: engine Inprocess::load_session_file(path)
   becomes load_session_mem(ptr,size) with the fd arriving via SCM_RIGHTS /
   dma-buf import instead of a filesystem path.

## Artifacts

- /tmp/rt_memfd2.cpp (harness; F/M dual import + comparison)
- /tmp/memfd_out3.log (20/20 identical token streams)
- blob: /tmp/m2/short_blob.bin (30B, session v9, 1288 tok)

## Next

Producer side: export prefill state via llama_state_get_data directly into the
shared window (skip save_file), pass the fd, measure handoff cost vs the file
path (expect: file write+read+fsync eliminated; import cost ~ mmap + set_data).
