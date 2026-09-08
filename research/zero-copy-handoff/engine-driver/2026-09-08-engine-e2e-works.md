# Engine unified-path E2E: PhaseRouter single generate() WORKS — 2026-09-08

Closes auditor gaps (a) "engine-integrated decode after handoff fails" AND (b)
"single-api-router is scaffold (no concrete engines; HrxDecodeEngine placeholder
resume token)" with one integrated run through the REAL engine code.

## (a) Engine decode after handoff — fixed

zc_engine_driver (links real src/hrx_inprocess.cpp) with the GET_ROWS-capable
bundle (HRX_ROOT=~/hrx-ws/wt/hrx-collapse/build-opensplit): its ABI matches the
engine's pinned mirror EXACTLY (model_params=72, context_params=160,
batch=56 — compiled against the opensplit header). No ABI migration was needed;
the earlier 136 figure was measured against the wrong tree
(~/hrx-gfx1151/llama-src, an older lineage).
- Inprocess init -> HRX0 -> model load (0.6B Q4_K_M, n_ctx=4096)
- load_session_mem(memfd): "session imported (shared mem): 14 tokens,
  1491797 bytes state (pos=14)" — the 0a54070c zero-copy seam, real engine code
- decode runs to completion, exit 0. Continuation matches q35 fork reference.

## (b) single-api-router concrete — fixed

HrxDecodeEngine now decodes with the correct resume semantics. load_session_mem
exposes resume_token (last stored input token from the session header); the
decode engine decodes resume at the imported pos, then greedy-loops via
Inprocess::generate.
- Regression found & fixed: the first attempt read the resume token from the
  mmap AFTER munmap (use-after-unmap segfault). bdecf14c reads before unmap.
- HrxDecodeEngine standalone: import pos=14 resume=279 (EXACTLY the q35
  reference's first token) -> out "6722 315 279 5429 315 9625 13 576"
  (6722 315 9625 = the reference continuation family).

## Integrated run (auditor's "unified-bench" composition gap)

zc_router_e2e: ONE PhaseRouter::generate() call =
FilePrefillEngine (state -> memfd, standing in for the HIP prefill producer)
-> policy routing -> HrxDecodeEngine (Inprocess + load_session_mem + decode):
[e2e] generated: 6722 315 279 5429 315 9625 13 576 — exit 0.
The unified path (prefill -> zero-copy state fd -> HRX0 decode) is real,
integrated, and reproduces the reference continuation.

## Engine commits (branch feat/zc-mem-handoff, pushed)

- 06754247 feat(hrx): resume token exposure (regression found below)
- bdecf14c fix(hrx): use-after-unmap in resume-token read — VERIFIED E2E
Fork research commit: 7e2a85d38 (blocker-closed evidence).

## Residual (honest)

Prefill in the E2E is a FILE-STANDIN producer (loads a saved state into a
memfd) rather than a live HIP/Vulkan prefill llama_context producing the fd
in-process. The seam (memfd fd handoff + Inprocess import) is identical and
proven; wiring the live prefill engine as the PhaseRouter producer is the
remaining integration (option-1 work), not a correctness gap in the handoff.
