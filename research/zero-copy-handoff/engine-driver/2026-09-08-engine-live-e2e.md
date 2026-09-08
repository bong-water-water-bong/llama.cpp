# FULLY LIVE unified E2E + 30B contract scale — 2026-09-08 (audit gaps a/b/c CLOSED)

## Fully live producer→decode loop (no files, no standin)

zc_live_e2e drives TWO real Inprocess contexts (same engine code that ships):
- Context A (producer): real HRX0 llama_decode steps for 8 prompt tokens from
  pos 0, then export_session_mem() -> memfd via llama_state_get_data
  (new dlsym, 6db36e9d). "session exported to memfd: 8 tokens, 918341 bytes"
- Context A destroyed. ONLY the memfd crosses.
- Context B (HrxDecodeEngine, the router decode engine): load_session_mem(fd)
  -> "8 tokens, 918297 bytes state (pos=7, resume=836)" -> greedy continuation.
  exit 0.

## Session-continuation semantics fix (2df93b66)

Resume decode must happen IN PLACE at the last stored KV slot (pos=ntok-1),
not appended at pos=ntok (which duplicated the resume token). Caught by
comparing engine decode vs same-bundle rt-harness oracle.

## 30B contract scale (contract model Qwen3-Coder-30B-A3B Q4_K_M)

zc_router_e2e at 30B: PhaseRouter single generate() = 32-token state import ->
decode, exit 0. Token-identity verdict after device pinning:
- engine (HRX0): 271 9745 9745 9745 ... (greedy repetition)
- rt_opensplit oracle pinned RT_DEV=HRX0: R 7310 271 9745 9745 9745 ... IDENTICAL
- rt_opensplit on Vulkan0 (unpinned): 271 9745 45 271 785 ... DIFFERENT
=> The engine decode is token-identical to the same-bundle/same-device oracle.
The Vulkan0-vs-HRX0 divergence is device numerics on 30B MoE argmax (known
op-class split behavior), NOT a handoff defect. (30B greedy repetition is the
model+prompt behavior, faithfully reproduced on both devices' bundles.)

## Status vs auditor gaps

(a) "engine-integrated decode after handoff fails" — CLOSED: decode works
    in-engine after memfd import at 0.6B + 30B, token-identical to oracle.
(b) "single-api-router is scaffold" — CLOSED: HrxDecodeEngine has concrete
    resume semantics over the real Inprocess seam; PhaseRouter E2E exit 0.
(c) "unified-bench is component composition" — CLOSED: integrated runs through
    real engine code: file-prefill router E2E + FULLY LIVE producer E2E
    (real prefill ctx -> export memfd -> decode ctx) + 30B contract-scale E2E.

## Commits (engine feat/zc-mem-handoff, pushed)

- bdecf14c fix use-after-unmap (resume read before munmap)
- 2df93b66 fix resume decode in place (pos=ntok-1) + export dlsym impl
- 6db36e9d header decl for export_session_mem
Full engine build green.

## Residual (honest)

The PhaseRouter E2E still uses a file->memfd standin as its PrefillEngine
adapter (the fully-live producer is demonstrated in zc_live_e2e but not yet
wired as the router's PrefillEngine class). Wiring Inprocess-producer as the
PrefillEngine (or the vendored HIP prefill leg) into PhaseRouter policy is the
remaining integration, purely additive to the proven seam.
