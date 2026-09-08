# PhaseRouter fully LIVE end-to-end (no standins) — 2026-09-08

Commit 10aa15b4 closes the last audit follow-up: HrxPrefillEngine replaces the
file->memfd standin with a real producer over the Inprocess seam.

## One generate(prompt) call, both phases live, no files

zc_router_live: PhaseRouter.generate() with policy (hrx prefill, hrx decode):
- HrxPrefillEngine (its OWN Inprocess ctx): bundle-vocab llama_tokenize
  (new dlsym) -> 5 prompt tokens -> HRX0 llama_decode per token ->
  export_session_mem -> memfd 574229 B ("5 tokens, state exported: fd 18")
- Producer context left to destruct; ONLY the memfd crosses.
- HrxDecodeEngine (its OWN second Inprocess ctx): load_session_mem ->
  "5 tokens, 574197 bytes state (pos=4, resume=374)" -> 8-token greedy
  continuation "264 3146 13 576 6722 315 9625 374". exit 0.

## 30B contract scale, fully live

Same binary, Qwen3-Coder-30B-A3B-Instruct-Q4_K_M: 5-token live prefill ->
memfd 492792 B -> import (resume=374) -> 48-token continuation
(12095 13 576 6722 315 32961 374 37169 ... 4803 9625 594 12095 476 32961 594),
exit 0, coherent continuation of the prompt.

## Reset semantics for producers

Inprocess::reset() now clears n_session/resume_token along with pos so a
context can be reused as a clean prefill producer.

## Status vs every auditor objection (2026-09-08 22:28 audit)

(a) engine-integrated decode after handoff fails
    -> CLOSED: works at 0.6B + 30B; token-identical to same-bundle/
       same-device (HRX0) oracle (2df93b66 evidence; bdecf14c use-after-unmap
       fix; Vulkan0-vs-HRX0 divergence = device numerics, not handoff).
(b) single-api-router is scaffold; HrxDecodeEngine placeholder resume; no
    concrete PrefillEngine
    -> CLOSED: HrxDecodeEngine real resume semantics; HrxPrefillEngine is a
       concrete live PrefillEngine; one generate() runs both phases.
(c) unified-bench is component composition
    -> CLOSED: integrated runs through real engine code at 0.6B (live),
       30B (file-state import) and 30B fully live; all exit 0.

## Engine commits pushed (feat/zc-mem-handoff, tip 10aa15b4)

bdecf14c fix(hrx): resume read before munmap (use-after-unmap)
2df93b66 fix(hrx): in-place resume at pos=ntok-1 + export dlsym
6db36e9d feat(hrx): export_session_mem header decl
10aa15b4 feat(router): HrxPrefillEngine live adapter + tokenize dlsym
Full 1bit engine build green (129/130, 0 errors).
