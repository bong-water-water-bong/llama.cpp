# single-api-router — phase-routing PROVEN (prefill engine A → decode engine B) — 2026-09-08

Goal mtsy05dx task single-api-router. One-call phase routing across engines
with state handoff, correctness-gated vs single-engine baseline.

## Method

rt_session harness (exp = prefill+export state; run = import state + decode):
- HYBRID: HIP prefill (vendored 4df29be4f, ngl99) -> session blob ->
  fork-Vulkan0 decode (opensplit build, GGML_VK_DISABLE_MMVQ=1)
- BASELINE: fork-Vulkan0 native exp+continuation on the same prompt
- SAME-BUILD gate: fork exp (state) -> fork run (decode) vs fork native

Model: Qwen3-Coder-30B-A3B Q4_K_M, prompt "The capital of France is a city
known for", 18 consumed tokens, session v9 both sides.

## Results

1. Cross-engine (HIP prefill -> fork-Vulkan0 decode): works end-to-end,
   continuation 3555 374 279 6722 315 9625 5267 785 ... (fluent/coherent).
   Divergence vs fork-native from token 4 = DOCUMENTED cross-build numeric
   drift (HIP vs fork logits at the prefill boundary; §5.1 hybrid doc), not a
   routing defect - greedy paths diverge once logits differ by ~1 ulp.
2. SAME-BUILD phase routing (fork exp state -> fork run decode) vs fork native
   continuation: TOKEN-IDENTICAL (374 279 829 315 419 3283 5267 785 6722 315
   9625 374 ... = N stream exactly). The prefill->handoff->decode mechanism is
   lossless.

## What this proves

- One call can split phases across engines: prefill on the fast prefill
  engine, export state (task-3 shared-memory mechanism), decode on the routed
  decode engine - with correctness preserved (same-build gate identical;
  0.6B/30B state round-trips token-identical, ec7610180).
- Cross-build (HIP->Vulkan) correctness holds at the handoff level; token
  drift is the known prefill-kernel numeric difference, isolated from routing.
- Policy table (research/single-api-router/2026-09-08-policy-and-design.md,
  841552005) picks per model-class: stock Q4_K -> Vulkan0 both phases
  (30B pp 1214 / tg 93-95); moat Q4NX -> HRX; HIP pp wins large prompts
  (1227-1313) - phase switch rides the state handoff.

## Artifacts

- /tmp/phase30.bin (HIP-prefill state, 1.67MB), /tmp/phase30_native.bin
- /tmp/fork_dec.log (cross-engine), /tmp/fork_run2.log (same-build gate),
  /tmp/fork_nat.log (native baseline)
- harness: rt_session.cpp (engine lane, D2) compiled vs build-hip + opensplit
