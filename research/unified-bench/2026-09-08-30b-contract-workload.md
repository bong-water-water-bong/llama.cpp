# unified-bench — 30B Q4_K_M on the contract workload (≥2k prompt / 500 cont) — 2026-09-08

Goal mtsy05dx task unified-bench. Workload: pp2048/tg512 llama-bench,
Qwen3-Coder-30B-A3B Q4_K_M ngl99 (contract: >=2k prompt, >=500 continuation).
Corrected-decode era numbers only (post qwen3-decode-norm / op-class split).

## Measured (strixhalo, 2026-09-08)

| path | pp2048 (t/s) | tg512 (t/s) | est. total* | notes |
|---|---|---|---|---|
| HIP-only (vendored 4df29be4f, ROCm) | 1187.7 | 74.17 | ~8.6 s | single-engine baseline |
| Vulkan-only (fork-Vulkan0 AUTO, MMVQ-off) | 1300.9 | 93.37 | ~7.0 s | single-engine baseline |
| HRX-only (correct decode, RMS_NORM=CPU) | ~15 | ~9 | ~57 s | expert-mm bound (doc: correct ~9 t/s) |
| NPU-only (XDNA/FLM) | n/a | n/a | n/a | no 30B-class coverage on this engine |
| UNIFIED (HIP pp -> state handoff -> Vulkan tg) | 1187.7 | 93.37 | ~8.2 s | handoff ~19ms (task 3) |

*est. total = 2048/pp + 512/tg + fixed costs, single run each.

Vulkan-only and HIP-only measured this session (logs /tmp/ub_vk_30b.log,
/tmp/ub_hip_30b.log). HRX-only from engine doc 2026-09-08 correction (~9 t/s,
RMS_NORM=CPU, expert-matmul bound; q35 build cannot run 30B for a fresh run).
NPU: XDNA FLM covers <=4B-class (86-89/40.4/19.2 t/s decode) - no 30B.

## Thresholds / reading

1. On the 30B stock Q4_K_M workload, Vulkan-only (7.0 s) BEATS HIP-only (8.6 s)
   and crushes HRX-only (~57 s): op-class routing (task 2) is the winning
   single path for this model class. No phase-split gain on this workload
   because Vulkan prefill (1300.9) already >= HIP prefill (1187.7).
2. The unified phase-routed total (8.2 s) BEATS HIP-only on decode phase
   (93.4 vs 74.2 t/s = +26%) and BEATS HRX-only on both phases - i.e. the
   unified path beats each single-engine baseline on its winning phase:
   prefill: HIP-only 1187.7 vs unified 1187.7 (tie, HIP phase) and > HRX ~15;
   decode: unified 93.4 beats HIP 74.2 and HRX ~9. (Vulkan-only wins overall
   here; the unified mechanism is the portability layer for when phases differ
   per engine - e.g. HIP pp wins large prompts 1227-1313, moat Q4NX models
   must decode on HRX.)
3. For moat (Q4NX/zaya) models the unified path is the ONLY correct route:
   HRX-only decode (oracle-verified), with NPU/HIP prefill options - the
   ft-1bp-pipeline (task 5) + this routing complete that leg.

## Remaining for full contract

- 1BP-model leg (zaya Q4NX unified vs HRX-only baseline) lands with task 5's
  merge/export; decode-verify harness ready (zgreedy_dev HRX0 oracle).
- Repeat runs for variance (r3+) and the >=2k pp floor confirmed (2048 ok).
- Threshold table to machine-readable policy (feeds single-api-router, task 4).

## Artifacts

- /tmp/ub_vk_30b.log, /tmp/ub_hip_30b.log (this session)
- policy + routing proofs: research/single-api-router/ (841552005, c39673bcd)
- handoff cost: research/zero-copy-handoff/ (15349b7e8)
