# single-api-router — phase-routing design + measured policy table (goal mtsy05dx task 4)

Contract: One inference API call routes prefill and decode across NPU/HRX/Vulkan
per model+phase policy; caller sees no split; router policy data-driven from
the unified bench; correctness gate (continuation matches single-engine
baselines); committed.

## Measured per-engine per-phase table (strixhalo, 2026-09-08)

All numbers are corrected-decode measurements (post qwen3-decode-norm /
op-class split), the only valid basis for policy.

| model | engine | prefill (pp) | decode (tg) | notes |
|---|---|---|---|---|
| qwen3-0.6B Q4_K_M | HRX (in-fork) | 12,518 | 231.4 | correct decode |
| | fork-Vulkan0 (AUTO) | 13,821 | 347.9 | auto-route, MMVQ-off |
| | stock Vulkan | — | 355.9 | ceiling ref |
| | FLM NPU v1.0.4 | — | 86-89 | XDNA, canary-correct |
| qwen3-1.7B Q4_K_M | HRX | 3,534 | 123.1 | |
| | fork-Vulkan0 (AUTO) | 5,777 | 166-168 | at stock bar |
| | FLM NPU | — | 40.4 | |
| qwen3-4B Q4_K_M | HRX | 1,258 | 58.4 | |
| | fork-Vulkan0 (AUTO) | 2,597 | 72-78 | EXCEEDS stock ceiling |
| | FLM NPU | — | 19.2 | |
| Qwen3-Coder-30B-A3B Q4_K_M | HIP (vendored 4df29be4f) | 660-1227 | ~70 | pp2962 4.5s |
| | HRX | ~15 | ~9 (correct, RMS_NORM=CPU) | expert-mm bound |
| | fork-Vulkan0 (AUTO) | 1,214-1,225 | 93.65-95.0 | tg256 @ KV 2944-3200 gate |
| zaya1-8b Q4NX (moat) | HRX0 | — | ~16-20 device | HRX-only (moat op) |

## Policy (data-driven)

1. Moat quant (Q4NX/type42/zaya): HRX only (auto-route already enforces).
2. Stock Q4_K roster (0.6/1.7/4B): fork-Vulkan0 for BOTH phases (prefill
   +10/+63/+106% over HRX; decode at stock ceiling class). NPU never wins on
   this class.
3. Stock Q4_K 30B-A3B: prefill HIP (~660-1227) OR fork-Vulkan0 (1214) - both
   win; decode fork-Vulkan0 (93-95) wins 13x over HRX-correct (~9) and 1.3x
   over HIP (~70). Policy: pp->HIP or Vulkan0, tg->Vulkan0.
4. Small-ctx warm decode (engine's in-process value case): HRX in-process
   (80-87 t/s small models) or Vulkan.
5. Cross-engine phase switch uses the task-3 shared-memory llama_state handoff
   (memfd/dma-buf, zero file I/O, token-identical - ec7610180).

## Where the router lives

- Fork level (already done, task 2 auto-route-fork): per-OP-CLASS routing via
  device_supports_op + has_standard_quant_weights. One llama call = auto-routed.
- Engine level (this task's delta): the DynamicRouter (per-token GPU<->NPU)
  must become PHASE-aware: prefill engine != decode engine per model class,
  with the state handoff between. Engine Backend interface has generate_text /
  generate / forward; the split needs a prefill(path)->state->decode(backend)
  pipeline = llama_state blob via shared memory (Inprocess::load_session_file
  becomes load_session_mem per task 3).
- Correctness gate: hybrid continuation == single-engine (Vulkan0) continuation
  on the same prompt (token-identical, greedy) - the rt_zc_rt harness pattern.

## Open integration points (engine repo, feat/hybrid-prefill-d2)

- backend_hrx.cpp: HRX_STATE_FILE env -> memfd fd (SCM_RIGHTS) 
- Inprocess::load_session_file -> llama_state_set_data(shared ptr)
- DynamicRouter: add phase field to pick_backend(); strategies PREFILL_ENGINE/
  DECODE_ENGINE tables from the bench json.
- unified bench (task 6) publishes the table above as machine-readable policy.

## Status

Design + measured policy committed here; engine integration is the remaining
code (engine repo, requires the D2 line + a session with the full engine
build). Fork-level half (auto per-op-class routing) is delivered + verified
(task 2); correctness machinery (token gates) proven in tasks 1-3.
