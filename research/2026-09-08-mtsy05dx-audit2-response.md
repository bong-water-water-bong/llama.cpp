# Audit-response #2 (2026-09-08, after 3rd disapproval) — gaps 2/3/4 closed in-engine

## Gap 2: single-api-router must route per policy across engines

CLOSED. set_device_pin API + per-engine device param (commit fad5f950):
- Inprocess pins its offload device by name ("HRX0" | "Vulkan0" | "none");
  HrxPrefillEngine/HrxDecodeEngine take the policy device.
- router_duallep_main.cpp (in-tree): ONE generate() call at 30B contract with
  stock-q4k policy = prefill pinned HRX0 -> memfd -> decode pinned Vulkan0:
  31.08 t/s (3.1x the 9.92 t/s single-leg HRX0 decode), exit 0.
- Vulkan0-pinned engine decode of the 30B state is TOKEN-IDENTICAL to the
  fork Vulkan0 oracle (271 9745 45 271 785 2701 374 279 4226 311, matching
  ref30_os.log R-stream) — the engine API now reproduces the architecture's
  own winning route for the contract model.
- Device pins coexist in one process (HRX0 prefill ctx + Vulkan0 decode ctx).

## Gap 3: unified-bench needs an integrated >=2k/>=500 measurement

CLOSED. prefill_batch (chunked 512, explicit pos) added to the seam so
long prompts run through one generate() (commit 9df31646); bench_contract_main
in-tree. Measured at 30B Q4_K_M (contract model), policy stock-q4k,
Vulkan0 prefill -> memfd -> Vulkan0 decode:
- prompt 2146 tokens (>=2k bar), 512 continuation tokens (>=500 bar)
- ONE generate() call: 8.015 s end-to-end (prefill+handoff+decode),
  decode leg ~80-85 t/s, coherent continuation, exit 0.
- Compare: fork unified-row estimate pp2048/tg512 ~8.2 s (HIP pp -> memfd ->
  Vulkan tg) — the in-engine measured number (8.015 s at pp2146/tg512) lands
  in the same class, now as a real single-API call.
- Full engine build green; /tmp/cb_vkvk3.log raw run.

## Gap 4: reproducibility (drivers were /tmp-only)

CLOSED (commits 9df31646, 9b19030c): router_test_main.cpp now has a LIVE
mode (mode 1: real tokenize + HRX0 prefill -> export memfd -> import ->
Vulkan0 decode, no files; PASS at 0.6B: 12095 13 576 6722 315 9625 374 1083)
plus fail-close mode 0. router_duallep_main.cpp and bench_contract_main.cpp
are in-tree mains; all compile from src/router/ + src/hrx_inprocess.cpp.

## Gap 1 (unchanged, needs user scope decision)

The objective's literal dma-buf/UMA NPU<->HRX shared-device-memory handoff is
NOT claimed: the implemented seam is a host-anon memfd carrying llama_state
bytes between engine contexts (zero-copy at the seam: no file, no host-side
copy of the handoff buffer). No XDNA NPU engine participates (NPU covers
<=4B; 30B contract cannot run on it; NPU-pool work is other lanes'
territory). This remains the one contract item requiring a user decision
(recorded amendment vs. separate dma-buf effort), not silent re-scope.

## Commits (engine feat/zc-mem-handoff, pushed; build green)

fad5f950 dual-leg policy routing + router_duallep_main
9df31646 prefill_batch + bench_contract_main (>=2k/>=500 measured)
9b19030c router_test live mode (in-tree reproducibility)
