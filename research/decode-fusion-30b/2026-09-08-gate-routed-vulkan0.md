# 30B Q4_K_M decode gate - CLOSED via op-class route (fork-Vulkan0) - 2026-09-08

Goal mtsy05dx-vz0y5n task decode-gate-routed. The launch-collapse lane executed
the op-class split (goal/hrx-collapse 40c42c3ee, user-authorized): the fork is
dual-backend HRX+Vulkan; stock Q4_K decode routes to fork-Vulkan0. This gate
measures the 30B-A3B Q4_K_M decode on that route (the old goal's numeric bar).

## Performance (llama-bench tg256 @ KV 2944-3200, ngl 99, r3, opensplit build 5c39c7d68)

| route | 30B decode (t/s) |
|---|---|
| old llama-build bundle HRX0 (ggml-hrx 0.9.11) | 40.0 |
| modern fork HRX-only (root cause: no fused MoE kernel fires) | 11.96 |
| fork-Vulkan0 (op-class route, GGML_VK_DISABLE_MMVQ=1) | 93.65 +- 0.32 |

Command: GGML_VK_DISABLE_MMVQ=1 llama-bench -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf -p 2944 -n 256 -ngl 99 -r 3 -dev Vulkan0
pp2944 1224.8 t/s, tg256 93.65 t/s. Gate bar >= 40 t/s: PASS (2.3x).

## Correctness (greedy argmax, temp 0, raw prompt, KV ~2921)

zgreedy_dev harness (CPU oracle = llama.cpp-m3cap pure-CPU build; device side =
opensplit build Vulkan0, GGML_VK_DISABLE_MMVQ=1). Prompt: 2921-token random
prose doc. 16 generated tokens at KV 2921-2937:

CPU oracle: 323 279 323 279 323 279 323 279 323 279 323 279 323 279 323 279
Vulkan0:    323 279 323 279 323 279 323 279 323 279 323 279 323 279 323 279
=> TOKEN-IDENTICAL 16/16. (Text degenerate "and the and the..." - random-word
prompt latching; identity is the gate, not fluency.)

0.6B cross-check: Vulkan0 == CPU oracle 12095/13/576/6722/315/9625/374/1083
(token-identical, matches lane oracle).

## Artifacts

- /tmp/zgreedy_dev.cpp (device-forced greedy harness; CPU oracle + Vulkan0)
- /tmp/gate30_cpu.log, /tmp/gate30_vk.log, /tmp/gate30b_vk.log (llama-bench)
- builds: ~/llama.cpp-m3cap/build (CPU oracle), ~/hrx-ws/wt/hrx-collapse/build-opensplit (dual-backend)

## Verdict

30B Q4_K_M decode >= 40 t/s at KV 2.9-3.2k is delivered by the op-class route
(fork-Vulkan0) at 93.65 t/s, correctness-gated token-identical vs CPU oracle.
The 30B stock-quant decode no longer depends on HRX fused-kernel engagement;
auto-route wiring (task auto-route-fork) makes this the default path.
