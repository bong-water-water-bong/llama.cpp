# qwen3-roster head-to-head: refreshed fork (HRX GPU) vs FastFlowLM (NPU) — same box

Date: 2026-09-05. Box: strixhalo (Ryzen AI MAX+ 395, gfx1151 iGPU via HRX vs
XDNA2 NPU via FLM). All numbers measured on this box.

## Method
- Ours: llama-bench, model fully offloaded to HRX0 (Radeon 8060S) with -ngl 99.
  pp512/tg128 = pure prefill/decode throughput (llama-bench internal timing).
  Real GPU verified (device registered + /dev/dri maps; see HRX-GPU-ENABLE.md).
- FLM: FastFlowLM v1.0.4 (NPU), flm serve + streaming chat/completions,
  decode = generated tokens (content+reasoning) / time-after-first-token at
  max_tokens=256, prompt 256 tokens. decode throughput measured over the full
  generation (FLM ignores early EOS, runs the full budget).

## Decode (tokens/s)

| model            | ours (HRX iGPU) | FLM (NPU) | ratio |
|------------------|----------------:|----------:|------:|
| Qwen3-0.6B       |          249.6 |      86-89 | ~2.9x |
| Qwen3-1.7B       |          122.4 |    40.4-40.8 | ~3.0x |
| Qwen3-4B         |           57.0 |       19.2 | ~3.0x |

## Prefill (tokens/s)

| model            | ours pp512 (HRX) | FLM blended* |
|------------------|-----------------:|-------------:|
| Qwen3-0.6B       |          12737 |          ~1250 |
| Qwen3-1.7B       |           3401 |          ~345 |
| Qwen3-4B         |           1241 |          ~200 |

*FLM prefill = prompt_tokens / TTFT (includes per-request overhead; not a pure
pp measurement). Directional only.

## TTFT (256-token prompt)

| model            | ours | FLM |
|------------------|-----:|----:|
| Qwen3-0.6B       | ~0.03-0.05 s | 0.6-0.84 s |
| Qwen3-1.7B       | (llama-bench) | 0.74-0.92 s |
| Qwen3-4B         | (llama-bench) | 1.2-1.4 s |

## Notes / caveats
- Different execution units (iGPU vs NPU) — this is a same-box, same-model-class
  throughput comparison of our open stack vs FLM, per the goal. Silicon differs;
  the numbers show our stacks decode path is ~3x FLMs on this hardware.
- Quant formats differ (GGUF Q4_K_M vs FLM q4nx) — same models/architectures.
- Ours: no DISABLE_* flags; clean offload, deterministic.
- Previous "HRX" numbers on this fork were CPU (see HRX-GPU-ENABLE.md) —
  treat all pre-2026-09-05 refreshed-fork throughput numbers as CPU.

## How to reproduce
    cd ~/hrx-ws/amd-hrx-graph
    export ROCMLIB=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
    export LD_LIBRARY_PATH=$ROCMLIB:build/bin
    ./build/bin/llama-bench -m ~/models/Qwen3-0.6B-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
    ./build/bin/llama-bench -m ~/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
    ./build/bin/llama-bench -m ~/models/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -p 512 -n 128 -r 3 -ngl 99
    # FLM side:
    /opt/fastflowlm/bin/flm serve qwen3:0.6b -p 8099   (also :1.7b -p 8101, :4b -p 8102)
    python3 research/flm-parity/flm_parity_bench.py --base http://127.0.0.1:PORT/v1 --model <tag> --ppl 256 --ngen 256 --reps 3
