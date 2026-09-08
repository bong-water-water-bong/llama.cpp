# ft-1bp-pipeline — status + verified links + remaining gap (goal mtsy05dx task 5)

Contract: served model (30B-A3B or zaya class) fine-tuned and converted to 1BP
end-to-end, loaded on the HRX custom-kernel path, decode verified vs reference;
pipeline repeatable per model.

## Verified links (this goal's prior tasks + engine lane)

1. LOAD on HRX custom kernels: zaya-q4nx-c43.gguf -> HRX0, all 40 layers
   device-assigned, decode == CPU oracle (9079/236761/107/2717/108/1882/735/1156,
   NaN-free) - verified in task auto-route-fork on q35-hrx-fix build.
2. CONVERT: q4nx container -> c43 GGUF exists (zaya-q4nx-c43.gguf, 7.48 GB,
   2026-09-03); ws12-hrx-loom has quantize-gguf-to-q4nx.cpp,
   make-q4nx-from-float.py, zaya-to-gguf.py (I8 tiles -> GGML_TYPE_Q4NX=43).
3. FINE-TUNE output EXISTS: /tmp/zaya_adapters.bin (67,820 B, 2026-09-08 12:40,
   ryzen engine checkout) - LoRA-style delta (CCA + MoE B/A matrices) from the
   engine-side CPU-expert trainer (zaya_train_main.cpp); CE 34.26 -> 21.33
   taught-code gate PASS (engine commits 81b24839/f4b9b5c3/dea14607 on
   feat/hrx-gfx1151-build).

## The gap (single missing link)

MERGE + EXPORT: the trainer's save_adapters writes B/A deltas; the code comment
(zaya_train_main.cpp) says "Real q4nx weights + merge/export land in the next
stage". There is no merge(adapter -> full weights) -> quantize -> GGUF converter
wired for the zaya arch end-to-end. This is the implementation work remaining:
  a. merge_adapters(): apply B/A LoRA deltas to the f32/zaya base per layer
     (CCA attn even idx, MoE odd idx per zaya_decode.cpp structure)
  b. requantize merged f32 -> I8 tiles -> GGML_TYPE_Q4NX (43) - reuse
     ws12-hrx-loom quantize-gguf-to-q4nx.cpp / make-q4nx-from-float.py
  c. zaya-to-gguf.py --q4nx -> c43 GGUF
  d. load on HRX0 (q35-hrx-fix / HRX2 build) + oracle-verify (zgreedy_dev,
     tok stream == CPU oracle; links 1-2 above are the acceptance harness)

## Pipeline repeatability

Per model: (1) engine CPU-expert FT (NPU_CPU_EXPERT=1, zaya_train_main) ->
adapter bin; (2) merge+requant (new code, step a-c above); (3) HRX load +
zgreedy oracle gate. The verify harness and load path are model-agnostic
(zgreedy_dev AUTO/HRX0 + rt_zc state tooling from tasks 3-4).

## Artifacts today

- adapter: /tmp/zaya_adapters.bin (ryzen + strixhalo copies)
- base: ~/zaya-f32.gguf (35 GB), ~/zaya-q4nx-c43.gguf (7.48 GB, verified)
- converters: ~/1bit-MONSTER/research/ws12-hrx-loom/
- verify: /tmp/zgreedy_dev (HRX0 oracle gate, task-2 verified)
