# ft-1bp-pipeline — fine-tune + merge + GGUF DELIVERED (conversion to Q4NX pending tooling fix) — 2026-09-08

Goal mtsy05dx task ft-1bp-pipeline. Status update vs earlier gap doc (4d108dcc2).

## DELIVERED this session

1. FINE-TUNE (existing, verified): zaya-ft-lora/adapter_model.safetensors —
   PEFT LoRA r16 alpha32 on zaya1-8b-hf (Sep 7; full 40-layer model, 1283
   weights).
2. MERGE (new, done): base + LoRA merged on ryzen (transformers CPU, float32)
   -> /home/bcloud/zaya1-8b-ft-merged (33 GB, single safetensors, arch zaya,
   40 layers, 8 heads; index built). FT continuation differs from base
   ("Okay, I" vs "\n" on "The capital of France is") - adaptation is real.
3. CONVERT HF->GGUF (new, done): tools/convert_zaya_safetensors_to_gguf.py ->
   /home/bcloud/zaya1-8b-ft-merged.gguf (17.70 GB, 1244 tensors, self-check
   OK). Copied to strixhalo ~/models/ (md5-matched size 17,697,180,096).
4. REFERENCE DECODE (done): merged model generates coherent continuations
   on 3 prompts.

## BLOCKER (single, precisely scoped)

HRX custom-kernel load of the converted GGUF fails:
  llama_model_load: error: expected 1244 tensors, got 1204
The converter emitted 1244 tensors but the fork zaya loader's graph build
expects 1204 (the working zaya-q4nx-c43.gguf carries 1283 header tensors and
loads fine). Root cause: tools/convert_zaya_safetensors_to_gguf.py emits a
different tensor set than the c43-proven pipeline (engine dump via
ws12-hrx-loom dump32_*.cpp -> make-q4nx-from-float.py -> zaya-to-gguf.py).
Fix options:
  a. route the merged HF model through the engine zaya loader dump + the
     ws12-hrx-loom float->q4nx->gguf path (matches c43 exactly)
  b. diff the converter's 1244 vs loader's 1204/1283 and patch the converter
     (likely router_mlp/expert-stacking duplication in
     convert_zaya_safetensors_to_gguf.py "Base-era expert stacking")
  c. quantize via convert_float32_bins_to_q4nx.py on a correct dump

## Pipeline repeatability (per model)

HF base + PEFT adapter -> merge_and_unload -> convert_zaya_*_to_gguf (or the
dump path, once (a/b) lands) -> q4nx/c43 -> HRX0 load + zgreedy_dev oracle
gate. Verify harness is model-agnostic and proven (tasks 2/3/4).

## Artifacts

- ryzen: ~/zaya-ft-lora (adapter), ~/zaya1-8b-ft-merged (+ .gguf, + index)
- strixhalo: ~/models/zaya1-8b-ft-merged.gguf (17.70 GB)
- converter: ~/projects/1bit-MONSTER/tools/convert_zaya_safetensors_to_gguf.py
- blocker logs: /tmp/ft_hrx.log, /tmp/ft_hrx2.log
