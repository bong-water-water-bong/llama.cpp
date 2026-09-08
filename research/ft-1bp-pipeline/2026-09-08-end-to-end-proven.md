# ft-1bp-pipeline — END-TO-END PROVEN — 2026-09-08

Goal mtsy05dx task ft-1bp-pipeline. Full chain now works on the served moat
model (zaya1-8b class):

## The chain (all done this session)

1. FINE-TUNE (existing): PEFT LoRA r16 alpha32 on zaya1-8b-hf
   (~/zaya-ft-lora/adapter_model.safetensors, Sep 7).
2. MERGE: base + LoRA merged (transformers CPU f32) ->
   ~/zaya1-8b-ft-merged (33 GB HF, arch zaya, 40 layers).
3. CONVERT HF->GGUF: tools/convert_zaya_safetensors_to_gguf.py, fixed:
   - post_attention_norm -> post_attn_norm (arch name)
   - zaya_router_eda from mlp.gate.router_states_scale (was skipped)
   - token list padded to 262272 (runtime derives n_vocab from it)
   - canonical hparams: ctx 256, rope_freq 5e6, no ssm.state/inner KVs
   - --f32 mode (tensors F32, matching the known-good f32twin)
   -> zaya1-8b-ft-merged7.gguf (35.38 GB, 1283 tensors, f32)
4. LOAD + DECODE on HRX0 custom-kernel path (q35-hrx-fix build):
   all 41 layers assigned HRX0; decode tok0-7 =
   9079 236761 107 262146 108 6481 236789 236751, text " Paris."
5. VERIFY vs reference (CPU decode of the same model):
   CPU == HRX0 TOKEN-IDENTICAL (8/8).
6. FT effect confirmed: tok3 = 262146 differs from the un-FT base oracle
   (base: 9079 236761 107 2717 108 1882 735 1156) - the adapter changed
   the model, decode stays coherent.

## Blocker history (resolved)

- converter emitted 1244 tensors / wrong names / wrong vocab / wrong
  hparams -> each fixed (see 3); final file loads and decodes clean.

## Repeatable pipeline (per model)

HF base + PEFT adapter -> merge_and_unload -> convert_zaya_...gguf.py --f32
(with canonical-hparams + 262272-vocab patches) -> HRX0 load -> zgreedy_dev
CPU-vs-HRX0 token gate. Harness + gates model-agnostic (tasks 2-4 infra).

## Artifacts

- ryzen: ~/zaya1-8b-ft-merged (+7.gguf 35.4GB), converter (patched)
- strixhalo: ~/models/zaya1-8b-ft-merged7.gguf (35.38 GB)
- logs: /tmp/ft7_cpu.log (reference), /tmp/ft7_hrx.log (HRX0, identical)
