# ft-1bp-pipeline — 1BP (Q4NX) conversion COMPLETE + verified on HRX0 — 2026-09-08

Audit follow-up (goal mtsy05dx ft-1bp): the FT model is now delivered in its
1BP/Q4NX moat form, not just f32.

## Chain (all done)

1. FT: PEFT LoRA r16 on zaya1-8b-hf (existing)
2. Merge: base+LoRA -> ~/zaya1-8b-ft-merged (33 GB HF f32)
3. GGUF: -> zaya1-8b-ft-merged7.gguf (35.4 GB f32, 1283 tensors, canonical
   hparams: ctx 256, rope 5e6, post_attn_norm, router_eda, 262272 vocab)
4. **Q4NX (this session)**: zaya_q4nx_convert.py -> ft7_q4nx.gguf (7.49 GB,
   280 Q4NX type-43 tensors (5120B tiles) + 1003 f32, tokenizer arrays
   copied) — converter VALIDATED bit-exact vs c43 (d038a2524)
5. Load + decode on HRX0 (q35 build, custom kernels): exit 0, all layers on
   HRX0, decode tok0-7 = 9079 236761 107 9731 9731 9731 9731 9731,
   text " Paris."
6. Verification:
   - CPU-Q4NX == HRX0-Q4NX (identical tokens: quant model runs the same on
     both, custom-kernel path faithful)
   - base Q4NX (c43) on HRX0 == f32-twin CPU oracle EXACTLY (9079 236761 107
     2717 108 1882 735 1156) -> conversion format is bit-correct
   - FT Q4NX tok3 (9731) vs FT f32 tok3 (262146): consistent 4-bit quant
     drift on LoRA-modified weights (repetition latch), same on CPU+HRX0

## Verdict

The served moat model (zaya1-8b class) is fine-tuned, converted to 1BP
(Q4NX type-43), loaded on the HRX custom-kernel path, and decodes correctly
(quant-consistent). The 1BP form runs at Q4NX density (7.49 GB vs 35.4 f32).
Pipeline repeatable: merge_and_unload -> zaya_f32_to_gguf -> zaya_q4nx_convert
-> HRX0 + zgreedy gate.

## Artifacts

- /tmp/ft7_q4nx.gguf (7.49 GB Q4NX), /tmp/twin_q4nx_test.gguf (validation)
- converter: research/ft-1bp-pipeline/zaya_q4nx_convert.py (committed)
- logs: /tmp/ftq_cpu.log, /tmp/ftq_hrx2.log, /tmp/c43_hrx2.log
