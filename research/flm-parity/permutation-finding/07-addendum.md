# Addendum: pre-attn chain probes (agent-f49062, 2026-09-07) - open sub-stage

Follows 06-LOCALIZATION.md (bug = pre-ffn for tokens >=1). Probes this session:

1. The whole pre-attn chain is HRX-dispatched and capturable: uid 10843 (per block) =
   the pre-attention program (input_norm/QKV/conv/flash/res_scale/residual/post_attn_norm).
   QK_dw-0 = 35840B = [7 x 1280] (ssm_conv over [state2 + QKraw6] = valid 8->7); QK_grp-0 =
   30720B = [6 x 1280] (grouped conv 7->6); Qcur/Kcur views captured; conv weights all F32
   in-file (ssm_conv1d [2,1280], cca_conv_grp [2,128,1280] = n_groups 10, bias [1280]).
   Two identical copies of the norm op exist per block (uids 10843 + 10854 zone) with
   identical outputs (graph duplication - same values).
2. Norm weight identity: HRX-bound blk.0.attn_norm.weight == gguf file bytes (mad 0.0).
3. ANOMALY: rms(w_attn_norm) = 1.33723 but the HRX input_norm-0 output rows have rms
   1.2925 (both copies). For a correct rmsnorm, out = x_hat*w with |x_hat|~1 => out rms
   ~= rms(w); eps corrections <= ~1%. 3.3% gap => either the norm KERNEL is wrong or its
   input is not what we assume (rms(x) ~0.012 implied). numpy reconstruction of the
   expected norm input (token embeddings, both vocab-fast/cont-token orientations) does
   NOT reproduce the HRX input_norm at ANY token (corr <= 0.25; embedding rows read from
   the file look sparse/coarse: values in ~0.034 steps with exact zeros - suspicious but
   the file type = F32; need a ground-truth embedding capture to resolve).
4. Flash-attn forcing (GGML_HRX_CPU_OPS=FLASH_ATTN_EXT, round-16 cell C) = UNSUPPORTED
   on the current tree: zaya falls back to non-flash attention whose KV-cache view
   (cache_v_l0 (view) (permuted) (transposed)) cannot be placed on HRX0 -> sched abort.
5. NOTE FOR THE FLEET: ~/zaya-decode/out_oracle now contains ONLY the 362 r04 files -
   the r30 decode-graph oracle series (3257 files) is GONE (deleted at some point since
   this session's start; not by me). Anyone relying on r30 decode oracles should
   regenerate or check the deletion.

## Where this leaves the hunt
- Airtight: gate_up mm + its kernel = correct; mm input x = wrong on-device for t>=1
  (and t0 in ~15 blocks); post_attn_norm (rmsnorm) = bit-consistent with its input;
  residual_post_attn = wrong for t>=1 => upstream (attention/conv/norm-input path).
- Open sub-stage: input_norm-0 (the attn_norm output, = the attention input) itself does
  not match a numpy norm of the file embeddings - either the attn_norm kernel, its input
  (embedding GET_ROWS result / recurrent-state add), or my embedding read is wrong.
  Discriminators for next: (a) capture the norm INPUT tensor (the value bound as cmd0 b0
  of uid 10843; likely named token_embd/unnamed - dump via bind_0 or exact-name filters,
  AVOID filter "token_embd" = would also dump the 2.1GB weight); (b) capture the decode
  (1-token) input_norm and compare its rms against rms(w) - the decode is coherent so its
  norm should be sane; (c) check the cca_state content (filter "cca_state") at prefill
  start - if the recurrent state is not zero-initialized, the norm input carries it.
