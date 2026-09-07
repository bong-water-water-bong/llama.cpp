# MAJOR LOCALIZATION (agent-f49062): gate_up mm EXONERATED - input x wrong upstream

Date: 2026-09-07. Tree: fix/hrx-ngl-init-order (WIP + full-act rd_act capture v2 + publish trace).

## The finding, in one line
The zaya gate_up (ffn_moe_gate_up MUL_MAT_ID) computes its matmul CORRECTLY on every
row; the rows a1..a5 (and a0 in a block-dependent subset) are wrong because the mm's
INPUT (post_attn_norm output, "x") is wrong in the device buffer for those tokens.
The corruption enters in residual_post_attn or its producers (attention/conv/SSM path
of the multi-token prefill).

## Proof chain (all measured, numpy, this session)
1. Ported the exact Q4NX dequant (ggml-quants.c tile layout: bf16 scales @[0..512),
   zeros @[512..1024), two's-complement int4 @[1024+lane*2048+c*8+bi]) + gguf tensor
   layout ([8192, tpe, n_expert] tile-framed, expert planes contiguous) to numpy.
2. VALIDATION A: W_down[blk.0] dequant x oracle swiglu (r03_005) reproduces the oracle
   down output (r03_006) EXACTLY (mad 0.000000) - dequant + tiling + plane order = right.
3. Extended the in-kernel rd_act capture to ALL 2048 k-values of every token
   (row_debug f32[8192 + token*2048 + k]; loom edit + corpus rebuild + capture).
4. VALIDATION B: W_gate_up[e3] (dequant, f16-truncated) x captured-x[0] reproduces the
   HRX row a0 EXACTLY (mad 0.00000) and the CPU oracle to f16 noise (0.00617).
   => captured x = exactly what the kernel used; the kernel's mma + publish = correct.
5. The bad row a1: W_gate_up[e5] x captured-x[1] reproduces HRX a1 EXACTLY (mad 0.0);
   the 16x6 candidate matrix (all experts x all tokens) has NO other match
   (next best mad 0.74). Since HRX a1 != oracle a1, captured-x[1] != oracle-x[1]:
   the kernel computed the right matmul on a wrong x. (partial-K variants: only full-K
   matches, so no k-window issue.)
6. The mm input BUFFER (d2h, same run) == the in-kernel fetch for all 6 tokens
   (mad 0.0001, f16 noise) => the buffer content itself is wrong for t>=1 (not a
   fetch/view misaddress, not a readback artifact).
7. x == rmsnorm(residual_post_attn) bit-exactly (mad 0.000000, all tokens) => the
   RMS_NORM op is fine; residual_post_attn (the scaled attention residual) is wrong
   for t>=1.
8. Consistent with the decode being grammatical: 1-token decodes use a different
   conv/state geometry; the corruption is specific to the 6-token prefill.

## What this means for the fleet
- The round-63..70 "mm1 per-partition compute" direction (b30173 WIP loom changes) was
  chasing the wrong op. The mul_mat_id kernel and its wmma path are CLEAN.
- The defect = in the pre-attention residual path for n_tokens > 1: candidates in order:
  (a) the ssm_conv / grouped conv (cca) handling of the multi-token conv_input
      ([conv_state(2) || QKraw(6)] -> conv over 8), incl. CPU<->HRX boundary copies
      for ops without native HRX kernels (round-16e precedent: "CPU->HRX boundary
      proven corrupt"; qwen fixed via GET_ROWS claim shape cap);
  (b) the flash-attn/wo projection for n_tokens > 1;
  (c) res_scale ops (elementwise - unlikely, but verify).
- a0 (token 0) block-dependence: blocks where even token 0 is wrong (3,6,7,10,19-21,
  26,29-30,35-39) => the corruption can also hit position 0 in those blocks (conv
  state or accumulated state from previous blocks).

## Next probes (cheapest first)
a. Capture QK_dw / QK_grp / conv_input / Qcur / Kcur (already named in zaya.cpp cbs)
   for block 0 and compare against a CPU-side numpy reconstruction of the conv
   (weights blk.0.ssm_conv1d.weight + cca_conv_grp + conv state) - identifies whether
   the conv output or the attention input is wrong for t>=1.
b. Verify which backend runs the ssm_conv/conv ops in the mixed graph (sched split
   listing; GGML_HRX_CPU_OPS forcing) - if they are CPU fallbacks fed by d2h of the
   HRX-written QK raw, the boundary copy is the suspect (round-16 family).

## Artifacts
- numpy port + probes: /tmp/probe_b2.py, /tmp/validate_down.py (recreate from this doc;
   committed copies under research/flm-parity/permutation-finding/ next commit).
- Captures: strixhalo /tmp/prg_dump_run1 (gate_up + down + full-act row_debug), plus
   /tmp/cap-*.log runs; current /tmp/prg_dump = xr run (row_debug + residual_post_attn
   + post_attn_norm for all 40 blocks).
