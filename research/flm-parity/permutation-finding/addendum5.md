## Addendum 5: full CPU-chain numpy reconstruction vs HRX QK_grp - no match
- Reconstructed the entire CPU-side conv chain in numpy with the corrected machinery
  (validated qk, tap-fastest conv weights, zero state): dw conv (8->7) then grouped
  conv (cca_conv_grp, 7->6, per-channel-group 2-tap). Compared vs the HRX QK_grp dump:
  rows 0-3 mad 2.9-6.3 (vs row rms 1.7-7.7) and rows 4-5 mad 1579/5228 (HRX garbage
  rms 4449/9575). No match.
- Signature: the HRX QK_dw rows rms [1.30, 1.29, 1.04, 0.60, 0.52, 85, 81] and QK_grp
  [7.7, 6.1, 3.6, 1.7, 4449, 9575] DECREASE across the sane rows while all validated
  upstream tensors (q/k projections, cur, embd) are magnitude-flat per token. The
  conv consumed data with a decreasing per-row magnitude - not the validated qk rows.
- Conclusion: the corruption = inside the conv command execution path (input assembly,
  the unidentified Kernel-kind executor, or the state rows as actually staged). The
  cross-backend copy of the raw qk = validated correct (addendum 4). All live
  instrumentation remains in the tree; the next step is to hook the conv command's
  input at launch time (the prepared-command executor, before the bindings are
  resolved to NULL host staging).
