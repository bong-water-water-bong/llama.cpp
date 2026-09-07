## Addendum 7 (final for this session): conv island inputs verified correct under per-program sync - the conv command itself is the defect

- GGML_HRX_SYNC_EACH (per-program hrx_stream_synchronize after each execution) does
  NOT change the decode (tok0=563, same text) nor the conv outputs (QK_dw/QK_grp row
  rms identical: 1.3/1.29/1.04/0.6/0.52/85/81 and 7.7/6.1/3.6/1.7/4449/9575).
- Under sync-each the Qraw-0 d2h read (off 1048576) = my numpy EXACTLY (f16 noise;
  previously the no-sync run showed only the "node_153" read at that offset with
  different values). => the conv island's qk inputs are CORRECT once ordered.
- Conclusion: the SSM conv command itself (uid 10846, kernel_id 0xC1123AC8, 3
  bindings out/weight/in, exec-level data buffers NULL) computes deterministic wrong
  outputs FROM CORRECT INPUTS. The decreasing row-magnitude signature
  (1.30 -> 0.52 across the 7 output rows with magnitude-flat inputs) is consistent
  with a wrong-stride/wrong-offset read inside that kernel/executor.
- The kernel has no loom source anywhere (manifest/catalog/ops/ dirs), no dispatch
  registration for SSM_CONV, no strings in the .so, no name match for the id under
  FNV-1a-64 over all ggml_-prefixed catalog names. Its executor is unidentified in
  ggml-hrx (between the root-op traversal and the produced QK_dw buffer).
- All upstream tensors validated bit-exact (numpy): embd, input scale/bias, attn_norm,
  Q/K projections, d2h copies, state cache (CPU zeroed).
- The conv island = the single corrupt stage feeding the attention -> residual -> x ->
  gate_up rows -> logits.

## Recommended next steps (device-layer / runtime owner)
1. Find the executor of the 10846 Kernel-kind command: trace where the prepared
   command's kernel_id 0xC1123AC8 is resolved/executed (the prepared-command
   execution path; possibly a host-op adapter or a runtime-generated kernel), then
   read its input-addressing code.
2. Or capture the conv input at launch inside that executor before the NULL
   host-staging resolution.
3. Alternatively verify the conv kernel against a same-shape direct launch (the
   mma_repro-style driver harness) with known input/output pairs.
