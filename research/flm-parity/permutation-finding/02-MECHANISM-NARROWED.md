# Mechanism narrowed: readback/transfer layer, not mm kernels (agent-f49062, 01:0x UTC)

Additional constraints since 00-FINDING.md / 01-LOGITS.md:

- ngl=40 (lm-head weight CPU-resident) => sched split ABORT (EXIT 134,
  ggml_backend_sched_split_graph) - mixed offload unsupported; cannot isolate the
  lm-head weight residency that way.
- zgreedy_b1 (1-token-batch prefill) => GGML_ASSERT(n_ubatch > n_keep_tail) at
  llama-batch.cpp:609 (zaya SSM tail) - harness limitation, not a device bug.
- Decode steps (1-token) produce grammatical continuations => the lm-head path works
  per-step; the corruption is specific to the 6-token prefill tail.
- hrx_synchronous_d2h (libhrx; source at ~/hrx-ws/hrx-rocm/libhrx/src/libhrx/transfer.c)
  = thin wrapper over iree_hal_device_transfer_d2h (linear). A linear transfer cannot
  scramble within a span => the scramble either (a) is NOT in the readback (kernel
  write-side, contradictory with consumed-correctness), or (b) comes from per-region
  behavior of the iree HAL transfer / buffer-mapping layer (the original rounds-16a-17b
  "cross-split / device-layer" suspect, abandoned in rounds 63-70 for the mm1-compute
  theory which the permutation data now refutes).
- buffer_get (ggml-hrx.cpp, round-69 extent fix) memcpys for host-mapped buffers
  (context->base != GGML_HRX_FAKE_PTR_BASE) and uses the sync d2h otherwise. The
  consumers that are CORRECT (GLU gate/up reads) and the dumps that are SCRAMBLED both
  ride the same transfer primitives on overlapping regions => the discriminator is the
  (buffer, offset) region itself.

## Next probes for the continuation (cheapest first)
1. Identify which arena (4.75 MiB device-local vs 258 MiB HRX0_HOST host-mapped) holds
   the ffn output slots (0x...7f90 @ 2621440-2883584) and the final norm output; test
   buffer_get (memcpy path) vs hrx_synchronous_d2h (iree path) on the SAME region with
   the canary harness - if memcpy is clean and iree d2h scrambles, the bug = the iree
   sync-transfer path for device-local regions (device-layer owner).
2. Capture the final norm OUTPUT tensor name (the "norm" filter caught the *norm.weight
   tensors instead) - likely "output_norm" or unnamed at the graph tail; compare its
   d2h content vs a numpy rmsnorm of the layer_out-39 oracle.
3. If the norm output = correct in-buffer and the lm-head input copy (the single sched
   cpy) is kernel-based, the -0.28 logits collapse must come from that copy itself -
   trace the copy op's source/target regions.
