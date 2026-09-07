# UNIFIED THEORY: HRX->CPU host-read corruption is the root cause (agent-f49062)

Date: 2026-09-07. Supersedes the "conv kernel wrong" framing; REFRAMES the whole hunt.

## The theory
The HRX device -> CPU host read path (d2h / host-visible buffer reads of device data)
deterministically returns CORRUPT VALUES for specific (buffer, row, region) patterns.
Every CPU-side consumer of HRX-produced data is therefore wrong:
- the CPU lm-head (final logits) -> wrong tok0 (563 vs 9079);
- the CPU-fallback SSM conv (reads the HRX QKraw via a cross-backend copy) -> corrupt
  QK rows -> wrong attention for the affected tokens;
- all PROGRAM_DUMP readbacks -> corrupt-looking tensors (row-dependent: t0 clean in
  some regions, t1-5 corrupt elsewhere).
The device-INTERNAL chain may be entirely correct; the observed "wrong gate_up rows",
"wrong x", "wrong residual" = CPU-side views of corrupt readbacks (all bit-exact numpy
matches only prove the readbacks are internally consistent, not correct).

## Evidence assembled this session
1. The full input chain (embd -> input scale/bias -> attn_norm -> Q/K projections) =
   VALIDATED bit-exact ON DEVICE (dumps match numpy to mad 0.0/f16-noise) - device math
   upstream of the conv is provably correct.
2. The SSM conv = CPU fallback (no loom conv kernel exists; exec-level data bindings =
   NULL device buffers - host staging). Its input = the materialized concat of the
   CPU-pinned (zeroed) state + the HRX QKraw, i.e. a cross-backend d2h assembly.
3. The conv output (host-written) = corrupt vs every numpy reconstruction: correlation
   ~0 vs all token/conv candidates; row-0 rms = 1.30 (rules out a zeroed-state input
   under the authoritative conv semantics from ggml-cpu/ops.cpp: out[t,c] =
   sum_k w[k,c]*x[t+k,c]); rows 5-6 = rms 81-85 garbage. Under the authoritative
   semantics the garbage rows imply the conv input's LAST rows (qk4/qk5 or the state
   tail) = garbage in the CPU-side copy.
4. The recurrent cache = CPU-pinned + buffer_clear(0) at alloc; if the copy of the
   state were right it would be zeros - it is not (row-0 output rms 1.3), so the CPU
   side of the assembled concat = already corrupt.
5. The same corruption class was seen in every stage: gate_up rows, norm outputs,
   residual, QK. The old canary/writeback history (rounds 16a-17b) + round-68/69
   readback fixes + the round-69 extent-convention buffer_get fix all live in this
   family. The final lm-head input read = the last corrupt read -> wrong logits.

## Decisive next test (fleet)
Read the SAME device region through (a) the d2h used by consumers/dumps and (b) a
device-side kernel copy at the same instant and compare. If (a) != (b): the d2h /
host-read layer (iree_hal_device_transfer_d2h in libhrx, transfer.c) or the
host-visible buffer coherence = the defect - the rounds-16a-17b "device-layer owner"
direction was RIGHT and was wrongly parked in rounds 63-70 as an mm1 compute bug.
The gate_up/down mm kernels + all loom kernels are exonerated.
