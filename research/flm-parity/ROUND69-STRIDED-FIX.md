
## Round 69 (executor lane, 428ab3) — FIX LANDED: strided buffer_get compact-gather replaced with the extent-convention single-span read

### The bug (round-68 finding, now fixed)
The uncommitted strided-gather in ggml-hrx.cpp buffer_get (added ~12:51, gather-era) filled
the CPU destination with the view's LOGICAL rows COMPACT (dst[t*2048+r], 49152B) while the
copy destination (ggml_dup_tensor_layout = same strided nb [4,16384,16384]) + the CPU
consumer read at the STRIDED positions (t*16384B). Result: token doubling (swiglu[t] =
token 2t's data: t0 ok, t1<-t2, t2<-t4, t3-5 <- the never-filled 40960B tail = stale,
repeated) — the divergence visible in all gather-era runs (13:13-13:18; the decode text
even changed vs the round-63 baseline).

### The fix (committed; ggml-hrx.cpp, my lane)
The strided branch now serves the view's MEMORY EXTENT with ONE parent-resolved span
(d2h of base_off..base_off+size through the view_src chain) = the ggml buffer_get
convention of every other backend (CPU/CUDA raw) = what the committed ae0590115-era code
did = the round-63 baseline semantics. The parent resolution (gather-era addition) is kept.

### Verification (f32twin, 5-token prefill, block 0)
- swiglu = silu(slot-gate)*slot-up, full rows, rms 1e-6 for ALL 6 tokens (was 0.35-0.75
  for t1-5 under the gather).
- moe_out-0 vs the CPU oracle (r04_006): tok0 mad 0.0011 (correct); t1-5 mad 0.08-0.23
  (still wrong, but now a CLEAN deterministic propagation of the mm1 slot values; the
  zero-tail artifact is gone).
- Decode text = back to the round-63 baseline variant ("is used to hide the problem. The").
- The round-64/66 "correct transient / leftover slot" observations = artifacts of the
  gather-era readback corruption; the round-63 closure + round-67 %wide (mm1 computes
  wrong values for 5/6 experts; d5694d's wmma per-partition compute = the target) =
  RE-CONFIRMED on a clean pipeline.

### Files
/tmp/fix_{gu,sw,moeout}.bin, /tmp/cap_swiglu_fix.log, /tmp/cap_fixtest.log, /tmp/allrun/,
~/zaya-captures-428ab3/. Fix = commit on fix/hrx-ngl-init-order (strided-branch revert to
extent-convention + keep parent resolution). The buffer_copy strided WIP = untouched
(separate path, not exercised here).
