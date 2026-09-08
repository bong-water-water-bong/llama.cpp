# Kernel round findings (2026-09-08 02:30+, agent-ec8072) - post-reboot, f49062 dead

## Ground truth re-established
- Device single-seq decode (conv CPU, SSM_CONV unclaimed): 6.4 t/s (156-160 ms/token,
  zgreedy_t per-token timing) - matches llama-bench 6.98. f49062's preserved comment
  claiming "12.65 claimed vs 15.15 CPU-forced t/s, verified oracle-exact" is NOT
  reproducible by direct measurement; those numbers appear to be from an unverifiable
  state (possibly ngl0-CPU = 16.08). Report rows (6.98-7.14) STAND.

## SSM_CONV kernel (a0ee433aa, f49062's, preserved): BROKEN at loom indexing
- Runtime: "Loom source indexing failed" (no diagnostics) on ggml_ssm_conv_f32
  compile. With the eager claim re-enabled for A/B, decode FAILS (kernel can't
  prepare). The kernel never compiled at runtime in this tree; f49062's comment
  claim-verification is unsubstantiated. Source structure reviewed line-by-line
  vs working kernels (get_rows/rmsnorm_binary) - idioms match; root cause of the
  no-diagnostics index failure NOT yet found. Reverted the claim; tree back to
  the oracle-exact preserved state.

## CONCAT kernel (mine, untracked WIP): progressed to compile + fire, faults
New ggml_concat_f32 loom kernel + dispatch (common/concat) built through the whole
pipeline. Debugging sequence: (1) manifest file-table = "files" not "sources";
(2) header include path ../dispatch-registry.h; (3) DispatchRegistration = 6 fields;
(4) workload args must match kernel root params; (5) dim-0 concat memory layout =
ne0-FASTEST (output[pos + (a+b)*ch], NOT a flat [src0|src1] block - the flat-copy
version compiled but CORRUPTED decode); (6) index.remu not a loom op -> index.rem.
Final state: kernel compiles + the dispatch fires (rows_a/rows_b/cols params) but
decode hits an AMDGPU memory fault (kernel or binding bug - unresolved; suspects:
a claimed concat shape the kernel mishandles, or the concat-output HRX->CPU d2h
path). Files: ggml/src/ggml-hrx/.../ops/concat_f32.loom + dispatch-concat.{cpp,h}
(untracked WIP, not wired).

## Next round (when taken up)
1. Fix ssm_conv_f32.loom indexing (get the loomc no-diagnostic failure diagnosed -
   possibly a source feature the indexer dislikes; compare against a build that
   indexes by bisecting the source).
2. Debug the concat AMDGPU fault (isolate which concat shape faults; check the
   kernel math against a CPU numpy ref of a claimed concat; verify the output
   binding/HRX->CPU d2h for concat outputs).
Both are pre-conditions for the launch-collapse conv-speed path; neither is a
quick fix. Ground-truth speed stands at ~6.4-7 t/s until the collapse lands.
## CONCAT kernel isolation results (2026-09-08 05:30, agent-ec8072) - WIP, needs loom-tooling round

Follow-on to KERNEL-ROUND-2026-09-08.md. Isolation A/B with the claim gated by
GGML_HRX_CONCAT_COLS:

- cols=6 concats (QKraw [1024,6]+[256,6] etc., ALL-HRX inputs): kernel compiles +
  fires (dumps prove CONCAT_QKraw execution) but decode CORRUPT (tok0=48873/236751,
  no fault). Two memory-layout hypotheses tested: (a) flat [src0|src1] block copy
  - wrong; (b) ne0-fastest strided with view<[cols]x[rows]> [ch,pos] - still wrong.
- The earlier all-concats AMDGPU fault = attributed to the conv_input concat
  (cols=1280, src0 = CPU-pinned recurrent state) - CPU-input staging/binding for
  newly claimed dispatches suspected (round-16e class).

Remaining suspect set (not yet discriminated):
1. ggml concat compute for THIS graph may not be plain dim-0 row stacking (zaya
   concats feed reshape/transpose chains; prefill vs decode shapes differ), or
2. the dispatch matcher's rows_a/rows_b/cols derivation is off for the graph
   values (e.g., src ne dims vs ggml concat op semantics), or
3. the concat OUTPUT value handoff HRX->CPU corrupts (concats feed the CPU conv
   region; dump tooling only captures CPU-executed ops so device-side output
   verification needs a device-side read-back harness).

NEXT (needs a loomc-diagnostics-capable session): verify the kernel math against
a CPU numpy ref of ONE captured concat (device dump unavailable for HRX-executed
nodes - add a GGML_HRX-side dump or run the concat CPU-forced and diff the sched
placement), then re-test. Both ssm_conv (indexing) and concat (values/fault)
block the launch-collapse conv-speed path. Files: ops/concat_f32.loom +
dispatch-concat.{cpp,h} = untracked WIP.
