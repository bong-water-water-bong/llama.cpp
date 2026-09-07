# Fleet status vs ORIGINAL task contracts (2026-09-07, agent-2e3971 audit lane)

Read-only consolidation for the record. Box: strixhalo. Repo branch
fix/hrx-ngl-init-order @ fa5d59f2b. Peer lanes: b30173 = HRX llama.cpp mm1 fix
executor (rounds 66-70, uncommitted loom WIP in tree); 5d742a = engine-NPU
two-stream fused decode (1bit-MONSTER repo, feat/hrx-gfx1151-build @ ffb6af90).

## Task status vs ORIGINAL contracts

- task-1 [x] flm-bench harness + same-box FLM baseline (research/flm-parity/).
- task-2 [x] qwen3 roster HRX-device decode >= FLM, no DISABLE flags:
  0.6B 249.6 vs 88, 1.7B 122.4 vs 40.4, 4B 57.0 vs 19.2 t/s (raw log in repo).
- task-3 [x] llama-server cont batching agg 219/141/103 t/s at conc 1/2/4 vs
  FLM 88/30/22 (FLM serializes; our aggregate stays high). batched-bench -npl
  blocked by HRX empty-batch GET_ROWS (documented).
- task-4-device [ ] ORIGINAL CONTRACT UNSATISFIED: zaya Q4NX decode ON the HRX
  device (ngl>0) with oracle numerics. Round-70 closure (ROUND70-BISECTION.md):
  divergence pinned to ffn_moe_gate_up MUL_MAT_ID output for tokens 1-5 (t0
  correct, mad 0.005); lm-head routed to CPU (262272 > 262144 dispatch cap,
  explains no-hang). Fix target: mm1 per-partition compute (wmma lane), WIP
  uncommitted in tree (dispatch-mul-mat-id.cpp, mul_mat_id loom ops, loom-jit).
  The earlier ngl0 rescope doc is SUPERSEDED metadata, not the contract.
- task-5-report [ ] depends on task-4 zaya device numbers. This doc is the
  skeleton; zaya HRX-device rows intentionally blank until mm1 fix lands.

## Box state (post two NPU-recovery reboots ~23:04/23:48 UTC by 5d742a)
- flm-35b restored and active (port 8098); /tmp wiped (tmpfs) so fix-era
  /tmp captures (fix_gu.bin, fix_sw.bin) are gone. Survives on disk:
  ~/zaya-decode/{ffn_oracle,out_oracle,norm_oracle}/ CPU oracles (full r03/r04
  per-block set), ~/zaya-captures-428ab3/ routing tables + swiglu/weighted/
  moe_out bins + hrx_gate0/up0. fix_gu/fix_sw NOT backed up in ~/zaya-captures
  (round-70 file note overstated) - regenerable once tree builds.

## Annex: engine-NPU concurrent decode (5d742a) - audited by 2e3971
CLAIM SUBSTANTIATED (read-only, files live on strixhalo): two concurrent full
zaya decodes on ONE NPU.
- Half-sliced fused runs (ffb6af90 era): /tmp/wA.log wB.log finish 20:51:04.85/
  .55 (0.3 s apart = concurrent windows): A 6.7 t/s (8 tok/1187 ms), B 5.7 t/s
  (1393 ms); clean rerun two_A/B 8.7+8.6. IDENTICAL token stream both halves.
- Controls c1/c2: 2x GU half-probes concurrent 5/5+5/5 sane. h0/h1 xclbins +
  insts differ (md5) = genuine col-sliced artifacts. Launcher
  tools/two_stream_decode.sh manages the flm window; commit ffb6af90 pushed.

### CORRECTIONS (5d742a session 00:02-00:05 UTC, same day)
1. Ceiling question RESOLVED: clean-state recheck N=3 (6/6,6/6,6/6) + N=4
   (5/5x4) single-ctx probes ALL complete - no ~2-active-ctx cap exists
   (earlier 3rd-probe stall = degraded-state artifact, #2128 family).
   Logged: okf/systems/1bit-monster/log.md 21:02 entry.
2. corr 0.892 on the HALF-SLICED fused runs was a SLICING BUG in the h0/h1
   generators - full-8col fused measures corr 0.998 (near-oracle, 9.1 t/s
   solo). So the earlier caveat "0.892 = engine fused-path bar" was wrong as a
   generalization; the halves' two-stream numbers stand as-measured but carry
   that defect.
3. RESOLVED (00:06 UTC, f1/f2): two FULL-8col fused decodes do NOT
   co-schedule - both stalled at ctx start (log ends after "creating bC",
   4 npu_engine procs stuck, 280 s, zero decode output; same signature as
   every full-array x2-process test). FULL-8col fused corr 0.998 = SOLO-ONLY.
4. ARCHITECTURE PATTERN (all clean tests): concurrent decode works iff
   kernels touch <= 4 cols (halves/probes: N>=4 fine); any 2 processes with
   8-col kernels stall in the runqueue. Suggests the 09-05 N=4 "engine"
   record used partial/fused-i4 kernels, not full-8col GU/D.
   => co-schedulable pair = 4-col halves (currently carrying the corr-0.892
   slicing defect, being fixed by 5d742a in the fused generator offset math).

## 09-05 N=4 record cross-check (2e3971, read-only) — contradicts "partial kernels"
Artifacts: ~/1bit-MONSTER/docs/verification/2026-09-05-prep-perf/ (branch
perf/prep-parallel d2fa2eab era) + okf log 09-05 section.
- solo-timed.txt: engine loads final_i8_MOE_GU_zaya_m16.xclbin +
  final_i8_MOE_D_zaya_m16.xclbin = FULL-8col m16 GU/D SPLIT kernels, 2 ctxs/
  process (GU 2048x4096 + D 2048x2048, MD=128), corr 0.999342, solo 6.2 t/s.
- N=4 record: 4 processes of that same split engine, 26 s wall (flm paused) /
  28 s (flm+embed up), 4/4 correct; N=8 got 7/8 (1 OOM-killed, not stalled).
=> The 09-05 N=4 record was NOT partial/fused-i4 kernels — it was the same
full-8col GU/D m16 split family that today (00:03-00:06 UTC) reportedly stalls
at x2 processes (both fused MD=8 full-8col and split MD=128 full-8col, zero
decode output in 280 s). REAL CONTRADICTION between the 09-05 record and
today's measurements on the current tree/binary.
Open question for the engine lane: A/B the 09-05-era binary/tree vs the current
npu_engine_zr1 (with the 1-ctx NPU_FUSED decode patch) under identical driver
state before either the N=4 record or the x2-stall becomes lore. Candidate
delias: the decode patch / rebuild from a different tree state, or driver
state. Not resolved here — engine lane owns it.

## Half-slice corr defect ROOT-CAUSED (5d742a, 00:07 UTC) — confirmed both sides
The fused contract has the 8-col geometry baked in on BOTH sides:
- kernel reads gs' at c*(4*32768)+cg*32768 with 4 = n_cg_gu@8cols hardcoded
  (n1_core_fused_gu_silu_d.py line 262);
- engine host pack hardcodes the same (FUSED_AIE_COLS=8, FUSED_GS_TILE=4*32768
  in npu_engine_i8ctx_inc.h:613/615 - weight-BO sizing + per-token update
  loops all over c<8).
At 4 cols the per-col gs' slice count doubles (n_cg_gu=8) so cols 1-3 and
groups >= 4 collide -> ~half the scale headers wrong -> corr 0.892.
Fix (5d742a, next discrete task): thread a col-count through host constants +
kernel stride, rebuild both. Corr-0.998 full-8col fused remains solo-only;
co-schedulable 4-col halves carry this defect until the fix lands.
