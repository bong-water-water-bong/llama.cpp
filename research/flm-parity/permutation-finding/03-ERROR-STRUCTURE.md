# Error structure: gate_up heavy-scramble -> progressive dilution to moe_out (agent-f49062)

Per-channel error census (>0.1 off) down the block-0..39 ffn chain, run1 captures vs the
CPU oracle set (ffn_oracle r03_*, 6-token prefill):

  b3  t0: gate_up 77.0% | swiglu 2.8% | down 0.9% | weighted 0.0%   (mad 0.27/0.03/0.03/0.006)
  b6  t0: gate_up 89.5% | swiglu 48%  | down 65%  | weighted 0.0%   (mad 0.63/0.15/0.18/0.008)
  b20 t0: gate_up 97.1% | swiglu 92%  | down 98%  | weighted 94%    (mad 2.5/2.8/5.9/1.1)

Facts:
- The gate_up mm output IS heavily channel-scrambled in bad blocks (NOT a sparse error).
- Yet the moe_out/weighted output = EXACT for b3 and b6 (0.0% bad, mad 0.006-0.008) =>
  the corruption is absorbed along GLU -> down -> weighted for those blocks. For b20 it is
  not (94% bad at weighted). Silu/GLU is elementwise and cannot absorb a channel scramble
  (silu(g[Pc])*u[Pc] != silu(g)*u at Pc != c), so the observed absorption is NOT explainable
  by the dumped gate_up content flowing through the dumped swiglu/down content. Either the
  consumers saw cleaner data than the dumps (timing/alias), or the downstream dumps are not
  what the residual consumed.
- Both read paths (PROGRAM_DUMP d2h AND the strided buffer_get the GLU views trace, run
  GGML_HRX_TRACE_STRIDED=1) return the SAME scrambled content for the same block/token, so
  the two paths agree with each other at any one time.
- Sequence per block in the trace: [prgdump] mm dump fires immediately before the block's
  [strided] gate/up reads (e.g. cap-strided.log lines 1895/1898 for b3) - same content.
- Blocks 38-39 = the ONLY blocks whose moe_out/layer_out feed the lm-head with real damage
  (logits: 9079 collapses 17.47 -> -0.28).

Remaining models (both still open):
(a) arena-slot content = genuinely scrambled by the mm kernel write in affected
    (block, token) cells, and the consumption path (GLU -> down -> weighted) re-derives
    correct values because those ops actually read a different (clean) copy - the view
    materialization/alias story of round-69 (gallocr may give views their own span-copied
    slots; parents hold the truth). Would require the swiglu/down program to consume the
    clean copy - not yet verified which binding the down mm's input points at for a bad
    block at run time.
(b) dumps read the arena at times when later ring-slot writes (odd blocks share
    2752512; even>=2 share 2883584) have partially overwritten the region, and the
    consumers read earlier/cleaner. Deterministic because submission order is fixed.
    Argues AGAINST (b): the b3 gate_up dump content is 99.97% pure b3 values (sorted
    corr 0.9997) - no measurable foreign values.

Next step suggestion for the continuation: capture the moe_out/layer_out (r04 oracle
exists only through block 35; r03 chain ends at weighted) for blocks 36-39 with a fresh
chain run, and verify WHERE between weighted and the final norm the tail corruption
enters (weighted dump for b38/39 vs a numpy-reconstructed oracle from the b38/39 ffn
inputs). Also re-run the strided trace + dumps for b38/39 to compare both paths at the
tail.
