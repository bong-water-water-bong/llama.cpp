
# Zaya ngl-sweep: mixed-mode overhead is offload-fraction-independent (agent-f49062)

## Measurement (zgreedy_t, 16 decode steps, oracle-exact at every working ngl)
- ngl=1:  5.43 t/s (184.1 ms/tok)   <- ONE layer offloaded = the full slowdown
- ngl=99: 5.82 t/s (171.7 ms/tok)
- ngl=10/20/30: CRASH in llama_context::sched_reserve (the mixed-ngl buffer
  sizing bug - the fix/hrx-compute-buffer-resize lane territory).
- ngl=0 (pure CPU): 14.4 t/s (69.2 ms) - the only fast config.

## Interpretation
With ANY HRX layer present, the decode pays ~110 ms/token extra even though
39/40 layers run on the CPU. The llama backend-sched's mixed mode executes
every CPU island as its own subgraph (per-layer conv/staging/ADD islands ->
~400-600 CPU subgraphs/token), each with a thread-pool launch + barrier, and
every HRX island as its own device subgraph (~640/token) with launch+sync.
The overhead is dominated by the SUBGRAPH COUNT on BOTH sides, not the
offload fraction. The pure-CPU single-subgraph run avoids all of it.

## Consequence for the 16.8 t/s device target
Reducing only the HRX-side count (loom conv kernels) is not enough: the
CPU-side islands must shrink too, or the mixed-mode CPU subgraph launches
still dominate. The levers, in order:
1. The residual-ADD ordering fix (round-16f) re-claims ~5-8 ADDs/block on the
   HRX side -> removes the same number of CPU-side islands.
2. SSM conv + grouped conv loom kernels -> removes the biggest islands (both
   sides) and lets the whole attention chain form one contiguous HRX run.
3. CONCAT/rope/GLU coverage for the remaining per-block islands.
4. Alternatively investigate the llama sched's mixed-mode CPU subgraph
   handling (thread-pool launch per subgraph) - fewer/wider CPU subgraphs
   would help even before the kernels land.

## Branch state
HEAD c7b726ffd (stable split uids) + docs up to 24. Zaya oracle-exact 5.6-5.8
t/s at ngl 1 and 99; CPU-only 14.4 (llama-bench tg64 17.45 vs FLM 16.8).
