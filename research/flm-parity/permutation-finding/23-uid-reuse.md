
# Stable-uid cache reuse: crashes on recorded-binding mismatch (reverted)

## Experiment
Replaced the sched's fresh-per-call split-graph uid (ggml-backend.cpp:1494,
ggml_graph_next_uid) with a structural hash of each split (op + ne + src-ops
+ op_params over all nodes) so the HRX graph-program cache's uid-keyed
fast-path would hit across decode tokens (it never does today; ~640 program
builds/token = ~180 ms of the 179 ms/token decode).

## Result
- zaya decode fails at step 0:
  "command 0 ... binding weight value=0 origin=GraphValue access=Read
   range=[0, 49152) is outside runtime binding length 8192"
  = a recorded program (6-token prefill binding span 49152) was replayed for a
  1-token decode subgraph whose runtime binding is 8192: the structural hash
  did not separate a prefill subgraph from a decode subgraph (or the reuse
  path binds the wrong generation of the same-shaped value), and the existing
  fast-path verification (match_trusted: node count + sentinels + external
  slots) does not compare binding byte lengths.
- Reverted to the fresh-uid baseline (working, oracle-exact, 5.6 t/s).

## Conclusion
The executor's cross-token program reuse is not safe yet: the recorded-graph
replay rebinds externals per call but a stale/incorrect program can pass the
trusted match and fail at execution with a binding-length error. A correct
reuse requires: (a) uid/hash keying that includes the binding sizes, (b) the
full structural match (value metadata) before reuse, and (c) understanding
why a prefill-span program is being replayed for the decode at all (the
subgraph compositions overlap between the prefill and the decode phases).
The 179 ms/token decode at 5.6 t/s remains the verified baseline; the
transient-arena + recorded-replay machinery (round-17 class) is the actual
fix surface, alongside the loom SSM-conv kernels for the structural
de-fragmentation.
