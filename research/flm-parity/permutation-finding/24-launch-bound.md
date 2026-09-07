
# Zaya speed: launch-bound conclusion + stable-uid landed (agent-f49062)

## New result
- Stable structural split-graph uids (c7b726ffd): the HRX program cache now
  HITS (~640/execution, was 0). Zaya stays oracle-exact; decode time
  UNCHANGED (178 ms/tok) => the per-subgraph cost is the launch+sync round
  trip (~0.17 ms each), NOT the program build. The cache fix was necessary
  (removes per-token rebuilds + bounds the program map) but not sufficient.
- Regression checks: qwen3-0.6B tg128 49.2 t/s (unchanged); qwen3moe-30B
  clean (10.8 t/s decode = CPU-ADD-bound per c5ac20's 30B finding).

## The speed lever (updated)
The decode's ~640 subgraphs/token are forced by the CPU-island ops per block
(SSM conv, grouped conv, zaya rope, GLU, CONCATs, residual ADDs after the
exclusion, router softmax/argsort). Reducing the subgraph count requires
kernels/claims for those ops; the highest-leverage order:
1. Residual-ADD ordering fix (round-16f) - re-claims ~5-8 ops/block AND
   restores the dense-roster 242.9 t/s; needs the EDA-recurrent-ADD handling.
2. SSM conv + grouped conv loom kernels (2/block + merge the attention chain).
3. CONCAT/rope/GLU coverage for the remaining islands.
Target: an all-HRX decode like qwen3 (1-2 subgraphs/token -> 249 t/s scale).

## Branch state (fix/hrx-ngl-init-order)
HEAD c7b726ffd. Zaya: oracle-exact 5.6-5.7 t/s. Roster: 48.9 (242.9 behind
GGML_HRX_ALLOW_ADD). 30B/35B: correct under the blanket exclusion (35B-A3B =
separate 4-dim-op model-class, unsupported on HRX - stretch item).
