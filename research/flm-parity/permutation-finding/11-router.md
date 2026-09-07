# ROUTER DIVERGENCE = THE ROOT CAUSE (agent-f49062, 2026-09-07)

## The finding
Block-0 prefill, 6 tokens. With the FULLY validated numpy machinery (embd ->
input scale/bias -> attn_norm -> projections -> d2h -> CPU conv -> attention), the
oracle gate_up rows (r03_004) match W[e]x[t] EXACTLY (mad 0.00000) under the routing:
  CPU routing (from the oracle): t0->e3, t1->e15, t2->e10, t3->e12, t4->e3, t5->e0
The HRX gate_up dump rows match W[e]x[t] EXACTLY (mad 0.006-0.009 = f16) under:
  HRX routing: t0->e3, t1->e5, t2->e12, t3->e7, t4->e2, t5->e4
Only t0 agrees (both e3). The mm kernel, the norms, the projections, the attention,
the conv island = ALL compute correctly. The MoE expert SELECTION differs between
the HRX run and the CPU oracle run for tokens 1-5.

## What this means
- Every prior "corruption" observation (x wrong for t>=1, gate_up rows wrong, swiglu/
  down cascades, conv-island garbage) was the downstream consequence of the mm
  computing the WRONG EXPERT projections (W[e5]x1 etc. vs the CPU's W[e15]x1...).
  The rows were "wrong vs the oracle" because the oracle = the CPU's (different)
  routing - the numerics were correct all along for the HRX's own routing.
- The round-70 "expert table" (e2,e3,e4,e5,e7,e12) = the HRX-side table = its own
  selection. Round-64's "experts match CPU" was evidently wrong or for another run.
- The routing (logits -> argsort -> ids -> expert/partition tables -> mul_mat_id)
  is the divergence site. The CPU router internals exist (r04_004 logits 17x6,
  r04_005 probs); the CPU selection (from the oracle!) is not the raw logits argmax
  (t0 argmax = e12 but routed = e3) - the zaya router uses a more complex rule
  (EDA/17th channel) - but whatever the rule, the HRX run's tables do not match the
  CPU run's selection.

## Next probes
1. Capture the HRX-side routing tables for block 0 (the expert_table content = the
   round-70 tuple source) and the router inputs the dispatch consumed (the argsort/
   ids value feeding dispatch-mul-mat-id's route_ids) - compare against the CPU's
   argsort/top-1 (r03_002 was a broken zeros capture; regenerate or use r04_005).
2. Determine where the zaya router's selection is computed (CPU-side softmax/argsort
   per the #2147 analysis vs HRX moe-router dispatches) and how the ids cross into
   the mm's routing bundle - the divergence likely = a CPU->HRX ids/argsort boundary
   corruption (the round-16e "CPU->HRX boundary corrupt" family) or a table-builder
   reading the wrong argsort row.
3. Verify with the routing fixed (force the CPU routing) whether the decode becomes
   oracle - this isolates routing from any remaining numeric issue.
