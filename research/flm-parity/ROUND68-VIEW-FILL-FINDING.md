
## Round 68 finding (executor lane, 428ab3) — the view materialization = stride-2 misread: first hard evidence for the view-fill theory

Within ONE run (13:18, f32twin 5-token; /tmp/allrun/): 
- mm slot (PROGRAM_DUMP of ffn_moe_gate_up-0, the PROVEN 40/40 external binding) = deterministic.
- CPU's actual gate/up reads (DUMPVIEW buffer_get captures hrx_gate0.bin/up0.bin) for t1-5 = the mm slot DECIMATED at double plane stride: cpu-plane t reads slot f32 idx t*8192 (= slot plane 2t, rows 0-2047): t0 = slot t0 (-1.4707, matches), t1 = slot-plane-2 values (0.1693...), t2 = slot-plane-4 (-0.8857...), t3-5 = ZEROS (f32 idx 24576+ = beyond the 98304B slot = never written).
- The CPU GLU = PERFECTLY self-consistent with its (mis-)reads: swiglu = silu(cpu-gate)*cpu-up rms 1e-6 for ALL 6 tokens; swiglu vs silu(slot)*slot-up rms 0.35-0.75 for t1-5.
=> whoever materializes the gate/up tensors for the CPU reads the mm output with the source plane stride of 8192 f32 (a [2048,12]-style layout) instead of 4096 f32 ([4096,6]), zero-filling planes past the end. The CPU then computes a correct GLU on mis-strided inputs. If the mm slot = the true (correct) output (round-64 numpy mad 0.003 — NOT contradicted by this evidence), the ENTIRE t1-5 divergence = this stride-2 view materialization. This = the first direct evidence for the round-66/13:00:54 "view fill" theory; my earlier "mm values wrong" inference was based on round-41 oracle heads that appear in NO capture (the CPU never receives them either).
- Files: /tmp/allrun/{10866_001_ffn_moe_gate_up-0.bin, 10868_001_ffn_moe_swiglu-0.bin, gate0.bin, up0.bin}; log /tmp/cap_all.log.
- Next: identify the materialization site (buffer_get offset/nb math in ggml-hrx.cpp or the ggml sched cross-backend view copy) + the authoritative CPU oracle to confirm the slot = correct.
