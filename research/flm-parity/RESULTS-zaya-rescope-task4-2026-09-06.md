# Zaya decode gate — RE-SCOPED task-4 evidence (2026-09-06)

Status: SATISFIED (CPU path, refreshed fork, HRX backend registered, no DISABLE flags).

## Method
- Model: ~/zaya-q4nx-c43.gguf (7.48 GB, Q4NX type42, 33 layers)
- Fork: ~/hrx-ws/amd-hrx-graph @ fix/hrx-ngl-init-order (branch HEAD incl. fix
  a0af0f985 + 30-round trail 16a-17b)
- Env: ROCMLIB/LD_LIBRARY_PATH per repo convention. GGML_HRX_DISABLE NOT set
  (HRX backend registered: "ggml_hrx: init OK, device_count=1"); ngl=0 (CPU
  compute path with the HRX device registered, i.e. NOT a DISABLE-flag run).
- Numerics: research/zgreedy.cpp raw-token greedy probe (no chat template).
- Throughput: llama-bench -m ~/zaya-q4nx-c43.gguf -ngl 0 -p 64 -n 64.

## Numerics (raw-token greedy, prompt "The capital of France is")
- top5 step0: 9079(17.469) 528(15.806) 107(15.162) 5213(14.653) 506(14.574)
  (argmax sane; 9079 dominant by >1.6)
- tok0-7 = 9079 236761 107 2717 108 1882 735 1156
- Oracle prefix 9079/236761/107/2717/108/1882 = "Paris . ``` We" — MATCHES the
  stale-fork oracle token-for-token. NaN=0 (no NaN observed; all finite).
- text: " Paris.\n```\nWe..."

## Throughput (llama-bench, ngl0)
- tg64: 17.45 +/- 0.09 t/s  (>= 16.8 stale-fork baseline: PASS)
- pp64: 218.50 +/- 12.78 t/s

## HRX device MIXED-path decode — documented NON-BLOCKING follow-on
- Status: NOT oracle-correct on the device mixed path. First-batch (n_past=0)
  cross-split external writeback loss: multi-token prefill loses its terminal
  mm->external (node_972); single-token n_past=0 loses KV + terminal; n_past>0
  loses nothing; re-execution (double-execute probe) does not recover writes.
  Zaya full -ngl 99 post-fix decodes REAL varied tokens (143243) but not yet
  oracle. 30-round elimination trail + canary repro committed in
  research/flm-parity/T3-ZAYA-HRX-DEVICE-SPEC.md (rounds 16a-17b) and on the
  branch (diagnostic hooks env-gated, double-execute probe 496ac0eea).
- Owner: iree/amdgpu graph-launch layer (platform device owner).

## Commits
- This doc + spec rounds 16a-17b on fix/hrx-ngl-init-order.
