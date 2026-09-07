## Addendum 6: conv kernel id + upload trace findings
- The conv command kernel_id = -1056636216 (0xC1123AC8) - not resolvable to any
  ggml_-named kernel in the loom-libs manifest / catalog inc by FNV-1a-64 of the name.
  The kernel_id hashing source text is unknown; the actual conv executor remains
  unidentified (no loom conv source exists anywhere in the corpus).
- buffer_set (h2d) post-write trace (GGML_HRX_TRACE_SET): the CPU->HRX uploads of the
  conv outputs are visible ("CPU#QK_grp-0 (permuted)#0" 30720B etc). The uploaded
  values == the HRX dumps (mad 2e-5) => the corruption is in the CPU-side data BEFORE
  the upload (the CPU conv island computed wrong values) OR in the conv input assembly.
- The Kraw CPU-side copy (d2h) = validated correct (addendum 4). So the conv island
  consumed correct qk and produced wrong output => the remaining candidates: (a) the
  state rows staged into the concat are not zeros, (b) the concat layout/order differs
  from [state2 || qk6], or (c) the conv kernel executes with different semantics than
  the ggml ops.cpp reference (unidentified executor).
- All tracing hooks live: convdbg (kernel_id + bindings), GGML_HRX_TRACE_SET/GET
  (post-read/write values), publish trace, full-act rd_act.
- Next: identify the hash_text input for kernel_id resolution (the command builder in
  graph-program-cache/dispatch that assigns the SSM_CONV kernel id), or capture the
  conv input at launch inside the executor before the NULL resolution.
