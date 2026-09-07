## Addendum 4: conv qk input copy = CORRECT - corruption not in the qk d2h
- Added a post-read trace to the plain (non-strided) HRX buffer_get (GGML_HRX_TRACE_GET,
  ggml-hrx.cpp, prints name/offset/size + 8 values AFTER the copy).
- The CPU-side copy of Kraw-0 (the conv/qk_mean consumer input) = my validated numpy
  Kraw to f16 noise (mad 0.011 over the head). => the cross-backend copy of the conv's
  qk source is CORRECT (at least for the k part; q part analogous - validated dumps).
- The state cache (CPU, zeroed at alloc) should also be correct.
=> The conv stage wrongness is NOT a corrupt qk d2h and NOT (per addendum 3) a zeroed
   state => the remaining suspects: (a) the conv kernel's own semantics/execution (the
   command is Kernel-kind with NULL exec-level data buffers - an unlocated code path
   between "no loom conv kernel" and the produced QK_dw), or (b) the state rows are NOT
   what the zeroed cache holds (rows 5-6 output garbage rms 81-85 requires garbage in
   the input tail under the authoritative ops.cpp semantics: out[t,c] = sum_k w[k,c] *
   x[t+k,c] over the 8-row [state2 + qk6] input).
- Next: locate the actual executor of the 10846 Kernel-kind command (search the runtime
   for how a Kernel command with no loom symbol executes; the prepared-command execution
   path), then capture its input at launch.
