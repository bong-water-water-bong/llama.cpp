## Addendum 3 (same session): conv input capture attempt + remaining state
- Added a convdbg hook (GGML_HRX_CONVDUMP, uid==10846) in dump_program_values that
  d2h's the prepared command bindings post-execution. Result: the conv input buffer
  (b2) content == the output content after the run (buffer reuse; the 35840B scratch is
  overwritten) - post-run input capture is contaminated; pre-execution capture would
  need a hook inside the command executor/fallback.
- Confirmed at the execution-context level: the conv program's data bindings carry
  NULL device buffers (only the weight is non-null) - the fallback path manages its
  own host staging, outside the window-scan and dump machinery.
- The conv output row-0 rms = 1.30 (NOT ~0) rules out a zeroed-state input under the
  state-first valid-conv convention: either the state the conv sees is non-zero
  (cache written by an earlier execution in the session, or the assembly reads a
  non-zero region), or the row mapping is not [state||tokens].
- Static: recurrent cache = CPU-pinned + buffer_clear(0) at alloc (llama-memory-
  recurrent.cpp). If the first llama_decode = the only execution before the conv, the
  state must be zero - unless the sched's concat assembly reads the wrong region.
- All instrumentation is live in the tree (convdbg hook, retuned windowscan, publish
  trace, full-act rd_act). The loom corpus has NO conv kernel; the conv = CPU fallback.

Next continuation: hook the command executor to capture the conv input AT LAUNCH
(before execution), or compare the effective conv input solved from the output
(y-b = w0*x_i + w1*x_{i+1} inversion) against the QKraw to quantify the corruption
pattern of the cross-backend qk copy.
