# HRX zero-output notes — device/base/arena trace (2026-09-06)

Author: ryzen agent (device-level pass). Companion to the round-16a..16p trail.
Symptom: qwen3-0.6B ngl99 with `GGML_HRX_CPU_OPS=GET_ROWS` (any CPU op mid-graph)
— HRX dispatches run (GPU busy 6–7% in the decode window) but all outputs read
back exact zeros (tok0-7=0), no error, no abort, exit 0.

## 1. Device layer (verified, rules OUT kernel/driver)

- dmesg is **completely clean** on both amdgpu and amdxdna during the repro
  (multiple runs, `dmesg -c` deltas): no GPU faults/resets/VM errors, no
  amdxdna errors. The repro opens `/dev/dri/renderD128` + `/dev/kfd` at init
  (+ accel0 briefly = GTT carve through amdxdna; that explains round-15
  `amdxdna_gem_shmem ENOMEM` under memory pressure, but no NPU hwctx is ever
  created and nothing executes on the NPU).
- gpu_busy_percent spikes 6–7% in the ~80 ms decode window (the whole repro is
  a 0.5 s / 8-token run — sample tightly or you watch a dead process).
- Conclusion: kernels dispatch and execute fault-free on the iGPU; writes
  vanish at the **userspace address/binding layer**, not in any driver.

## 2. TransientArena lifecycle (runtime/transient-arena.cpp/.h)

- Grow-only allocator. `ensure_capacity_locked`: on growth it stream-syncs,
  `hrx_buffer_release(buffer_)`, allocates a NEW device buffer, bumps
  `allocation_id_` (per-arena monotonic counter = the "gen"; gen=4 ⇒ 3
  in-session reallocs ⇒ real handle churn). `clear()` releases + resets id.
- Consumers bind from a `TransientArenaAllocationRef {buffer_, capacity_,
  allocation_id_}` snapshot; the executor tracks
  `bound_transient_arena_allocation_id` and rebinds ALL transient bindings
  when it changes (command-program-executor.cpp:985-1014,
  `bind_prepared_command_program_transients`).
- Arena buffers do NOT carry a ggml `context->generation` (that counter is for
  ggml backend buffers, set once at create from `g_allocation_generation`).
  Do not key arena validity on `context->generation`.

## 3. Fake-base offset math (backend-buffer-binding.cpp, ggml-hrx.cpp)

- `offset = tensor->data - context->base`; device buffers `base =
  GGML_HRX_FAKE_PTR_BASE (0x1000)` ⇒ offsets are pure fake-address deltas,
  STABLE across arena reallocs. Host-visible buffers use the real mapped host
  ptr as base.
- `resolve_value_buffer` has two paths:
  - HRX buffer (incl. HRX0_HOST, host-visible): `binding.buffer =
    context->buffer`, offset, `identity = context->identity`, `generation =
    context->generation`.
  - plain host buffer (ggml-cpu): `binding.host_data = base`, host_offset,
    `generation = 1`, identity = hash(buffer/base addrs). → **a plain
    ggml-cpu activation CAN resolve** via this path (valid binding), so a
    resolve failure is NOT automatic for embd-in-ggml-cpu.
- `bind_external_value_buffers` (graph-executor.cpp:49-75): iterates
  `match.external_bindings`, resolves each; on resolve FAILURE it logs
  "external value %d is not bound" and **still pushes an empty (buffer=null)
  binding** — kernels then read/write an unbound slot ⇒ exact zeros.

## 4. External-slot membership (graph-program-cache.cpp)

- `external_slots_` entries are {value, kind(Node|Source), node_index,
  source_index} — references into the graph NODE array. Node kind → the node
  itself; Source kind → that node's `src[source_index]`.
- `match_trusted_graph` resolves slots against the CURRENT graph and builds
  `external_bindings` from `external_slots_` (plus host-staging at :245-256).
- Round-16p symptom (missing CPU-produced activations in the binding lists,
  e.g. split#3 no node_973/974, split#1 no embd) is therefore either:
  (a) the slot never entered `external_slots_` at build time, or
  (b) it is a slot but `resolve_external_slot`/`resolve_value_buffer` fails at
      match time.

## 5. leaves/mask (bind) vs embd (missing) — buffer-type angle

At import time leaves/mask live in the **HRX0_HOST buft** (host-visible HRX,
device path, `direct_host_binding`), embd in a **plain ggml-cpu buffer** (host
fallback path). Both paths can produce valid bindings, so buffer type alone
should NOT cause (b); the discriminating check is whether the CPU-produced
activation (an INTERNAL node — output of an unclaimed op) is walked into
`external_slots_` at all. Leaves are graph inputs (node-array leaves); embd is
an internal node → suspect (a): slot construction walks claimed-op inputs but
only registers inputs it can classify (leaves/weights/mask), dropping
internal-CPU-output nodes.

Check for the dumps: does embd appear in `external_bindings` as an
unbound/null-buffer entry ((b), resolve failure) or is it absent entirely
((a), slot never built)?

## 6. Pre-eliminated by round 16k

Cache/replay staleness (bindings_hash gaps, recorded-descriptor replay) is NOT
the defect: always-fresh prepare (no cache, no replay, fresh lease + fresh
binding table) still yields zeros from the first token, and the missing-input
symptom is present at first BUILD (prefill split#1). Pure first-record defect.
