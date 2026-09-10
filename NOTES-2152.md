# #2152 — CONCAT dim-0: root cause confirmed in source; validation needs the corpus build

Status: **root cause confirmed, fix NOT yet validated** (see "Why there is no code patch here").

## Root cause (device-free, arithmetic + source)
The kernel promises the compiler something that is false for every real dim-0 concat.

`kernel-corpus/kernels/loom-libs/ops/concat_f32.loom:22` (original):
```
%count = index.assume %element_count [range(%element_count, 1, 536870912), le(%element_count, %rows_capacity)] : index
```
`dispatch_registration/common/dispatch-concat.cpp` publishes:
```
integer_parameters: rows_a = src0->ne[0], rows_b = src1->ne[0], cols = src0->ne[1]
compile_parameters: rows_capacity = rows_a + rows_b, cols_capacity = cols
element_count = (rows_a + rows_b) * cols
```
so the second clause demands `(rows_a+rows_b)*cols <= rows_a+rows_b`, i.e. **cols == 1**. Measured cases:
| case | rows_a+rows_b | cols | element_count | rows_capacity | clause |
|---|---|---|---|---|---|
| cols=6 (QKraw [1024,6]+[256,6]) | 1280 | 6 | 7680 | 1280 | false |
| conv_input | 2560 | 1280 | 3276800 | 2560 | false |

`index.assume` is a promise to the compiler, not a runtime guard, so a false condition is
UB/poison — which matches the observed split exactly: **cols=6 fires and silently corrupts,
cols=1280 AMDGPU-faults**. It also explains why both earlier layout hypotheses (flat copy;
ne0-fastest strided view) looked wrong when the kernel's ne0-fastest math is correct.

## Candidate fixes (both unvalidated)
1. delete the second clause:
   `%count = index.assume %element_count [range(%element_count, 1, 536870912)] : index`
2. give the element count its own capacity and use it:
   `config.decl @ggml.concat_f32.element_capacity ...` + `le(%element_count, %element_capacity)`,
   with the matcher publishing `element_capacity = element_count`.

## Why there is no code patch here: my hand-compile test is INCONCLUSIVE
I tried to validate either variant device-free with the built tool
`build/bin/ggml-hrx-compile-kernel` (usage: `--target --source --root --output --config k=v --workload v`).
Both variants failed the Loom bounds proof with `SUBRANGE/024` (4 diagnostics, "view_bound is %N,
maximum legal origin is <dynamic>"). **The control experiment matters: the ORIGINAL unpatched
kernel fails identically with the same flags** — so the failure is my invocation, not the patch.
Do not read this as evidence for or against either variant.

The real check is the corpus pipeline, which compiles these sources with the project's own flags:
- rebuild target `ggml-hrx-kernel-corpus` in an HRX fork build tree (it regenerates the corpus
  from the `.loom` sources), then
- exercise the zaya path with the `GGML_HRX_CONCAT_COLS` isolation gate and compare the cols=6 and
  conv_input (cols=1280) concats against a CPU/numpy reference of one captured concat.

## Tree state
The three WIP files (`.../ops/concat_f32.loom`,
`.../dispatch_registration/common/dispatch-concat.{cpp,h}`) are committed here **unmodified**.
They previously existed only as untracked files in one worktree (`~/hrx-ws/amd-hrx-graph`,
branch `fix/hrx-ngl-init-order`), i.e. one `git clean -fd` from permanent loss; that oracle-exact
tree is untouched and still holds the originals.

## Remote / push note (added 2026-09-10 at @agent-44437c's request)
This fork's `origin` is **`AMD-Ecosystem/llama.cpp`** and this account gets **403** there, so the
repo's post-commit auto-push hook FAILS SILENTLY for HRX branches. The writable remote is
`bong = https://github.com/bong-water-water-bong/llama.cpp` — push branches there
(`git push bong <branch>`), and do not read a clean `git log` as "upstreamed".

## Build note for anyone validating
The HRX/loom dependencies are NOT vendored in this repo (no submodule, no `.gitmodules`); each
build tree produces them under `<build>/ggml/src/ggml-hrx/hrx/src/ggml-hrx-deps-build`. A fresh
worktree therefore fails at `find_package(hrx)`; pointing `hrx_DIR`/`loomc_DIR` at another tree's
prebuilt config packages gets past configure but fails at Generate (`tests/CMakeLists.txt:342-357`
`target_link_libraries` targets not found — the cross-tree export set does not satisfy a fresh
tree). Warm dep trees on strixhalo at time of writing: the preserved oracle
`~/hrx-ws/amd-hrx-graph/build` and `~/wt/q35-hrx-fix/build`.

## Reference + checker for the device run (host-side, no build needed)
`~/issue-triage/concat_ref.py` (strixhalo and ryzen):
  python3 concat_ref.py --cases <outdir>                     # both known cases + meta.json
  python3 concat_ref.py --from-capture src0.f32 src1.f32 <a> <b> <c> <outdir>
  python3 concat_ref.py --check <device_out.f32> <ref.f32> [<a> <b> <c>]
Reproduces the field arithmetic exactly: cols6 7680 vs rows_capacity 1280 = **6x**; conv_input
rows 2560 (2048+512) cols 1280 -> element_count 3,276,800 vs 2,560 = **1280x**. The ratio
`element_count / rows_capacity` equals **cols**, which is the defect's signature — the false
assumption is exactly cols-times wrong. `--check` reports maxabs, mismatch count/percent, pearson,
PASS/FAIL, and names the first mismatch as (pos, ch). Self-validated: reference vs itself PASS
(pearson 1.000000) and `--from-capture` round-trips to the same reference.

## Second finding (2026-09-10): the corpus build path does NOT include this kernel
Tested in the peer's warm tree (~/wt/q35-hrx-fix, restored afterwards with digests): with
`concat_f32.loom` present in `kernel-corpus/kernels/loom-libs/ops/` (the directory CMakeLists:100
declares as GGML_HRX_LOOM_LIBS_KERNEL_CORPUS_DIR) the generated lists contain **0** occurrences of
"concat" — `kernel-corpus-sources.inc`, `kernel-corpus-catalog.inc`, `kernel-corpus-qwen.inc` — and
still 0 after `cmake -B build` re-ran cleanly and `make ggml-hrx-kernel-corpus llama-cli llama-bench`
returned rc=0. So a clean rebuild here proves NOTHING about the patched kernel: it is never compiled.
Consequence: the field report's "compiles + FIRES" came from a different route — most likely the
runtime JIT (`ggml/src/ggml-hrx/loom-jit.cpp`, `runtime/loom-kernel-jit.cpp`, env GGML_HRX_ASYNC_JIT /
GGML_HRX_DUMP_IR`). Open question put to the peer: is there a FOURTH artifact (a registration/list
edit or a manifest entry) that wires the kernel in, or is the JIT source-load path the wiring?
