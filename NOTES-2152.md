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
