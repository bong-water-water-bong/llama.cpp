#include "graph-executor.h"
#include <unordered_set>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "backend-buffer-binding.h"
#include "hrx-interop-utils.h"
#include "ggml-impl.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <utility>
#include <vector>

namespace ggml::hrx {

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

Status GraphExecutor::context_valid_for_graph_programs() const {
    Status status;
    if (context_.device == nullptr) {
        status.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        status.log("missing HRX target");
    }
    return status;
}

Status GraphExecutor::context_valid_for_execution() const {
    return context_valid_for_graph_programs();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    result.status = context_valid_for_graph_programs();
    if (!result.status.success()) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.status.append(support.status);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    Status                             status;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        CommandProgramBinding binding;
        binding.value = external.value;
        const bool resolved = ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding);
        if (resolved) {
            binding.buffer     = value_binding.buffer;
            binding.host_data  = value_binding.host_data;
            binding.offset     = value_binding.offset;
            binding.length     = value_binding.length;
            binding.identity   = value_binding.identity;
            binding.generation = value_binding.generation;
            binding.capacity   = value_binding.capacity;
            binding.weight     = value_binding.weight;
        } else {
            status.log("external value %d is not bound", external.value.value);
        }
        // [eb4f0b 2026-09-06] per-external trace: which list entries lose their
        // CPU-produced activations (embd / node_973 etc.) vs leaves/weights.
        if (getenv("GGML_HRX_DUMP_WRITEBIND")) {
            const ggml_tensor * t = external.tensor;
            ggml_backend_buffer_t tb = t ? (t->view_src ? t->view_src->buffer : t->buffer) : nullptr;
            fprintf(stderr,
                    "[hrxext] value=%d name=%s resolved=%d nbytes=%zu buf=%p hostbuf=%d data=%p base=%p\n",
                    external.value.value, t ? ggml_get_name(t) : "?", resolved ? 1 : 0,
                    t ? (size_t) ggml_nbytes(t) : 0u, (void*) tb,
                    tb ? (ggml_backend_buffer_is_host(tb) ? 1 : 0) : -1,
                    (void*) (t ? t->data : nullptr),
                    (void*) (tb ? ggml_backend_buffer_get_base(tb) : nullptr));
            fflush(stderr);
        }
        if (getenv("GGML_HRX_TRACE_1336") &&
            (external.value.value == 1336 || (binding.length == 20480 && binding.offset == 1572864))) {
            fprintf(stderr,
                    "[trA] ext value=%d buf=%p(iree) host=%p off=%zu len=%zu gen=%llu id=%llu tensor=%s wrapper=%p\n",
                    external.value.value, (void*)binding.buffer, (void*)binding.host_data, binding.offset,
                    binding.length, (unsigned long long)binding.generation, (unsigned long long)binding.identity,
                    external.tensor ? ggml_get_name(external.tensor) : "?", (void*)external.tensor);
            fflush(stderr);
        }
        bindings.push_back(binding);
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), status);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    if (graph.n_nodes == 0) {
        result.code = GGML_STATUS_SUCCESS;
        return result;
    }
    result.status = context_valid_for_execution();
    if (!result.status.success()) {
        return result;
    }

    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    GraphProgramLookup   lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    if (!lookup.valid()) {
        result.status.append(lookup.status);
        result.status.append(lookup.match.status);
        if (result.status.success()) {
            result.status.log("build HRX graph program failed");
        }
        return result;
    }

    const bool use_graph_prepared =
        !lookup.program->has_prepared_program() || lookup.program->can_use_prepared_fast_path(graph);
    GraphProgramMatch binding_match = std::move(lookup.match);
    if (use_graph_prepared && lookup.program->has_prepared_program()) {
        binding_match = lookup.program->match_host_staging_graph(graph);
        if (!binding_match.valid()) {
            result.status.append(binding_match.status);
            return result;
        }
    }

    if (getenv("GGML_HRX_DUMP_WRITEBIND")) {
        // Leaf-vs-external discriminator: which graph leaves (cross-split /
        // CPU-produced inputs consumed by this HRX graph) are NOT in the
        // binding_match external list. If embd / node_973 show as
        // consumed-but-unlisted -> cache-hit slot mismatch confirmed.
        std::vector<const ggml_tensor *> leaves;
        for (int i = 0; i < graph.n_nodes; i++) {
            const ggml_tensor * node = graph.nodes[i];
            if (node == nullptr) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                const ggml_tensor * src = node->src[j];
                if (src == nullptr) continue;
                const ggml_tensor * root = src->view_src ? src->view_src : src;
                bool in_graph = false;
                for (int k = 0; k < graph.n_nodes && !in_graph; k++) {
                    const ggml_tensor * n2 = graph.nodes[k];
                    if (n2 == src || (n2->view_src && n2->view_src == root)) in_graph = true;
                }
                if (!in_graph) {
                    bool seen = false;
                    for (const ggml_tensor * L : leaves) if (L == src || L == root) seen = true;
                    if (!seen) leaves.push_back(root);
                }
            }
        }
        for (const ggml_tensor * leaf : leaves) {
            bool is_external = false;
            for (const GraphProgramExternalBinding & ext : binding_match.external_bindings) {
                const ggml_tensor * rt = ext.tensor->view_src ? ext.tensor->view_src : ext.tensor;
                if (rt == leaf) { is_external = true; break; }
            }
            ggml_backend_buffer_t lb = leaf ? leaf->buffer : nullptr;
            fprintf(stderr, "[hrxleaf] %s external=%d buf=%p data=%p nbytes=%zu\n",
                    leaf ? ggml_get_name(leaf) : "?", is_external ? 1 : 0, (void*) lb,
                    (void*) (leaf ? leaf->data : nullptr), leaf ? (size_t) ggml_nbytes(leaf) : 0u);
        }
        fprintf(stderr, "[hrxleaf] externals_total=%zu leaves_total=%zu\n",
                binding_match.external_bindings.size(), leaves.size());
        fflush(stderr);
    }

    CommandProgramBindings bindings = bind_external_value_buffers(binding_match);
    if (!bindings.valid()) {
        result.status.append(bindings.status);
        return result;
    }
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.kernel_executables,
        &context_.transient_arena,
        &context_.host_transfers,
        &context_.host_weights,
    };
    // [eb4f0b 2026-09-06] GGML_HRX_DOUBLE_EXECUTE probe: the canary's FIRST
    // execution of a freshly built program loses writebacks (KV + terminal
    // externals read zero post-sync at n_past=0; later executions of the same
    // program land — rounds 17a/17b). If a second launch of the same program
    // with the same bindings lands its writes, the loss is first-launch
    // writeback (warm/barrier fix) rather than a semantic error. Safe for
    // dense qwen (KV slot at the same n_past re-written identically); NOT for
    // recurrent-state models (state would double-apply) — probe only.
    bool double_exec = getenv("GGML_HRX_DOUBLE_EXECUTE") != nullptr;
    bool do_second = false;
    if (double_exec && use_graph_prepared) {
        static std::unordered_set<uint64_t> warmed_uids;
        const uint64_t uid = lookup.program->uid();
        do_second = warmed_uids.insert(uid).second;
    }
    const PreparedCommandProgramCacheExecutionResult execution =
        [&]() -> PreparedCommandProgramCacheExecutionResult {
        auto run_once = [&]() {
            return use_graph_prepared ?
                lookup.program->execute_with_result(execution_context, bindings) :
                context_.prepared_programs.execute_with_result(execution_context, lookup.program->uid(),
                                                               lookup.program->command_shape(),
                                                               lookup.program->commands(), bindings);
        };
        if (do_second) {
            PreparedCommandProgramCacheExecutionResult warm = run_once();
            if (!warm.success) { return warm; }
            fprintf(stderr, "[dblexec] warmed uid=%llu once; running again\n",
                    (unsigned long long) lookup.program->uid());
        }
        return run_once();
    }();
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }


    // [b30173 2026-09-06] GGML_HRX_WINDOWSCAN probe: post-execute forensic on the
    // FIRST program execution (batch-0 prefill). node_972's terminal write reads
    // exact zero at its bound slot (gen-4 @1572864) though the kernel is configured
    // identically to working mms. H1: the wmma store lands at a DELTA address
    // (stale offset/base in the wmma launch path). Dump +-4MB around the slot and
    // scan 20480B-aligned windows for plausible mm-output patterns (nonzero,
    // not-constant, sane f32). Also persist the window to /tmp/window.bin so an
    // offline CPU-oracle correlation can run without re-running the device.
    if (std::getenv("GGML_HRX_WINDOWSCAN") && execution.success) {
        static int window_scans = 0;
        if (window_scans == 0) {
            window_scans = 1;
            const CommandProgramBinding * slot = nullptr;
            for (const CommandProgramBinding & b : bindings.bindings()) {
                if (b.buffer != nullptr && b.length == 20480 && !b.weight) {
                    slot = &b;  // node_972's 5-token prefill output
                    break;
                }
            }
            if (slot != nullptr) {
                fprintf(stderr, "[win] slot buf=%p off=%zu len=%zu identity=%llu gen=%llu\n",
                        (void*)slot->buffer, slot->offset, slot->length,
                        (unsigned long long)slot->identity, (unsigned long long)slot->generation);
                const size_t half = 4194304;  // +-4MB
                const size_t win_base = slot->offset > half ? slot->offset - half : 0;
                const size_t win_size = (slot->offset - win_base) + half + slot->length;
                std::vector<uint8_t> win(win_size);
                if (ErrorResult error = take_status(hrx_synchronous_d2h(
                        context_.device->device, slot->buffer, win_base, win.data(), win_size))) {
                    fprintf(stderr, "[win] d2h window failed: %s\n", error->c_str());
                } else {
                    FILE * wf = fopen("/tmp/window.bin", "wb");
                    if (wf) { fwrite(win.data(), 1, win_size, wf); fclose(wf); }
                    fprintf(stderr, "[win] window saved /tmp/window.bin base=%zu size=%zu\n", win_base, win_size);
                    const size_t step = slot->length;  // 20480
                    size_t hits = 0, cand = 0;
                    for (size_t base = 0; base + step <= win_size; base += step, ++cand) {
                        const float * f = (const float *)(win.data() + base);
                        const size_t nf = step / sizeof(float);
                        float sum = 0.f, maxv = 0.f, minv = 0.f;
                        bool any = false, all_same = true;
                        sum = f[0]; maxv = minv = f[0];
                        for (size_t i = 1; i < nf; ++i) {
                            sum += f[i];
                            if (f[i] != f[0]) all_same = false;
                            if (f[i] > maxv) maxv = f[i];
                            if (f[i] < minv) minv = f[i];
                            if (f[i] != 0.f) any = true;
                        }
                        const float mean = sum / (float)nf;
                        float var = 0.f;
                        for (size_t i = 0; i < nf; ++i) var += (f[i]-mean)*(f[i]-mean);
                        var /= (float)nf;
                        if (any && !all_same && var > 1e-6f && var < 1e6f) {
                            const long long delta = (long long)base - (long long)(slot->offset - win_base);
                            if (hits < 20) {
                                fprintf(stderr, "[win] HIT cand=%zu delta=%+lld f32=%.5g %.5g %.5g %.5g %.5g %.5g var=%.3g\n",
                                        cand, delta, f[0], f[1], f[2], f[3], f[4], f[5], var);
                            }
                            ++hits;
                        }
                    }
                    fprintf(stderr, "[win] scan done: %zu candidates, %zu plausible hits\n", cand, hits);
                }
                fflush(stderr);
            }
        }
    }
    result.code = GGML_STATUS_SUCCESS;
    return result;
}

}  // namespace ggml::hrx
