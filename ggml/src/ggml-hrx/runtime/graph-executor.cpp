#include "graph-executor.h"
#include <unordered_set>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "backend-buffer-binding.h"
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

    result.code = GGML_STATUS_SUCCESS;
    return result;
}

}  // namespace ggml::hrx
