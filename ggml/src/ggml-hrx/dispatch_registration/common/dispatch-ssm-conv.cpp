// Dispatch for the SSM (Mamba-style) 2-tap depthwise conv, f32.
// Contract (CONV-CLOSED-FORM.md, agent-ec8072): input [ncs = n_t+2, d_inner]
// contiguous (pos, c) at pos + c*ncs; weight [2, d_inner] tap-interleaved
// w[tap][c] = file[2c+tap]; output [d_inner, n_t] at c + t*d_inner;
// y[c][t] = w0[c]x[t][c] + w1[c]x[t+1][c] (bias is a later ADD node).
#include "dispatch-ssm-conv.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kSsmConvF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_ssm_conv_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool match_ssm_conv_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SSM_CONV || node->inputs.size() != 2) {
        return false;
    }

    const Value * output = graph_value(context.graph, node->output);
    const Value * input  = graph_value(context.graph, node->inputs[0]);
    const Value * weight = graph_value(context.graph, node->inputs[1]);
    if (output == nullptr || input == nullptr || weight == nullptr) {
        return false;
    }

    if (input->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input->contiguous || !weight->contiguous || !output->contiguous || output->element_count <= 0) {
        return false;
    }

    // Contract shapes: input [ncs, d_inner, 1], weight [2, d_inner], output [d_inner, n_t, 1]
    // (ggml semantics: n_t = ncs - d_conv + 1 = ncs - 1 for 2 taps)
    const int64_t d_conv  = weight->ne[0];
    const int64_t d_inner = weight->ne[1];
    if (d_conv != 2 || d_inner <= 0 || d_inner > 262144 || input->ne[1] != d_inner ||
        input->ne[0] <= d_conv || input->ne[0] > 2050 || output->ne[0] != d_inner ||
        output->ne[1] != input->ne[0] - d_conv + 1 || output->ne[2] != 1 || output->ne[3] != 1) {
        return false;
    }
    const int64_t token_count = output->ne[1];
    if (token_count <= 0 || token_count > 2048) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSsmConvF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", token_count);
    dispatch.kernel.integer_parameters.emplace("d_inner", d_inner);
    dispatch.kernel.compile_parameters.emplace("ggml.ssm_conv_f32.token_capacity",
                                               std::to_string(token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.ssm_conv_f32.d_inner_capacity",
                                               std::to_string(d_inner));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ weight->id, 0, weight->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_ssm_conv_dispatches(DispatchRegistryBuilder & builder) {
    builder.add({
        "common.ssm_conv_f32",
        GGML_OP_SSM_CONV,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_ssm_conv_f32_dispatch,
    });
}

}  // namespace ggml::hrx
