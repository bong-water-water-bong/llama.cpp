// Dispatch for the GGML_OP_CONCAT dim-0 (row-stacking) concat of two 2D f32 inputs.
// Contract: src0 [rows_a, cols] + src1 [rows_b, cols] -> output [rows_a+rows_b, cols],
// all contiguous f32. Covers the zaya cca_conv_input concat (state+token rows) and
// the QKraw/Vcur row concats -> keeps the conv region device-side. (agent-ec8072)
#include "dispatch-concat.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kConcatF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_concat_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool match_concat_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2) {
        return false;
    }

    const Value * output = graph_value(context.graph, node->output);
    const Value * src0   = graph_value(context.graph, node->inputs[0]);
    const Value * src1   = graph_value(context.graph, node->inputs[1]);
    if (output == nullptr || src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !src0->contiguous || !src1->contiguous || !output->contiguous) {
        return false;
    }
    // Dim-0 concat of two 2D matrices with a shared column count.
    if (src0->ne[1] != src1->ne[1] || output->ne[1] != src0->ne[1] || output->ne[0] != src0->ne[0] + src1->ne[0]) {
        return false;
    }
    const int64_t rows_a = src0->ne[0];
    const int64_t rows_b = src1->ne[0];
    const int64_t cols   = src0->ne[1];
    if (rows_a <= 0 || rows_b <= 0 || cols <= 0 || rows_a > 4096 || rows_b > 4096 || cols > 65536) {
        return false;
    }
    if (src0->ne[2] != 1 || src1->ne[2] != 1 || output->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[3] != 1 ||
        output->ne[3] != 1) {
        return false;
    }

    // #2152 (c): a device dispatch must not be claimed for host-resident inputs —
    // ValueBufferBinding carries both forms and host data needs residency/staging
    // before execution. The cca_conv_input concat (src0 = CPU-pinned recurrent
    // state) reached this dispatch and produced the AMDGPU fault.
    auto device_bindable = [](const Value * v) {
        return v->buffer.has_value() && v->buffer->buffer != nullptr && v->buffer->host_data == nullptr;
    };
    if (!device_bindable(src0) || !device_bindable(src1) || !device_bindable(output)) {
        return false;
    }

    const int64_t element_count = (rows_a + rows_b) * cols;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kConcatF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", element_count);
    dispatch.kernel.integer_parameters.emplace("rows_a", rows_a);
    dispatch.kernel.integer_parameters.emplace("rows_b", rows_b);
    dispatch.kernel.integer_parameters.emplace("cols", cols);
    dispatch.kernel.compile_parameters.emplace("ggml.concat_f32.rows_capacity", std::to_string(rows_a + rows_b));
    dispatch.kernel.compile_parameters.emplace("ggml.concat_f32.cols_capacity", std::to_string(cols));
    dispatch.bindings.push_back({ src0->id, 0, src0->byte_count });
    dispatch.bindings.push_back({ src1->id, 0, src1->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_concat_dispatches(DispatchRegistryBuilder & builder) {
    builder.add({
        "common.concat_f32",
        GGML_OP_CONCAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_concat_f32_dispatch,
    });
}

}  // namespace ggml::hrx
