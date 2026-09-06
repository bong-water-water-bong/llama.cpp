#include "graph.h"

#include "ggml-impl.h"

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ggml::hrx {
namespace {

// A value is TRANSIENT only when it is BOTH produced and consumed inside this
// node set (computed by an in-set node, read by another). Anything consumed
// here but produced OUTSIDE the set (a prior CPU/other-backend split computed
// it, e.g. zaya/qwen CPU GET_ROWS feeding an HRX layer) must stay EXTERNAL so
// it binds from its real ggml buffer (host_data staging / device copy) instead
// of being freshly zero-allocated in the transient arena. Previously only
// consumption was checked, so CPU-produced activations consumed by the set were
// misclassified Transient -> kernels read zeros (round 16r).
static bool tensor_is_external(const ggml_tensor *                                  tensor,
                               const std::unordered_map<const ggml_tensor *, int> & use_counts,
                               const std::unordered_set<const ggml_tensor *> &      produced_here) {
    if (tensor->op == GGML_OP_NONE) {
        return true;
    }
    const auto found = use_counts.find(tensor);
    if (found == use_counts.end() || found->second == 0) {
        return true;
    }
    // consumed by an in-set node: transient only if also produced here
    return produced_here.find(tensor) == produced_here.end();
}

}  // namespace

GraphIndex GraphIndex::build(const Graph & graph) {
    GraphIndex                     index;
    const std::vector<GraphNode> & nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode & node = nodes[i];
        index.node_indices_.emplace(&node, i);
        index.producers_.emplace(node.output.value, &node);
        for (ValueId input : node.inputs) {
            index.consumers_[input.value].push_back(&node);
        }
    }
    return index;
}

const GraphNode * GraphIndex::producer(ValueId value) const {
    const auto found = producers_.find(value.value);
    return found == producers_.end() ? nullptr : found->second;
}

const std::vector<const GraphNode *> & GraphIndex::consumers(ValueId value) const {
    static const std::vector<const GraphNode *> empty;
    const auto                                  found = consumers_.find(value.value);
    return found == consumers_.end() ? empty : found->second;
}

bool GraphIndex::has_single_consumer(ValueId value) const {
    return consumers(value).size() == 1;
}

bool GraphIndex::node_index(const GraphNode * node, size_t & index) const {
    const auto found = node_indices_.find(node);
    if (found == node_indices_.end()) {
        return false;
    }
    index = found->second;
    return true;
}

Graph::Graph(const Graph & other) : values_(other.values_), nodes_(other.nodes_) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(const Graph & other) {
    if (this == &other) {
        return *this;
    }
    values_ = other.values_;
    nodes_  = other.nodes_;
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

Graph::Graph(Graph && other) : values_(std::move(other.values_)), nodes_(std::move(other.nodes_)) {
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
}

Graph & Graph::operator=(Graph && other) {
    if (this == &other) {
        return *this;
    }
    values_ = std::move(other.values_);
    nodes_  = std::move(other.nodes_);
    index_.reset();
    if (other.has_index()) {
        index_ = GraphIndex::build(*this);
    }
    return *this;
}

GraphNode & Graph::add_node(ggml_op op, ValueId output, std::vector<ValueId> inputs) {
    index_.reset();
    GraphNode node;
    node.op     = op;
    node.output = output;
    node.inputs = std::move(inputs);
    nodes_.push_back(std::move(node));
    return nodes_.back();
}

Status Graph::build_index() {
    index_ = GraphIndex::build(*this);
    return {};
}

const GraphIndex & Graph::index() const {
    assert(index_.has_value());
    return *index_;
}

GraphImportResult import_ggml_graph(const ggml_cgraph & graph) {
    GraphImportResult                            result;
    std::unordered_map<const ggml_tensor *, int> use_counts;
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr) {
            result.status.log("ggml graph contains a null node");
            return result;
        }
        for (const ggml_tensor * source : node->src) {
            if (source != nullptr) {
                ++use_counts[source];
            }
        }
    }

    std::unordered_set<const ggml_tensor *> produced_here;
    produced_here.reserve(static_cast<size_t>(graph.n_nodes));
    for (int i = 0; i < graph.n_nodes; ++i) {
        produced_here.insert(graph.nodes[i]);
    }

    ValueMap & values = result.graph.values();
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor *  node = graph.nodes[i];
        std::vector<ValueId> inputs;
        for (const ggml_tensor * source : node->src) {
            if (source == nullptr) {
                continue;
            }
            const ValueKind kind =
                tensor_is_external(source, use_counts, produced_here) ? ValueKind::External : ValueKind::Transient;
            inputs.push_back(values.get_or_add_tensor_value(source, kind));
        }

        const ValueKind output_kind =
            tensor_is_external(node, use_counts, produced_here) ? ValueKind::External : ValueKind::Transient;
        const ValueId   output      = values.get_or_add_tensor_value(node, output_kind);
        GraphNode &     graph_node  = result.graph.add_node(node->op, output, std::move(inputs));
        graph_node.params           = import_op_params(*node);
        // Pure relayout ops never produce data: when the ggml view_src chain
        // does not directly name the in-graph input (cont->reshape->view
        // chains, e.g. zaya flash-attn Vcur), alias the output to its single
        // input when it is a contiguous same-element relayout. Otherwise the
        // dispatch scheduler cannot elide the op and no VIEW/RESHAPE dispatch
        // exists -> "unsupported HRX node". (1bit-MONSTER zaya port)
        if (graph_node.inputs.size() == 1 && is_layout_alias_op(node->op)) {
            values.force_alias_relayout(output, graph_node.inputs[0], node->view_offs);
        }
    }

    result.status.append(result.graph.build_index());
    return result;
}

bool is_layout_alias_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

bool is_layout_alias_node(const Graph & graph, const GraphNode & node) {
    return is_layout_alias_op(node.op) && node.inputs.size() == 1 &&
           graph.values().same_storage(node.output, node.inputs[0]);
}

}  // namespace ggml::hrx
