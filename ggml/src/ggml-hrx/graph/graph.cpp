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

// Full-graph use count for a tensor. The cgraph handed to graph_compute is a
// ggml_graph_view slice of the scheduler's full graph and SHARES the full
// graph's use_counts / visited_hash_set (ggml_graph_view aliases
// cgraph0->use_counts). A produced value whose FULL use count exceeds its
// in-slice use count is consumed by a node in a LATER split (e.g. a
// CPU-forced residual add) that reads its ggml-buffer slot after this slice
// computes. Those values must stay External (written to the ggml slot) even
// when they also have in-slice consumers; otherwise the post-slice reader gets
// a never-written slot (round 21: l_out-26 all-zeros at gen-4@0,
// split-externalization gap; node_972 externalized because it has no in-slice
// consumers).
static int32_t full_graph_use_count(const struct ggml_cgraph & graph, const ggml_tensor * tensor) {
    if (tensor == nullptr || graph.use_counts == nullptr || graph.visited_hash_set.used == nullptr) {
        return 0;
    }
    const size_t pos = ggml_hash_find(&graph.visited_hash_set, tensor);
    if (pos >= graph.visited_hash_set.size) {
        return 0;
    }
    if (!ggml_bitset_get(graph.visited_hash_set.used, pos)) {
        return 0;
    }
    return graph.use_counts[pos];
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
    auto is_layout_alias = [](ggml_op op) {
        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    };
    // consumers[V] = in-slice nodes whose srcs include V.
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> consumers;
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr) {
            result.status.log("ggml graph contains a null node");
            return result;
        }
        for (const ggml_tensor * source : node->src) {
            if (source != nullptr) {
                ++use_counts[source];
                consumers[source].push_back(node);
            }
        }
    }

    std::unordered_set<const ggml_tensor *> produced_here;
    produced_here.reserve(static_cast<size_t>(graph.n_nodes));
    for (int i = 0; i < graph.n_nodes; ++i) {
        produced_here.insert(graph.nodes[i]);
    }

    // alias_only_external: a produced value whose in-slice consumers are ALL
    // layout aliases, and none of those aliases' outputs (recursively) is
    // consumed by a real (non-alias) in-slice node, is read solely through
    // slice-exiting aliases (cross-slice readers like the CPU swiglu). Its
    // real ggml slot must be written -> External. If an alias-descendant is
    // consumed by a real in-slice op (Kcur -> permuted views -> in-slice
    // flash-attn), the value stays Transient (fused path, round 30).
    std::unordered_set<const ggml_tensor *> alias_only_external;
    for (int i = 0; i < graph.n_nodes; ++i) {
        const ggml_tensor * node = graph.nodes[i];
        if (node == nullptr || !produced_here.count(node)) {
            continue;
        }
        const auto uc = use_counts.find(node);
        if (uc == use_counts.end() || uc->second == 0) {
            continue;  // no in-slice consumer: handled by the terminal rule
        }
        bool reaches_real = false;
        {

            std::unordered_set<const ggml_tensor *> seen;
            // BFS over alias consumers starting from the node's own consumers.
            std::vector<const ggml_tensor *> frontier = consumers[node];
            while (!frontier.empty() && !reaches_real) {
                const ggml_tensor * cur = frontier.back();
                frontier.pop_back();
                if (!seen.insert(cur).second) {
                    continue;
                }
                if (!is_layout_alias(cur->op)) {
                    reaches_real = true;  // a real in-slice consumer exists
                    break;
                }
                const auto cc = consumers.find(cur);
                if (cc != consumers.end()) {
                    for (const ggml_tensor * next : cc->second) {
                        frontier.push_back(next);
                    }
                }
            }
        }
        if (!reaches_real) {
            alias_only_external.insert(node);
            if (std::getenv("GGML_HRX_TRACE_EXT")) {
                const char * nm = ggml_get_name(node);
                fprintf(stderr, "[ext] ALIAS-ONLY-EXTERNAL name=%s op=%s\n", nm ? nm : "?", ggml_op_name(node->op));
            }
        }
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

        const auto        local_use   = use_counts.find(node);
        const bool        consumed_outside =
            full_graph_use_count(graph, node) > (local_use == use_counts.end() ? 0 : local_use->second);
        if (std::getenv("GGML_HRX_TRACE_EXT")) {
            const char * nm = ggml_get_name(node);
            if (nm != nullptr && (strstr(nm, "ffn_moe_gate") != nullptr || strstr(nm, "ffn_moe_up") != nullptr ||
                                  strstr(nm, "l_out") == nm || strstr(nm, "node_") == nm)) {
                const auto  lu = use_counts.find(node);
                const auto  fu = full_graph_use_count(graph, node);
                fprintf(stderr, "[ext] name=%s ext=%d consumed_outside=%d local_use=%d full_use=%d\n",
                        nm, tensor_is_external(node, use_counts, produced_here) ? 1 : 0,
                        consumed_outside ? 1 : 0, (int)(lu == use_counts.end() ? 0 : lu->second), (int)fu);
            }
        }
        const bool alias_only = alias_only_external.count(node) != 0;
        const ValueKind output_kind =
            (tensor_is_external(node, use_counts, produced_here) || consumed_outside || alias_only)
                ? ValueKind::External
                : ValueKind::Transient;
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
