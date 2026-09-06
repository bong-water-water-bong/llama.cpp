#pragma once

#include "dispatch/command-program.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/prepared-command-program-cache.h"
#include "status.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

struct GraphProgramExternalBinding {
    ValueId             value;
    const ggml_tensor * tensor = nullptr;
};

enum class GraphProgramExternalSlotKind {
    Node,
    Source,
};

struct GraphProgramExternalSlot {
    ValueId                      value;
    GraphProgramExternalSlotKind kind         = GraphProgramExternalSlotKind::Node;
    size_t                       node_index   = 0;
    int32_t                      source_index = -1;
};

struct GraphProgramMatch {
    std::vector<GraphProgramExternalBinding> external_bindings;
    Status                                   status;

    bool valid() const { return status.success(); }
};

class GraphProgram {
  public:
    GraphProgram(uint64_t                        uid,
                 std::string                     target,
                 std::unique_ptr<Graph>          graph,
                 std::unique_ptr<CommandProgram> commands,
                 std::string                     command_shape,
                 std::unordered_map<int32_t, std::string> generated_value_names);

    uint64_t uid() const { return uid_; }

    const std::string & target() const { return target_; }

    const std::string & command_shape() const { return command_shape_; }

    const Graph & graph() const { return *graph_; }

    Graph & graph() { return *graph_; }

    const CommandProgram & commands() const { return *commands_; }

    CommandProgram & commands() { return *commands_; }

    GraphProgramMatch match_current_graph(const ggml_cgraph & graph) const;
    GraphProgramMatch match_trusted_graph(const ggml_cgraph & graph, bool bind_external = true) const;
    GraphProgramMatch match_host_staging_graph(const ggml_cgraph & graph) const;

    Status capture_external_slots(const ggml_cgraph & graph, const GraphProgramMatch & match);

    bool has_prepared_program() const;
    bool can_use_prepared_fast_path(const ggml_cgraph & graph) const;

    PreparedCommandProgramCacheStats prepared_stats() const;

    PreparedCommandProgramCacheExecutionResult execute_with_result(const CommandProgramExecutionContext & context,
                                                                   const CommandProgramBindings &         bindings);

    // Env-gated (GGML_HRX_PROGRAM_DUMP=<comma-separated name substrings>)
    // post-execution dump of in-program value regions (external ggml slots +
    // transient-arena regions), keyed by program uid + per-uid execution
    // ordinal so cross-run comparisons align on (uid, ordinal) instead of the
    // scheduler's divergent run counters. Writes
    // /tmp/prg_dump/<uid>_<ordinal>_<name>.bin. Transient/generated-resource
    // bindings are named from generated_value_names_ (common.moe_routing.*);
    // the positional bind_<index> tag remains a filter alias. (eb4f0b, round
    // 57: readbacks cannot reach in-program values; this is the
    // executor-side mechanism.)
    void dump_program_values(const CommandProgramExecutionContext & context) const;

  private:
    const GraphProgramExternalSlot * find_external_slot(ValueId value) const;
    const ggml_tensor *              resolve_external_slot(const ggml_cgraph &              graph,
                                                           const GraphProgramExternalSlot & slot,
                                                           Status &                         status) const;

    uint64_t                        uid_ = 0;
    std::string                     target_;
    std::unique_ptr<Graph>          graph_;
    std::unique_ptr<CommandProgram> commands_;
    std::string                     command_shape_;
    // ValueId -> generated-resource name for plan transients that have no ggml
    // tensor and no graph Value entry (moe routing expert_table / partition_table
    // / row_debug, e.g. "common.moe_routing.*"). Threaded from the scheduler plan
    // at program build; used by the env-gated in-program dump to name values that
    // otherwise only have positional bind_<index> tags.
    std::unordered_map<int32_t, std::string> generated_value_names_;

    std::vector<GraphProgramExternalSlot> external_slots_;
    std::unordered_map<int32_t, size_t>   external_slot_by_value_;
    const ggml_tensor * const *           fast_path_nodes_ = nullptr;
    mutable std::mutex                    prepared_mutex_;
    PreparedCommandProgram                prepared_;
    RecordedCommandGraph                  recorded_;
    PreparedCommandProgramCacheStats      prepared_stats_;
    bool                                  has_prepared_ = false;
};

struct GraphProgramCacheStats {
    uint64_t builds                  = 0;
    uint64_t hits                    = 0;
    uint64_t prepared_program_builds = 0;
    uint64_t prepared_program_hits   = 0;
};

struct GraphProgramLookup {
    GraphProgram *                program = nullptr;
    std::unique_ptr<GraphProgram> uncached_program;
    GraphProgramMatch             match;
    Status                        status;

    bool valid() const { return program != nullptr && status.success() && match.valid(); }
};

struct GraphProgramSupportResult {
    bool   supported = false;
    Status status;

    bool valid() const { return supported && status.success(); }
};

class GraphProgramCache {
  public:
    bool can_execute(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target) const;

    GraphProgramSupportResult check_support(const ggml_cgraph &  graph,
                                            const KernelCorpus & corpus,
                                            const std::string &  target) const;

    GraphProgramLookup get_or_build(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target);

    GraphProgramCacheStats stats() const;

    void clear();

  private:
    GraphProgramLookup build_from_imported(const ggml_cgraph &  graph,
                                           Graph &&             imported_graph,
                                           const KernelCorpus & corpus,
                                           const std::string &  target);

    std::unique_ptr<GraphProgram> build_program_from_imported(uint64_t             uid,
                                                              Graph &&             imported_graph,
                                                              const KernelCorpus & corpus,
                                                              const std::string &  target,
                                                              Status &             errors) const;

    mutable std::mutex                                          mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<GraphProgram>> programs_;
    GraphProgram *                                              last_program_ = nullptr;
    GraphProgramCacheStats                                      stats_;
};

bool can_execute_standalone_op_as_graph(const ggml_tensor * op, const std::string & target);

}  // namespace ggml::hrx
