#pragma once

#include "llama-kv-cache-paged.h"

#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Heuristic scorer: chunk importance by recency + decay curve.
// Blended with attention feedback from the main model's last layer.
// ---------------------------------------------------------------------------
class llama_kv_paged_heuristic_scorer : public llama_kv_paged_scorer_i {
public:
    llama_kv_paged_heuristic_scorer();
    std::vector<float> score(uint32_t n_chunks, uint32_t pool_capacity_chunks) const override;
};

// ---------------------------------------------------------------------------
// Factory: create the best available scorer.
// If a drafter model path is given, creates a model-based scorer (future).
// Otherwise returns a heuristic scorer.
// ---------------------------------------------------------------------------
std::unique_ptr<llama_kv_paged_scorer_i> llama_kv_paged_scorer_create(
    const char * drafter_model_path = nullptr,
    int          n_gpu_layers       = 0);
