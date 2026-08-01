#include "llama-kv-cache-paged-scorer.h"

#include "llama-impl.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Heuristic scorer
// ---------------------------------------------------------------------------

llama_kv_paged_heuristic_scorer::llama_kv_paged_heuristic_scorer() {
}

std::vector<float> llama_kv_paged_heuristic_scorer::score(
    uint32_t n_chunks, uint32_t pool_capacity_chunks) const {

    if (n_chunks == 0) return {};

    // Simple heuristic: score each chunk based on distance from the end.
    // The last chunks (most recent tokens) get the highest scores.
    // This approximates LRU behavior with a smooth decay.
    //
    // Score formula: score(c) = sigmoid((n_chunks - c - pool_capacity/2) / (pool_capacity/4))
    // This gives:
    //   - The last pool_capacity/2 chunks: scores near 1.0
    //   - The first n_chunks - pool_capacity chunks: scores near 0.0
    //   - A smooth transition in between

    std::vector<float> scores(n_chunks, 0.0f);
    const float mid   = (float)n_chunks - (float)pool_capacity_chunks * 0.5f;
    const float scale = (float)pool_capacity_chunks * 0.25f;
    if (scale < 1.0f) {
        // Pool is tiny relative to context: just score the last pool_capacity chunks
        for (uint32_t c = 0; c < n_chunks; ++c) {
            scores[c] = (c >= n_chunks - pool_capacity_chunks) ? 1.0f : 0.0f;
        }
    } else {
        for (uint32_t c = 0; c < n_chunks; ++c) {
            const float x = ((float)c - mid) / scale;
            scores[c] = 1.0f / (1.0f + std::exp(-x));
        }
    }

    return scores;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<llama_kv_paged_scorer_i> llama_kv_paged_scorer_create(
    const char * drafter_model_path,
    int          n_gpu_layers) {

    (void) n_gpu_layers; // unused until drafter scorer is implemented

    if (drafter_model_path && strlen(drafter_model_path) > 0) {
        LLAMA_LOG_INFO("%s: drafter model scorer requested but not yet implemented, "
                       "falling back to heuristic scorer\n", __func__);
    }

    auto scorer = std::make_unique<llama_kv_paged_heuristic_scorer>();
    LLAMA_LOG_INFO("%s: created heuristic KV cache scorer\n", __func__);
    return scorer;
}
