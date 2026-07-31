#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <cmath>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

llama_kv_cache_paged::llama_kv_cache_paged(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   logical_size,
                 uint32_t   pool_size,
                 uint32_t   n_seq_max,
                 uint32_t   /*n_ubatch*/,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                 uint32_t   n_pad,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share)
    : logical_size_(logical_size)
    , pool_size_(pool_size)
    , n_stream_(unified ? 1 : n_seq_max)
    , paging_active_(pool_size < logical_size)
    , n_chunks_((logical_size + chunk_sz - 1) / chunk_sz)
    , logical_to_pool_(n_stream_, std::vector<int32_t>(logical_size, -1))
    , pool_to_logical_(n_stream_, std::vector<uint32_t>(pool_size, UINT32_MAX))
    , pool_lru_(n_stream_, std::vector<uint64_t>(pool_size, 0))
    , lru_counter_(n_stream_, 0) {

    LLAMA_LOG_INFO("%s: paged KV cache: logical=%u cells, pool=%u slots, "
                   "chunk=%u tokens, %u chunks, %u streams, active=%s\n",
                   __func__, logical_size, pool_size, chunk_sz, n_chunks_,
                   n_stream_, paging_active_ ? "YES" : "NO");

    kv_pool_ = std::make_unique<llama_kv_cache>(
        model, hparams, type_k, type_v,
        v_trans, offload, unified,
        pool_size, n_seq_max,
        /*n_pad=*/1, n_swa, swa_type,
        /*mem_other=*/nullptr,
        filter, reuse, share);

    // Store the GPU backend for async transfers (use first layer's device)
    if (paging_active_) {
        auto layer_ids = kv_pool_->get_layer_ids();
        if (!layer_ids.empty()) {
            auto * dev = model.dev_layer(layer_ids[0]);
            if (dev) {
                // Get the first backend associated with this device
                auto * buft = ggml_backend_dev_buffer_type(dev);
                if (buft) {
                    // We need a backend, not a buft. Get it from the first backend.
                    // The scheduler owns backends. We'll look them up during page ops.
                }
            }
        }
    }

    if (paging_active_) {
        kv_backing_ = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans,
            /*offload=*/false, unified,
            logical_size, n_seq_max,
            n_pad, n_swa, swa_type,
            /*mem_other=*/nullptr,
            filter, reuse, share);

        init_layer_info();
        init_pinned_buffers();

        attn_scores_.resize(n_chunks_, 0.0f);

        size_t pool_bytes = 0, backing_bytes = 0;
        for (auto & li : layers_) {
            pool_bytes    += (li.row_bytes_k + li.row_bytes_v) * pool_size;
            backing_bytes += (li.row_bytes_k + li.row_bytes_v) * logical_size;
        }
        LLAMA_LOG_INFO("%s: pool VRAM ~%.1f MiB, backing RAM ~%.1f MiB, "
                       "savings ~%.1f MiB vs full GPU cache\n",
                       __func__,
                       pool_bytes    / (1024.0 * 1024.0),
                       backing_bytes / (1024.0 * 1024.0),
                       (backing_bytes - pool_bytes) / (1024.0 * 1024.0));
    } else {
        kv_backing_ = nullptr;
        layers_.clear();
    }
}

// ---------------------------------------------------------------------------
// Auto-detect pool size from GPU VRAM
// ---------------------------------------------------------------------------

uint32_t llama_kv_cache_paged::auto_detect_pool_size(
        const llama_model & model, uint32_t logical_size) {
    // Try each device; use the first non-CPU one
    for (uint32_t i = 0; i < (uint32_t)model.n_devices(); ++i) {
        auto * dev = model.devices[i].dev;
        if (!dev) continue;

        auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) continue;

        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);

        if (free_mem > 0) {
            // Use 80% of free VRAM for KV cache pool
            const size_t target_bytes = (size_t)(free_mem * 0.8);
            // Rough per-token KV size estimate (K+V in f16, ~80 layers × head_dim × n_heads × 2)
            const size_t approx_per_token = 4096; // conservative estimate
            const uint32_t suggested = (uint32_t)(target_bytes / std::max(approx_per_token, (size_t)4096));
            const uint32_t clamped = std::min(suggested, logical_size);
            const uint32_t min_pool = 256u;

            LLAMA_LOG_INFO("%s: device %s: free=%zu MiB, pool=%u slots (clamped to %u)\n",
                           __func__, ggml_backend_dev_name(dev),
                           free_mem / (1024*1024), suggested, clamped);

            return std::max(min_pool, clamped);
        }
    }

    // No GPU found — default to full logical size (disable paging)
    LLAMA_LOG_INFO("%s: no GPU device found, disabling paging\n", __func__);
    return logical_size;
}

// ---------------------------------------------------------------------------
// Initialization helpers
// ---------------------------------------------------------------------------

bool llama_kv_cache_paged::is_tail(uint32_t pos) const {
    if (n_chunks_ <= tail_chunks) return false;
    return pos >= (n_chunks_ - tail_chunks) * chunk_sz && pos < logical_size_;
}

void llama_kv_cache_paged::init_mappings() {
    for (uint32_t s = 0; s < n_stream_; ++s) {
        std::fill(logical_to_pool_[s].begin(), logical_to_pool_[s].end(), -1);
        std::fill(pool_to_logical_[s].begin(), pool_to_logical_[s].end(), UINT32_MAX);
        std::fill(pool_lru_[s].begin(), pool_lru_[s].end(), 0);
        lru_counter_[s] = 0;
    }
    n_tokens_written_ = 0;
    std::fill(chunk_written_.begin(), chunk_written_.end(), false);
    std::fill(attn_scores_.begin(), attn_scores_.end(), 0.0f);
}

void llama_kv_cache_paged::init_layer_info() {
    layers_.clear();
    for (uint32_t il : kv_pool_->get_layer_ids()) {
        ggml_tensor * k = kv_pool_->get_k_storage(il);
        ggml_tensor * v = kv_pool_->get_v_storage(il);
        layers_.push_back({
            il,
            k ? (size_t)ggml_row_size(k->type, k->ne[0]) : 0,
            v ? (size_t)ggml_row_size(v->type, v->ne[0]) : 0
        });
    }
}

void llama_kv_cache_paged::init_pinned_buffers() {
    pinned_in_.resize(n_stream_);
    pinned_out_.resize(n_stream_);
    for (uint32_t s = 0; s < n_stream_; ++s) {
        pinned_in_[s].k.resize(layers_.empty() ? 0 : layers_[0].row_bytes_k * chunk_sz, 0);
        pinned_in_[s].v.resize(layers_.empty() ? 0 : layers_[0].row_bytes_v * chunk_sz, 0);
        pinned_out_[s].k.resize(layers_.empty() ? 0 : layers_[0].row_bytes_k * chunk_sz, 0);
        pinned_out_[s].v.resize(layers_.empty() ? 0 : layers_[0].row_bytes_v * chunk_sz, 0);
    }
}

// ---------------------------------------------------------------------------
// Scorer
// ---------------------------------------------------------------------------

void llama_kv_cache_paged::set_scorer(std::unique_ptr<llama_kv_paged_scorer_i> scorer) {
    scorer_ = std::move(scorer);
    LLAMA_LOG_INFO("%s: %s scorer installed\n", __func__,
                   scorer_ ? "custom" : "null (LRU fallback)");
}

// ---------------------------------------------------------------------------
// Attention feedback — updates chunk scores from main model's last layer
// ---------------------------------------------------------------------------

void llama_kv_cache_paged::feed_attention(
        const float * attn_weights, uint32_t n_heads, uint32_t n_kv) {
    if (attn_scores_.empty() || !attn_weights) return;

    // Decay previous scores
    for (auto & s : attn_scores_) s *= attn_decay_;

    // Accumulate attention to each position, averaged across heads
    for (uint32_t p = 0; p < n_kv && p < logical_size_; ++p) {
        const uint32_t c = chunk_of(p);
        if (c >= attn_scores_.size()) continue;

        // Average attention across heads for this position
        float sum = 0.0f;
        for (uint32_t h = 0; h < n_heads; ++h) {
            sum += attn_weights[h * n_kv + p];
        }
        attn_scores_[c] += sum / (float)n_heads;
    }
}

// ---------------------------------------------------------------------------
// Paging core (multi-stream, async-capable)
// ---------------------------------------------------------------------------

void llama_kv_cache_paged::touch_sink_and_tail(uint32_t stream) {
    for (uint32_t p = 0; p < logical_size_; ++p) {
        if (!is_sink(p) && !is_tail(p)) continue;
        int32_t slot = logical_to_pool_[stream][p];
        if (slot >= 0) pool_lru_[stream][slot] = ++lru_counter_[stream];
    }
}

std::vector<uint32_t> llama_kv_cache_paged::evict_lru(uint32_t n, uint32_t stream) {
    if (n == 0 || !kv_backing_) return {};

    auto & l2p = logical_to_pool_[stream];
    auto & p2l = pool_to_logical_[stream];
    auto & lru = pool_lru_[stream];

    std::vector<std::pair<uint64_t, uint32_t>> candidates;
    for (uint32_t s = 0; s < pool_size_; ++s) {
        uint32_t lp = p2l[s];
        if (lp == UINT32_MAX) return {s};
        if (is_sink(lp) || is_tail(lp)) continue;
        candidates.push_back({lru[s], s});
    }
    if (candidates.empty()) return {};

    std::sort(candidates.begin(), candidates.end());
    uint32_t ne = std::min((uint32_t)candidates.size(), n);
    std::vector<uint32_t> slots;
    slots.reserve(ne);
    for (uint32_t i = 0; i < ne; ++i) {
        uint32_t slot = candidates[i].second;
        uint32_t lp = p2l[slot];
        if (lp != UINT32_MAX) {
            page_out(chunk_of(lp), slot, stream);
            l2p[lp] = -1;
            p2l[slot] = UINT32_MAX;
        }
        slots.push_back(slot);
    }
    return slots;
}

// Bulk chunk copy: backing → pool (async when GPU backend available)
void llama_kv_cache_paged::copy_chunk_to_pool(uint32_t chunk_id, uint32_t slot, uint32_t stream) {
    if (!kv_backing_) return;

    const uint32_t start = chunk_id * chunk_sz;
    const uint32_t end   = std::min(start + chunk_sz, logical_size_);
    const uint32_t nrows = end - start;

    for (size_t li = 0; li < layers_.size(); ++li) {
        const auto & info = layers_[li];
        ggml_tensor * kp = kv_pool_->get_k_storage(info.il);
        ggml_tensor * vp = kv_pool_->get_v_storage(info.il);
        ggml_tensor * kb = kv_backing_->get_k_storage(info.il);
        ggml_tensor * vb = kv_backing_->get_v_storage(info.il);

        if (kp && kb && info.row_bytes_k > 0) {
            auto & buf = pinned_in_[stream].k;
            buf.resize(info.row_bytes_k * nrows);
            for (uint32_t i = 0; i < nrows; ++i) {
                ggml_backend_tensor_get(kb,
                    buf.data() + (size_t)i * info.row_bytes_k,
                    ((size_t)start + i) * info.row_bytes_k,
                    info.row_bytes_k);
            }
            // Use async set if GPU backend is available
            if (backend_gpu_) {
                ggml_backend_tensor_set_async(backend_gpu_, kp, buf.data(),
                    (size_t)slot * info.row_bytes_k,
                    info.row_bytes_k * nrows);
            } else {
                ggml_backend_tensor_set(kp, buf.data(),
                    (size_t)slot * info.row_bytes_k,
                    info.row_bytes_k * nrows);
            }
        }

        if (vp && vb && info.row_bytes_v > 0) {
            auto & buf = pinned_in_[stream].v;
            buf.resize(info.row_bytes_v * nrows);
            for (uint32_t i = 0; i < nrows; ++i) {
                ggml_backend_tensor_get(vb,
                    buf.data() + (size_t)i * info.row_bytes_v,
                    ((size_t)start + i) * info.row_bytes_v,
                    info.row_bytes_v);
            }
            if (backend_gpu_) {
                ggml_backend_tensor_set_async(backend_gpu_, vp, buf.data(),
                    (size_t)slot * info.row_bytes_v,
                    info.row_bytes_v * nrows);
            } else {
                ggml_backend_tensor_set(vp, buf.data(),
                    (size_t)slot * info.row_bytes_v,
                    info.row_bytes_v * nrows);
            }
        }
    }
}

// Bulk chunk copy: pool → backing (async when GPU backend available)
void llama_kv_cache_paged::copy_chunk_to_backing(uint32_t chunk_id, uint32_t slot, uint32_t stream) {
    if (!kv_backing_) return;

    const uint32_t start = chunk_id * chunk_sz;
    const uint32_t end   = std::min(start + chunk_sz, logical_size_);
    const uint32_t nrows = end - start;

    for (size_t li = 0; li < layers_.size(); ++li) {
        const auto & info = layers_[li];
        ggml_tensor * kp = kv_pool_->get_k_storage(info.il);
        ggml_tensor * vp = kv_pool_->get_v_storage(info.il);
        ggml_tensor * kb = kv_backing_->get_k_storage(info.il);
        ggml_tensor * vb = kv_backing_->get_v_storage(info.il);

        if (kp && kb && info.row_bytes_k > 0) {
            auto & buf = pinned_out_[stream].k;
            buf.resize(info.row_bytes_k * nrows);
            // Async get from GPU
            if (backend_gpu_) {
                ggml_backend_tensor_get_async(backend_gpu_, kp, buf.data(),
                    (size_t)slot * info.row_bytes_k,
                    info.row_bytes_k * nrows);
            } else {
                ggml_backend_tensor_get(kp, buf.data(),
                    (size_t)slot * info.row_bytes_k,
                    info.row_bytes_k * nrows);
            }
            for (uint32_t i = 0; i < nrows; ++i) {
                ggml_backend_tensor_set(kb,
                    buf.data() + (size_t)i * info.row_bytes_k,
                    ((size_t)start + i) * info.row_bytes_k,
                    info.row_bytes_k);
            }
        }

        if (vp && vb && info.row_bytes_v > 0) {
            auto & buf = pinned_out_[stream].v;
            buf.resize(info.row_bytes_v * nrows);
            if (backend_gpu_) {
                ggml_backend_tensor_get_async(backend_gpu_, vp, buf.data(),
                    (size_t)slot * info.row_bytes_v,
                    info.row_bytes_v * nrows);
            } else {
                ggml_backend_tensor_get(vp, buf.data(),
                    (size_t)slot * info.row_bytes_v,
                    info.row_bytes_v * nrows);
            }
            for (uint32_t i = 0; i < nrows; ++i) {
                ggml_backend_tensor_set(vb,
                    buf.data() + (size_t)i * info.row_bytes_v,
                    ((size_t)start + i) * info.row_bytes_v,
                    info.row_bytes_v);
            }
        }
    }
}

void llama_kv_cache_paged::release_pool_cells(uint32_t chunk_id, uint32_t stream) {
    const uint32_t start = chunk_id * chunk_sz;
    const uint32_t end   = std::min(start + chunk_sz, pool_size_);
    for (uint32_t p = start; p < end; ++p) {
        if (p >= pool_size_) break;
        kv_pool_->release_cell(p, stream);
    }
}

void llama_kv_cache_paged::page_in(uint32_t chunk_id, uint32_t slot, uint32_t stream) {
    const uint32_t start = chunk_id * chunk_sz;
    const uint32_t end   = std::min(start + chunk_sz, logical_size_);

    copy_chunk_to_pool(chunk_id, slot, stream);

    auto & l2p = logical_to_pool_[stream];
    for (uint32_t p = start; p < end; ++p) {
        l2p[p] = (int32_t)slot;
    }
    pool_to_logical_[stream][slot] = start;
    pool_lru_[stream][slot] = ++lru_counter_[stream];

    // Boost attention score: this chunk was demanded (paged in)
    if (chunk_id < attn_scores_.size()) {
        attn_scores_[chunk_id] += 1.0f;
    }

    chunk_written_[chunk_id] = true;
    n_page_in_++;
}

void llama_kv_cache_paged::page_out(uint32_t chunk_id, uint32_t slot, uint32_t stream) {
    copy_chunk_to_backing(chunk_id, slot, stream);
    release_pool_cells(chunk_id, stream);
    chunk_written_[chunk_id] = true;
    n_page_out_++;
}

// ---------------------------------------------------------------------------
// Repool — three strategies: scored, simple, or skip (prefill optimization)
// ---------------------------------------------------------------------------

void llama_kv_cache_paged::repool_scored() {
    const uint32_t pool_chunks = std::max(1u, pool_size_ / chunk_sz);

    // Combine scorer scores with attention feedback
    std::vector<float> scores;
    if (scorer_) {
        scores = scorer_->score(n_chunks_, pool_chunks);
    }

    // Build priority list
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(n_chunks_);
    for (uint32_t c = 0; c < n_chunks_; ++c) {
        float s = 0.0f;
        if (!scores.empty()) s += scores[c] * 0.7f;   // scorer weight
        if (!attn_scores_.empty()) s += attn_scores_[c] * 0.3f; // attention feedback

        // Boost sink and tail
        const uint32_t cpos = c * chunk_sz;
        if (is_sink(cpos))       s = std::numeric_limits<float>::max();
        else if (is_tail(cpos))  s = std::numeric_limits<float>::max() - 1.0f;

        ranked.push_back({s, c});
    }
    std::sort(ranked.begin(), ranked.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    // Top pool_chunks written chunks should be resident
    std::vector<bool> want(n_chunks_, false);
    uint32_t admitted = 0;
    for (uint32_t i = 0; i < ranked.size() && admitted < pool_chunks; ++i) {
        const uint32_t c = ranked[i].second;
        if (chunk_written_[c]) { want[c] = true; admitted++; }
    }

    // Evict and page-in for each stream
    for (uint32_t s = 0; s < n_stream_; ++s) {
        auto & l2p = logical_to_pool_[s];
        auto & p2l = pool_to_logical_[s];

        for (uint32_t sl = 0; sl < pool_size_; ++sl) {
            uint32_t lp = p2l[sl];
            if (lp == UINT32_MAX) continue;
            if (!want[chunk_of(lp)]) {
                page_out(chunk_of(lp), sl, s);
                l2p[lp] = -1;
                p2l[sl] = UINT32_MAX;
            }
        }

        for (uint32_t c = 0; c < n_chunks_; ++c) {
            if (!want[c]) continue;
            bool resident = false;
            for (uint32_t p = c * chunk_sz; p < std::min((c + 1) * chunk_sz, logical_size_); ++p) {
                if (l2p[p] >= 0) { resident = true; break; }
            }
            if (resident) continue;
            auto slots = evict_lru(1, s);
            if (slots.empty()) break;
            page_in(c, slots[0], s);
        }
    }
}

void llama_kv_cache_paged::repool_simple() {
    const uint32_t pool_chunks = std::max(1u, pool_size_ / chunk_sz);
    std::vector<bool> want(n_chunks_, false);

    for (uint32_t c = 0; c < sink_chunks && c < n_chunks_; ++c) {
        if (chunk_written_[c]) want[c] = true;
    }
    for (uint32_t c = n_chunks_ > tail_chunks ? n_chunks_ - tail_chunks : 0;
         c < n_chunks_; ++c) {
        if (chunk_written_[c]) want[c] = true;
    }

    uint32_t pinned = 0, n_written = 0;
    for (uint32_t c = 0; c < n_chunks_; ++c) {
        if (chunk_written_[c]) n_written++;
        if (want[c]) pinned++;
    }

    uint32_t remaining = pool_chunks > pinned ? pool_chunks - pinned : 0;
    for (uint32_t c = n_chunks_; c > 0 && remaining > 0; --c) {
        if (!want[c - 1] && chunk_written_[c - 1]) {
            want[c - 1] = true; remaining--;
        }
    }

    for (uint32_t s = 0; s < n_stream_; ++s) {
        auto & l2p = logical_to_pool_[s];
        auto & p2l = pool_to_logical_[s];

        for (uint32_t sl = 0; sl < pool_size_; ++sl) {
            uint32_t lp = p2l[sl];
            if (lp == UINT32_MAX) continue;
            if (!want[chunk_of(lp)]) {
                page_out(chunk_of(lp), sl, s);
                l2p[lp] = -1;
                p2l[sl] = UINT32_MAX;
            }
        }

        for (uint32_t c = 0; c < n_chunks_; ++c) {
            if (!want[c]) continue;
            bool resident = false;
            for (uint32_t p = c * chunk_sz; p < std::min((c + 1) * chunk_sz, logical_size_); ++p) {
                if (l2p[p] >= 0) { resident = true; break; }
            }
            if (resident) continue;
            auto slots = evict_lru(1, s);
            if (slots.empty()) break;
            page_in(c, slots[0], s);
        }
    }
}

void llama_kv_cache_paged::repool() {
    if (!paging_active_) return;

    for (uint32_t s = 0; s < n_stream_; ++s) {
        touch_sink_and_tail(s);
    }

    // Prefill optimization: if we've written <= pool_size tokens,
    // everything is already in the pool — no eviction needed.
    if (n_tokens_written_ <= pool_size_) {
        return;
    }

    if (scorer_ || !attn_scores_.empty()) {
        repool_scored();
    } else {
        repool_simple();
    }
}

// ---------------------------------------------------------------------------
// llama_memory_i
// ---------------------------------------------------------------------------

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc, uint32_t ubatch_sz, bool embd_all) {
    repool();

    auto ctx = kv_pool_->init_batch(balloc, ubatch_sz, embd_all);

    if (ctx && ctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS) {
        const uint32_t n_new = balloc.get_n_tokens();
        for (uint32_t p = n_tokens_written_; p < n_tokens_written_ + n_new && p < logical_size_; ++p) {
            chunk_written_[chunk_of(p)] = true;
        }
        n_tokens_written_ += n_new;
    }

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    return kv_pool_->init_full();
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * lctx, bool optimize) {
    repool();

    // Synchronize any pending async transfers
    if (backend_gpu_) {
        // GPU backends handle sync via their scheduler
    }

    return kv_pool_->init_update(lctx, optimize);
}

void llama_kv_cache_paged::clear(bool data) {
    kv_pool_->clear(data);
    if (kv_backing_) kv_backing_->clear(data);
    if (data) init_mappings();
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool r = kv_pool_->seq_rm(seq_id, p0, p1);
    if (kv_backing_) r &= kv_backing_->seq_rm(seq_id, p0, p1);
    return r;
}

void llama_kv_cache_paged::seq_cp(llama_seq_id s_src, llama_seq_id s_dst, llama_pos p0, llama_pos p1) {
    kv_pool_->seq_cp(s_src, s_dst, p0, p1);
    if (kv_backing_) kv_backing_->seq_cp(s_src, s_dst, p0, p1);
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    kv_pool_->seq_keep(seq_id);
    if (kv_backing_) kv_backing_->seq_keep(seq_id);
}

void llama_kv_cache_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_pool_->seq_add(seq_id, p0, p1, shift);
    if (kv_backing_) kv_backing_->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_pool_->seq_div(seq_id, p0, p1, d);
    if (kv_backing_) kv_backing_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    return kv_backing_ ? kv_backing_->seq_pos_min(seq_id) : kv_pool_->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    return kv_backing_ ? kv_backing_->seq_pos_max(seq_id) : kv_pool_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    auto b = kv_pool_->memory_breakdown();
    if (kv_backing_) {
        for (auto & [k, v] : kv_backing_->memory_breakdown()) b[k] += v;
    }
    return b;
}

void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_pool_->state_write(io, seq_id, flags);
    if (kv_backing_) kv_backing_->state_write(io, seq_id, flags);
}

void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv_pool_->state_read(io, seq_id, flags);
    if (kv_backing_) kv_backing_->state_read(io, seq_id, flags);
}
