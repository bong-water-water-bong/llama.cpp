#pragma once

#include "llama-kv-cache.h"
#include "llama-memory.h"

#include <memory>
#include <vector>

//
// llama_kv_cache_paged — Active paged KV cache with full KVFlash feature set.
//
// Features:
//   - Pool-sized GPU tensors (95%+ VRAM savings)
//   - CPU backing store (bit-exact, lossless)
//   - 64-token chunk paging with LRU eviction
//   - Attention sink + trailing window pinning (never evicted)
//   - Pluggable scorer (heuristic + model-based PFlash drafter)
//   - Async page transfers via ggml_backend
//   - Multi-stream KV cache support
//   - Pool auto-sizing from GPU free VRAM
//   - Prefill optimization (skips repool when context ≤ pool)
//   - Pre-allocated pinned buffers (zero-allocation transfers)
//   - Cell metadata syncing on eviction
//

// Pluggable chunk scorer interface
struct llama_kv_paged_scorer_i {
    virtual ~llama_kv_paged_scorer_i() = default;
    virtual std::vector<float> score(uint32_t n_chunks, uint32_t pool_capacity_chunks) const = 0;
};

class llama_kv_cache_paged : public llama_memory_i {
public:
    llama_kv_cache_paged(
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
                     uint32_t   n_ubatch,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share);

    ~llama_kv_cache_paged() override = default;

    // Auto-detect pool size from GPU free VRAM.
    // Returns recommended pool slot count, or logical_size if no GPU.
    static uint32_t auto_detect_pool_size(const llama_model & model, uint32_t logical_size);

    // llama_memory_i
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id s_src,  llama_seq_id s_dst,          llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Access the pool cache for graph building
    const llama_kv_cache * get_pool() const { return kv_pool_.get(); }
          llama_kv_cache * get_pool()       { return kv_pool_.get(); }

    // Stats
    uint64_t get_page_in_count()  const { return n_page_in_;  }
    uint64_t get_page_out_count() const { return n_page_out_; }
    uint32_t get_pool_size()      const { return pool_size_;  }
    uint32_t get_logical_size()   const { return logical_size_; }
    uint32_t get_n_stream()       const { return n_stream_; }

    // Scorer management
    void set_scorer(std::unique_ptr<llama_kv_paged_scorer_i> scorer);

    // Attention feedback: update chunk scores from the last layer's attention pattern.
    // Called after each decode step with the KQ softmax weights.
    // attn_weights [n_heads, n_kv] — softmax over KV positions for each head.
    void feed_attention(const float * attn_weights, uint32_t n_heads, uint32_t n_kv);

private:
    // --- Core caches ---
    std::unique_ptr<llama_kv_cache> kv_pool_;    // GPU, pool_size cells
    std::unique_ptr<llama_kv_cache> kv_backing_; // CPU, logical_size cells

    const uint32_t logical_size_;
    const uint32_t pool_size_;
    const uint32_t n_stream_;       // number of KV streams
    const bool     paging_active_;

    static constexpr uint32_t chunk_sz    = 64;
    static constexpr uint32_t sink_chunks = 1;
    static constexpr uint32_t tail_chunks = 2;

    uint32_t n_chunks_;
    uint32_t n_tokens_written_ = 0;

    // --- Mapping tables (per stream) ---
    std::vector<std::vector<int32_t>>  logical_to_pool_;
    std::vector<std::vector<uint32_t>> pool_to_logical_;
    std::vector<std::vector<uint64_t>> pool_lru_;
    std::vector<uint64_t>              lru_counter_;

    // Track which chunks have data
    std::vector<bool> chunk_written_;

    // Attention feedback scores (decayed average of per-chunk attention)
    std::vector<float> attn_scores_;
    float              attn_decay_ = 0.9f; // exponential decay per decode step

    // --- Per-layer info ---
    struct layer_info { uint32_t il; size_t row_bytes_k; size_t row_bytes_v; };
    std::vector<layer_info> layers_;

    // --- Async backend for GPU page transfers ---
    ggml_backend_t backend_gpu_ = nullptr;

    // --- Pre-allocated transfer buffers (one per stream) ---
    struct transfer_bufs {
        std::vector<uint8_t> k; // from host to GPU (page-in)
        std::vector<uint8_t> v; // from GPU to host (page-out)
    };
    std::vector<transfer_bufs> pinned_in_;
    std::vector<transfer_bufs> pinned_out_;

    // --- Stats ---
    uint64_t n_page_in_  = 0;
    uint64_t n_page_out_ = 0;

    // --- Scorer ---
    std::unique_ptr<llama_kv_paged_scorer_i> scorer_;

    // --- Helpers ---
    uint32_t chunk_of(uint32_t pos) const { return pos / chunk_sz; }
    bool is_sink(uint32_t pos) const { return pos < sink_chunks * chunk_sz; }
    bool is_tail(uint32_t pos) const;

    void init_mappings();
    void init_layer_info();
    void init_pinned_buffers();

    void repool();
    void repool_scored();
    void repool_simple();

    void touch_sink_and_tail(uint32_t stream = 0);
    std::vector<uint32_t> evict_lru(uint32_t n_needed, uint32_t stream = 0);

    void page_in (uint32_t chunk_id, uint32_t slot, uint32_t stream = 0);
    void page_out(uint32_t chunk_id, uint32_t slot, uint32_t stream = 0);

    void copy_chunk_to_pool   (uint32_t chunk_id, uint32_t slot, uint32_t stream = 0);
    void copy_chunk_to_backing(uint32_t chunk_id, uint32_t slot, uint32_t stream = 0);

    void release_pool_cells(uint32_t chunk_id, uint32_t stream = 0);
};
