#include "ggml-hrx2.h"

#include "ggml-hrx2-catalog.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "hrx_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

static constexpr size_t    GGML_HRX2_ALIGNMENT     = 256;
static constexpr uintptr_t GGML_HRX2_FAKE_PTR_BASE = 0x200000000ull;
static constexpr size_t    GGML_HRX2_STAGING_ARENA_DEFAULT_SIZE = 8 * 1024 * 1024;
static constexpr uint64_t  GGML_HRX2_DEFAULT_DISPATCHES_PER_SUBMIT = 12;
static constexpr uint64_t  GGML_HRX2_DEFAULT_MAX_MUL_MAT_BYTES_PER_SUBMIT = 100ull * 1000ull * 1000ull;

struct ggml_backend_hrx2_device_context;

struct ggml_backend_hrx2_staging_arena {
    hrx_stream_t stream = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * mapped = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::vector<hrx_buffer_t> retired_buffers;
};

struct ggml_backend_hrx2_device_scratch {
    hrx_buffer_t buffer = nullptr;
    size_t capacity = 0;
    std::vector<hrx_buffer_t> retired_buffers;
};

enum class ggml_backend_hrx2_scratch_state {
    available,
    in_use,
    retired,
};

struct ggml_backend_hrx2_scratch_buffer {
    hrx_buffer_t buffer = nullptr;
    size_t size = 0;
    ggml_backend_hrx2_scratch_state state = ggml_backend_hrx2_scratch_state::available;
};

struct ggml_backend_hrx2_buffer_type_context {
    ggml_backend_hrx2_device_context * device_context = nullptr;
    std::string name;
};

struct ggml_backend_hrx2_buffer_context {
    ggml_backend_hrx2_device_context * device_context = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * base = nullptr;
};

struct ggml_backend_hrx2_device_context {
    hrx_device_t device = nullptr;
    hrx_stream_t active_stream = nullptr;
    hrx_stream_t transfer_stream = nullptr;
    std::mutex streams_mutex;
    std::vector<hrx_stream_t> live_streams;
    std::vector<ggml_backend_hrx2_staging_arena> staging_arenas;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    // zero-copy: the DEFAULT buft is host-visible (activations shared with the
    // CPU); the WEIGHT buft is device-local (NPU-only, fake base). The
    // device-local default from before is kept for weights via get_extra_bufts.
    ggml_backend_hrx2_buffer_type_context buffer_type_context;
    ggml_backend_buffer_type buffer_type = {};
    ggml_backend_hrx2_buffer_type_context host_buffer_type_context;
    ggml_backend_buffer_type host_buffer_type = {};
    ggml_backend_hrx2_buffer_type_context weight_buffer_type_context;
    ggml_backend_buffer_type weight_buffer_type = {};
    ggml_backend_hrx2_catalog_ptr catalog;
    std::vector<const ggml_backend_hrx2_kernel_route *> rms_norm_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> rms_norm_mul_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> add_rms_norm_mul_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q8_0_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> quantize_q8_1_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_f32_f32_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> q4nx_dequant_routes;
    // persistent Q4NX fused-dispatch scratch (never released mid-flight:
    // kernels are async on the stream, so per-dispatch alloc/free raced the
    // allocator and corrupted memory)
    hrx_buffer_t q4nx_scl = nullptr;
    size_t q4nx_scl_cap = 0;
    hrx_buffer_t q4nx_w = nullptr;
    size_t q4nx_w_cap = 0;
    hrx_buffer_t q4nx_ids = nullptr;   // host-visible scratch for MUL_MAT_ID_Q4NX ids
    size_t q4nx_ids_cap = 0;
    hrx_buffer_t q4nx_tbl = nullptr;   // device scratch: tbl-tiled mm src1_cols/dst_cols tables
    size_t q4nx_tbl_cap = 0;
    // Grown-out q4nx scratch buffers, freed only after a stream sync. The
    // kernels reading q4nx_scl/q4nx_w/q4nx_ids are async on the stream; the
    // allocator unmaps/recycles a released buffer immediately, so releasing on
    // growth while the previous op's kernels are still in flight caused
    // gfxhub TCP read faults (shader reads of unmapped GTT pages). Retire here
    // and free in ggml_backend_hrx2_sync_streams, after every stream synced.
    std::vector<hrx_buffer_t> q4nx_retired;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q4_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q4_k_swiglu_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_id_q4_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q5_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_id_q5_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_q6_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_id_q6_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_f16_f32_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_mat_f16_f32_cont_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> copy_f32_f16_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> cont_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> cont_set_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> swiglu_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> set_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> rope_set_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> add_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> mul_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> div_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> scale_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> clamp_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> sum_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_q8_0_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_q4_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_q5_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_q6_k_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> get_rows_moe_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> argsort_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> rope_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> soft_max_routes;
    std::vector<const ggml_backend_hrx2_kernel_route *> flash_attn_fa0_routes;
    std::unordered_map<std::string, std::unique_ptr<ggml_backend_hrx2_provider>> providers;
    std::unordered_set<std::string> provider_failures;
};

struct ggml_backend_hrx2_context {
    ggml_backend_hrx2_device_context * device_context = nullptr;
    hrx_stream_t stream = nullptr;
    std::string name;
    uint64_t last_total_mul_mat_bytes = 0;
    uint64_t submitted_dispatches = 0;
    uint64_t mul_mat_bytes = 0;
    uint64_t total_mul_mat_bytes = 0;
    uint64_t mul_mat_bytes_per_submit = 0;
    uint64_t submit_count = 0;
    uint64_t submit_flush_count = 0;
    const ggml_tensor * submit_last_node = nullptr;
    std::vector<ggml_backend_hrx2_scratch_buffer> scratch_buffers;
    ggml_backend_hrx2_device_scratch q8_1_scratch;
    ggml_backend_hrx2_device_scratch route_scratch;
    const ggml_tensor * q8_1_cached_src = nullptr;
    bool q8_1_cached_use_x4 = false;
    size_t q8_1_cached_size = 0;
    uint32_t q8_1_cached_ne00 = 0;
    uint32_t q8_1_cached_s01 = 0;
    uint32_t q8_1_cached_s02 = 0;
    uint32_t q8_1_cached_s03 = 0;
    uint32_t q8_1_cached_ne0 = 0;
    uint32_t q8_1_cached_ne1 = 0;
    uint32_t q8_1_cached_ne2 = 0;
    uint32_t q8_1_cached_blocks = 0;
    uint32_t q8_1_cached_z_count = 0;
    hrx_buffer_ref_t q8_1_cached_ref = {};
};

static thread_local ggml_backend_hrx2_context * g_hrx2_active_graph_context = nullptr;
static thread_local const ggml_tensor * g_hrx2_active_graph_node = nullptr;

static void ggml_backend_hrx2_unregister_stream(ggml_backend_hrx2_device_context * device_context, hrx_stream_t stream);

struct ggml_backend_hrx2_reg_context {
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx2_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx2_reg_context() {
        for (auto & device_context : device_contexts) {
            device_context->providers.clear();
            if (device_context->transfer_stream) {
                hrx_status_t status = hrx_stream_synchronize(device_context->transfer_stream);
                if (!hrx_status_is_ok(status)) {
                    hrx_status_ignore(status);
                }
                ggml_backend_hrx2_unregister_stream(device_context.get(), device_context->transfer_stream);
                hrx_stream_release(device_context->transfer_stream);
                device_context->transfer_stream = nullptr;
            }
            if (device_context->q4nx_scl) hrx_buffer_release(device_context->q4nx_scl);
            if (device_context->q4nx_w)   hrx_buffer_release(device_context->q4nx_w);
            if (device_context->q4nx_ids) hrx_buffer_release(device_context->q4nx_ids);
            if (device_context->q4nx_tbl) hrx_buffer_release(device_context->q4nx_tbl);
            for (hrx_buffer_t b : device_context->q4nx_retired) hrx_buffer_release(b);
            device_context->q4nx_retired.clear();
            if (device_context->device) {
                hrx_device_release(device_context->device);
            }
        }
        if (gpu_initialized) {
            hrx_status_ignore(hrx_gpu_shutdown());
        }
    }
};

struct ggml_backend_hrx2_rms_norm_constants {
    uint32_t ncols;
    uint32_t nrows;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t src_nb1;
    uint32_t src_nb2;
    uint32_t src_nb3;
    uint32_t dst_nb1;
    uint32_t dst_nb2;
    uint32_t dst_nb3;
    float eps;
};

static_assert(sizeof(ggml_backend_hrx2_rms_norm_constants) == 44);

struct ggml_backend_hrx2_mul_mat_constants {
    uint32_t k;
    uint32_t rows;
    uint32_t cols;
};

static_assert(sizeof(ggml_backend_hrx2_mul_mat_constants) == 12);

struct ggml_backend_hrx2_quantize_q8_1_constants {
    uint32_t ne00;
    uint32_t s01;
    uint32_t s02;
    uint32_t s03;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
};

static_assert(sizeof(ggml_backend_hrx2_quantize_q8_1_constants) == 28);

struct ggml_backend_hrx2_quantize_q8_1_shape {
    uint32_t blocks = 0;
    uint32_t ne1 = 0;
    uint32_t z_count = 0;
};

struct ggml_backend_hrx2_rope_constants {
    float freq_base;
    float freq_scale;
    float attn_factor;
};

static_assert(sizeof(ggml_backend_hrx2_rope_constants) == 12);

struct ggml_backend_hrx2_soft_max_constants {
    float scale;
};

static_assert(sizeof(ggml_backend_hrx2_soft_max_constants) == 4);

struct ggml_backend_hrx2_flash_attn_fa0_constants {
    int64_t D;
    int64_t KV;
    int64_t N;
    int64_t H;
    int64_t H_KV;
    int64_t S;
    int64_t q_nb1;
    int64_t q_nb2;
    int64_t q_nb3;
    int64_t k_nb1;
    int64_t k_nb2;
    int64_t k_nb3;
    int64_t v_nb0;
    int64_t v_nb1;
    int64_t v_nb2;
    int64_t v_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    int64_t mask_nb0;
    int64_t mask_nb1;
    int64_t mask_nb3;
    float scale;
    int32_t has_mask;
    float max_bias;
    float m0;
    float m1;
    float logit_softcap;
    int32_t n_head_log2;
    int32_t has_sinks;
};

static_assert(sizeof(ggml_backend_hrx2_flash_attn_fa0_constants) == 208);

struct ggml_backend_hrx2_copy_shape {
    uint32_t n = 0;
};

struct ggml_backend_hrx2_mul_mat_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct ggml_backend_hrx2_mul_mat_q4_k_swiglu_fusion {
    const ggml_tensor * x = nullptr;
    const ggml_tensor * gate = nullptr;
    ggml_backend_hrx2_mul_mat_shape shape = {};
};

struct ggml_backend_hrx2_mul_mat_q4_k_packed_swiglu_fusion {
    const ggml_tensor * mul_mat = nullptr;
    ggml_backend_hrx2_mul_mat_shape shape = {};
};

struct ggml_backend_hrx2_mul_mat_id_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t nexperts = 0;
    uint32_t nselected = 0;
    uint32_t ntokens = 0;
    uint32_t src1_selected_stride = 0;
    uint32_t src1_token_stride = 0;
    uint32_t idx_token_stride = 0;
    uint32_t dst_token_stride = 0;
};

struct ggml_backend_hrx2_mul_mat_f16_shape {
    uint32_t k = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t dst_ne2 = 0;
    uint32_t dst_ne3 = 0;
    uint32_t src0_ne2 = 0;
    uint32_t src0_ne3 = 0;
    uint32_t src0_stride_row = 0;
    uint32_t src0_stride_ne2 = 0;
    uint32_t src0_stride_ne3 = 0;
    uint32_t src1_stride_col = 0;
    uint32_t src1_stride_ne2 = 0;
    uint32_t src1_stride_ne3 = 0;
    uint32_t dst_stride_col = 0;
    uint32_t dst_stride_ne2 = 0;
    uint32_t dst_stride_ne3 = 0;
};

struct ggml_backend_hrx2_rms_norm_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
};

struct ggml_backend_hrx2_rms_norm_mul_fusion {
    const ggml_tensor * rms_norm = nullptr;
    const ggml_tensor * mul = nullptr;
    const ggml_tensor * weight = nullptr;
    ggml_backend_hrx2_rms_norm_shape shape = {};
};

struct ggml_backend_hrx2_add_rms_norm_mul_fusion {
    const ggml_tensor * add = nullptr;
    const ggml_tensor * rms_norm = nullptr;
    const ggml_tensor * mul = nullptr;
    const ggml_tensor * weight = nullptr;
    ggml_backend_hrx2_rms_norm_shape shape = {};
};

struct ggml_backend_hrx2_set_rows_shape {
    uint32_t nc = 0;
    uint32_t nr = 0;
    uint32_t ne02 = 0;
    uint32_t ne03 = 0;
    uint32_t ne1 = 0;
    uint32_t ne11 = 0;
    uint32_t ne12 = 0;
    uint32_t src0_nb1 = 0;
    uint32_t src0_nb2 = 0;
    uint32_t src0_nb3 = 0;
    uint32_t idx_nb0 = 0;
    uint32_t idx_nb1 = 0;
    uint32_t idx_nb2 = 0;
    uint32_t dst_nb1 = 0;
    uint32_t dst_nb2 = 0;
    uint32_t dst_nb3 = 0;
};

struct ggml_backend_hrx2_pointwise_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t src0_row_stride = 0;
    uint32_t src1_row_stride = 0;
    uint32_t src1_ncols = 0;
};

struct ggml_backend_hrx2_sum_rows_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t src0_row_stride = 0;
};

struct ggml_backend_hrx2_get_rows_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t src0_nrows = 0;
    uint32_t src0_row_stride = 0;
    uint32_t idx_row_stride = 0;
    uint32_t dst_row_stride = 0;
};

struct ggml_backend_hrx2_get_rows_moe_shape {
    uint32_t nexperts = 0;
    uint32_t nselected = 0;
    uint32_t ntokens = 0;
    uint32_t src0_token_stride = 0;
    uint32_t idx_token_stride = 0;
    uint32_t dst_token_stride = 0;
};

struct ggml_backend_hrx2_argsort_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
};

struct ggml_backend_hrx2_rope_shape {
    uint32_t ncols = 0;
    uint32_t n_dims = 0;
    uint32_t mode = 0;
    uint32_t nheads = 0;
    uint32_t ntokens = 0;
    uint32_t nrows = 0;
    uint32_t src0_head_stride = 0;
    uint32_t src0_token_stride = 0;
    uint32_t dst_head_stride = 0;
    uint32_t dst_token_stride = 0;
    uint32_t pos_token_stride = 0;
};

struct ggml_backend_hrx2_rope_set_rows_fusion {
    const ggml_tensor * rope = nullptr;
    const ggml_tensor * view = nullptr;
    const ggml_tensor * set_rows = nullptr;
    ggml_backend_hrx2_rope_shape rope_shape = {};
    ggml_backend_hrx2_set_rows_shape set_rows_shape = {};
};

struct ggml_backend_hrx2_soft_max_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t ne01 = 0;
    uint32_t ne02 = 0;
    uint32_t mask_nb1 = 1;
    uint32_t mask_nb2 = 1;
    uint32_t mask_nb3 = 1;
    uint32_t mask_ne1 = 1;
    uint32_t mask_ne2 = 1;
    uint32_t mask_ne3 = 1;
    bool has_mask = false;
};

struct ggml_backend_hrx2_flash_attn_fa0_shape {
    uint32_t D = 0;
    uint32_t KV = 0;
    uint32_t N = 0;
    uint32_t H = 0;
    uint32_t H_KV = 0;
    uint32_t S = 0;
};

struct ggml_backend_hrx2_cont_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t ne1 = 0;
    uint32_t ne2 = 0;
    uint32_t src_nb1 = 0;
    uint32_t src_nb2 = 0;
    uint32_t src_nb3 = 0;
};

struct ggml_backend_hrx2_cont_set_rows_fusion {
    const ggml_tensor * cont = nullptr;
    const ggml_tensor * set_rows = nullptr;
    ggml_backend_hrx2_cont_shape cont_shape = {};
    ggml_backend_hrx2_set_rows_shape set_rows_shape = {};
};

struct ggml_backend_hrx2_swiglu_shape {
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    enum ggml_glu_op glu_op = GGML_GLU_OP_SWIGLU;
    bool split_sources = false;
};

struct ggml_backend_hrx2_provider_plan {
    const ggml_backend_hrx2_kernel_route * route = nullptr;
    std::string cache_key;
    std::vector<ggml_backend_hrx2_config_binding> config_bindings;
};

static bool ggml_backend_hrx2_q4_k_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape);
static bool ggml_backend_hrx2_q4_k_q8_1_x4_mmq_enabled();
static bool ggml_backend_hrx2_q8_0_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape);
static bool ggml_backend_hrx2_q5_k_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape);
static bool ggml_backend_hrx2_q5_k_q8_1_x4_prompt_enabled();
static bool ggml_backend_hrx2_q6_k_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape);
static bool ggml_backend_hrx2_q6_k_q8_1_x4_prompt_enabled();
static bool ggml_backend_hrx2_route_uses_q8_1_rhs(const ggml_backend_hrx2_kernel_route * route);
static bool ggml_backend_hrx2_route_uses_q8_1_x4_rhs(const ggml_backend_hrx2_kernel_route * route);
static uint32_t ggml_backend_hrx2_provider_workgroup_size_x(const ggml_backend_hrx2_provider * provider);
static bool ggml_backend_hrx2_dispatch_quantize_q8_1(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * src,
        bool use_x4,
        hrx_buffer_ref_t * out_q8_1_ref);
static bool ggml_backend_hrx2_cont_route_copies_vec4(const ggml_backend_hrx2_kernel_route * route);
static bool ggml_backend_hrx2_extract_cont_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_cont_shape * out_shape);

static bool ggml_hrx2_check(hrx_status_t status, const char * expression, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length = 0;
    if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) && message) {
        GGML_LOG_ERROR("HRX2: %s failed at %s:%d: %.*s\n", expression, file, line, (int) length, message);
        hrx_status_free_message(message);
    } else {
        GGML_LOG_ERROR("HRX2: %s failed at %s:%d\n", expression, file, line);
    }
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX2_CHECK(expr) ggml_hrx2_check((expr), #expr, __FILE__, __LINE__)

static bool ggml_backend_hrx2_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool ggml_backend_hrx2_fusion_enabled() {
    return !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_FUSION");
}

static bool ggml_backend_hrx2_trace_graph_enabled() {
    return ggml_backend_hrx2_env_enabled("GGML_HRX2_TRACE_GRAPH");
}

static uint64_t ggml_backend_hrx2_u64_from_env(const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<uint64_t>(parsed);
}

static const char * ggml_backend_hrx2_glu_op_key(enum ggml_glu_op glu_op) {
    switch (glu_op) {
        case GGML_GLU_OP_REGLU: return "REGLU";
        case GGML_GLU_OP_GEGLU: return "GEGLU";
        case GGML_GLU_OP_SWIGLU: return "SWIGLU";
        case GGML_GLU_OP_SWIGLU_OAI: return "SWIGLU_OAI";
        case GGML_GLU_OP_GEGLU_ERF: return "GEGLU_ERF";
        case GGML_GLU_OP_GEGLU_QUICK: return "GEGLU_QUICK";
        case GGML_GLU_OP_COUNT: break;
    }
    return "UNKNOWN";
}

static std::string ggml_backend_hrx2_json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[7] = {};
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    escaped += buffer;
                } else {
                    escaped += static_cast<char>(c);
                }
                break;
        }
    }
    return escaped;
}

static std::string ggml_backend_hrx2_json_kv(const char * key, const std::string & value) {
    std::string result = "\"";
    result += key;
    result += "\":\"";
    result += ggml_backend_hrx2_json_escape(value);
    result += "\"";
    return result;
}

static std::string ggml_backend_hrx2_json_kv(const char * key, uint64_t value) {
    std::string result = "\"";
    result += key;
    result += "\":";
    result += std::to_string(value);
    return result;
}

static uint64_t ggml_backend_hrx2_now_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

static void ggml_backend_hrx2_trace_event(const char * event, const std::string & fields_json) {
    const char * trace_path = std::getenv("GGML_HRX2_TRACE_JSONL");
    const bool trace_log = ggml_backend_hrx2_env_enabled("GGML_HRX2_TRACE_ROUTES");
    if ((!trace_path || trace_path[0] == '\0') && !trace_log) {
        return;
    }

    std::string line = "{\"event\":\"";
    line += event ? event : "";
    line += "\",\"t_us\":" + std::to_string(ggml_backend_hrx2_now_us());
    if (!fields_json.empty()) {
        line += ",";
        line += fields_json;
    }
    line += "}";

    if (trace_path && trace_path[0] != '\0') {
        static std::string output_path;
        static std::ofstream output;
        if (!output.is_open() || output_path != trace_path) {
            if (output.is_open()) {
                output.close();
            }
            output_path = trace_path;
            output.open(output_path, std::ios::out | std::ios::app);
            if (!output) {
                GGML_LOG_WARN("%s: failed to open GGML_HRX2_TRACE_JSONL=%s\n", __func__, trace_path);
            }
        }
        if (output) {
            output << line << '\n';
        }
    }
    if (trace_log) {
        GGML_LOG_INFO("HRX2_TRACE: %s\n", line.c_str());
    }
}


static uint64_t ggml_backend_hrx2_dispatches_per_submit() {
    return ggml_backend_hrx2_u64_from_env(
        "GGML_HRX2_DISPATCHES_PER_SUBMIT",
        GGML_HRX2_DEFAULT_DISPATCHES_PER_SUBMIT);
}

static uint64_t ggml_backend_hrx2_max_mul_mat_bytes_per_submit() {
    return ggml_backend_hrx2_u64_from_env(
        "GGML_HRX2_MAX_MUL_MAT_BYTES_PER_SUBMIT",
        GGML_HRX2_DEFAULT_MAX_MUL_MAT_BYTES_PER_SUBMIT);
}

static size_t ggml_backend_hrx2_align_up(size_t value, size_t alignment) {
    GGML_ASSERT(alignment > 0);
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static size_t ggml_backend_hrx2_staging_arena_capacity() {
    const uint64_t requested = ggml_backend_hrx2_u64_from_env(
        "GGML_HRX2_STAGING_ARENA_SIZE", GGML_HRX2_STAGING_ARENA_DEFAULT_SIZE);
    const size_t capacity = static_cast<size_t>(std::max<uint64_t>(requested, GGML_HRX2_ALIGNMENT));
    return ggml_backend_hrx2_align_up(capacity, GGML_HRX2_ALIGNMENT);
}

static void ggml_backend_hrx2_reset_staging_arena_locked(ggml_backend_hrx2_staging_arena & arena) {
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena.retired_buffers.clear();
    arena.offset = 0;
}

static void ggml_backend_hrx2_release_staging_arena_locked(ggml_backend_hrx2_staging_arena & arena) {
    if (arena.buffer) {
        hrx_buffer_release(arena.buffer);
    }
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena = {};
}

static ggml_backend_hrx2_staging_arena * ggml_backend_hrx2_find_staging_arena_locked(
        ggml_backend_hrx2_device_context * device_context,
        hrx_stream_t stream) {
    for (auto & arena : device_context->staging_arenas) {
        if (arena.stream == stream) {
            return &arena;
        }
    }
    return nullptr;
}

static ggml_backend_hrx2_staging_arena * ggml_backend_hrx2_get_staging_arena_locked(
        ggml_backend_hrx2_device_context * device_context,
        hrx_stream_t stream) {
    if (auto * arena = ggml_backend_hrx2_find_staging_arena_locked(device_context, stream)) {
        return arena;
    }
    device_context->staging_arenas.push_back({});
    auto & arena = device_context->staging_arenas.back();
    arena.stream = stream;
    return &arena;
}

static void ggml_backend_hrx2_register_stream(ggml_backend_hrx2_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    if (std::find(device_context->live_streams.begin(), device_context->live_streams.end(), stream) ==
            device_context->live_streams.end()) {
        device_context->live_streams.push_back(stream);
    }
}

static void ggml_backend_hrx2_unregister_stream(ggml_backend_hrx2_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    auto & streams = device_context->live_streams;
    streams.erase(std::remove(streams.begin(), streams.end(), stream), streams.end());
    auto & arenas = device_context->staging_arenas;
    auto arena_it = std::find_if(
        arenas.begin(), arenas.end(),
        [stream](const ggml_backend_hrx2_staging_arena & arena) { return arena.stream == stream; });
    if (arena_it != arenas.end()) {
        ggml_backend_hrx2_release_staging_arena_locked(*arena_it);
        arenas.erase(arena_it);
    }
    if (device_context->active_stream == stream) {
        device_context->active_stream = nullptr;
    }
}

static hrx_stream_t ggml_backend_hrx2_retain_active_stream(ggml_backend_hrx2_device_context * device_context) {
    if (!device_context) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t stream = device_context->active_stream;
    if (!stream) {
        stream = device_context->transfer_stream;
    }
    if (stream) {
        hrx_stream_retain(stream);
    }
    return stream;
}

static bool ggml_backend_hrx2_sync_streams(ggml_backend_hrx2_device_context * device_context) {
    if (!device_context) {
        return true;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    bool ok = true;
    for (hrx_stream_t stream : device_context->live_streams) {
        ok = GGML_HRX2_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx2_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx2_reset_staging_arena_locked(*arena);
        }
    }
    // Every stream is quiescent now: safe to free grown-out q4nx scratch.
    for (hrx_buffer_t b : device_context->q4nx_retired) {
        hrx_buffer_release(b);
    }
    device_context->q4nx_retired.clear();
    return ok;
}

static bool ggml_backend_hrx2_sync_graph_entry_streams(
        ggml_backend_hrx2_device_context * device_context,
        hrx_stream_t graph_stream) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t streams[] = {
        device_context->active_stream,
        device_context->transfer_stream,
    };

    bool ok = true;
    for (hrx_stream_t stream : streams) {
        if (!stream || stream == graph_stream) {
            continue;
        }
        ok = GGML_HRX2_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx2_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx2_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx2_prepare_stream_signal(
        hrx_stream_t stream,
        hrx_semaphore_t * semaphore,
        uint64_t * signal_value,
        hrx_semaphore_list_t * wait_list,
        hrx_semaphore_list_t * signal_list,
        hrx_semaphore_t * wait_semaphores,
        uint64_t * wait_values,
        hrx_semaphore_t * signal_semaphores,
        uint64_t * signal_values) {
    hrx_timeline_point_t position = {};
    if (!GGML_HRX2_CHECK(hrx_stream_flush(stream)) ||
        !GGML_HRX2_CHECK(hrx_stream_get_timeline_position(stream, &position)) ||
        !GGML_HRX2_CHECK(hrx_stream_get_semaphore(stream, semaphore))) {
        return false;
    }

    *signal_value = position.value + 1;
    if (position.value > 0) {
        wait_semaphores[0] = *semaphore;
        wait_values[0] = position.value;
        *wait_list = {
            /* .semaphores = */ wait_semaphores,
            /* .values     = */ wait_values,
            /* .count      = */ 1,
        };
    } else {
        *wait_list = {};
    }

    signal_semaphores[0] = *semaphore;
    signal_values[0] = *signal_value;
    *signal_list = {
        /* .semaphores = */ signal_semaphores,
        /* .values     = */ signal_values,
        /* .count      = */ 1,
    };
    return true;
}

static bool ggml_backend_hrx2_finish_stream_signal(hrx_stream_t stream, uint64_t signal_value) {
    uint64_t advanced_value = 0;
    if (!GGML_HRX2_CHECK(hrx_stream_advance_timeline(stream, &advanced_value))) {
        return false;
    }
    if (advanced_value != signal_value) {
        GGML_LOG_ERROR("%s: stream timeline advanced to %" PRIu64 ", expected %" PRIu64 "\n",
                __func__, advanced_value, signal_value);
        return false;
    }
    return GGML_HRX2_CHECK(hrx_stream_wait(stream));
}

static bool ggml_backend_hrx2_queue_fill_stream_sync(
        ggml_backend_hrx2_device_context * device_context,
        hrx_buffer_t buffer,
        size_t offset,
        size_t size,
        const void * pattern,
        size_t pattern_size) {
    hrx_stream_t stream = ggml_backend_hrx2_retain_active_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX2 stream registered for synchronous fill\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx2_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX2_CHECK(hrx_queue_fill(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, buffer, offset, size, pattern, pattern_size));
    ok = ok && ggml_backend_hrx2_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx2_queue_copy_stream_sync(
        ggml_backend_hrx2_device_context * device_context,
        hrx_buffer_t src,
        size_t src_offset,
        hrx_buffer_t dst,
        size_t dst_offset,
        size_t size) {
    hrx_stream_t stream = ggml_backend_hrx2_retain_active_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX2 stream registered for synchronous copy\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx2_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX2_CHECK(hrx_queue_copy(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, src, src_offset, dst, dst_offset, size));
    ok = ok && ggml_backend_hrx2_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx2_ensure_staging_buffer_locked(
        ggml_backend_hrx2_device_context * device_context,
        ggml_backend_hrx2_staging_arena * arena,
        size_t required_capacity) {
    if (arena->buffer && arena->capacity >= required_capacity && arena->mapped) {
        return true;
    }

    if (arena->buffer) {
        arena->retired_buffers.push_back(arena->buffer);
        arena->buffer = nullptr;
        arena->mapped = nullptr;
        arena->capacity = 0;
        arena->offset = 0;
    }

    const size_t capacity = ggml_backend_hrx2_align_up(
        std::max(required_capacity, ggml_backend_hrx2_staging_arena_capacity()),
        GGML_HRX2_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type           = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        /* .access         = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT |
                               HRX_BUFFER_USAGE_MAPPING_SCOPED |
                               HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, capacity, &arena->buffer))) {
        return false;
    }

    void * mapped = nullptr;
    if (!GGML_HRX2_CHECK(hrx_buffer_map(arena->buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, capacity, &mapped))) {
        hrx_buffer_release(arena->buffer);
        arena->buffer = nullptr;
        return false;
    }
    arena->mapped = static_cast<uint8_t *>(mapped);
    arena->capacity = capacity;
    arena->offset = 0;
    return true;
}

static void ggml_backend_hrx2_release_device_scratch(ggml_backend_hrx2_device_scratch & scratch) {
    if (scratch.buffer) {
        hrx_buffer_release(scratch.buffer);
    }
    for (hrx_buffer_t buffer : scratch.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    scratch = {};
}

static void ggml_backend_hrx2_release_retired_device_scratch(ggml_backend_hrx2_device_scratch & scratch) {
    for (hrx_buffer_t buffer : scratch.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    scratch.retired_buffers.clear();
}

static bool ggml_backend_hrx2_ensure_device_scratch(
        ggml_backend_hrx2_context * context,
        ggml_backend_hrx2_device_scratch * scratch,
        size_t required_capacity,
        hrx_buffer_ref_t * out_ref) {
    if (!context || !scratch || !out_ref || required_capacity == 0) {
        return false;
    }
    const size_t capacity = ggml_backend_hrx2_align_up(required_capacity, GGML_HRX2_ALIGNMENT);
    if (!scratch->buffer || scratch->capacity < capacity) {
        if (scratch->buffer) {
            scratch->retired_buffers.push_back(scratch->buffer);
            scratch->buffer = nullptr;
            scratch->capacity = 0;
        }
        hrx_buffer_params_t params = {
            /* .type           = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access         = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        };
        if (!GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(
                hrx_device_allocator(context->device_context->device), params, capacity, &scratch->buffer))) {
            return false;
        }
        scratch->capacity = capacity;
    }
    *out_ref = {
        /* .buffer = */ scratch->buffer,
        /* .offset = */ 0,
        /* .length = */ required_capacity,
    };
    return true;
}

static void ggml_backend_hrx2_retire_in_use_scratch_buffers(ggml_backend_hrx2_context * context) {
    for (ggml_backend_hrx2_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.state == ggml_backend_hrx2_scratch_state::in_use) {
            scratch.state = ggml_backend_hrx2_scratch_state::retired;
        }
    }
}

static void ggml_backend_hrx2_recycle_scratch_buffers(ggml_backend_hrx2_context * context) {
    for (ggml_backend_hrx2_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.state != ggml_backend_hrx2_scratch_state::available) {
            scratch.state = ggml_backend_hrx2_scratch_state::available;
        }
    }
}

static void ggml_backend_hrx2_release_scratch_buffers(ggml_backend_hrx2_context * context) {
    for (ggml_backend_hrx2_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.buffer) {
            hrx_buffer_release(scratch.buffer);
            scratch.buffer = nullptr;
        }
        scratch.size = 0;
        scratch.state = ggml_backend_hrx2_scratch_state::available;
    }
    context->scratch_buffers.clear();
}

static bool ggml_backend_hrx2_request_scratch_buffer(
        ggml_backend_hrx2_context * context,
        size_t size,
        hrx_buffer_ref_t * out_ref) {
    if (!context || !out_ref || size == 0) {
        return false;
    }

    ggml_backend_hrx2_retire_in_use_scratch_buffers(context);

    ggml_backend_hrx2_scratch_buffer * selected = nullptr;
    for (ggml_backend_hrx2_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.state != ggml_backend_hrx2_scratch_state::available || scratch.size < size) {
            continue;
        }
        if (!selected || scratch.size < selected->size) {
            selected = &scratch;
        }
    }

    if (!selected) {
        hrx_buffer_params_t params = {
            /* .type           = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access         = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        };
        hrx_buffer_t buffer = nullptr;
        if (!GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(
                hrx_device_allocator(context->device_context->device), params, size, &buffer))) {
            return false;
        }
        context->scratch_buffers.push_back({
            /* .buffer = */ buffer,
            /* .size   = */ size,
            /* .state  = */ ggml_backend_hrx2_scratch_state::available,
        });
        selected = &context->scratch_buffers.back();
    }

    selected->state = ggml_backend_hrx2_scratch_state::in_use;
    *out_ref = {
        /* .buffer = */ selected->buffer,
        /* .offset = */ 0,
        /* .length = */ size,
    };
    return true;
}

static bool ggml_backend_hrx2_ensure_route_scratch(
        ggml_backend_hrx2_context * context,
        size_t size,
        hrx_buffer_ref_t * out_ref) {
    return ggml_backend_hrx2_ensure_device_scratch(context, &context->route_scratch, size, out_ref);
}

static uint64_t ggml_backend_hrx2_node_mul_mat_bytes(const ggml_tensor * node) {
    if (!node || !node->src[0]) {
        return 0;
    }
    if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }
    return static_cast<uint64_t>(ggml_nbytes(node->src[0]));
}

static void ggml_backend_hrx2_begin_submit_batch(ggml_backend_hrx2_context * context) {
    if (!context) {
        return;
    }
    const uint64_t max_bytes = ggml_backend_hrx2_max_mul_mat_bytes_per_submit();
    const uint64_t last_scaled = context->last_total_mul_mat_bytes / 40u;
    context->submitted_dispatches = 0;
    context->mul_mat_bytes = 0;
    context->total_mul_mat_bytes = 0;
    context->mul_mat_bytes_per_submit = std::min(max_bytes, last_scaled);
    context->submit_count = 0;
    context->submit_flush_count = 0;
    context->submit_last_node = nullptr;
}

static hrx_status_t ggml_backend_hrx2_maybe_submit_batch_after_dispatch(hrx_stream_t stream) {
    ggml_backend_hrx2_context * context = g_hrx2_active_graph_context;
    if (!context || stream != context->stream || ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_SUBMIT_BATCHING")) {
        return hrx_ok_status();
    }

    context->submitted_dispatches++;
    const ggml_tensor * node = g_hrx2_active_graph_node;
    if (node && node != context->submit_last_node) {
        const uint64_t matmul_bytes = ggml_backend_hrx2_node_mul_mat_bytes(node);
        context->submit_last_node = node;
        context->mul_mat_bytes += matmul_bytes;
        context->total_mul_mat_bytes += matmul_bytes;
    }

    const uint64_t dispatches_per_submit = ggml_backend_hrx2_dispatches_per_submit();
    const bool dispatch_threshold =
        dispatches_per_submit != 0 && context->submitted_dispatches >= dispatches_per_submit;
    const bool byte_threshold =
        context->mul_mat_bytes_per_submit != 0 && context->mul_mat_bytes >= context->mul_mat_bytes_per_submit;
    if (!dispatch_threshold && !byte_threshold) {
        return hrx_ok_status();
    }

    const uint64_t start_us = ggml_backend_hrx2_now_us();
    ggml_backend_hrx2_trace_event(
        "submit_batch_flush_begin",
        ggml_backend_hrx2_json_kv("dispatches", context->submitted_dispatches) + "," +
        ggml_backend_hrx2_json_kv("mul_mat_bytes", context->mul_mat_bytes) + "," +
        ggml_backend_hrx2_json_kv("mul_mat_bytes_per_submit", context->mul_mat_bytes_per_submit) + "," +
        ggml_backend_hrx2_json_kv("submit_count", context->submit_count));
    hrx_status_t status = ::hrx_stream_flush(stream);
    ggml_backend_hrx2_trace_event(
        "submit_batch_flush_end",
        ggml_backend_hrx2_json_kv("status_ok", static_cast<uint64_t>(hrx_status_is_ok(status) ? 1 : 0)) + "," +
        ggml_backend_hrx2_json_kv("elapsed_us", ggml_backend_hrx2_now_us() - start_us));
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    context->submitted_dispatches = 0;
    context->mul_mat_bytes = 0;
    if (context->submit_count < 3) {
        context->mul_mat_bytes_per_submit *= 2;
    }
    context->submit_count++;
    context->submit_flush_count++;
    return hrx_ok_status();
}

static hrx_status_t ggml_backend_hrx2_traced_stream_dispatch(
        hrx_stream_t stream,
        hrx_executable_t executable,
        uint32_t export_ordinal,
        const hrx_dispatch_config_t * config,
        const void * constants,
        size_t constants_size,
        const hrx_buffer_ref_t * bindings,
        size_t binding_count,
        uint32_t flags) {
    GGML_UNUSED(constants);
    GGML_UNUSED(bindings);
    const uint64_t start_us = ggml_backend_hrx2_now_us();
    ggml_backend_hrx2_trace_event(
        "hrx_stream_dispatch_begin",
        ggml_backend_hrx2_json_kv("export_ordinal", export_ordinal) + "," +
        ggml_backend_hrx2_json_kv("workgroups_x", config ? config->workgroup_count[0] : 0) + "," +
        ggml_backend_hrx2_json_kv("workgroups_y", config ? config->workgroup_count[1] : 0) + "," +
        ggml_backend_hrx2_json_kv("workgroups_z", config ? config->workgroup_count[2] : 0) + "," +
        ggml_backend_hrx2_json_kv("workgroup_size_x", config ? config->workgroup_size[0] : 0) + "," +
        ggml_backend_hrx2_json_kv("constants_size", static_cast<uint64_t>(constants_size)) + "," +
        ggml_backend_hrx2_json_kv("binding_count", static_cast<uint64_t>(binding_count)) + "," +
        ggml_backend_hrx2_json_kv("flags", static_cast<uint64_t>(flags)));

    hrx_status_t status = ::hrx_stream_dispatch(
        stream,
        executable,
        export_ordinal,
        config,
        constants,
        constants_size,
        bindings,
        binding_count,
        flags);

    ggml_backend_hrx2_trace_event(
        "hrx_stream_dispatch_end",
        ggml_backend_hrx2_json_kv("status_ok", static_cast<uint64_t>(hrx_status_is_ok(status) ? 1 : 0)) + "," +
        ggml_backend_hrx2_json_kv("elapsed_us", ggml_backend_hrx2_now_us() - start_us));
    if (hrx_status_is_ok(status)) {
        status = ggml_backend_hrx2_maybe_submit_batch_after_dispatch(stream);
    }
    if (hrx_status_is_ok(status) && ggml_backend_hrx2_env_enabled("GGML_HRX2_SYNC_AFTER_DISPATCH")) {
        const uint64_t sync_start_us = ggml_backend_hrx2_now_us();
        ggml_backend_hrx2_trace_event("hrx_stream_synchronize_begin", ggml_backend_hrx2_json_kv("reason", "after_dispatch"));
        status = ::hrx_stream_synchronize(stream);
        ggml_backend_hrx2_trace_event(
            "hrx_stream_synchronize_end",
            ggml_backend_hrx2_json_kv("reason", "after_dispatch") + "," +
            ggml_backend_hrx2_json_kv("status_ok", static_cast<uint64_t>(hrx_status_is_ok(status) ? 1 : 0)) + "," +
            ggml_backend_hrx2_json_kv("elapsed_us", ggml_backend_hrx2_now_us() - sync_start_us));
    }
    return status;
}

static hrx_status_t ggml_backend_hrx2_traced_stream_synchronize(hrx_stream_t stream) {
    const uint64_t start_us = ggml_backend_hrx2_now_us();
    ggml_backend_hrx2_trace_event("hrx_stream_synchronize_begin", "");
    const hrx_status_t status = ::hrx_stream_synchronize(stream);
    ggml_backend_hrx2_trace_event(
        "hrx_stream_synchronize_end",
        ggml_backend_hrx2_json_kv("status_ok", static_cast<uint64_t>(hrx_status_is_ok(status) ? 1 : 0)) + "," +
        ggml_backend_hrx2_json_kv("elapsed_us", ggml_backend_hrx2_now_us() - start_us));
    return status;
}

static hrx_status_t ggml_backend_hrx2_traced_stream_flush(hrx_stream_t stream) {
    const uint64_t start_us = ggml_backend_hrx2_now_us();
    ggml_backend_hrx2_trace_event("hrx_stream_flush_begin", "");
    const hrx_status_t status = ::hrx_stream_flush(stream);
    ggml_backend_hrx2_trace_event(
        "hrx_stream_flush_end",
        ggml_backend_hrx2_json_kv("status_ok", static_cast<uint64_t>(hrx_status_is_ok(status) ? 1 : 0)) + "," +
        ggml_backend_hrx2_json_kv("elapsed_us", ggml_backend_hrx2_now_us() - start_us));
    return status;
}

#define hrx_stream_dispatch ggml_backend_hrx2_traced_stream_dispatch
#define hrx_stream_synchronize ggml_backend_hrx2_traced_stream_synchronize
#define hrx_stream_flush ggml_backend_hrx2_traced_stream_flush

static ggml_backend_hrx2_device_context * ggml_backend_hrx2_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx2_device_context *>(dev->context);
}

static ggml_backend_hrx2_context * ggml_backend_hrx2_get_context(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx2_context *>(backend->context);
}

static bool ggml_backend_hrx2_route_available(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route) {
    return route && (route->target_key.empty() || route->target_key == device_context->architecture);
}

static std::string ggml_backend_hrx2_base_cache_key(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route) {
    std::string cache_key = route ? route->id : std::string();
    cache_key += "|target=";
    cache_key += device_context ? device_context->architecture : std::string();
    return cache_key;
}

static ggml_backend_hrx2_provider * ggml_backend_hrx2_get_provider(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const std::vector<ggml_backend_hrx2_config_binding> & config_bindings,
        const std::string & cache_key) {
    if (!ggml_backend_hrx2_route_available(device_context, route) || !device_context->catalog) {
        return nullptr;
    }

    auto existing = device_context->providers.find(cache_key);
    if (existing != device_context->providers.end()) {
        ggml_backend_hrx2_trace_event(
            "provider_cache",
            ggml_backend_hrx2_json_kv("status", "hit") + "," +
            ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", cache_key));
        return existing->second.get();
    }
    if (device_context->provider_failures.find(cache_key) != device_context->provider_failures.end()) {
        ggml_backend_hrx2_trace_event(
            "provider_cache",
            ggml_backend_hrx2_json_kv("status", "failed_memo") + "," +
            ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", cache_key));
        return nullptr;
    }

    ggml_backend_hrx2_trace_event(
        "provider_cache",
        ggml_backend_hrx2_json_kv("status", "miss") + "," +
        ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
        ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("cache_key", cache_key));

    const ggml_backend_hrx2_device_info jit_device = {
        /* .device       = */ device_context->device,
        /* .architecture = */ device_context->architecture.c_str(),
    };
    auto provider = ggml_backend_hrx2_load_provider(jit_device, *device_context->catalog, *route, config_bindings, cache_key);
    if (!provider) {
        ggml_backend_hrx2_trace_event(
            "provider_compile",
            ggml_backend_hrx2_json_kv("status", "failed") + "," +
            ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", cache_key));
        device_context->provider_failures.insert(cache_key);
        return nullptr;
    }
    ggml_backend_hrx2_provider * provider_ptr = provider.get();
    ggml_backend_hrx2_trace_event(
        "provider_compile",
        ggml_backend_hrx2_json_kv("status", "success") + "," +
        ggml_backend_hrx2_json_kv("route_id", route->id) + "," +
        ggml_backend_hrx2_json_kv("target_key", device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("cache_key", cache_key) + "," +
        ggml_backend_hrx2_json_kv("compile_report_bytes", static_cast<uint64_t>(provider_ptr->compile_report_json.size())) + "," +
        ggml_backend_hrx2_json_kv("manifest_bytes", static_cast<uint64_t>(provider_ptr->manifest_json.size())));
    device_context->providers.emplace(cache_key, std::move(provider));
    return provider_ptr;
}

static ggml_backend_hrx2_buffer_context * ggml_backend_hrx2_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx2_buffer_context *>(buffer->context);
}

static ggml_backend_hrx2_buffer_type_context * ggml_backend_hrx2_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx2_buffer_type_context *>(buft->context);
}

static size_t ggml_backend_hrx2_total_memory(hrx_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_HRX2_CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_hrx2_device_string_property(hrx_device_t device, hrx_device_property_t property) {
    std::array<char, 128> value = {};
    if (!GGML_HRX2_CHECK(hrx_device_get_property(device, property, value.data(), value.size()))) {
        return {};
    }
    return std::string(value.data());
}

static std::string ggml_backend_hrx2_device_description(hrx_device_t device) {
    std::string name = ggml_backend_hrx2_device_string_property(device, HRX_DEVICE_PROPERTY_NAME);
    std::string arch = ggml_backend_hrx2_device_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
    if (name.empty()) {
        name = "HRX GPU";
    }
    if (!arch.empty()) {
        name += " (";
        name += arch;
        name += ")";
    }
    return name;
}

static void * ggml_backend_hrx2_buffer_get_base(ggml_backend_buffer_t buffer);

static size_t ggml_backend_hrx2_tensor_offset(const ggml_backend_hrx2_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static bool ggml_backend_hrx2_tensor_buffer_ref(const ggml_tensor * tensor, hrx_buffer_ref_t * out_ref) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx2_buffer_get_base) {
        return false;
    }

    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    const size_t offset = ggml_backend_hrx2_tensor_offset(context, tensor);
    const size_t length = ggml_nbytes(tensor);
    if (!context->buffer || offset > buffer->size || length > buffer->size - offset) {
        return false;
    }

    *out_ref = {
        /* .buffer = */ context->buffer,
        /* .offset = */ offset,
        /* .length = */ length,
    };
    return true;
}

static bool ggml_backend_hrx2_stage_and_copy_tensor(
        ggml_backend_hrx2_buffer_context * context,
        const ggml_tensor * tensor,
        const void * data,
        size_t buffer_offset,
        size_t buffer_size,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: upload for tensor %s exceeds HRX2 buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx2_retain_active_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX2 stream available for tensor upload\n", __func__);
        return false;
    }


    std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
    auto * arena = ggml_backend_hrx2_get_staging_arena_locked(context->device_context, stream);
    if (!arena ||
        !ggml_backend_hrx2_ensure_staging_buffer_locked(
            context->device_context,
            arena,
            ggml_backend_hrx2_staging_arena_capacity())) {
        hrx_stream_release(stream);
        return false;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t uploaded = 0;
    bool ok = true;
    while (uploaded < size) {
        size_t staging_offset = ggml_backend_hrx2_align_up(arena->offset, GGML_HRX2_ALIGNMENT);
        if (staging_offset >= arena->capacity) {
            ok = GGML_HRX2_CHECK(hrx_stream_flush(stream)) && GGML_HRX2_CHECK(hrx_stream_wait(stream));
            if (!ok) {
                break;
            }
            ggml_backend_hrx2_reset_staging_arena_locked(*arena);
            staging_offset = 0;
        }

        const size_t available = arena->capacity - staging_offset;
        const size_t chunk_size = std::min(size - uploaded, available);
        if (chunk_size == 0) {
            GGML_LOG_ERROR("%s: HRX2 staging arena has no available space\n", __func__);
            ok = false;
            break;
        }

        std::memcpy(arena->mapped + staging_offset, bytes + uploaded, chunk_size);
        ok = GGML_HRX2_CHECK(hrx_stream_copy_buffer(
            stream,
            arena->buffer,
            staging_offset,
            context->buffer,
            buffer_offset + uploaded,
            chunk_size));
        if (!ok) {
            break;
        }

        arena->offset = ggml_backend_hrx2_align_up(staging_offset + chunk_size, GGML_HRX2_ALIGNMENT);
        uploaded += chunk_size;
    }

    // 1bit-MONSTER fix: make the upload synchronous so tensors uploaded on
    // the transfer stream (or any non-graph stream) are visible to graph
    // dispatches on other streams. The async staging copies previously raced
    // the graph stream and the staging arena could be reused mid-flight,
    // corrupting cross-backend activation copies (e.g. the Q4NX op's src1).
    if (ok) {
        ok = GGML_HRX2_CHECK(hrx_stream_flush(stream)) && GGML_HRX2_CHECK(hrx_stream_wait(stream));
        if (ok) {
            if (auto * arena_sync = ggml_backend_hrx2_find_staging_arena_locked(context->device_context, stream)) {
                ggml_backend_hrx2_reset_staging_arena_locked(*arena_sync);
            }
        }
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx2_copy_tensor_to_staging(
        ggml_backend_hrx2_buffer_context * context,
        const ggml_tensor * tensor,
        size_t buffer_offset,
        size_t buffer_size,
        void * data,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: readback for tensor %s exceeds HRX2 buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }


    hrx_stream_t stream = ggml_backend_hrx2_retain_active_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX2 stream available for tensor readback\n", __func__);
        return false;
    }

    auto * out_bytes = static_cast<uint8_t *>(data);
    size_t copied = 0;
    bool ok = true;
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        auto * arena = ggml_backend_hrx2_get_staging_arena_locked(context->device_context, stream);
        if (!arena ||
            !ggml_backend_hrx2_ensure_staging_buffer_locked(
                context->device_context,
                arena,
                ggml_backend_hrx2_staging_arena_capacity())) {
            hrx_stream_release(stream);
            return false;
        }

        while (copied < size) {
            size_t staging_offset = ggml_backend_hrx2_align_up(arena->offset, GGML_HRX2_ALIGNMENT);
            if (staging_offset >= arena->capacity) {
                ok = GGML_HRX2_CHECK(hrx_stream_synchronize(stream));
                if (!ok) {
                    break;
                }
                ggml_backend_hrx2_reset_staging_arena_locked(*arena);
                staging_offset = 0;
            }

            const size_t chunk_size = std::min(size - copied, arena->capacity - staging_offset);
            if (chunk_size == 0) {
                GGML_LOG_ERROR("%s: HRX2 staging arena has no available space\n", __func__);
                ok = false;
                break;
            }

            ok = GGML_HRX2_CHECK(hrx_stream_copy_buffer(
                stream,
                context->buffer,
                buffer_offset + copied,
                arena->buffer,
                staging_offset,
                chunk_size));
            if (!ok) {
                break;
            }

            ok = GGML_HRX2_CHECK(hrx_stream_synchronize(stream));
            if (!ok) {
                break;
            }
            std::memcpy(out_bytes + copied, arena->mapped + staging_offset, chunk_size);
            copied += chunk_size;
            ggml_backend_hrx2_reset_staging_arena_locked(*arena);
        }
    }

    hrx_stream_release(stream);
    return ok;
}

static const char * ggml_backend_hrx2_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_hrx2_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_hrx2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (context->buffer) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_hrx2_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx2_get_buffer_context(buffer)->base;
}

static void ggml_backend_hrx2_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }
    if (!ggml_backend_hrx2_sync_streams(context->device_context)) {
        return;
    }
    const size_t buffer_offset = ggml_backend_hrx2_tensor_offset(context, tensor) + offset;
    const bool ok = ggml_backend_hrx2_queue_fill_stream_sync(
        context->device_context, context->buffer, buffer_offset, size, &value, sizeof(value));
    GGML_UNUSED(ok);
}

static void ggml_backend_hrx2_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }
    const size_t buffer_offset = ggml_backend_hrx2_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx2_stage_and_copy_tensor(context, tensor, data, buffer_offset, buffer->size, size)) {
        GGML_LOG_ERROR("%s: failed to upload tensor %s through HRX2 staging\n", __func__, tensor->name);
    }
}

static void ggml_backend_hrx2_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }
    const size_t buffer_offset = ggml_backend_hrx2_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx2_copy_tensor_to_staging(context, tensor, buffer_offset, buffer->size, data, size)) {
        GGML_LOG_ERROR("%s: failed to read tensor %s through HRX2 staging\n", __func__, tensor->name);
    }
}

static bool ggml_backend_hrx2_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (!src_buffer || src_buffer->iface.get_base != ggml_backend_hrx2_buffer_get_base) {
        return false;
    }

    auto * dst_context = ggml_backend_hrx2_get_buffer_context(buffer);
    auto * src_context = ggml_backend_hrx2_get_buffer_context(src_buffer);
    if (dst_context->device_context != src_context->device_context ||
        !dst_context->buffer || !src_context->buffer) {
        return false;
    }

    if (!ggml_backend_hrx2_sync_streams(dst_context->device_context)) {
        return false;
    }

    const size_t src_offset = ggml_backend_hrx2_tensor_offset(src_context, src);
    const size_t dst_offset = ggml_backend_hrx2_tensor_offset(dst_context, dst);
    const size_t size = ggml_nbytes(src);
    if (dst_offset > buffer->size || size > buffer->size - dst_offset) {
        GGML_LOG_ERROR(
            "%s: destination tensor %s exceeds HRX2 buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, dst ? dst->name : "<unknown>", dst_offset, size, buffer->size);
        return false;
    }
    if (src_offset > src_buffer->size || size > src_buffer->size - src_offset) {
        GGML_LOG_ERROR(
            "%s: source tensor %s exceeds HRX2 buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, src ? src->name : "<unknown>", src_offset, size, src_buffer->size);
        return false;
    }

    return ggml_backend_hrx2_queue_copy_stream_sync(
        dst_context->device_context,
        src_context->buffer,
        src_offset,
        dst_context->buffer,
        dst_offset,
        size);
}

static void ggml_backend_hrx2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_hrx2_get_buffer_context(buffer);
    if (buffer->size == 0 || !context->buffer) {
        return;
    }
    if (!ggml_backend_hrx2_sync_streams(context->device_context)) {
        return;
    }
    const bool ok = ggml_backend_hrx2_queue_fill_stream_sync(
        context->device_context, context->buffer, 0, buffer->size, &value, sizeof(value));
    GGML_UNUSED(ok);
}

static const ggml_backend_buffer_i ggml_backend_hrx2_buffer_i = {
    /* .free_buffer   = */ ggml_backend_hrx2_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx2_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx2_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx2_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx2_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_hrx2_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx2_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_hrx2_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx2_get_buft_context(buft);

    // DEVICE-LOCAL default buft (weights): the NPU reads these directly; the
    // CPU never touches them (fake base). Activations use the separate
    // host-visible buft (see host_buffer_type) for zero-copy compute.
    hrx_buffer_params_t params = {
        /* .type           = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
        /* .access         = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT,
        /* .queue_affinity = */ 0,
    };
    hrx_buffer_t hrx_buffer = nullptr;
    if (size != 0 && !GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device), params, size, &hrx_buffer))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx2_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ hrx_buffer,
        /* .base           = */ reinterpret_cast<uint8_t *>(GGML_HRX2_FAKE_PTR_BASE),
    };
    if (!context) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_hrx2_buffer_i, context, size);
    if (!buffer) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        delete context;
    }
    return buffer;
}

static ggml_backend_buffer_t ggml_backend_hrx2_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx2_get_buft_context(buft);

    hrx_buffer_params_t params = {
        /* .type           = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE | HRX_MEMORY_TYPE_HOST_COHERENT,
        /* .access         = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT |
                               HRX_BUFFER_USAGE_MAPPING_SCOPED |
                               HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    hrx_buffer_t hrx_buffer = nullptr;
    if (size != 0 && !GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device), params, size, &hrx_buffer))) {
        return nullptr;
    }

    void * mapped = nullptr;
    if (size != 0 && !GGML_HRX2_CHECK(hrx_buffer_map(
            hrx_buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &mapped))) {
        hrx_buffer_release(hrx_buffer);
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx2_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ hrx_buffer,
        /* .base           = */ static_cast<uint8_t *>(mapped),
    };
    if (!context) {
        hrx_buffer_release(hrx_buffer);
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_hrx2_buffer_i, context, size);
    if (!buffer) {
        hrx_buffer_release(hrx_buffer);
        delete context;
    }
    return buffer;
}

static bool ggml_backend_hrx2_host_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;  // zero-copy: host-visible buffers the CPU can read/write in place
}

static size_t ggml_backend_hrx2_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX2_ALIGNMENT;
}

static size_t ggml_backend_hrx2_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * context = ggml_backend_hrx2_get_buft_context(buft)->device_context;
    return context->memory_total ? context->memory_total : std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_hrx2_host_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx2_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx2_host_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx2_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx2_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ ggml_backend_hrx2_host_buft_is_host,
};

static const ggml_backend_buffer_type_i ggml_backend_hrx2_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx2_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx2_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx2_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx2_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_hrx2_device_buffer_type(ggml_backend_dev_t dev) {
    // CONFIG E: DEFAULT buft = host-visible (activations GTT)
    return &ggml_backend_hrx2_get_device_context(dev)->host_buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hrx2_device_host_buffer_type(ggml_backend_dev_t dev) {
    return &ggml_backend_hrx2_get_device_context(dev)->host_buffer_type;
}

static bool ggml_backend_hrx2_supports_rms_norm(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_RMS_NORM &&
           src0 &&
           op->view_src == nullptr &&
           op->type == GGML_TYPE_F32 &&
           src0->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_q8_0(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q8_0 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_f32_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    // batches (ne2/ne3): GQA attention broadcasts src0's kv-head batch over
    // src1's q-head batch. Each batch is a contiguous [k, rows]/[k, cols]/
    // [rows, cols] block (nb1 == packed row stride); batch strides may be
    // strided (KV-cache views), handled via nb2/nb3 in dispatch.
    // require src1->ne[2] % src0->ne[2] == 0 with src0->ne[2] <= src1->ne[2].
    const int64_t inner_src0 = k * rows;
    const int64_t inner_src1 = k * cols;
    const int64_t inner_dst = rows * cols;
    const int64_t n_batch = src1->ne[2] * src1->ne[3];
    if (n_batch > 1) {
        const int64_t n_batch_src0 = src0->ne[2] * src0->ne[3];
        if (src0->ne[3] != src1->ne[3] ||
            src0->ne[3] != op->ne[3] ||
            n_batch_src0 > n_batch ||
            (n_batch % n_batch_src0) != 0) {
            return false;
        }
        // contiguous within each batch: nb1 == packed row stride
        if (src0->nb[1] != (int64_t) k * (int64_t) sizeof(float) ||
            src1->nb[1] != (int64_t) k * (int64_t) sizeof(float) ||
            op->nb[1] != (int64_t) rows * (int64_t) sizeof(float)) {
            return false;
        }
    }
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k > 0 && rows > 0 && cols > 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_q4_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q4_K ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_id_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op,
        ggml_type src0_type) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    if (op->op != GGML_OP_MUL_MAT_ID ||
        !src0 ||
        !src1 ||
        !src2 ||
        op->view_src != nullptr ||
        src0->type != src0_type ||
        src1->type != GGML_TYPE_F32 ||
        src2->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t nexperts = src0->ne[2];
    const int64_t nselected = src2->ne[0];
    const int64_t ntokens = op->ne[2];
    const int64_t block_size = ggml_blck_size(src0->type);
    return k > 0 &&
           rows > 0 &&
           nexperts > 0 &&
           nselected > 0 &&
           ntokens > 0 &&
           src0->ne[0] == src1->ne[0] &&
           src0->ne[3] == 1 &&
           (src1->ne[1] == 1 || src1->ne[1] == nselected) &&
           src1->ne[2] == ntokens &&
           src1->ne[3] == 1 &&
           src2->ne[1] == ntokens &&
           src2->ne[2] == 1 &&
           src2->ne[3] == 1 &&
           op->ne[0] == rows &&
           op->ne[1] == nselected &&
           op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           src1->nb[0] == sizeof(float) &&
           src1->nb[1] % sizeof(float) == 0 &&
           src1->nb[2] % sizeof(float) == 0 &&
           src2->nb[0] == sizeof(int32_t) &&
           src2->nb[1] % sizeof(int32_t) == 0 &&
           op->nb[0] == sizeof(float) &&
           op->nb[1] == static_cast<size_t>(rows) * sizeof(float) &&
           op->nb[2] % sizeof(float) == 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           nexperts <= std::numeric_limits<uint32_t>::max() &&
           nselected <= std::numeric_limits<uint32_t>::max() &&
           ntokens <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[1] / sizeof(float) <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[2] / sizeof(float) <= std::numeric_limits<uint32_t>::max() &&
           src2->nb[1] / sizeof(int32_t) <= std::numeric_limits<uint32_t>::max() &&
           op->nb[2] / sizeof(float) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q4_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k(device_context, op, GGML_TYPE_Q4_K);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q5_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k(device_context, op, GGML_TYPE_Q5_K);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q6_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k(device_context, op, GGML_TYPE_Q6_K);
}

static bool ggml_backend_hrx2_supports_mul_mat_q6_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q6_K ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_q5_k(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q5_K ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    const int64_t block_size = ggml_blck_size(src0->type);
    return src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           block_size > 0 &&
           (k % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           k >= 0 && rows >= 0 && cols >= 0 &&
           k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_mul_mat_f16_f32_batched(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F16 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        src0->nb[0] != ggml_type_size(GGML_TYPE_F16) ||
        src1->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float)) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];
    if (k <= 0 ||
        rows <= 0 ||
        cols <= 0 ||
        src0->ne[0] != src1->ne[0] ||
        op->ne[0] != src0->ne[1] ||
        op->ne[1] != src1->ne[1] ||
        op->ne[2] != src1->ne[2] ||
        op->ne[3] != src1->ne[3] ||
        src0->ne[2] <= 0 ||
        src0->ne[3] <= 0 ||
        op->ne[2] <= 0 ||
        op->ne[3] <= 0 ||
        op->ne[2] % src0->ne[2] != 0 ||
        op->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    return k <= std::numeric_limits<uint32_t>::max() &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max() &&
           op->ne[2] <= std::numeric_limits<uint32_t>::max() &&
           op->ne[3] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[2] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[3] <= std::numeric_limits<uint32_t>::max() &&
           src0->nb[1] <= std::numeric_limits<uint32_t>::max() &&
           src0->nb[2] <= std::numeric_limits<uint32_t>::max() &&
           src0->nb[3] <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[1] <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[2] <= std::numeric_limits<uint32_t>::max() &&
           src1->nb[3] <= std::numeric_limits<uint32_t>::max() &&
           op->nb[1] <= std::numeric_limits<uint32_t>::max() &&
           op->nb[2] <= std::numeric_limits<uint32_t>::max() &&
           op->nb[3] <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_u32(int64_t value, uint32_t * out_value) {
    if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool ggml_backend_hrx2_u32_size(size_t value, uint32_t * out_value) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool ggml_backend_hrx2_flat_row_stride_f32(
        const ggml_tensor * tensor,
        uint32_t * out_row_stride) {
    if (!tensor ||
        tensor->type != GGML_TYPE_F32 ||
        tensor->nb[0] != sizeof(float) ||
        tensor->ne[0] <= 0 ||
        tensor->nb[1] % sizeof(float) != 0) {
        return false;
    }
    const size_t row_stride = tensor->nb[1] / sizeof(float);
    if (row_stride < static_cast<size_t>(tensor->ne[0]) ||
        !ggml_backend_hrx2_u32_size(row_stride, out_row_stride)) {
        return false;
    }
    if (tensor->ne[2] > 1) {
        if (tensor->nb[2] % sizeof(float) != 0 ||
            tensor->nb[2] / sizeof(float) != row_stride * static_cast<size_t>(tensor->ne[1])) {
            return false;
        }
    }
    if (tensor->ne[3] > 1) {
        if (tensor->nb[3] % sizeof(float) != 0 ||
            tensor->nb[3] / sizeof(float) != row_stride * static_cast<size_t>(tensor->ne[1]) * static_cast<size_t>(tensor->ne[2])) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx2_supports_set_rows(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    return op->op == GGML_OP_SET_ROWS &&
           src0 &&
           src1 &&
           src2 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I64 &&
           src2->type == op->type &&
           (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           op->ne[1] > 0 &&
           src1->ne[0] == src0->ne[1] &&
           src0->ne[2] % src1->ne[1] == 0 &&
           src0->ne[3] % src1->ne[2] == 0 &&
           src1->ne[3] == 1 &&
           ggml_is_contiguous_rows(src0) &&
           ggml_is_contiguous_rows(op);
}

static bool ggml_backend_hrx2_supports_pointwise_binary(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if ((op->op != GGML_OP_ADD && op->op != GGML_OP_MUL && op->op != GGML_OP_DIV) ||
        !src0 || !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        !ggml_is_contiguous(op) ||
        src0->ne[0] <= 0 ||
        ggml_nrows(src0) <= 0 ||
        src0->ne[0] > std::numeric_limits<uint32_t>::max() ||
        ggml_nrows(src0) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) > 1073741824ULL) {
        return false;
    }

    uint32_t src0_row_stride = 0;
    uint32_t src1_row_stride = 0;
    if (!ggml_backend_hrx2_flat_row_stride_f32(src0, &src0_row_stride)) {
        return false;
    }

    const int64_t nrows = ggml_nrows(src0);
    const bool same_shape_row_strided =
        ggml_are_same_shape(src1, src0) &&
        ggml_backend_hrx2_flat_row_stride_f32(src1, &src1_row_stride);
    const bool row_broadcast =
        src1->ne[0] == src0->ne[0] &&
        ggml_nrows(src1) == 1 &&
        src1->nb[0] == sizeof(float);
    const bool col_broadcast =
        src1->ne[0] == 1 &&
        ggml_nrows(src1) == nrows &&
        ggml_backend_hrx2_flat_row_stride_f32(src1, &src1_row_stride);
    const bool scalar_broadcast =
        src1->ne[0] == 1 &&
        ggml_nrows(src1) == 1 &&
        src1->nb[0] == sizeof(float);
    return same_shape_row_strided || row_broadcast || col_broadcast || scalar_broadcast;
}

static bool ggml_backend_hrx2_supports_scale(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    // zero-size op (worst-case reserve builds empty state views): trivially
    // supported (the dispatch copies nothing)
    if (op->op == GGML_OP_SCALE && src0 && ggml_nelements(src0) == 0) {
        return true;
    }
    // allow in-place scale (e.g. the recurrent-state zeroing in build_rs):
    // op->view_src points at a view of src0 that shares its buffer; the
    // dispatch resolves both through view_src to the same region and the
    // kernel writes in place. Contiguity + shape still gate acceptance.
    return op->op == GGML_OP_SCALE &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_clamp(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_CLAMP &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_sum_rows(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    uint32_t src0_row_stride = 0;
    return op->op == GGML_OP_SUM_ROWS &&
           src0 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           op->ne[0] == 1 &&
           op->ne[1] == src0->ne[1] &&
           op->ne[2] == src0->ne[2] &&
           op->ne[3] == src0->ne[3] &&
           ggml_backend_hrx2_flat_row_stride_f32(src0, &src0_row_stride) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_get_rows_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const uint64_t max_dense_elements = 1073741824ULL;
    return op->op == GGML_OP_GET_ROWS &&
           src0 &&
           src1 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[1] > 0 &&
           src0->ne[2] == 1 &&
           src0->ne[3] == 1 &&
           src1->ne[0] == op->ne[1] &&
           src1->ne[1] == 1 &&
           src1->ne[2] == 1 &&
           src1->ne[3] == 1 &&
           op->ne[2] == 1 &&
           op->ne[3] == 1 &&
           src0->ne[0] > 0 &&
           src1->ne[0] > 0 &&
           src0->nb[0] == sizeof(float) &&
           src1->nb[0] == sizeof(int32_t) &&
           op->nb[0] == sizeof(float) &&
           src0->nb[1] % sizeof(float) == 0 &&
           op->nb[1] % sizeof(float) == 0 &&
           src0->nb[1] / sizeof(float) == static_cast<size_t>(src0->ne[0]) &&
           op->nb[1] / sizeof(float) == static_cast<size_t>(op->ne[0]) &&
           src0->nb[1] / sizeof(float) <= 65536 &&
           op->nb[1] / sizeof(float) <= 65536 &&
           ggml_is_contiguous(op) &&
           static_cast<uint64_t>(src0->ne[0]) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src1->ne[0]) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(src1->ne[0]) <= max_dense_elements;
}

static bool ggml_backend_hrx2_supports_get_rows_quantized(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op,
        ggml_type src0_type) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const uint64_t max_dense_elements = 1073741824ULL;
    const int64_t block_size = ggml_blck_size(src0_type);
    return op->op == GGML_OP_GET_ROWS &&
           src0 &&
           src1 &&
           op->view_src == nullptr &&
           src0->type == src0_type &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           block_size > 0 &&
           src0->ne[0] == op->ne[0] &&
           (src0->ne[0] % block_size) == 0 &&
           src0->ne[1] > 0 &&
           src0->ne[2] == 1 &&
           src0->ne[3] == 1 &&
           src1->ne[0] == op->ne[1] &&
           src1->ne[1] == 1 &&
           src1->ne[2] == 1 &&
           src1->ne[3] == 1 &&
           op->ne[2] == 1 &&
           op->ne[3] == 1 &&
           src0->ne[0] > 0 &&
           src1->ne[0] > 0 &&
           src1->nb[0] == sizeof(int32_t) &&
           op->nb[0] == sizeof(float) &&
           src0->nb[1] == ggml_row_size(src0_type, src0->ne[0]) &&
           op->nb[1] % sizeof(float) == 0 &&
           op->nb[1] / sizeof(float) == static_cast<size_t>(op->ne[0]) &&
           op->nb[1] / sizeof(float) <= 65536 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           static_cast<uint64_t>(src0->ne[0]) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src1->ne[0]) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(src1->ne[0]) <= max_dense_elements;
}

static bool ggml_backend_hrx2_supports_get_rows_moe_weights(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return op->op == GGML_OP_GET_ROWS &&
           src0 &&
           src1 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == 1 &&
           src0->ne[1] == 128 &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           src1->ne[0] == op->ne[1] &&
           src1->ne[1] == op->ne[2] &&
           src1->ne[2] == op->ne[3] &&
           src1->ne[3] == 1 &&
           op->ne[0] == 1 &&
           op->ne[1] == 8 &&
           op->ne[2] >= 1 &&
           op->ne[3] == 1 &&
           src0->nb[0] == sizeof(float) &&
           src0->nb[1] == sizeof(float) &&
           src0->nb[2] % sizeof(float) == 0 &&
           src1->nb[0] == sizeof(int32_t) &&
           src1->nb[1] % sizeof(int32_t) == 0 &&
           op->nb[0] == sizeof(float) &&
           op->nb[1] == sizeof(float) &&
           op->nb[2] % sizeof(float) == 0 &&
           src0->nb[2] / sizeof(float) >= static_cast<size_t>(src0->ne[1]) &&
           src1->nb[1] / sizeof(int32_t) >= static_cast<size_t>(src1->ne[0]) &&
           op->nb[2] / sizeof(float) >= static_cast<size_t>(op->ne[1]) &&
           op->ne[2] <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_supports_cont(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_CONT &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->nb[0] == sizeof(float) &&
           ggml_is_contiguous(op) &&
           ggml_nelements(src0) == ggml_nelements(op) &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           src0->ne[0] > 0 &&
           ggml_nrows(src0) > 0 &&
           src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_extract_copy_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_copy_shape * out_shape) {
    if (!out_shape) {
        return false;
    }
    uint32_t n = 0;
    if (!ggml_backend_hrx2_u32(ggml_nelements(op), &n)) {
        return false;
    }
    *out_shape = { /* .n = */ n };
    return true;
}

static bool ggml_backend_hrx2_supports_cpy(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_CPY ||
        !src0 ||
        !src1 ||
        ggml_nelements(src0) != ggml_nelements(op) ||
        ggml_nbytes(src1) != ggml_nbytes(op) ||
        !ggml_is_contiguous(op)) {
        return false;
    }

    if (src0->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F16 &&
        op->type == GGML_TYPE_F16 &&
        ggml_is_contiguous(src0) &&
        device_context) {
        ggml_backend_hrx2_copy_shape shape;
        if (!ggml_backend_hrx2_extract_copy_shape(op, &shape)) {
            return false;
        }
        for (const auto * route : device_context->copy_f32_f16_routes) {
            if (ggml_backend_hrx2_route_available(device_context, route) &&
                shape.n >= route->ncols_min &&
                shape.n <= route->ncols_max) {
                return true;
            }
        }
    }

    return src0->type == src1->type &&
           src0->type == op->type &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           (ggml_is_contiguous(src0) || src0->nb[0] == ggml_type_size(src0->type));
}

static bool ggml_backend_hrx2_supports_rope_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    const int32_t n_dims = ggml_get_op_params_i32(op, 1);
    const int32_t mode = ggml_get_op_params_i32(op, 2);
    if (op->op != GGML_OP_ROPE ||
        !src0 ||
        !src1 ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) ||
        ggml_get_op_params_f32(op, 7) != 0.0f ||
        n_dims <= 0 ||
        n_dims > src0->ne[0] ||
        (n_dims % 2) != 0 ||
        (!src2 && n_dims != src0->ne[0]) ||
        src0->ne[0] <= 0 ||
        (src0->ne[0] % 2) != 0 ||
        src0->ne[1] <= 0 ||
        src0->ne[2] <= 0 ||
        src0->ne[3] != 1 ||
        src1->ne[0] != src0->ne[2] ||
        src1->ne[1] != 1 ||
        src1->ne[2] != 1 ||
        src1->ne[3] != 1 ||
        src0->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float) ||
        src1->nb[0] != sizeof(int32_t) ||
        src0->nb[1] % sizeof(float) != 0 ||
        src0->nb[2] % sizeof(float) != 0 ||
        op->nb[1] % sizeof(float) != 0 ||
        op->nb[2] % sizeof(float) != 0 ||
        src1->nb[0] % sizeof(int32_t) != 0 ||
        src1->nb[1] % sizeof(int32_t) != 0) {
        return false;
    }

    if (src2) {
        if (src2->type != GGML_TYPE_F32 ||
            src2->ne[0] != n_dims / 2 ||
            src2->ne[1] != 1 ||
            src2->ne[2] != 1 ||
            src2->ne[3] != 1 ||
            src2->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(src2)) {
            return false;
        }
    }

    const uint64_t total_pairs =
        static_cast<uint64_t>(src0->ne[0] / 2) *
        static_cast<uint64_t>(src0->ne[1]) *
        static_cast<uint64_t>(src0->ne[2]);
    return src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[1] <= std::numeric_limits<uint32_t>::max() &&
           src0->ne[2] <= std::numeric_limits<uint32_t>::max() &&
           total_pairs <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_soft_max_f32(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    if (op->op != GGML_OP_SOFT_MAX ||
        !src0 ||
        src2 != nullptr ||
        op->view_src != nullptr ||
        src0->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(op) ||
        ggml_get_op_params_f32(op, 1) != 0.0f ||
        src0->ne[0] <= 0) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) ||
            src1->ne[0] != src0->ne[0] ||
            src1->ne[1] < src0->ne[1] ||
            src1->ne[2] <= 0 ||
            src1->ne[3] <= 0 ||
            (src0->ne[2] % src1->ne[2]) != 0 ||
            (src0->ne[3] % src1->ne[3]) != 0 ||
            src1->nb[1] % sizeof(float) != 0 ||
            src1->nb[2] % sizeof(float) != 0 ||
            src1->nb[3] % sizeof(float) != 0) {
            return false;
        }
    }

    return src0->ne[0] <= std::numeric_limits<uint32_t>::max() &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<uint64_t>(src0->ne[0]) * static_cast<uint64_t>(ggml_nrows(src0)) <= 1073741824ULL;
}

static bool ggml_backend_hrx2_supports_swiglu(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const enum ggml_glu_op glu_op = ggml_get_glu_op(op);
    if (op->op != GGML_OP_GLU ||
        !src0 ||
        op->view_src != nullptr ||
        (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) ||
        ggml_get_op_params_i32(op, 1) != 0 ||
        src0->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        src0->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(op) ||
        op->ne[0] <= 0 ||
        ggml_nrows(op) <= 0 ||
        op->ne[0] > std::numeric_limits<uint32_t>::max() ||
        ggml_nrows(op) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (src1) {
        if (src1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) ||
            !ggml_are_same_shape(src0, op) ||
            !ggml_are_same_shape(src1, op)) {
            return false;
        }
    } else if (src0->ne[0] != op->ne[0] * 2 ||
               ggml_nrows(src0) != ggml_nrows(op)) {
        return false;
    }
    const uint64_t total = static_cast<uint64_t>(op->ne[0]) * static_cast<uint64_t>(ggml_nrows(op));
    return total <= 536870912ULL;
}

static bool ggml_backend_hrx2_is_pow2_i64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static bool ggml_backend_hrx2_supports_argsort_f32_i32_desc(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    GGML_UNUSED(device_context);
    const ggml_tensor * src0 = op->src[0];
    return op->op == GGML_OP_ARGSORT &&
           src0 &&
           op->view_src == nullptr &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_I32 &&
           ggml_get_op_params_i32(op, 0) == GGML_SORT_ORDER_DESC &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           src0->ne[0] > 0 &&
           src0->ne[0] <= 256 &&
           ggml_nrows(src0) > 0 &&
           ggml_nrows(src0) <= std::numeric_limits<uint32_t>::max();
}

static bool ggml_backend_hrx2_mul_mat_q8_0_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q8_0(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_f32_f32_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_f32_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_q4_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q4_k(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_extract_mul_mat_q4_k_swiglu_fusion(
        const ggml_tensor * first,
        const ggml_tensor * second,
        const ggml_tensor * swiglu,
        ggml_backend_hrx2_mul_mat_q4_k_swiglu_fusion * out_fusion) {
    if (!first ||
        !second ||
        !swiglu ||
        !out_fusion ||
        first->op != GGML_OP_MUL_MAT ||
        second->op != GGML_OP_MUL_MAT ||
        swiglu->op != GGML_OP_GLU ||
        ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(swiglu, 1) != 0 ||
        !swiglu->src[0] ||
        !swiglu->src[1] ||
        swiglu->src[0]->op != GGML_OP_MUL_MAT ||
        swiglu->src[1]->op != GGML_OP_MUL_MAT ||
        !((swiglu->src[0] == first && swiglu->src[1] == second) ||
          (swiglu->src[0] == second && swiglu->src[1] == first))) {
        return false;
    }

    const ggml_tensor * x = swiglu->src[0];
    const ggml_tensor * gate = swiglu->src[1];
    if (!x->src[0] ||
        !x->src[1] ||
        !gate->src[0] ||
        !gate->src[1] ||
        x->src[1] != gate->src[1] ||
        x->src[0]->type != GGML_TYPE_Q4_K ||
        gate->src[0]->type != GGML_TYPE_Q4_K ||
        !ggml_are_same_shape(x, gate) ||
        !ggml_are_same_shape(x, swiglu) ||
        swiglu->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(swiglu)) {
        return false;
    }

    ggml_backend_hrx2_mul_mat_shape x_shape;
    ggml_backend_hrx2_mul_mat_shape gate_shape;
    if (!ggml_backend_hrx2_mul_mat_q4_k_shape(x, &x_shape) ||
        !ggml_backend_hrx2_mul_mat_q4_k_shape(gate, &gate_shape) ||
        x_shape.k != gate_shape.k ||
        x_shape.rows != gate_shape.rows ||
        x_shape.cols != gate_shape.cols) {
        return false;
    }

    *out_fusion = {
        /* .x     = */ x,
        /* .gate  = */ gate,
        /* .shape = */ x_shape,
    };
    return true;
}

static bool ggml_backend_hrx2_extract_mul_mat_q4_k_packed_swiglu_fusion(
        const ggml_tensor * mul_mat,
        const ggml_tensor * swiglu,
        ggml_backend_hrx2_mul_mat_q4_k_packed_swiglu_fusion * out_fusion) {
    if (!mul_mat ||
        !swiglu ||
        !out_fusion ||
        mul_mat->op != GGML_OP_MUL_MAT ||
        swiglu->op != GGML_OP_GLU ||
        swiglu->src[0] != mul_mat ||
        swiglu->src[1] != nullptr ||
        ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(swiglu, 1) != 0 ||
        swiglu->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(swiglu)) {
        return false;
    }

    ggml_backend_hrx2_mul_mat_shape mul_mat_shape;
    if (!ggml_backend_hrx2_mul_mat_q4_k_shape(mul_mat, &mul_mat_shape) ||
        mul_mat_shape.rows != swiglu->ne[0] * 2 ||
        mul_mat_shape.cols != ggml_nrows(swiglu) ||
        swiglu->ne[0] <= 0 ||
        ggml_nrows(swiglu) <= 0 ||
        swiglu->ne[0] > std::numeric_limits<uint32_t>::max() ||
        ggml_nrows(swiglu) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    *out_fusion = {
        /* .mul_mat = */ mul_mat,
        /* .shape   = */ {
            /* .k    = */ mul_mat_shape.k,
            /* .rows = */ static_cast<uint32_t>(swiglu->ne[0]),
            /* .cols = */ static_cast<uint32_t>(ggml_nrows(swiglu)),
        },
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_id_k_shape(
        const ggml_tensor * op,
        ggml_type src0_type,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    if (!ggml_backend_hrx2_supports_mul_mat_id_k(nullptr, op, src0_type)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    out_shape->k = static_cast<uint32_t>(src0->ne[0]);
    out_shape->rows = static_cast<uint32_t>(src0->ne[1]);
    out_shape->nexperts = static_cast<uint32_t>(src0->ne[2]);
    out_shape->nselected = static_cast<uint32_t>(src2->ne[0]);
    out_shape->ntokens = static_cast<uint32_t>(op->ne[2]);
    out_shape->src1_selected_stride = src1->ne[1] == 1 ? 0 : static_cast<uint32_t>(src1->nb[1] / sizeof(float));
    out_shape->src1_token_stride = static_cast<uint32_t>(src1->nb[2] / sizeof(float));
    out_shape->idx_token_stride = static_cast<uint32_t>(src2->nb[1] / sizeof(int32_t));
    out_shape->dst_token_stride = static_cast<uint32_t>(op->nb[2] / sizeof(float));
    return true;
}

static bool ggml_backend_hrx2_mul_mat_id_q4_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    return ggml_backend_hrx2_mul_mat_id_k_shape(op, GGML_TYPE_Q4_K, out_shape);
}

static bool ggml_backend_hrx2_mul_mat_id_q5_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    return ggml_backend_hrx2_mul_mat_id_k_shape(op, GGML_TYPE_Q5_K, out_shape);
}

static bool ggml_backend_hrx2_mul_mat_id_q6_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_id_shape * out_shape) {
    return ggml_backend_hrx2_mul_mat_id_k_shape(op, GGML_TYPE_Q6_K, out_shape);
}

static bool ggml_backend_hrx2_mul_mat_q6_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q6_k(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_mul_mat_q5_k_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_q5_k(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    *out_shape = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    return true;
}

static bool ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_mul_mat_f16_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_mul_mat_f16_f32_batched(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_mul_mat_f16_shape shape = {};
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.k) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.rows) ||
        !ggml_backend_hrx2_u32(src1->ne[1], &shape.cols) ||
        !ggml_backend_hrx2_u32(op->ne[2], &shape.dst_ne2) ||
        !ggml_backend_hrx2_u32(op->ne[3], &shape.dst_ne3) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.src0_ne2) ||
        !ggml_backend_hrx2_u32(src0->ne[3], &shape.src0_ne3) ||
        !ggml_backend_hrx2_u32_size(src0->nb[1], &shape.src0_stride_row) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2], &shape.src0_stride_ne2) ||
        !ggml_backend_hrx2_u32_size(src0->nb[3], &shape.src0_stride_ne3) ||
        !ggml_backend_hrx2_u32_size(src1->nb[1], &shape.src1_stride_col) ||
        !ggml_backend_hrx2_u32_size(src1->nb[2], &shape.src1_stride_ne2) ||
        !ggml_backend_hrx2_u32_size(src1->nb[3], &shape.src1_stride_ne3) ||
        !ggml_backend_hrx2_u32_size(op->nb[1], &shape.dst_stride_col) ||
        !ggml_backend_hrx2_u32_size(op->nb[2], &shape.dst_stride_ne2) ||
        !ggml_backend_hrx2_u32_size(op->nb[3], &shape.dst_stride_ne3)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_mul_mat_f16_f32_cont_fusion(
        const ggml_tensor * mul_mat,
        const ggml_tensor * permute,
        const ggml_tensor * cont,
        ggml_backend_hrx2_mul_mat_f16_shape * out_shape) {
    if (!out_shape ||
        !mul_mat ||
        !permute ||
        !cont ||
        mul_mat->op != GGML_OP_MUL_MAT ||
        permute->op != GGML_OP_PERMUTE ||
        cont->op != GGML_OP_CONT ||
        permute->src[0] != mul_mat ||
        cont->src[0] != permute ||
        cont->type != GGML_TYPE_F32 ||
        cont->view_src != nullptr ||
        !ggml_is_contiguous(cont)) {
        return false;
    }

    ggml_backend_hrx2_mul_mat_f16_shape shape = {};
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(mul_mat, &shape)) {
        return false;
    }

    return shape.rows > 0 &&
           shape.dst_ne2 > 0 &&
           shape.dst_ne3 == 1 &&
           permute->ne[0] == mul_mat->ne[0] &&
           permute->ne[1] == mul_mat->ne[2] &&
           permute->ne[2] == mul_mat->ne[1] &&
           permute->ne[3] == mul_mat->ne[3] &&
           cont->ne[0] == static_cast<int64_t>(shape.rows) * static_cast<int64_t>(shape.dst_ne2) &&
           cont->ne[1] == shape.cols &&
           cont->ne[2] == shape.dst_ne3 &&
           cont->ne[3] == 1 &&
           cont->nb[0] == static_cast<int64_t>(sizeof(float)) &&
           cont->nb[1] == static_cast<size_t>(cont->ne[0]) * sizeof(float) &&
           (*out_shape = shape, true);
}

static bool ggml_backend_hrx2_extract_flash_attn_fa0_fusion(
        const ggml_tensor * kq,
        const ggml_tensor * soft_max,
        const ggml_tensor * kqv,
        const ggml_tensor * permute,
        const ggml_tensor * cont,
        ggml_backend_hrx2_flash_attn_fa0_shape * out_shape) {
    if (!out_shape ||
        !kq ||
        !soft_max ||
        !kqv ||
        !permute ||
        !cont ||
        kq->op != GGML_OP_MUL_MAT ||
        soft_max->op != GGML_OP_SOFT_MAX ||
        kqv->op != GGML_OP_MUL_MAT ||
        permute->op != GGML_OP_PERMUTE ||
        cont->op != GGML_OP_CONT ||
        soft_max->src[0] != kq ||
        kqv->src[1] != soft_max ||
        permute->src[0] != kqv ||
        cont->src[0] != permute ||
        !kq->src[0] ||
        !kq->src[1] ||
        !soft_max->src[1] ||
        !kqv->src[0] ||
        kq->src[0]->type != GGML_TYPE_F16 ||
        kq->src[1]->type != GGML_TYPE_F32 ||
        kq->type != GGML_TYPE_F32 ||
        soft_max->type != GGML_TYPE_F32 ||
        soft_max->src[1]->type != GGML_TYPE_F32 ||
        kqv->src[0]->type != GGML_TYPE_F16 ||
        kqv->type != GGML_TYPE_F32 ||
        cont->type != GGML_TYPE_F32 ||
        cont->view_src != nullptr ||
        !ggml_is_contiguous(cont) ||
        soft_max->src[2] != nullptr ||
        ggml_get_op_params_f32(soft_max, 1) != 0.0f) {
        return false;
    }

    const ggml_tensor * k = kq->src[0];
    const ggml_tensor * q = kq->src[1];
    const ggml_tensor * mask = soft_max->src[1];
    const ggml_tensor * v = kqv->src[0];

    if (k->ne[0] != 128 ||
        q->ne[0] != 128 ||
        v->ne[1] != 128 ||
        kqv->ne[0] != 128 ||
        permute->ne[0] != 128 ||
        kq->ne[0] != k->ne[1] ||
        kq->ne[1] != q->ne[1] ||
        kq->ne[2] != q->ne[2] ||
        kq->ne[3] != q->ne[3] ||
        soft_max->ne[0] != kq->ne[0] ||
        soft_max->ne[1] != kq->ne[1] ||
        soft_max->ne[2] != kq->ne[2] ||
        soft_max->ne[3] != kq->ne[3] ||
        v->ne[0] != kq->ne[0] ||
        v->ne[2] != k->ne[2] ||
        kqv->ne[1] != q->ne[1] ||
        kqv->ne[2] != q->ne[2] ||
        kqv->ne[3] != q->ne[3] ||
        permute->ne[1] != q->ne[2] ||
        permute->ne[2] != q->ne[1] ||
        permute->ne[3] != q->ne[3] ||
        cont->ne[0] != q->ne[0] * q->ne[2] ||
        cont->ne[1] != q->ne[1] ||
        cont->ne[2] != q->ne[3] ||
        cont->ne[3] != 1 ||
        mask->ne[0] < kq->ne[0] ||
        mask->ne[1] < kq->ne[1] ||
        mask->ne[2] <= 0 ||
        mask->ne[3] <= 0 ||
        (q->ne[2] % k->ne[2]) != 0 ||
        q->ne[3] != 1 ||
        k->ne[3] != 1 ||
        v->ne[3] != 1 ||
        q->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        k->nb[0] != static_cast<int64_t>(sizeof(ggml_fp16_t)) ||
        v->nb[0] != static_cast<int64_t>(sizeof(ggml_fp16_t)) ||
        mask->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        cont->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        cont->nb[1] != static_cast<size_t>(cont->ne[0]) * sizeof(float) ||
        q->ne[1] <= 0 ||
        q->ne[2] <= 0 ||
        k->ne[1] <= 0 ||
        k->ne[2] <= 0 ||
        k->ne[2] > 16) {
        return false;
    }

    ggml_backend_hrx2_flash_attn_fa0_shape shape = {};
    if (!ggml_backend_hrx2_u32(k->ne[0], &shape.D) ||
        !ggml_backend_hrx2_u32(k->ne[1], &shape.KV) ||
        !ggml_backend_hrx2_u32(q->ne[1], &shape.N) ||
        !ggml_backend_hrx2_u32(q->ne[2], &shape.H) ||
        !ggml_backend_hrx2_u32(k->ne[2], &shape.H_KV) ||
        !ggml_backend_hrx2_u32(q->ne[3], &shape.S)) {
        return false;
    }

    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_rms_norm_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_rms_norm_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_rms_norm(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    *out_shape = {
        /* .ncols = */ static_cast<uint32_t>(src0->ne[0]),
        /* .nrows = */ static_cast<uint32_t>(ggml_nrows(src0)),
    };
    return true;
}

static bool ggml_backend_hrx2_extract_rms_norm_mul_fusion(
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul,
        ggml_backend_hrx2_rms_norm_mul_fusion * out_fusion) {
    if (!out_fusion ||
        !rms_norm ||
        !mul ||
        rms_norm->op != GGML_OP_RMS_NORM ||
        mul->op != GGML_OP_MUL ||
        mul->view_src != nullptr ||
        mul->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(rms_norm, mul) ||
        !ggml_is_contiguous(mul)) {
        return false;
    }

    ggml_backend_hrx2_rms_norm_shape shape;
    if (!ggml_backend_hrx2_extract_rms_norm_shape(rms_norm, &shape)) {
        return false;
    }

    const ggml_tensor * weight = nullptr;
    if (mul->src[0] == rms_norm) {
        weight = mul->src[1];
    } else if (mul->src[1] == rms_norm) {
        weight = mul->src[0];
    } else {
        return false;
    }

    if (!weight ||
        weight->type != GGML_TYPE_F32 ||
        weight->view_src != nullptr ||
        !ggml_is_contiguous(weight) ||
        weight->ne[0] != rms_norm->ne[0] ||
        ggml_nrows(weight) != 1 ||
        weight->nb[0] != sizeof(float)) {
        return false;
    }

    *out_fusion = {
        /* .rms_norm = */ rms_norm,
        /* .mul      = */ mul,
        /* .weight   = */ weight,
        /* .shape    = */ shape,
    };
    return true;
}

static bool ggml_backend_hrx2_extract_add_rms_norm_mul_fusion(
        const ggml_tensor * add,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul,
        ggml_backend_hrx2_add_rms_norm_mul_fusion * out_fusion) {
    if (!out_fusion ||
        !add ||
        !rms_norm ||
        !mul ||
        add->op != GGML_OP_ADD ||
        add->view_src != nullptr ||
        add->type != GGML_TYPE_F32 ||
        !add->src[0] ||
        !add->src[1] ||
        add->src[0]->type != GGML_TYPE_F32 ||
        add->src[1]->type != GGML_TYPE_F32 ||
        add->src[0]->view_src != nullptr ||
        add->src[1]->view_src != nullptr ||
        !ggml_are_same_shape(add->src[0], add) ||
        !ggml_are_same_shape(add->src[1], add) ||
        !ggml_is_contiguous(add->src[0]) ||
        !ggml_is_contiguous(add->src[1]) ||
        !ggml_is_contiguous(add) ||
        add->src[0]->nb[0] != sizeof(float) ||
        add->src[1]->nb[0] != sizeof(float) ||
        add->nb[0] != sizeof(float) ||
        rms_norm->src[0] != add) {
        return false;
    }

    ggml_backend_hrx2_rms_norm_mul_fusion rms_mul_fusion;
    if (!ggml_backend_hrx2_extract_rms_norm_mul_fusion(rms_norm, mul, &rms_mul_fusion)) {
        return false;
    }

    *out_fusion = {
        /* .add      = */ add,
        /* .rms_norm = */ rms_norm,
        /* .mul      = */ mul,
        /* .weight   = */ rms_mul_fusion.weight,
        /* .shape    = */ rms_mul_fusion.shape,
    };
    return true;
}

static bool ggml_backend_hrx2_extract_set_rows_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_set_rows_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_set_rows(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_set_rows_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.nc) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.nr) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ne02) ||
        !ggml_backend_hrx2_u32(src0->ne[3], &shape.ne03) ||
        !ggml_backend_hrx2_u32(op->ne[1], &shape.ne1) ||
        !ggml_backend_hrx2_u32(src1->ne[1], &shape.ne11) ||
        !ggml_backend_hrx2_u32(src1->ne[2], &shape.ne12) ||
        src0->nb[1] % ggml_type_size(src0->type) != 0 ||
        src0->nb[2] % ggml_type_size(src0->type) != 0 ||
        src0->nb[3] % ggml_type_size(src0->type) != 0 ||
        src1->nb[0] % sizeof(int64_t) != 0 ||
        src1->nb[1] % sizeof(int64_t) != 0 ||
        src1->nb[2] % sizeof(int64_t) != 0 ||
        op->nb[1] % ggml_type_size(op->type) != 0 ||
        op->nb[2] % ggml_type_size(op->type) != 0 ||
        op->nb[3] % ggml_type_size(op->type) != 0 ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / ggml_type_size(src0->type), &shape.src0_nb1) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / ggml_type_size(src0->type), &shape.src0_nb2) ||
        !ggml_backend_hrx2_u32_size(src0->nb[3] / ggml_type_size(src0->type), &shape.src0_nb3) ||
        !ggml_backend_hrx2_u32_size(src1->nb[0] / sizeof(int64_t), &shape.idx_nb0) ||
        !ggml_backend_hrx2_u32_size(src1->nb[1] / sizeof(int64_t), &shape.idx_nb1) ||
        !ggml_backend_hrx2_u32_size(src1->nb[2] / sizeof(int64_t), &shape.idx_nb2) ||
        !ggml_backend_hrx2_u32_size(op->nb[1] / ggml_type_size(op->type), &shape.dst_nb1) ||
        !ggml_backend_hrx2_u32_size(op->nb[2] / ggml_type_size(op->type), &shape.dst_nb2) ||
        !ggml_backend_hrx2_u32_size(op->nb[3] / ggml_type_size(op->type), &shape.dst_nb3)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_pointwise_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_pointwise_shape * out_shape) {
    if (!out_shape) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_pointwise_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows)) {
        return false;
    }
    if (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL || op->op == GGML_OP_DIV) {
        if (!ggml_backend_hrx2_supports_pointwise_binary(nullptr, op)) {
            return false;
        }
        if (!ggml_backend_hrx2_flat_row_stride_f32(src0, &shape.src0_row_stride)) {
            return false;
        }
        if (ggml_are_same_shape(src1, src0)) {
            if (!ggml_backend_hrx2_flat_row_stride_f32(src1, &shape.src1_row_stride) ||
                !ggml_backend_hrx2_u32(src1->ne[0], &shape.src1_ncols)) {
                return false;
            }
        } else if (src1->ne[0] == 1 && ggml_nrows(src1) == ggml_nrows(src0)) {
            if (!ggml_backend_hrx2_flat_row_stride_f32(src1, &shape.src1_row_stride) ||
                !ggml_backend_hrx2_u32(src1->ne[0], &shape.src1_ncols)) {
                return false;
            }
        } else if ((src1->ne[0] == src0->ne[0] || src1->ne[0] == 1) && ggml_nrows(src1) == 1) {
            shape.src1_row_stride = 0;
            if (!ggml_backend_hrx2_u32(src1->ne[0], &shape.src1_ncols)) {
                return false;
            }
        } else {
            return false;
        }
    } else if (op->op == GGML_OP_SCALE) {
        if (!ggml_backend_hrx2_supports_scale(nullptr, op)) {
            return false;
        }
        shape.src0_row_stride = shape.ncols;
        shape.src1_row_stride = 0;
        shape.src1_ncols = 1;
    } else if (op->op == GGML_OP_CLAMP) {
        if (!ggml_backend_hrx2_supports_clamp(nullptr, op)) {
            return false;
        }
        shape.src0_row_stride = shape.ncols;
        shape.src1_row_stride = 0;
        shape.src1_ncols = 1;
    } else {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_sum_rows_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_sum_rows_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_sum_rows(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    ggml_backend_hrx2_sum_rows_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) ||
        !ggml_backend_hrx2_flat_row_stride_f32(src0, &shape.src0_row_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_get_rows_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_get_rows_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_get_rows_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_get_rows_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(src1->ne[0], &shape.nrows) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.src0_nrows) ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / sizeof(float), &shape.src0_row_stride) ||
        !ggml_backend_hrx2_u32_size(src1->nb[0] / sizeof(int32_t), &shape.idx_row_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[1] / sizeof(float), &shape.dst_row_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_get_rows_quantized_shape(
        const ggml_tensor * op,
        ggml_type src0_type,
        ggml_backend_hrx2_get_rows_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_get_rows_quantized(nullptr, op, src0_type)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_get_rows_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(src1->ne[0], &shape.nrows) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.src0_nrows) ||
        !ggml_backend_hrx2_u32_size(src0->nb[1], &shape.src0_row_stride) ||
        !ggml_backend_hrx2_u32_size(src1->nb[0] / sizeof(int32_t), &shape.idx_row_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[1] / sizeof(float), &shape.dst_row_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_get_rows_moe_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_get_rows_moe_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_get_rows_moe_weights(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_get_rows_moe_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[1], &shape.nexperts) ||
        !ggml_backend_hrx2_u32(op->ne[1], &shape.nselected) ||
        !ggml_backend_hrx2_u32(op->ne[2], &shape.ntokens) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src0_token_stride) ||
        !ggml_backend_hrx2_u32_size(src1->nb[1] / sizeof(int32_t), &shape.idx_token_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[2] / sizeof(float), &shape.dst_token_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_argsort_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_argsort_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_argsort_f32_i32_desc(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    ggml_backend_hrx2_argsort_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_rope_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_rope_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_rope_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_rope_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_get_op_params_i32(op, 1), &shape.n_dims) ||
        !ggml_backend_hrx2_u32(ggml_get_op_params_i32(op, 2), &shape.mode) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.nheads) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ntokens) ||
        !ggml_backend_hrx2_u32(src0->ne[1] * src0->ne[2], &shape.nrows) ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / sizeof(float), &shape.src0_head_stride) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src0_token_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[1] / sizeof(float), &shape.dst_head_stride) ||
        !ggml_backend_hrx2_u32_size(op->nb[2] / sizeof(float), &shape.dst_token_stride) ||
        !ggml_backend_hrx2_u32_size(src1->nb[0] / sizeof(int32_t), &shape.pos_token_stride)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_rope_set_rows_fusion(
        const ggml_tensor * rope,
        const ggml_tensor * view,
        const ggml_tensor * set_rows,
        ggml_backend_hrx2_rope_set_rows_fusion * out_fusion) {
    if (!out_fusion ||
        !rope ||
        !view ||
        !set_rows ||
        rope->op != GGML_OP_ROPE ||
        view->op != GGML_OP_VIEW ||
        set_rows->op != GGML_OP_SET_ROWS ||
        rope->src[2] == nullptr ||
        set_rows->src[0] != view ||
        set_rows->src[1] == nullptr ||
        rope->type != GGML_TYPE_F32 ||
        view->type != GGML_TYPE_F32 ||
        set_rows->type != GGML_TYPE_F16 ||
        set_rows->src[1]->type != GGML_TYPE_I64 ||
        rope->view_src != nullptr ||
        view->view_offs != 0 ||
        (view->view_src != rope && view->src[0] != rope)) {
        return false;
    }

    ggml_backend_hrx2_rope_shape rope_shape;
    ggml_backend_hrx2_set_rows_shape set_rows_shape;
    if (!ggml_backend_hrx2_extract_rope_shape(rope, &rope_shape) ||
        !ggml_backend_hrx2_extract_set_rows_shape(set_rows, &set_rows_shape)) {
        return false;
    }

    if (view->ne[0] != rope->ne[0] * rope->ne[1] ||
        view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 ||
        view->ne[3] != 1 ||
        view->nb[0] != sizeof(float) ||
        view->nb[1] != rope->nb[2] ||
        set_rows_shape.nc != rope_shape.ncols * rope_shape.nheads ||
        set_rows_shape.nr != rope_shape.ntokens ||
        set_rows_shape.ne02 != 1 ||
        set_rows_shape.ne03 != 1) {
        return false;
    }

    *out_fusion = {
        /* .rope           = */ rope,
        /* .view           = */ view,
        /* .set_rows       = */ set_rows,
        /* .rope_shape     = */ rope_shape,
        /* .set_rows_shape = */ set_rows_shape,
    };
    return true;
}

static bool ggml_backend_hrx2_tensor_is_or_zero_offset_view_source(
        const ggml_tensor * tensor,
        const ggml_tensor * base) {
    for (const ggml_tensor * cur = tensor; cur != nullptr;) {
        if (cur == base) {
            return true;
        }
        if (cur->view_offs != 0) {
            return false;
        }
        if (cur->view_src) {
            cur = cur->view_src;
            continue;
        }
        if ((cur->op == GGML_OP_VIEW ||
             cur->op == GGML_OP_RESHAPE ||
             cur->op == GGML_OP_PERMUTE ||
             cur->op == GGML_OP_TRANSPOSE) &&
            cur->src[0]) {
            cur = cur->src[0];
            continue;
        }
        return false;
    }
    return false;
}

static bool ggml_backend_hrx2_extract_cont_set_rows_fusion(
        const ggml_tensor * cont,
        const ggml_tensor * set_rows,
        ggml_backend_hrx2_cont_set_rows_fusion * out_fusion) {
    if (!out_fusion ||
        !cont ||
        !set_rows ||
        cont->op != GGML_OP_CONT ||
        set_rows->op != GGML_OP_SET_ROWS ||
        set_rows->src[0] == nullptr ||
        set_rows->src[1] == nullptr ||
        cont->type != GGML_TYPE_F32 ||
        set_rows->type != GGML_TYPE_F16 ||
        set_rows->src[1]->type != GGML_TYPE_I64 ||
        cont->view_src != nullptr ||
        (cont->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        !ggml_backend_hrx2_tensor_is_or_zero_offset_view_source(set_rows->src[0], cont)) {
        return false;
    }

    ggml_backend_hrx2_cont_shape cont_shape;
    ggml_backend_hrx2_set_rows_shape set_rows_shape;
    if (!ggml_backend_hrx2_extract_cont_shape(cont, &cont_shape) ||
        !ggml_backend_hrx2_extract_set_rows_shape(set_rows, &set_rows_shape)) {
        return false;
    }

    if (cont_shape.ncols != 128 ||
        set_rows_shape.nc != 1 ||
        set_rows_shape.ne02 != 1 ||
        set_rows_shape.ne03 != 1 ||
        set_rows_shape.ne1 > 1048576 ||
        ggml_nelements(set_rows->src[0]) != ggml_nelements(cont)) {
        return false;
    }

    *out_fusion = {
        /* .cont           = */ cont,
        /* .set_rows       = */ set_rows,
        /* .cont_shape     = */ cont_shape,
        /* .set_rows_shape = */ set_rows_shape,
    };
    return true;
}

static bool ggml_backend_hrx2_extract_soft_max_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_soft_max_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_soft_max_f32(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_backend_hrx2_soft_max_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.ne01) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ne02)) {
        return false;
    }
    shape.has_mask = src1 != nullptr;
    if (src1) {
        if (!ggml_backend_hrx2_u32_size(src1->nb[1] / sizeof(float), &shape.mask_nb1) ||
            !ggml_backend_hrx2_u32_size(src1->nb[2] / sizeof(float), &shape.mask_nb2) ||
            !ggml_backend_hrx2_u32_size(src1->nb[3] / sizeof(float), &shape.mask_nb3) ||
            !ggml_backend_hrx2_u32(src1->ne[1], &shape.mask_ne1) ||
            !ggml_backend_hrx2_u32(src1->ne[2], &shape.mask_ne2) ||
            !ggml_backend_hrx2_u32(src1->ne[3], &shape.mask_ne3)) {
            return false;
        }
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_cont_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_cont_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_cont(nullptr, op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    ggml_backend_hrx2_cont_shape shape;
    if (!ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) ||
        !ggml_backend_hrx2_u32(src0->ne[1], &shape.ne1) ||
        !ggml_backend_hrx2_u32(src0->ne[2], &shape.ne2) ||
        src0->nb[1] % sizeof(float) != 0 ||
        src0->nb[2] % sizeof(float) != 0 ||
        src0->nb[3] % sizeof(float) != 0 ||
        !ggml_backend_hrx2_u32_size(src0->nb[1] / sizeof(float), &shape.src_nb1) ||
        !ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src_nb2) ||
        !ggml_backend_hrx2_u32_size(src0->nb[3] / sizeof(float), &shape.src_nb3)) {
        return false;
    }
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_extract_swiglu_shape(
        const ggml_tensor * op,
        ggml_backend_hrx2_swiglu_shape * out_shape) {
    if (!out_shape || !ggml_backend_hrx2_supports_swiglu(nullptr, op)) {
        return false;
    }
    ggml_backend_hrx2_swiglu_shape shape;
    if (!ggml_backend_hrx2_u32(op->ne[0], &shape.ncols) ||
        !ggml_backend_hrx2_u32(ggml_nrows(op), &shape.nrows)) {
        return false;
    }
    shape.glu_op = ggml_get_glu_op(op);
    shape.split_sources = op->src[1] != nullptr;
    *out_shape = shape;
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_copy_shape & shape) {
    return route &&
           shape.n >= route->ncols_min &&
           shape.n <= route->ncols_max;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape) {
    if (!route ||
        shape.k < route->k_min || shape.k > route->k_max ||
        shape.rows < route->rows_min || shape.rows > route->rows_max ||
        shape.cols < route->cols_min || shape.cols > route->cols_max) {
        return false;
    }
    if (route->k_pow2_guard != 0) {
        const bool k_pow2 = ggml_backend_hrx2_is_pow2_i64(shape.k);
        if ((route->k_pow2_guard > 0) != k_pow2) {
            return false;
        }
    }
    if (route->all_pot_guard != 0) {
        const bool all_pot =
            ggml_backend_hrx2_is_pow2_i64(shape.k) &&
            ggml_backend_hrx2_is_pow2_i64(shape.rows) &&
            ggml_backend_hrx2_is_pow2_i64(shape.cols);
        if ((route->all_pot_guard > 0) != all_pot) {
            return false;
        }
    }
    if (route->k_multiple_of_guard != 0 && (shape.k % route->k_multiple_of_guard) != 0) {
        return false;
    }
    if (route->rows_multiple_of_guard != 0 && (shape.rows % route->rows_multiple_of_guard) != 0) {
        return false;
    }
    if (route->cols_multiple_of_guard != 0 && (shape.cols % route->cols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_id_shape & shape) {
    if (!route ||
        shape.k < route->k_min || shape.k > route->k_max ||
        shape.rows < route->rows_min || shape.rows > route->rows_max ||
        shape.nselected < route->cols_min || shape.nselected > route->cols_max ||
        shape.ntokens < route->nrows_min || shape.ntokens > route->nrows_max) {
        return false;
    }
    if (route->k_multiple_of_guard != 0 && (shape.k % route->k_multiple_of_guard) != 0) {
        return false;
    }
    if (route->cols_multiple_of_guard != 0 && (shape.nselected % route->cols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_f16_shape & shape) {
    if (!route ||
        shape.k < route->k_min || shape.k > route->k_max ||
        shape.rows < route->rows_min || shape.rows > route->rows_max ||
        shape.cols < route->cols_min || shape.cols > route->cols_max) {
        return false;
    }
    if (route->k_pow2_guard != 0) {
        const bool k_pow2 = ggml_backend_hrx2_is_pow2_i64(shape.k);
        if ((route->k_pow2_guard > 0) != k_pow2) {
            return false;
        }
    }
    if (route->all_pot_guard != 0) {
        const bool all_pot =
            ggml_backend_hrx2_is_pow2_i64(shape.k) &&
            ggml_backend_hrx2_is_pow2_i64(shape.rows) &&
            ggml_backend_hrx2_is_pow2_i64(shape.cols);
        if ((route->all_pot_guard > 0) != all_pot) {
            return false;
        }
    }
    if (route->k_multiple_of_guard != 0 && (shape.k % route->k_multiple_of_guard) != 0) {
        return false;
    }
    if (route->rows_multiple_of_guard != 0 && (shape.rows % route->rows_multiple_of_guard) != 0) {
        return false;
    }
    if (route->cols_multiple_of_guard != 0 && (shape.cols % route->cols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_set_rows_shape & shape) {
    if (!route ||
        shape.nc < route->ncols_min || shape.nc > route->ncols_max ||
        shape.nr < route->nrows_min || shape.nr > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.nc % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rms_norm_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_pointwise_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    if (route->pointwise_src0_row_stride_eq_ncols && shape.src0_row_stride != shape.ncols) {
        return false;
    }
    if (route->pointwise_src1_row_stride_eq_ncols && shape.src1_row_stride != shape.ncols) {
        return false;
    }
    if (route->pointwise_src1_row_stride_eq_zero && shape.src1_row_stride != 0) {
        return false;
    }
    if (route->pointwise_src1_ncols_eq_ncols && shape.src1_ncols != shape.ncols) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_sum_rows_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_get_rows_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_get_rows_moe_shape & shape) {
    if (!route ||
        shape.nselected < route->ncols_min || shape.nselected > route->ncols_max ||
        shape.ntokens < route->nrows_min || shape.ntokens > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.nselected % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_argsort_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rope_shape & shape) {
    if (!route ||
        (!route->supports_mode.empty() &&
            !((route->supports_mode == "NORMAL" && shape.mode == GGML_ROPE_TYPE_NORMAL) ||
              (route->supports_mode == "NEOX"   && shape.mode == GGML_ROPE_TYPE_NEOX))) ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        (route->n_dims_min != 0 && shape.n_dims < route->n_dims_min) ||
        (route->n_dims_max != 0 && shape.n_dims > route->n_dims_max) ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max ||
        shape.nheads < route->rows_min || shape.nheads > route->rows_max ||
        shape.ntokens < route->cols_min || shape.ntokens > route->cols_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_soft_max_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max ||
        route->binding_count != (shape.has_mask ? 3u : 2u)) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_flash_attn_fa0_shape & shape) {
    if (!route ||
        shape.KV < route->k_min || shape.KV > route->k_max ||
        shape.N < route->rows_min || shape.N > route->rows_max ||
        shape.H < route->cols_min || shape.H > route->cols_max) {
        return false;
    }
    if (route->k_multiple_of_guard != 0 && (shape.KV % route->k_multiple_of_guard) != 0) {
        return false;
    }
    if (route->rows_multiple_of_guard != 0 && (shape.N % route->rows_multiple_of_guard) != 0) {
        return false;
    }
    if (route->cols_multiple_of_guard != 0 && (shape.H % route->cols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_cont_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_route_shape_matches(
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_swiglu_shape & shape) {
    if (!route ||
        shape.ncols < route->ncols_min || shape.ncols > route->ncols_max ||
        shape.nrows < route->nrows_min || shape.nrows > route->nrows_max) {
        return false;
    }
    if (route->ncols_multiple_of_guard != 0 && (shape.ncols % route->ncols_multiple_of_guard) != 0) {
        return false;
    }
    if (!route->supports_glu_op.empty() && route->supports_glu_op != ggml_backend_hrx2_glu_op_key(shape.glu_op)) {
        return false;
    }
    return true;
}

static bool ggml_backend_hrx2_make_copy_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_copy_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.copy.n") {
            binding.value = std::to_string(shape.n);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|n=" + std::to_string(shape.n);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_rms_norm_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rms_norm_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_pointwise_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_pointwise_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }
    if (!ggml_backend_hrx2_env_enabled("GGML_HRX2_ENABLE_POINTWISE_2D") &&
        route->id.find("_2d_") != std::string::npos) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.pointwise.src0_row_stride") {
            binding.value = std::to_string(shape.src0_row_stride);
        } else if (spec.value_source == "shape.pointwise.src1_row_stride") {
            binding.value = std::to_string(shape.src1_row_stride);
        } else if (spec.value_source == "shape.pointwise.src1_ncols") {
            binding.value = std::to_string(shape.src1_ncols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|src0_row_stride=" + std::to_string(shape.src0_row_stride);
        plan.cache_key += "|src1_row_stride=" + std::to_string(shape.src1_row_stride);
        plan.cache_key += "|src1_ncols=" + std::to_string(shape.src1_ncols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_sum_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_sum_rows_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.sum_rows.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.sum_rows.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.sum_rows.src0_row_stride") {
            binding.value = std::to_string(shape.src0_row_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|src0_row_stride=" + std::to_string(shape.src0_row_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_get_rows_moe_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_get_rows_moe_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.get_rows_moe.nexperts") {
            binding.value = std::to_string(shape.nexperts);
        } else if (spec.value_source == "shape.get_rows_moe.nselected") {
            binding.value = std::to_string(shape.nselected);
        } else if (spec.value_source == "shape.get_rows_moe.ntokens") {
            binding.value = std::to_string(shape.ntokens);
        } else if (spec.value_source == "shape.get_rows_moe.src0_token_stride") {
            binding.value = std::to_string(shape.src0_token_stride);
        } else if (spec.value_source == "shape.get_rows_moe.idx_token_stride") {
            binding.value = std::to_string(shape.idx_token_stride);
        } else if (spec.value_source == "shape.get_rows_moe.dst_token_stride") {
            binding.value = std::to_string(shape.dst_token_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|nexperts=" + std::to_string(shape.nexperts);
        plan.cache_key += "|nselected=" + std::to_string(shape.nselected);
        plan.cache_key += "|ntokens=" + std::to_string(shape.ntokens);
        plan.cache_key += "|src0_token_stride=" + std::to_string(shape.src0_token_stride);
        plan.cache_key += "|idx_token_stride=" + std::to_string(shape.idx_token_stride);
        plan.cache_key += "|dst_token_stride=" + std::to_string(shape.dst_token_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_argsort_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_argsort_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.argsort.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.argsort.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_rope_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rope_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.rope.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.rope.n_dims") {
            binding.value = std::to_string(shape.n_dims);
        } else if (spec.value_source == "shape.rope.nheads") {
            binding.value = std::to_string(shape.nheads);
        } else if (spec.value_source == "shape.rope.ntokens") {
            binding.value = std::to_string(shape.ntokens);
        } else if (spec.value_source == "shape.rope.src0_head_stride") {
            binding.value = std::to_string(shape.src0_head_stride);
        } else if (spec.value_source == "shape.rope.src0_token_stride") {
            binding.value = std::to_string(shape.src0_token_stride);
        } else if (spec.value_source == "shape.rope.dst_head_stride") {
            binding.value = std::to_string(shape.dst_head_stride);
        } else if (spec.value_source == "shape.rope.dst_token_stride") {
            binding.value = std::to_string(shape.dst_token_stride);
        } else if (spec.value_source == "shape.rope.pos_token_stride") {
            binding.value = std::to_string(shape.pos_token_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|n_dims=" + std::to_string(shape.n_dims);
        plan.cache_key += "|mode=" + std::to_string(shape.mode);
        plan.cache_key += "|nheads=" + std::to_string(shape.nheads);
        plan.cache_key += "|ntokens=" + std::to_string(shape.ntokens);
        plan.cache_key += "|src0_head_stride=" + std::to_string(shape.src0_head_stride);
        plan.cache_key += "|src0_token_stride=" + std::to_string(shape.src0_token_stride);
        plan.cache_key += "|dst_head_stride=" + std::to_string(shape.dst_head_stride);
        plan.cache_key += "|dst_token_stride=" + std::to_string(shape.dst_token_stride);
        plan.cache_key += "|pos_token_stride=" + std::to_string(shape.pos_token_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_rope_set_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_rope_set_rows_fusion & fusion,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, fusion.rope_shape) ||
        route->binding_count != 5) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.rope.ncols") {
            binding.value = std::to_string(fusion.rope_shape.ncols);
        } else if (spec.value_source == "shape.rope.n_dims") {
            binding.value = std::to_string(fusion.rope_shape.n_dims);
        } else if (spec.value_source == "shape.rope.nheads") {
            binding.value = std::to_string(fusion.rope_shape.nheads);
        } else if (spec.value_source == "shape.rope.ntokens") {
            binding.value = std::to_string(fusion.rope_shape.ntokens);
        } else if (spec.value_source == "shape.rope.src0_head_stride") {
            binding.value = std::to_string(fusion.rope_shape.src0_head_stride);
        } else if (spec.value_source == "shape.rope.src0_token_stride") {
            binding.value = std::to_string(fusion.rope_shape.src0_token_stride);
        } else if (spec.value_source == "shape.rope.dst_head_stride") {
            binding.value = std::to_string(fusion.rope_shape.dst_head_stride);
        } else if (spec.value_source == "shape.rope.dst_token_stride") {
            binding.value = std::to_string(fusion.rope_shape.dst_token_stride);
        } else if (spec.value_source == "shape.rope.pos_token_stride") {
            binding.value = std::to_string(fusion.rope_shape.pos_token_stride);
        } else if (spec.value_source == "shape.set_rows.nc") {
            binding.value = std::to_string(fusion.set_rows_shape.nc);
        } else if (spec.value_source == "shape.set_rows.nr") {
            binding.value = std::to_string(fusion.set_rows_shape.nr);
        } else if (spec.value_source == "shape.set_rows.ne02") {
            binding.value = std::to_string(fusion.set_rows_shape.ne02);
        } else if (spec.value_source == "shape.set_rows.ne03") {
            binding.value = std::to_string(fusion.set_rows_shape.ne03);
        } else if (spec.value_source == "shape.set_rows.ne1") {
            binding.value = std::to_string(fusion.set_rows_shape.ne1);
        } else if (spec.value_source == "shape.set_rows.ne11") {
            binding.value = std::to_string(fusion.set_rows_shape.ne11);
        } else if (spec.value_source == "shape.set_rows.ne12") {
            binding.value = std::to_string(fusion.set_rows_shape.ne12);
        } else if (spec.value_source == "shape.set_rows.src0_nb1") {
            binding.value = std::to_string(fusion.set_rows_shape.src0_nb1);
        } else if (spec.value_source == "shape.set_rows.src0_nb2") {
            binding.value = std::to_string(fusion.set_rows_shape.src0_nb2);
        } else if (spec.value_source == "shape.set_rows.src0_nb3") {
            binding.value = std::to_string(fusion.set_rows_shape.src0_nb3);
        } else if (spec.value_source == "shape.set_rows.idx_nb0") {
            binding.value = std::to_string(fusion.set_rows_shape.idx_nb0);
        } else if (spec.value_source == "shape.set_rows.idx_nb1") {
            binding.value = std::to_string(fusion.set_rows_shape.idx_nb1);
        } else if (spec.value_source == "shape.set_rows.idx_nb2") {
            binding.value = std::to_string(fusion.set_rows_shape.idx_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb1") {
            binding.value = std::to_string(fusion.set_rows_shape.dst_nb1);
        } else if (spec.value_source == "shape.set_rows.dst_nb2") {
            binding.value = std::to_string(fusion.set_rows_shape.dst_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb3") {
            binding.value = std::to_string(fusion.set_rows_shape.dst_nb3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(fusion.rope_shape.ncols);
        plan.cache_key += "|n_dims=" + std::to_string(fusion.rope_shape.n_dims);
        plan.cache_key += "|mode=" + std::to_string(fusion.rope_shape.mode);
        plan.cache_key += "|nheads=" + std::to_string(fusion.rope_shape.nheads);
        plan.cache_key += "|ntokens=" + std::to_string(fusion.rope_shape.ntokens);
        plan.cache_key += "|src0_head_stride=" + std::to_string(fusion.rope_shape.src0_head_stride);
        plan.cache_key += "|src0_token_stride=" + std::to_string(fusion.rope_shape.src0_token_stride);
        plan.cache_key += "|pos_token_stride=" + std::to_string(fusion.rope_shape.pos_token_stride);
        plan.cache_key += "|set_rows_ne1=" + std::to_string(fusion.set_rows_shape.ne1);
        plan.cache_key += "|idx_nb0=" + std::to_string(fusion.set_rows_shape.idx_nb0);
        plan.cache_key += "|idx_nb1=" + std::to_string(fusion.set_rows_shape.idx_nb1);
        plan.cache_key += "|idx_nb2=" + std::to_string(fusion.set_rows_shape.idx_nb2);
        plan.cache_key += "|dst_nb1=" + std::to_string(fusion.set_rows_shape.dst_nb1);
        plan.cache_key += "|dst_nb2=" + std::to_string(fusion.set_rows_shape.dst_nb2);
        plan.cache_key += "|dst_nb3=" + std::to_string(fusion.set_rows_shape.dst_nb3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_soft_max_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_soft_max_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.soft_max.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.soft_max.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.soft_max.ne01") {
            binding.value = std::to_string(shape.ne01);
        } else if (spec.value_source == "shape.soft_max.ne02") {
            binding.value = std::to_string(shape.ne02);
        } else if (spec.value_source == "shape.soft_max.mask_nb1") {
            binding.value = std::to_string(shape.mask_nb1);
        } else if (spec.value_source == "shape.soft_max.mask_nb2") {
            binding.value = std::to_string(shape.mask_nb2);
        } else if (spec.value_source == "shape.soft_max.mask_nb3") {
            binding.value = std::to_string(shape.mask_nb3);
        } else if (spec.value_source == "shape.soft_max.mask_ne1") {
            binding.value = std::to_string(shape.mask_ne1);
        } else if (spec.value_source == "shape.soft_max.mask_ne2") {
            binding.value = std::to_string(shape.mask_ne2);
        } else if (spec.value_source == "shape.soft_max.mask_ne3") {
            binding.value = std::to_string(shape.mask_ne3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|ne01=" + std::to_string(shape.ne01);
        plan.cache_key += "|ne02=" + std::to_string(shape.ne02);
        plan.cache_key += "|mask=" + std::to_string(shape.has_mask ? 1 : 0);
        if (shape.has_mask) {
            plan.cache_key += "|mask_nb1=" + std::to_string(shape.mask_nb1);
            plan.cache_key += "|mask_nb2=" + std::to_string(shape.mask_nb2);
            plan.cache_key += "|mask_nb3=" + std::to_string(shape.mask_nb3);
            plan.cache_key += "|mask_ne1=" + std::to_string(shape.mask_ne1);
            plan.cache_key += "|mask_ne2=" + std::to_string(shape.mask_ne2);
            plan.cache_key += "|mask_ne3=" + std::to_string(shape.mask_ne3);
        }
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_flash_attn_fa0_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_flash_attn_fa0_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);
    plan.cache_key += "|D=" + std::to_string(shape.D);
    plan.cache_key += "|KV=" + std::to_string(shape.KV);
    plan.cache_key += "|N=" + std::to_string(shape.N);
    plan.cache_key += "|H=" + std::to_string(shape.H);
    plan.cache_key += "|H_KV=" + std::to_string(shape.H_KV);
    plan.cache_key += "|S=" + std::to_string(shape.S);

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_cont_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_cont_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.cont.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.cont.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.cont.ne1") {
            binding.value = std::to_string(shape.ne1);
        } else if (spec.value_source == "shape.cont.ne2") {
            binding.value = std::to_string(shape.ne2);
        } else if (spec.value_source == "shape.cont.src_nb1") {
            binding.value = std::to_string(shape.src_nb1);
        } else if (spec.value_source == "shape.cont.src_nb2") {
            binding.value = std::to_string(shape.src_nb2);
        } else if (spec.value_source == "shape.cont.src_nb3") {
            binding.value = std::to_string(shape.src_nb3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|ne1=" + std::to_string(shape.ne1);
        plan.cache_key += "|ne2=" + std::to_string(shape.ne2);
        plan.cache_key += "|src_nb1=" + std::to_string(shape.src_nb1);
        plan.cache_key += "|src_nb2=" + std::to_string(shape.src_nb2);
        plan.cache_key += "|src_nb3=" + std::to_string(shape.src_nb3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_swiglu_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_swiglu_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }
    if ((shape.split_sources && route->binding_count != 3) ||
        (!shape.split_sources && route->binding_count != 2)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.swiglu.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.swiglu.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|glu_op=";
        plan.cache_key += ggml_backend_hrx2_glu_op_key(shape.glu_op);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q8_0_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source == "shape.q8_full_unroll_factor") {
            const uint32_t block_step = route->workgroup_size[0] / 8;
            if (block_step == 0 || (shape.k / 32) % block_step != 0) {
                return false;
            }
            binding.value = std::to_string((shape.k / 32) / block_step);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_quantize_q8_1_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_quantize_q8_1_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan || !ggml_backend_hrx2_route_available(device_context, route)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.q8_1.blocks") {
            binding.value = std::to_string(shape.blocks);
        } else if (spec.value_source == "shape.q8_1.ne1") {
            binding.value = std::to_string(shape.ne1);
        } else if (spec.value_source == "shape.q8_1.z_count") {
            binding.value = std::to_string(shape.z_count);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|blocks=" + std::to_string(shape.blocks);
        plan.cache_key += "|ne1=" + std::to_string(shape.ne1);
        plan.cache_key += "|z_count=" + std::to_string(shape.z_count);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q4_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_id_q4_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_id_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.mul_mat_id.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.mul_mat_id.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.mul_mat_id.nexperts") {
            binding.value = std::to_string(shape.nexperts);
        } else if (spec.value_source == "shape.mul_mat_id.nselected") {
            binding.value = std::to_string(shape.nselected);
        } else if (spec.value_source == "shape.mul_mat_id.ntokens") {
            binding.value = std::to_string(shape.ntokens);
        } else if (spec.value_source == "shape.mul_mat_id.src1_selected_stride") {
            binding.value = std::to_string(shape.src1_selected_stride);
        } else if (spec.value_source == "shape.mul_mat_id.src1_token_stride") {
            binding.value = std::to_string(shape.src1_token_stride);
        } else if (spec.value_source == "shape.mul_mat_id.idx_token_stride") {
            binding.value = std::to_string(shape.idx_token_stride);
        } else if (spec.value_source == "shape.mul_mat_id.dst_token_stride") {
            binding.value = std::to_string(shape.dst_token_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|nexperts=" + std::to_string(shape.nexperts);
        plan.cache_key += "|nselected=" + std::to_string(shape.nselected);
        plan.cache_key += "|ntokens=" + std::to_string(shape.ntokens);
        plan.cache_key += "|src1_selected_stride=" + std::to_string(shape.src1_selected_stride);
        plan.cache_key += "|src1_token_stride=" + std::to_string(shape.src1_token_stride);
        plan.cache_key += "|idx_token_stride=" + std::to_string(shape.idx_token_stride);
        plan.cache_key += "|dst_token_stride=" + std::to_string(shape.dst_token_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q6_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_q5_k_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_set_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_set_rows_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.set_rows.nc") {
            binding.value = std::to_string(shape.nc);
        } else if (spec.value_source == "shape.set_rows.nr") {
            binding.value = std::to_string(shape.nr);
        } else if (spec.value_source == "shape.set_rows.ne02") {
            binding.value = std::to_string(shape.ne02);
        } else if (spec.value_source == "shape.set_rows.ne03") {
            binding.value = std::to_string(shape.ne03);
        } else if (spec.value_source == "shape.set_rows.ne1") {
            binding.value = std::to_string(shape.ne1);
        } else if (spec.value_source == "shape.set_rows.ne11") {
            binding.value = std::to_string(shape.ne11);
        } else if (spec.value_source == "shape.set_rows.ne12") {
            binding.value = std::to_string(shape.ne12);
        } else if (spec.value_source == "shape.set_rows.src0_nb1") {
            binding.value = std::to_string(shape.src0_nb1);
        } else if (spec.value_source == "shape.set_rows.src0_nb2") {
            binding.value = std::to_string(shape.src0_nb2);
        } else if (spec.value_source == "shape.set_rows.src0_nb3") {
            binding.value = std::to_string(shape.src0_nb3);
        } else if (spec.value_source == "shape.set_rows.idx_nb0") {
            binding.value = std::to_string(shape.idx_nb0);
        } else if (spec.value_source == "shape.set_rows.idx_nb1") {
            binding.value = std::to_string(shape.idx_nb1);
        } else if (spec.value_source == "shape.set_rows.idx_nb2") {
            binding.value = std::to_string(shape.idx_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb1") {
            binding.value = std::to_string(shape.dst_nb1);
        } else if (spec.value_source == "shape.set_rows.dst_nb2") {
            binding.value = std::to_string(shape.dst_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb3") {
            binding.value = std::to_string(shape.dst_nb3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|nc=" + std::to_string(shape.nc);
        plan.cache_key += "|nr=" + std::to_string(shape.nr);
        plan.cache_key += "|ne02=" + std::to_string(shape.ne02);
        plan.cache_key += "|ne03=" + std::to_string(shape.ne03);
        plan.cache_key += "|ne1=" + std::to_string(shape.ne1);
        plan.cache_key += "|ne11=" + std::to_string(shape.ne11);
        plan.cache_key += "|ne12=" + std::to_string(shape.ne12);
        plan.cache_key += "|src0_nb1=" + std::to_string(shape.src0_nb1);
        plan.cache_key += "|src0_nb2=" + std::to_string(shape.src0_nb2);
        plan.cache_key += "|src0_nb3=" + std::to_string(shape.src0_nb3);
        plan.cache_key += "|idx_nb0=" + std::to_string(shape.idx_nb0);
        plan.cache_key += "|idx_nb1=" + std::to_string(shape.idx_nb1);
        plan.cache_key += "|idx_nb2=" + std::to_string(shape.idx_nb2);
        plan.cache_key += "|dst_nb1=" + std::to_string(shape.dst_nb1);
        plan.cache_key += "|dst_nb2=" + std::to_string(shape.dst_nb2);
        plan.cache_key += "|dst_nb3=" + std::to_string(shape.dst_nb3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_cont_set_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_cont_set_rows_fusion & fusion,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, fusion.set_rows_shape) ||
        route->binding_count != 3) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.cont.ncols") {
            binding.value = std::to_string(fusion.cont_shape.ncols);
        } else if (spec.value_source == "shape.cont.ne1") {
            binding.value = std::to_string(fusion.cont_shape.ne1);
        } else if (spec.value_source == "shape.cont.ne2") {
            binding.value = std::to_string(fusion.cont_shape.ne2);
        } else if (spec.value_source == "shape.cont.src_nb1") {
            binding.value = std::to_string(fusion.cont_shape.src_nb1);
        } else if (spec.value_source == "shape.cont.src_nb2") {
            binding.value = std::to_string(fusion.cont_shape.src_nb2);
        } else if (spec.value_source == "shape.cont.src_nb3") {
            binding.value = std::to_string(fusion.cont_shape.src_nb3);
        } else if (spec.value_source == "shape.set_rows.nc") {
            binding.value = std::to_string(fusion.set_rows_shape.nc);
        } else if (spec.value_source == "shape.set_rows.nr") {
            binding.value = std::to_string(fusion.set_rows_shape.nr);
        } else if (spec.value_source == "shape.set_rows.ne02") {
            binding.value = std::to_string(fusion.set_rows_shape.ne02);
        } else if (spec.value_source == "shape.set_rows.ne03") {
            binding.value = std::to_string(fusion.set_rows_shape.ne03);
        } else if (spec.value_source == "shape.set_rows.ne1") {
            binding.value = std::to_string(fusion.set_rows_shape.ne1);
        } else if (spec.value_source == "shape.set_rows.ne11") {
            binding.value = std::to_string(fusion.set_rows_shape.ne11);
        } else if (spec.value_source == "shape.set_rows.ne12") {
            binding.value = std::to_string(fusion.set_rows_shape.ne12);
        } else if (spec.value_source == "shape.set_rows.src0_nb1") {
            binding.value = std::to_string(fusion.set_rows_shape.src0_nb1);
        } else if (spec.value_source == "shape.set_rows.src0_nb2") {
            binding.value = std::to_string(fusion.set_rows_shape.src0_nb2);
        } else if (spec.value_source == "shape.set_rows.src0_nb3") {
            binding.value = std::to_string(fusion.set_rows_shape.src0_nb3);
        } else if (spec.value_source == "shape.set_rows.idx_nb0") {
            binding.value = std::to_string(fusion.set_rows_shape.idx_nb0);
        } else if (spec.value_source == "shape.set_rows.idx_nb1") {
            binding.value = std::to_string(fusion.set_rows_shape.idx_nb1);
        } else if (spec.value_source == "shape.set_rows.idx_nb2") {
            binding.value = std::to_string(fusion.set_rows_shape.idx_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb1") {
            binding.value = std::to_string(fusion.set_rows_shape.dst_nb1);
        } else if (spec.value_source == "shape.set_rows.dst_nb2") {
            binding.value = std::to_string(fusion.set_rows_shape.dst_nb2);
        } else if (spec.value_source == "shape.set_rows.dst_nb3") {
            binding.value = std::to_string(fusion.set_rows_shape.dst_nb3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|cont_ncols=" + std::to_string(fusion.cont_shape.ncols);
        plan.cache_key += "|cont_nrows=" + std::to_string(fusion.cont_shape.nrows);
        plan.cache_key += "|cont_ne1=" + std::to_string(fusion.cont_shape.ne1);
        plan.cache_key += "|cont_ne2=" + std::to_string(fusion.cont_shape.ne2);
        plan.cache_key += "|cont_src_nb1=" + std::to_string(fusion.cont_shape.src_nb1);
        plan.cache_key += "|cont_src_nb2=" + std::to_string(fusion.cont_shape.src_nb2);
        plan.cache_key += "|cont_src_nb3=" + std::to_string(fusion.cont_shape.src_nb3);
        plan.cache_key += "|set_nc=" + std::to_string(fusion.set_rows_shape.nc);
        plan.cache_key += "|set_nr=" + std::to_string(fusion.set_rows_shape.nr);
        plan.cache_key += "|set_ne1=" + std::to_string(fusion.set_rows_shape.ne1);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_get_rows_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_get_rows_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.get_rows.ncols") {
            binding.value = std::to_string(shape.ncols);
        } else if (spec.value_source == "shape.get_rows.nrows") {
            binding.value = std::to_string(shape.nrows);
        } else if (spec.value_source == "shape.get_rows.src0_nrows") {
            binding.value = std::to_string(shape.src0_nrows);
        } else if (spec.value_source == "shape.get_rows.src0_row_stride") {
            binding.value = std::to_string(shape.src0_row_stride);
        } else if (spec.value_source == "shape.get_rows.idx_row_stride") {
            binding.value = std::to_string(shape.idx_row_stride);
        } else if (spec.value_source == "shape.get_rows.dst_row_stride") {
            binding.value = std::to_string(shape.dst_row_stride);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|ncols=" + std::to_string(shape.ncols);
        plan.cache_key += "|nrows=" + std::to_string(shape.nrows);
        plan.cache_key += "|src0_nrows=" + std::to_string(shape.src0_nrows);
        plan.cache_key += "|src0_row_stride=" + std::to_string(shape.src0_row_stride);
        plan.cache_key += "|idx_row_stride=" + std::to_string(shape.idx_row_stride);
        plan.cache_key += "|dst_row_stride=" + std::to_string(shape.dst_row_stride);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_make_mul_mat_f16_f32_plan(
        const ggml_backend_hrx2_device_context * device_context,
        const ggml_backend_hrx2_kernel_route * route,
        const ggml_backend_hrx2_mul_mat_f16_shape & shape,
        ggml_backend_hrx2_provider_plan * out_plan) {
    if (!out_plan ||
        !ggml_backend_hrx2_route_available(device_context, route) ||
        !ggml_backend_hrx2_route_shape_matches(route, shape)) {
        return false;
    }

    ggml_backend_hrx2_provider_plan plan;
    plan.route = route;
    plan.cache_key = ggml_backend_hrx2_base_cache_key(device_context, route);

    if (!route->specialization_mode.empty() && route->specialization_mode != "jit_config") {
        return false;
    }
    for (const auto & spec : route->config_bindings) {
        ggml_backend_hrx2_config_binding binding;
        binding.key = spec.key;
        if (spec.value_source == "shape.mul_mat_f16.k") {
            binding.value = std::to_string(shape.k);
        } else if (spec.value_source == "shape.mul_mat_f16.rows") {
            binding.value = std::to_string(shape.rows);
        } else if (spec.value_source == "shape.mul_mat_f16.cols") {
            binding.value = std::to_string(shape.cols);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_ne2") {
            binding.value = std::to_string(shape.dst_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_ne3") {
            binding.value = std::to_string(shape.dst_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_ne2") {
            binding.value = std::to_string(shape.src0_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_ne3") {
            binding.value = std::to_string(shape.src0_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_stride_row") {
            binding.value = std::to_string(shape.src0_stride_row);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_stride_ne2") {
            binding.value = std::to_string(shape.src0_stride_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.src0_stride_ne3") {
            binding.value = std::to_string(shape.src0_stride_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.src1_stride_col") {
            binding.value = std::to_string(shape.src1_stride_col);
        } else if (spec.value_source == "shape.mul_mat_f16.src1_stride_ne2") {
            binding.value = std::to_string(shape.src1_stride_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.src1_stride_ne3") {
            binding.value = std::to_string(shape.src1_stride_ne3);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_stride_col") {
            binding.value = std::to_string(shape.dst_stride_col);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_stride_ne2") {
            binding.value = std::to_string(shape.dst_stride_ne2);
        } else if (spec.value_source == "shape.mul_mat_f16.dst_stride_ne3") {
            binding.value = std::to_string(shape.dst_stride_ne3);
        } else if (spec.value_source.empty()) {
            binding.value = spec.value;
        } else {
            return false;
        }
        plan.config_bindings.push_back(std::move(binding));
    }
    if (route->specialization_mode == "jit_config") {
        plan.cache_key += "|k=" + std::to_string(shape.k);
        plan.cache_key += "|rows=" + std::to_string(shape.rows);
        plan.cache_key += "|cols=" + std::to_string(shape.cols);
        plan.cache_key += "|dst_ne2=" + std::to_string(shape.dst_ne2);
        plan.cache_key += "|dst_ne3=" + std::to_string(shape.dst_ne3);
        for (const auto & binding : plan.config_bindings) {
            plan.cache_key += "|";
            plan.cache_key += binding.key;
            plan.cache_key += "=";
            plan.cache_key += binding.value;
        }
    }

    *out_plan = std::move(plan);
    return true;
}

static bool ggml_backend_hrx2_supports_mul_mat_q8_0_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q8_0_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q8_0_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q8_0_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_q4_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q4_k_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q4_k_routes) {
        if (ggml_backend_hrx2_route_uses_q8_1_rhs(route) &&
            !ggml_backend_hrx2_q4_k_q8_1_prompt_enabled(shape)) {
            continue;
        }
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) &&
            !ggml_backend_hrx2_q4_k_q8_1_x4_mmq_enabled()) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q4_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_q4_k_swiglu_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * first,
        const ggml_tensor * second,
        const ggml_tensor * swiglu) {
    ggml_backend_hrx2_mul_mat_q4_k_swiglu_fusion fusion;
    if (!ggml_backend_hrx2_extract_mul_mat_q4_k_swiglu_fusion(first, second, swiglu, &fusion)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q4_k_swiglu_routes) {
        if (route->binding_count != 4) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q4_k_plan(device_context, route, fusion.shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_q4_k_packed_swiglu_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * mul_mat,
        const ggml_tensor * swiglu) {
    ggml_backend_hrx2_mul_mat_q4_k_packed_swiglu_fusion fusion;
    if (!ggml_backend_hrx2_extract_mul_mat_q4_k_packed_swiglu_fusion(mul_mat, swiglu, &fusion)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q4_k_swiglu_routes) {
        if (route->binding_count != 3) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q4_k_plan(device_context, route, fusion.shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_f32_f32_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_f32_f32_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_f32_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q4_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_id_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op,
        ggml_type src0_type,
        const std::vector<const ggml_backend_hrx2_kernel_route *> & routes) {
    ggml_backend_hrx2_mul_mat_id_shape shape;
    if (!ggml_backend_hrx2_mul_mat_id_k_shape(op, src0_type, &shape)) {
        return false;
    }
    for (const auto * route : routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_id_q4_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q4_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k_route(
        device_context, op, GGML_TYPE_Q4_K, device_context->mul_mat_id_q4_k_routes);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q5_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k_route(
        device_context, op, GGML_TYPE_Q5_K, device_context->mul_mat_id_q5_k_routes);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q6_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx2_supports_mul_mat_id_k_route(
        device_context, op, GGML_TYPE_Q6_K, device_context->mul_mat_id_q6_k_routes);
}

static bool ggml_backend_hrx2_supports_mul_mat_q6_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q6_k_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q6_k_routes) {
        if (ggml_backend_hrx2_route_uses_q8_1_rhs(route) &&
            !ggml_backend_hrx2_q6_k_q8_1_prompt_enabled(shape)) {
            continue;
        }
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) &&
            !ggml_backend_hrx2_q6_k_q8_1_x4_prompt_enabled()) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q6_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_q5_k_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q5_k_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_q5_k_routes) {
        if (ggml_backend_hrx2_route_uses_q8_1_rhs(route) &&
            !ggml_backend_hrx2_q5_k_q8_1_prompt_enabled(shape)) {
            continue;
        }
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) &&
            !ggml_backend_hrx2_q5_k_q8_1_x4_prompt_enabled()) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_q5_k_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_set_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_set_rows_shape shape;
    if (!ggml_backend_hrx2_extract_set_rows_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->set_rows_routes) {
        if ((op->type == GGML_TYPE_F16 && route->export_name != "hrx2_set_rows_f32_f16") ||
            (op->type == GGML_TYPE_F32 && route->export_name != "hrx2_set_rows_f32_f32")) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_set_rows_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_cont_set_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * cont,
        const ggml_tensor * set_rows) {
    ggml_backend_hrx2_cont_set_rows_fusion fusion;
    if (!ggml_backend_hrx2_extract_cont_set_rows_fusion(cont, set_rows, &fusion)) {
        return false;
    }
    for (const auto * route : device_context->cont_set_rows_routes) {
        if (route->export_name != "hrx2_cont_set_rows_f32_f16") {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_cont_set_rows_plan(device_context, route, fusion, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_f16_f32_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_mul_mat_f16_shape shape;
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_f16_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_f16_f32_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_mul_mat_f16_f32_cont_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * mul_mat,
        const ggml_tensor * permute,
        const ggml_tensor * cont) {
    ggml_backend_hrx2_mul_mat_f16_shape shape;
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_cont_fusion(mul_mat, permute, cont, &shape)) {
        return false;
    }
    for (const auto * route : device_context->mul_mat_f16_f32_cont_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_mul_mat_f16_f32_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_rms_norm_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_rms_norm_shape shape;
    if (!ggml_backend_hrx2_extract_rms_norm_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->rms_norm_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rms_norm_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_rms_norm_mul_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    ggml_backend_hrx2_rms_norm_mul_fusion fusion;
    if (!ggml_backend_hrx2_extract_rms_norm_mul_fusion(rms_norm, mul, &fusion)) {
        return false;
    }
    for (const auto * route : device_context->rms_norm_mul_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rms_norm_plan(device_context, route, fusion.shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_add_rms_norm_mul_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * add,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    ggml_backend_hrx2_add_rms_norm_mul_fusion fusion;
    if (!ggml_backend_hrx2_extract_add_rms_norm_mul_fusion(add, rms_norm, mul, &fusion)) {
        return false;
    }
    for (const auto * route : device_context->add_rms_norm_mul_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rms_norm_plan(device_context, route, fusion.shape, &plan)) {
            return true;
        }
    }
    return false;
}

static const std::vector<const ggml_backend_hrx2_kernel_route *> * ggml_backend_hrx2_pointwise_routes(
        const ggml_backend_hrx2_device_context * device_context,
        enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD:
            return &device_context->add_routes;
        case GGML_OP_MUL:
            return &device_context->mul_routes;
        case GGML_OP_DIV:
            return &device_context->div_routes;
        case GGML_OP_SCALE:
            return &device_context->scale_routes;
        case GGML_OP_CLAMP:
            return &device_context->clamp_routes;
        default:
            return nullptr;
    }
}

static bool ggml_backend_hrx2_supports_pointwise_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    if (ggml_nelements(op) == 0) {
        return true; // zero-size no-op
    }
    ggml_backend_hrx2_pointwise_shape shape;
    if (!ggml_backend_hrx2_extract_pointwise_shape(op, &shape)) {
        return false;
    }
    const auto * routes = ggml_backend_hrx2_pointwise_routes(device_context, op->op);
    if (!routes) {
        return false;
    }
    for (const auto * route : *routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_pointwise_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_sum_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_sum_rows_shape shape;
    if (!ggml_backend_hrx2_extract_sum_rows_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->sum_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_sum_rows_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_get_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_get_rows_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->get_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_get_rows_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_get_rows_quantized_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op,
        ggml_type src0_type,
        const std::vector<const ggml_backend_hrx2_kernel_route *> & routes) {
    ggml_backend_hrx2_get_rows_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_quantized_shape(op, src0_type, &shape)) {
        return false;
    }
    for (const auto * route : routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_get_rows_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_get_rows_moe_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_get_rows_moe_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_moe_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->get_rows_moe_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_get_rows_moe_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_argsort_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_argsort_shape shape;
    if (!ggml_backend_hrx2_extract_argsort_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->argsort_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_argsort_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_rope_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    const bool has_freq_factors = op->src[2] != nullptr;
    ggml_backend_hrx2_rope_shape shape;
    if (!ggml_backend_hrx2_extract_rope_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->rope_routes) {
        if ((has_freq_factors && route->binding_count != 4) ||
            (!has_freq_factors && route->binding_count != 3)) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rope_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_rope_set_rows_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * rope,
        const ggml_tensor * set_rows) {
    const ggml_tensor * view = set_rows ? set_rows->src[0] : nullptr;
    ggml_backend_hrx2_rope_set_rows_fusion fusion;
    if (!ggml_backend_hrx2_extract_rope_set_rows_fusion(rope, view, set_rows, &fusion)) {
        return false;
    }
    for (const auto * route : device_context->rope_set_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_rope_set_rows_plan(device_context, route, fusion, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_tensor_is_or_views(
        const ggml_tensor * tensor,
        const ggml_tensor * base) {
    for (const ggml_tensor * cur = tensor; cur != nullptr; cur = cur->view_src) {
        if (cur == base) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_can_fuse_rope_set_rows(
        const ggml_cgraph * cgraph,
        int node_index) {
    if (!cgraph || node_index + 1 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * rope = cgraph->nodes[node_index];
    const ggml_tensor * set_rows = cgraph->nodes[node_index + 1];
    if (!rope ||
        !set_rows ||
        rope->op != GGML_OP_ROPE ||
        set_rows->op != GGML_OP_SET_ROWS ||
        (rope->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (set_rows->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (rope->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const ggml_tensor * view = set_rows->src[0];
    if (!view || !ggml_backend_hrx2_tensor_is_or_views(view, rope)) {
        return false;
    }

    int uses = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = node->src[src_idx];
            if (!ggml_backend_hrx2_tensor_is_or_views(src, rope)) {
                continue;
            }
            ++uses;
            if (i != node_index + 1 || src != view) {
                return false;
            }
        }
    }
    return uses == 1;
}

static bool ggml_backend_hrx2_can_fuse_cont_set_rows(
        const ggml_cgraph * cgraph,
        int cont_index,
        int set_rows_index) {
    if (!cgraph ||
        cont_index < 0 ||
        set_rows_index <= cont_index ||
        set_rows_index >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * cont = cgraph->nodes[cont_index];
    const ggml_tensor * set_rows = cgraph->nodes[set_rows_index];
    if (!cont ||
        !set_rows ||
        cont->op != GGML_OP_CONT ||
        set_rows->op != GGML_OP_SET_ROWS ||
        (cont->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (set_rows->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (cont->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        !ggml_backend_hrx2_tensor_is_or_zero_offset_view_source(set_rows->src[0], cont)) {
        return false;
    }

    int set_rows_uses = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (!node) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = node->src[src_idx];
            if (!ggml_backend_hrx2_tensor_is_or_zero_offset_view_source(src, cont)) {
                continue;
            }
            if (i == set_rows_index && src == set_rows->src[0]) {
                ++set_rows_uses;
                continue;
            }
            if (i > cont_index &&
                i < set_rows_index &&
                (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE) &&
                ggml_backend_hrx2_tensor_is_or_zero_offset_view_source(node, cont)) {
                continue;
            }
            {
                return false;
            }
        }
    }
    return set_rows_uses == 1;
}

static bool ggml_backend_hrx2_supports_soft_max_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_soft_max_shape shape;
    if (!ggml_backend_hrx2_extract_soft_max_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->soft_max_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_soft_max_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_flash_attn_fa0_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * kq,
        const ggml_tensor * soft_max,
        const ggml_tensor * kqv,
        const ggml_tensor * permute,
        const ggml_tensor * cont) {
    if (ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_F16_FA0_ATTENTION_FUSION")) {
        return false;
    }
    ggml_backend_hrx2_flash_attn_fa0_shape shape;
    if (!ggml_backend_hrx2_extract_flash_attn_fa0_fusion(kq, soft_max, kqv, permute, cont, &shape)) {
        return false;
    }
    for (const auto * route : device_context->flash_attn_fa0_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_flash_attn_fa0_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_cont_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_cont_shape shape;
    if (!ggml_backend_hrx2_extract_cont_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->cont_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_cont_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_hrx2_supports_swiglu_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx2_swiglu_shape shape;
    if (!ggml_backend_hrx2_extract_swiglu_shape(op, &shape)) {
        return false;
    }
    for (const auto * route : device_context->swiglu_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (ggml_backend_hrx2_make_swiglu_plan(device_context, route, shape, &plan)) {
            return true;
        }
    }
    return false;
}

static std::string ggml_backend_hrx2_tensor_summary(const ggml_tensor * tensor) {
    if (!tensor) {
        return "null";
    }
    std::string result = "'";
    result += tensor->name;
    result += "' type=";
    result += ggml_type_name(tensor->type);
    result += " op=";
    result += ggml_op_name(tensor->op);
    result += " ne=[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i) {
            result += ",";
        }
        result += std::to_string(tensor->ne[i]);
    }
    result += "] nb=[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i) {
            result += ",";
        }
        result += std::to_string(tensor->nb[i]);
    }
    result += "]";
    result += tensor->view_src ? " view=true" : " view=false";
    result += tensor->buffer ? " buffer=true" : " buffer=false";
    return result;
}

static ggml_status ggml_backend_hrx2_dispatch_rms_norm(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_rms_norm_shape shape;
    if (!ggml_backend_hrx2_extract_rms_norm_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid RMS_NORM shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: RMS_NORM tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(eps));
    ggml_backend_hrx2_rms_norm_constants constants = {
        /* .ncols   = */ static_cast<uint32_t>(src0->ne[0]),
        /* .nrows   = */ static_cast<uint32_t>(ggml_nrows(src0)),
        /* .ne1     = */ static_cast<uint32_t>(src0->ne[1]),
        /* .ne2     = */ static_cast<uint32_t>(src0->ne[2]),
        /* .src_nb1 = */ static_cast<uint32_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<uint32_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<uint32_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<uint32_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<uint32_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<uint32_t>(dst->nb[3]),
        /* .eps     = */ eps,
    };

    for (const auto * route : context->device_context->rms_norm_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rms_norm_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "RMS_NORM") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length == sizeof(eps)) {
            constant_data = &eps;
            constant_size = sizeof(eps);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: RMS_NORM route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.nrows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                1,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "RMS_NORM") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", constants.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", constants.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: RMS_NORM provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_rms_norm_mul(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    ggml_backend_hrx2_rms_norm_mul_fusion fusion;
    if (!ggml_backend_hrx2_extract_rms_norm_mul_fusion(rms_norm, mul, &fusion)) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("family", "rms_norm_mul_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "fusion_shape"));
        return GGML_STATUS_FAILED;
    }

    const ggml_tensor * src0 = rms_norm->src[0];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(fusion.weight, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(mul, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: RMS_NORM_MUL tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));

    for (const auto * route : context->device_context->rms_norm_mul_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rms_norm_plan(context->device_context, route, fusion.shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "RMS_NORM_MUL") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", fusion.shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", fusion.shape.nrows));
            continue;
        }
        if (provider->route.constant_byte_length != sizeof(eps)) {
            GGML_LOG_ERROR(
                "HRX2: RMS_NORM_MUL route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (fusion.shape.nrows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                1,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "RMS_NORM_MUL") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", fusion.shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", fusion.shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &eps,
                sizeof(eps),
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: RMS_NORM_MUL provider is not available for ncols=%u nrows=%u\n",
            fusion.shape.ncols, fusion.shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_add_rms_norm_mul(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * add,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    ggml_backend_hrx2_add_rms_norm_mul_fusion fusion;
    if (!ggml_backend_hrx2_extract_add_rms_norm_mul_fusion(add, rms_norm, mul, &fusion)) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("family", "add_rms_norm_mul_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "fusion_shape"));
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(add->src[0], &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(add->src[1], &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(add, &bindings[2]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(fusion.weight, &bindings[3]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(mul, &bindings[4])) {
        GGML_LOG_ERROR("HRX2: ADD_RMS_NORM_MUL tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));

    for (const auto * route : context->device_context->add_rms_norm_mul_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rms_norm_plan(context->device_context, route, fusion.shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "ADD_RMS_NORM_MUL") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", fusion.shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", fusion.shape.nrows));
            continue;
        }
        if (provider->route.constant_byte_length != sizeof(eps)) {
            GGML_LOG_ERROR(
                "HRX2: ADD_RMS_NORM_MUL route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (fusion.shape.nrows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                1,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "ADD_RMS_NORM_MUL") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", fusion.shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", fusion.shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &eps,
                sizeof(eps),
                bindings,
                5,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: ADD_RMS_NORM_MUL provider is not available for ncols=%u nrows=%u\n",
            fusion.shape.ncols, fusion.shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_cont(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_cont_shape shape;
    if (!ggml_backend_hrx2_extract_cont_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid CONT shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: CONT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->cont_routes) {
        if (ggml_backend_hrx2_cont_route_copies_vec4(route) &&
            ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_CONT_VEC4")) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_cont_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "CONT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }
        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR("HRX2: CONT route %s has unsupported constant byte length %u\n",
                    provider->route.id.c_str(), provider->route.constant_byte_length);
            continue;
        }

        const bool copies_vec4 = ggml_backend_hrx2_cont_route_copies_vec4(&provider->route);
        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint64_t dispatch_items = copies_vec4 ? (total + 3) / 4 : total;
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((dispatch_items + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "CONT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("copies_vec4", copies_vec4 ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: CONT provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_cpy(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    if (!ggml_backend_hrx2_supports_cpy(context->device_context, dst)) {
        GGML_LOG_ERROR("HRX2: invalid CPY shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t src_ref = {};
    hrx_buffer_ref_t dst_ref = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &src_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("HRX2: CPY tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    const size_t size = ggml_nbytes(dst);
    if (size == 0) {
        return GGML_STATUS_SUCCESS;
    }

    if (src0->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F16 &&
        ggml_is_contiguous(src0)) {
        ggml_backend_hrx2_copy_shape shape;
        if (!ggml_backend_hrx2_extract_copy_shape(dst, &shape)) {
            return GGML_STATUS_FAILED;
        }
        hrx_buffer_ref_t bindings[2] = { src_ref, dst_ref };
        for (const auto * route : context->device_context->copy_f32_f16_routes) {
            ggml_backend_hrx2_provider_plan plan;
            if (!ggml_backend_hrx2_make_copy_plan(context->device_context, route, shape, &plan)) {
                continue;
            }
            const auto * provider = ggml_backend_hrx2_get_provider(
                context->device_context,
                plan.route,
                plan.config_bindings,
                plan.cache_key);
            if (!provider || provider->route.constant_byte_length != 0) {
                continue;
            }
            const uint32_t workgroup_size =
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
            hrx_dispatch_config_t config = {
                /* .workgroup_count = */ { static_cast<uint32_t>((shape.n + workgroup_size - 1) / workgroup_size), 1, 1 },
                /* .workgroup_size  = */ { workgroup_size, 1, 1 },
                /* .subgroup_size   = */ 0,
            };
            ggml_backend_hrx2_trace_event(
                "dispatch",
                ggml_backend_hrx2_json_kv("op", "CPY") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
                ggml_backend_hrx2_json_kv("n", shape.n) + "," +
                ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
                ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));
            if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                    context->stream,
                    provider->executable,
                    provider->export_ordinal,
                    &config,
                    nullptr,
                    0,
                    bindings,
                    2,
                    HRX_DISPATCH_FLAG_NONE))) {
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }
        GGML_LOG_ERROR("HRX2: no F32->F16 CPY provider matched n=%u\n", shape.n);
        return GGML_STATUS_FAILED;
    }

    if (ggml_is_contiguous(src0)) {
        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "CPY") + "," +
            ggml_backend_hrx2_json_kv("route_id", "cpy_contiguous_stream") + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("nbytes", size) + "," +
            ggml_backend_hrx2_json_kv("nrows", ggml_nrows(src0)) + "," +
            ggml_backend_hrx2_json_kv("src_type", ggml_type_name(src0->type)));
        if (src_ref.buffer == dst_ref.buffer && src_ref.offset == dst_ref.offset) {
            return GGML_STATUS_SUCCESS;
        }
        if (!GGML_HRX2_CHECK(hrx_stream_copy_buffer(
                context->stream,
                src_ref.buffer,
                src_ref.offset,
                dst_ref.buffer,
                dst_ref.offset,
                size))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    if (src0->type == GGML_TYPE_F32) {
        ggml_backend_hrx2_cont_shape shape;
        if (ggml_backend_hrx2_u32(src0->ne[0], &shape.ncols) &&
            ggml_backend_hrx2_u32(ggml_nrows(src0), &shape.nrows) &&
            ggml_backend_hrx2_u32(src0->ne[1], &shape.ne1) &&
            ggml_backend_hrx2_u32(src0->ne[2], &shape.ne2) &&
            src0->nb[1] % sizeof(float) == 0 &&
            src0->nb[2] % sizeof(float) == 0 &&
            src0->nb[3] % sizeof(float) == 0 &&
            ggml_backend_hrx2_u32_size(src0->nb[1] / sizeof(float), &shape.src_nb1) &&
            ggml_backend_hrx2_u32_size(src0->nb[2] / sizeof(float), &shape.src_nb2) &&
            ggml_backend_hrx2_u32_size(src0->nb[3] / sizeof(float), &shape.src_nb3)) {
            hrx_buffer_ref_t bindings[2] = { src_ref, dst_ref };
            for (const auto * route : context->device_context->cont_routes) {
                ggml_backend_hrx2_provider_plan plan;
                if (!ggml_backend_hrx2_make_cont_plan(context->device_context, route, shape, &plan)) {
                    continue;
                }
                const auto * provider = ggml_backend_hrx2_get_provider(
                    context->device_context,
                    plan.route,
                    plan.config_bindings,
                    plan.cache_key);
                if (!provider || provider->route.constant_byte_length != 0) {
                    continue;
                }
                const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
                const uint32_t workgroup_size =
                    provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
                hrx_dispatch_config_t config = {
                    /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
                    /* .workgroup_size  = */ { workgroup_size, 1, 1 },
                    /* .subgroup_size   = */ 0,
                };
                ggml_backend_hrx2_trace_event(
                    "dispatch",
                    ggml_backend_hrx2_json_kv("op", "CPY") + "," +
                    ggml_backend_hrx2_json_kv("route_id", "cpy_strided_f32_cont_route") + "," +
                    ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                    ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
                    ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                    ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
                    ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
                    ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));
                if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                        context->stream,
                        provider->executable,
                        provider->export_ordinal,
                        &config,
                        nullptr,
                        0,
                        bindings,
                        2,
                        HRX_DISPATCH_FLAG_NONE))) {
                    return GGML_STATUS_FAILED;
                }
                return GGML_STATUS_SUCCESS;
            }
        }
    }

    ggml_backend_hrx2_trace_event(
        "dispatch",
        ggml_backend_hrx2_json_kv("op", "CPY") + "," +
        ggml_backend_hrx2_json_kv("route_id", "cpy_strided_rows_stream") + "," +
        ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("nbytes", size) + "," +
        ggml_backend_hrx2_json_kv("nrows", ggml_nrows(src0)) + "," +
        ggml_backend_hrx2_json_kv("src_type", ggml_type_name(src0->type)));
    const size_t row_size = ggml_row_size(src0->type, src0->ne[0]);
    size_t dst_offset = dst_ref.offset;
    for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                const size_t src_offset =
                    src_ref.offset +
                    static_cast<size_t>(i1) * src0->nb[1] +
                    static_cast<size_t>(i2) * src0->nb[2] +
                    static_cast<size_t>(i3) * src0->nb[3];
                if (!GGML_HRX2_CHECK(hrx_stream_copy_buffer(
                        context->stream,
                        src_ref.buffer,
                        src_offset,
                        dst_ref.buffer,
                        dst_offset,
                        row_size))) {
                    return GGML_STATUS_FAILED;
                }
                dst_offset += row_size;
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx2_dispatch_swiglu(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_swiglu_shape shape;
    if (!ggml_backend_hrx2_extract_swiglu_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid GLU shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(dst->src[1]).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    uint32_t binding_count = 0;
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0])) {
        GGML_LOG_ERROR("HRX2: GLU tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }
    if (shape.split_sources) {
        if (!ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
            GGML_LOG_ERROR("HRX2: GLU tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 3;
    } else {
        if (!ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
            GGML_LOG_ERROR("HRX2: GLU tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 2;
    }

    for (const auto * route : context->device_context->swiglu_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_swiglu_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GLU") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("glu_op", ggml_backend_hrx2_glu_op_key(shape.glu_op)) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }
        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR("HRX2: GLU route %s has unsupported constant byte length %u\n",
                    provider->route.id.c_str(), provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("glu_op", ggml_backend_hrx2_glu_op_key(shape.glu_op)) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                binding_count,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: GLU provider is not available for glu_op=%s ncols=%u nrows=%u\n",
            ggml_backend_hrx2_glu_op_key(shape.glu_op), shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_pointwise(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (ggml_nelements(dst) == 0) {
        return GGML_STATUS_SUCCESS; // zero-size no-op
    }
    ggml_backend_hrx2_pointwise_shape shape;
    if (!ggml_backend_hrx2_extract_pointwise_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid pointwise shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str());
        return GGML_STATUS_FAILED;
    }

    const auto * routes = ggml_backend_hrx2_pointwise_routes(context->device_context, dst->op);
    if (!routes) {
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    uint32_t binding_count = 0;
    struct scale_constants {
        float scale;
        float bias;
    } scale = {};
    const void * constant_data = nullptr;
    size_t constant_size = 0;

    if (dst->op == GGML_OP_SCALE || dst->op == GGML_OP_CLAMP) {
        if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
            GGML_LOG_ERROR("HRX2: unary pointwise tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 2;
        std::memcpy(&scale, dst->op_params, sizeof(scale));
        constant_data = &scale;
        constant_size = sizeof(scale);
    } else {
        if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
            !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
            GGML_LOG_ERROR("HRX2: pointwise tensor is not backed by HRX2 buffers\n");
            return GGML_STATUS_FAILED;
        }
        binding_count = 3;
    }

    for (const auto * route : *routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_pointwise_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", ggml_op_name(dst->op)) + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != constant_size) {
            GGML_LOG_ERROR(
                "HRX2: pointwise route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                constant_size);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        const bool use_2d_dispatch = provider->route.cols_per_workgroup > 1;
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                use_2d_dispatch ?
                    static_cast<uint32_t>((shape.ncols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup) :
                    static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
                use_2d_dispatch ?
                    static_cast<uint32_t>((shape.nrows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup) :
                    1,
                1,
            },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", ggml_op_name(dst->op)) + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                bindings,
                binding_count,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: pointwise provider is not available for op=%s ncols=%u nrows=%u\n",
            ggml_op_name(dst->op), shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_sum_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_sum_rows_shape shape;
    if (!ggml_backend_hrx2_extract_sum_rows_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid SUM_ROWS shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: SUM_ROWS tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->sum_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_sum_rows_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "SUM_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: SUM_ROWS route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { shape.nrows, 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "SUM_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: SUM_ROWS provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_get_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_get_rows_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid GET_ROWS F32 shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: GET_ROWS F32 tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->get_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_get_rows_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("family", "get_rows_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: GET_ROWS F32 route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("family", "get_rows_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("src0_nrows", shape.src0_nrows) + "," +
            ggml_backend_hrx2_json_kv("src0_row_stride", shape.src0_row_stride) + "," +
            ggml_backend_hrx2_json_kv("idx_row_stride", shape.idx_row_stride) + "," +
            ggml_backend_hrx2_json_kv("dst_row_stride", shape.dst_row_stride) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: GET_ROWS F32 provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_get_rows_quantized(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst,
        ggml_type src0_type,
        const char * family,
        const char * type_label,
        const std::vector<const ggml_backend_hrx2_kernel_route *> & routes) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_get_rows_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_quantized_shape(dst, src0_type, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid GET_ROWS %s shape during dispatch: dst=%s src0=%s src1=%s\n",
                type_label,
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: GET_ROWS %s tensor is not backed by HRX2 buffers\n", type_label);
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_get_rows_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("family", family) + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->export_info.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: GET_ROWS %s route %s has constant byte length %u but dispatch has none\n",
                type_label,
                plan.route->id.c_str(),
                provider->export_info.constant_byte_length);
            return GGML_STATUS_FAILED;
        }

        const uint32_t workgroup_size = provider->export_info.workgroup_size[0] != 0 ?
            provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        const uint64_t total = static_cast<uint64_t>(shape.ncols) * static_cast<uint64_t>(shape.nrows);
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream,
            provider->executable,
            provider->export_ordinal,
            &config,
            nullptr,
            0,
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("src0_nrows", shape.src0_nrows) + "," +
            ggml_backend_hrx2_json_kv("src0_row_stride", shape.src0_row_stride) + "," +
            ggml_backend_hrx2_json_kv("idx_row_stride", shape.idx_row_stride) + "," +
            ggml_backend_hrx2_json_kv("dst_row_stride", shape.dst_row_stride) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: GET_ROWS %s provider is not available for ncols=%u nrows=%u\n", type_label, shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_get_rows_moe(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_get_rows_moe_shape shape;
    if (!ggml_backend_hrx2_extract_get_rows_moe_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid GET_ROWS MoE shape during dispatch: dst=%s src0=%s src1=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: GET_ROWS MoE tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->get_rows_moe_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_get_rows_moe_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
                ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
                ggml_backend_hrx2_json_kv("ntokens", shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: GET_ROWS MoE route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total = static_cast<uint64_t>(shape.nselected) * static_cast<uint64_t>(shape.ntokens);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size), 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
            ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
            ggml_backend_hrx2_json_kv("ntokens", shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("src0_token_stride", shape.src0_token_stride) + "," +
            ggml_backend_hrx2_json_kv("idx_token_stride", shape.idx_token_stride) + "," +
            ggml_backend_hrx2_json_kv("dst_token_stride", shape.dst_token_stride) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: GET_ROWS MoE provider is not available for nexperts=%u nselected=%u ntokens=%u\n",
            shape.nexperts, shape.nselected, shape.ntokens);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_argsort(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_backend_hrx2_argsort_shape shape;
    if (!ggml_backend_hrx2_extract_argsort_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid ARGSORT shape during dispatch: dst=%s src0=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: ARGSORT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->argsort_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_argsort_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "ARGSORT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: ARGSORT route %s has constant byte length %u but dispatch has none\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { shape.nrows, 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "ARGSORT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: ARGSORT provider is not available for ncols=%u nrows=%u\n", shape.ncols, shape.nrows);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_rope(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    ggml_backend_hrx2_rope_shape shape;
    if (!ggml_backend_hrx2_extract_rope_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid ROPE shape during dispatch: dst=%s src0=%s src1=%s src2=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str(),
                ggml_backend_hrx2_tensor_summary(src2).c_str());
        return GGML_STATUS_FAILED;
    }

    const bool has_freq_factors = src2 != nullptr;
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        (has_freq_factors && !ggml_backend_hrx2_tensor_buffer_ref(src2, &bindings[2])) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[has_freq_factors ? 3 : 2])) {
        GGML_LOG_ERROR("HRX2: ROPE tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }


    ggml_backend_hrx2_rope_constants constants = {};
    std::memcpy(&constants.freq_base,   reinterpret_cast<const int32_t *>(dst->op_params) + 5, sizeof(float));
    std::memcpy(&constants.freq_scale,  reinterpret_cast<const int32_t *>(dst->op_params) + 6, sizeof(float));
    std::memcpy(&constants.attn_factor, reinterpret_cast<const int32_t *>(dst->op_params) + 8, sizeof(float));

    for (const auto * route : context->device_context->rope_routes) {
        const uint32_t expected_binding_count = has_freq_factors ? 4u : 3u;
        if (route->binding_count != expected_binding_count) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rope_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "ROPE") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("n_dims", shape.n_dims) + "," +
                ggml_backend_hrx2_json_kv("mode", shape.mode) + "," +
                ggml_backend_hrx2_json_kv("nheads", shape.nheads) + "," +
                ggml_backend_hrx2_json_kv("ntokens", shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != sizeof(constants)) {
            GGML_LOG_ERROR(
                "HRX2: ROPE route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                sizeof(constants));
            continue;
        }

        const uint64_t total_pairs =
            static_cast<uint64_t>(shape.ncols / 2) *
            static_cast<uint64_t>(shape.nheads) *
            static_cast<uint64_t>(shape.ntokens);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total_pairs + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "ROPE") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("n_dims", shape.n_dims) + "," +
            ggml_backend_hrx2_json_kv("mode", shape.mode) + "," +
            ggml_backend_hrx2_json_kv("nheads", shape.nheads) + "," +
            ggml_backend_hrx2_json_kv("ntokens", shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("has_freq_factors", has_freq_factors ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                expected_binding_count,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }

        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: ROPE provider is not available for ncols=%u nheads=%u ntokens=%u\n",
        shape.ncols,
        shape.nheads,
        shape.ntokens);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_rope_set_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * rope,
        const ggml_tensor * set_rows) {
    const ggml_tensor * view = set_rows ? set_rows->src[0] : nullptr;
    ggml_backend_hrx2_rope_set_rows_fusion fusion;
    if (!ggml_backend_hrx2_extract_rope_set_rows_fusion(rope, view, set_rows, &fusion)) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("fusion", "ROPE_SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("reason", "fusion_shape"));
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(rope->src[0], &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(rope->src[1], &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(rope->src[2], &bindings[2]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(set_rows->src[1], &bindings[3]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(set_rows, &bindings[4])) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("fusion", "ROPE_SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref"));
        GGML_LOG_ERROR("HRX2: ROPE_SET_ROWS tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_rope_constants constants = {};
    std::memcpy(&constants.freq_base,   reinterpret_cast<const int32_t *>(rope->op_params) + 5, sizeof(float));
    std::memcpy(&constants.freq_scale,  reinterpret_cast<const int32_t *>(rope->op_params) + 6, sizeof(float));
    std::memcpy(&constants.attn_factor, reinterpret_cast<const int32_t *>(rope->op_params) + 8, sizeof(float));

    for (const auto * route : context->device_context->rope_set_rows_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_rope_set_rows_plan(context->device_context, route, fusion, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "ROPE_SET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", fusion.rope_shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("n_dims", fusion.rope_shape.n_dims) + "," +
                ggml_backend_hrx2_json_kv("mode", fusion.rope_shape.mode) + "," +
                ggml_backend_hrx2_json_kv("nheads", fusion.rope_shape.nheads) + "," +
                ggml_backend_hrx2_json_kv("ntokens", fusion.rope_shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != sizeof(constants)) {
            GGML_LOG_ERROR(
                "HRX2: ROPE_SET_ROWS route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                sizeof(constants));
            continue;
        }

        const uint64_t total_pairs =
            static_cast<uint64_t>(fusion.rope_shape.ncols / 2) *
            static_cast<uint64_t>(fusion.rope_shape.nheads) *
            static_cast<uint64_t>(fusion.rope_shape.ntokens);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total_pairs + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "ROPE_SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", fusion.rope_shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("n_dims", fusion.rope_shape.n_dims) + "," +
            ggml_backend_hrx2_json_kv("mode", fusion.rope_shape.mode) + "," +
            ggml_backend_hrx2_json_kv("nheads", fusion.rope_shape.nheads) + "," +
            ggml_backend_hrx2_json_kv("ntokens", fusion.rope_shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("set_rows_ne1", fusion.set_rows_shape.ne1) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                5,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: ROPE_SET_ROWS provider is not available for ncols=%u nheads=%u ntokens=%u dst=%s\n",
        fusion.rope_shape.ncols,
        fusion.rope_shape.nheads,
        fusion.rope_shape.ntokens,
        ggml_backend_hrx2_tensor_summary(set_rows).c_str());
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_cont_set_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * cont,
        const ggml_tensor * set_rows) {
    ggml_backend_hrx2_cont_set_rows_fusion fusion;
    if (!ggml_backend_hrx2_extract_cont_set_rows_fusion(cont, set_rows, &fusion)) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("fusion", "CONT_SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("reason", "fusion_shape"));
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(cont->src[0], &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(set_rows->src[1], &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(set_rows, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("fusion", "CONT_SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref"));
        GGML_LOG_ERROR("HRX2: CONT_SET_ROWS tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->cont_set_rows_routes) {
        if (route->export_name != "hrx2_cont_set_rows_f32_f16") {
            continue;
        }

        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_cont_set_rows_plan(context->device_context, route, fusion, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "CONT_SET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("cont_ncols", fusion.cont_shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("set_rows_nr", fusion.set_rows_shape.nr));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: CONT_SET_ROWS route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total =
            static_cast<uint64_t>(fusion.set_rows_shape.nc) *
            static_cast<uint64_t>(fusion.set_rows_shape.nr) *
            static_cast<uint64_t>(fusion.set_rows_shape.ne02) *
            static_cast<uint64_t>(fusion.set_rows_shape.ne03);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ {
                workgroup_size,
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "CONT_SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("cont_ncols", fusion.cont_shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("cont_nrows", fusion.cont_shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("set_rows_nc", fusion.set_rows_shape.nc) + "," +
            ggml_backend_hrx2_json_kv("set_rows_nr", fusion.set_rows_shape.nr) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: CONT_SET_ROWS provider is not available for cont_ncols=%u set_rows_nc=%u set_rows_nr=%u\n",
        fusion.cont_shape.ncols,
        fusion.set_rows_shape.nc,
        fusion.set_rows_shape.nr);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_soft_max(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_backend_hrx2_soft_max_shape shape;
    if (!ggml_backend_hrx2_extract_soft_max_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid SOFT_MAX shape during dispatch: dst=%s src0=%s src1=%s src2=%s\n",
                ggml_backend_hrx2_tensor_summary(dst).c_str(),
                ggml_backend_hrx2_tensor_summary(src0).c_str(),
                ggml_backend_hrx2_tensor_summary(src1).c_str(),
                ggml_backend_hrx2_tensor_summary(dst->src[2]).c_str());
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, shape.has_mask ? &bindings[2] : &bindings[1])) {
        GGML_LOG_ERROR("HRX2: SOFT_MAX tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }
    if (shape.has_mask && !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1])) {
        GGML_LOG_ERROR("HRX2: SOFT_MAX mask tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_soft_max_constants constants = {};
    std::memcpy(&constants.scale, dst->op_params, sizeof(float));

    for (const auto * route : context->device_context->soft_max_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_soft_max_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "SOFT_MAX") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
                ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
                ggml_backend_hrx2_json_kv("has_mask", shape.has_mask ? 1 : 0));
            continue;
        }

        if (provider->route.constant_byte_length != sizeof(constants)) {
            GGML_LOG_ERROR(
                "HRX2: SOFT_MAX route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                sizeof(constants));
            continue;
        }

        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { shape.nrows, 1, 1 },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "SOFT_MAX") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("ncols", shape.ncols) + "," +
            ggml_backend_hrx2_json_kv("nrows", shape.nrows) + "," +
            ggml_backend_hrx2_json_kv("has_mask", shape.has_mask ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                shape.has_mask ? 3 : 2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: SOFT_MAX provider is not available for ncols=%u nrows=%u has_mask=%d\n",
        shape.ncols,
        shape.nrows,
        shape.has_mask ? 1 : 0);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_flash_attn_fa0(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * kq,
        const ggml_tensor * soft_max,
        const ggml_tensor * kqv,
        const ggml_tensor * permute,
        const ggml_tensor * cont) {
    ggml_backend_hrx2_flash_attn_fa0_shape shape;
    if (!ggml_backend_hrx2_extract_flash_attn_fa0_fusion(kq, soft_max, kqv, permute, cont, &shape)) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("fusion", "FLASH_ATTN_FA0") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape"));
        GGML_LOG_ERROR("HRX2: invalid FLASH_ATTN_FA0 fusion shape\n");
        return GGML_STATUS_FAILED;
    }

    const ggml_tensor * k = kq->src[0];
    const ggml_tensor * q = kq->src[1];
    const ggml_tensor * mask = soft_max->src[1];
    const ggml_tensor * v = kqv->src[0];
    hrx_buffer_ref_t bindings[6] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(q, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(k, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(v, &bindings[2]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(mask, &bindings[3]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(q, &bindings[4]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(cont, &bindings[5])) {
        ggml_backend_hrx2_trace_event(
            "fusion_reject",
            ggml_backend_hrx2_json_kv("fusion", "FLASH_ATTN_FA0") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref"));
        GGML_LOG_ERROR("HRX2: FLASH_ATTN_FA0 tensors are not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    float scale = 1.0f;
    std::memcpy(&scale, soft_max->op_params, sizeof(scale));
    ggml_backend_hrx2_flash_attn_fa0_constants constants = {
        /* .D               = */ static_cast<int64_t>(shape.D),
        /* .KV              = */ static_cast<int64_t>(shape.KV),
        /* .N               = */ static_cast<int64_t>(shape.N),
        /* .H               = */ static_cast<int64_t>(shape.H),
        /* .H_KV            = */ static_cast<int64_t>(shape.H_KV),
        /* .S               = */ static_cast<int64_t>(shape.S),
        /* .q_nb1           = */ static_cast<int64_t>(q->nb[1]),
        /* .q_nb2           = */ static_cast<int64_t>(q->nb[2]),
        /* .q_nb3           = */ static_cast<int64_t>(q->nb[3]),
        /* .k_nb1           = */ static_cast<int64_t>(k->nb[1]),
        /* .k_nb2           = */ static_cast<int64_t>(k->nb[2]),
        /* .k_nb3           = */ static_cast<int64_t>(k->nb[3]),
        /* .v_nb0           = */ static_cast<int64_t>(v->nb[0]),
        /* .v_nb1           = */ static_cast<int64_t>(v->nb[1]),
        /* .v_nb2           = */ static_cast<int64_t>(v->nb[2]),
        /* .v_nb3           = */ static_cast<int64_t>(v->nb[3]),
        /* .dst_nb1         = */ static_cast<int64_t>(shape.D) * static_cast<int64_t>(sizeof(float)),
        /* .dst_nb2         = */ static_cast<int64_t>(cont->nb[1]),
        /* .dst_nb3         = */ static_cast<int64_t>(cont->nb[3]),
        /* .mask_nb0        = */ static_cast<int64_t>(mask->nb[0]),
        /* .mask_nb1        = */ static_cast<int64_t>(mask->nb[1]),
        /* .mask_nb3        = */ static_cast<int64_t>(mask->nb[3]),
        /* .scale           = */ scale,
        /* .has_mask        = */ 1,
        /* .max_bias        = */ 0.0f,
        /* .m0              = */ 1.0f,
        /* .m1              = */ 1.0f,
        /* .logit_softcap   = */ 0.0f,
        /* .n_head_log2     = */ 0,
        /* .has_sinks       = */ 0,
    };

    for (const auto * route : context->device_context->flash_attn_fa0_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_flash_attn_fa0_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "FLASH_ATTN_FA0") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("KV", shape.KV) + "," +
                ggml_backend_hrx2_json_kv("N", shape.N) + "," +
                ggml_backend_hrx2_json_kv("H", shape.H));
            continue;
        }
        if (provider->route.constant_byte_length != sizeof(constants)) {
            GGML_LOG_ERROR(
                "HRX2: FLASH_ATTN_FA0 route %s has constant byte length %u but dispatch has %zu\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length,
                sizeof(constants));
            continue;
        }

        const uint32_t workgroup_size = provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.N + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                shape.H,
                shape.S,
            },
            /* .workgroup_size  = */ { workgroup_size, 1, 1 },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "FLASH_ATTN_FA0") + "," +
            ggml_backend_hrx2_json_kv("fusion", "KQ_SOFTMAX_KQV_CONT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("D", shape.D) + "," +
            ggml_backend_hrx2_json_kv("KV", shape.KV) + "," +
            ggml_backend_hrx2_json_kv("N", shape.N) + "," +
            ggml_backend_hrx2_json_kv("H", shape.H) + "," +
            ggml_backend_hrx2_json_kv("H_KV", shape.H_KV) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                &constants,
                sizeof(constants),
                bindings,
                6,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: FLASH_ATTN_FA0 provider is not available for D=%u KV=%u N=%u H=%u H_KV=%u\n",
        shape.D,
        shape.KV,
        shape.N,
        shape.H,
        shape.H_KV);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q8_0(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q8_0_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q8_0 shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    for (const auto * route : context->device_context->mul_mat_q8_0_routes) {
        const bool use_q8_1_rhs = ggml_backend_hrx2_route_uses_q8_1_rhs(route);
        if (use_q8_1_rhs && !ggml_backend_hrx2_q8_0_q8_1_prompt_enabled(shape)) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q8_0_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_buffer_ref_t route_bindings[3] = { bindings[0], bindings[1], bindings[2] };
        if (use_q8_1_rhs) {
            hrx_buffer_ref_t q8_1_ref = {};
            if (!ggml_backend_hrx2_dispatch_quantize_q8_1(
                    context,
                    src1,
                    ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route),
                    &q8_1_ref)) {
                ggml_backend_hrx2_trace_event(
                    "dispatch_failed",
                    ggml_backend_hrx2_json_kv("op", "QUANTIZE") + "," +
                    ggml_backend_hrx2_json_kv("family", "quantize_q8_1_f32") + "," +
                    ggml_backend_hrx2_json_kv("reason", "q8_1_quantize") + "," +
                    ggml_backend_hrx2_json_kv("route_id", provider->route.id));
                continue;
            }
            route_bindings[1] = q8_1_ref;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                route_bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q8_0 provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

// Q4NX MUL_MAT: fused dequant + f32 matmul on device (1bit-MONSTER Q4NX type).
// src0 is a GGML_TYPE_Q4NX tensor with ne = [256, rows] (rows = 32 * n_tiles,
// each 5120-byte tile = one ggml block). The tile blob is tile-major
// ([scales 512][zeros 512][packed 4096] per tile); we assemble the
// section-major views the q4nx_dequant_f32 kernel expects, dequant on the
// device into an f32 scratch, then run the mul_mat_f32_f32 route with the
// scratch as src0. Single tile-column width (k = 256) for now.
static bool ggml_backend_hrx2_q4nx_scratch_grow(
        ggml_backend_hrx2_device_context * device_context,
        hrx_buffer_t * out, size_t * cap, size_t needed) {
    if (*out && *cap >= needed) {
        return true;
    }
    if (*out) {
        // Do NOT release here: the previous op's kernels may still be reading
        // this buffer (async on the stream, no per-op sync). Retire it and let
        // ggml_backend_hrx2_sync_streams free it after all streams are synced.
        device_context->q4nx_retired.push_back(*out);
        *out = nullptr;
    }
    hrx_allocator_t alloc = hrx_device_allocator(device_context->device);
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        0,
    };
    if (!GGML_HRX2_CHECK(hrx_allocator_allocate_buffer(alloc, params, needed, out))) {
        return false;
    }
    *cap = needed;
    return true;
}

static bool ggml_backend_hrx2_supports_mul_mat_q4nx_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->op != GGML_OP_MUL_MAT_Q4NX ||
        !src0 || !src1 || op->view_src != nullptr ||
        src0->type != GGML_TYPE_Q4NX ||
        src1->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32) {
        return false;
    }
    // src0: [8192, n_tiles] (each 8192-elem row = one 5120-byte tile, in
    // (tile_row, tile_col) order); src1: F32 [in, cols] with in = k (the
    // logical reduction dim, multiple of 256); dst: F32 [n_tiles/n_tc*32,
    // cols] where n_tc = in/256 column tiles.
    const int64_t in     = src1->ne[0];
    const int64_t n_tiles = src0->ne[1];
    if (src0->ne[0] != GGML_Q4NX_TILE_COLS * GGML_Q4NX_TILE_ROWS ||
        in <= 0 || in % GGML_Q4NX_TILE_COLS != 0) {
        return false;
    }
    const int64_t n_tc = in / GGML_Q4NX_TILE_COLS;
    if (n_tc <= 0 || n_tiles % n_tc != 0) {
        return false;
    }
    const int64_t n_tr = n_tiles / n_tc;
    const int64_t rows = n_tr * GGML_Q4NX_TILE_ROWS;
    const int64_t cols = src1->ne[1];
    return op->ne[0] == rows &&
           op->ne[1] == cols &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           n_tiles > 0 && cols > 0 &&
           rows <= std::numeric_limits<uint32_t>::max() &&
           cols <= std::numeric_limits<uint32_t>::max() &&
           !device_context->q4nx_dequant_routes.empty() &&
           !device_context->mul_mat_f32_f32_routes.empty();
}

// Q4NX slice helper: dequant tiles [tile_base, tile_base+n_tiles) of src0 and
// f32 matmul against src1 columns [src1_col, src1_col+cols), writing into
// dst columns [dst_col, dst_col+cols). b_w holds the dequantized weight
// [rows, k]. Shared by the whole-tensor and per-slot MUL_MAT_ID paths.
static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4nx_mm(
        ggml_backend_hrx2_context * context,
        hrx_buffer_t b_w,
        size_t w_bytes,
        const hrx_buffer_ref_t & src1_ref,
        const hrx_buffer_ref_t & dst_ref,
        uint32_t k,
        uint32_t rows,
        uint32_t cols,
        uint32_t src1_col,    // src1 column offset (elements, multiple of k)
        uint32_t dst_col) {   // dst column offset (elements, multiple of rows)
    ggml_backend_hrx2_device_context * device_context = context->device_context;
    const uint32_t wg_size = 256;

    // f32 matmul provider: for cols >= 8 prefer the 8-column tiled route
    // (each workgroup computes one row across 8 columns, so dequantized-
    // weight reads drop ~8x); for cols < 8 (decode) the naive per-column
    // route has no wasted lanes and is faster.
    const ggml_backend_hrx2_kernel_route * mm_route = nullptr;
    if (cols >= 8) {
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_f32_f32_ggml_tiled") { mm_route = r; break; }
        }
    }
    if (!mm_route) {
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_f32_f32_ggml") { mm_route = r; break; }
        }
    }
    if (!mm_route) {
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX: mul_mat_f32_f32_ggml[_tiled] route missing\n");
        return GGML_STATUS_FAILED;
    }
    std::vector<ggml_backend_hrx2_config_binding> mm_cfg;
    mm_cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
    mm_cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
    mm_cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
    mm_cfg.push_back({"@hrx2.tuning.workgroup_size", std::to_string(wg_size)});
    const std::string mm_key = ggml_backend_hrx2_base_cache_key(device_context, mm_route) +
        "-q4nx-mm-r" + std::to_string(rows) + "-c" + std::to_string(cols) +
        "-k" + std::to_string(k) + "-wg" + std::to_string(wg_size);
    auto * mm = ggml_backend_hrx2_get_provider(device_context, mm_route, mm_cfg, mm_key);
    if (!mm) {
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX f32 provider unavailable\n");
        return GGML_STATUS_FAILED;
    }
    if (mm->route.constant_byte_length != 0) {
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX f32 route has unsupported constants\n");
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t mm_bindings[3] = {
        { b_w,    0, w_bytes },
        { src1_ref.buffer, src1_ref.offset + (size_t) src1_col * sizeof(float), (size_t) cols * k * sizeof(float) },
        { dst_ref.buffer,  dst_ref.offset  + (size_t) dst_col  * sizeof(float), (size_t) cols * rows * sizeof(float) },
    };
    // tiled route launches (rows, ceil(cols/8)) workgroups; the naive route
    // launches (rows, cols)
    const uint32_t mm_cols = (mm_route->id == "mul_mat_f32_f32_ggml_tiled") ? (cols + 7) / 8 : cols;
    hrx_dispatch_config_t mm_config = {
        { rows, mm_cols, 1 },
        { mm->export_info.workgroup_size[0] ? mm->export_info.workgroup_size[0] : mm->route.workgroup_size[0], 1, 1 },
        0,
    };
    if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream, mm->executable, mm->export_ordinal,
            &mm_config, nullptr, 0, mm_bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

// Q4NX slice helper: dequant tiles [tile_base, tile_base+n_tiles) of src0 and
// run the f32 matmul against src1 columns [src1_col, src1_col+cols), writing
// into dst columns [dst_col, dst_col+cols). Used by both the 2-D MUL_MAT_Q4NX
// (whole tensor) and MUL_MAT_ID_Q4NX (one expert per call).
static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice(
        ggml_backend_hrx2_context * context,
        const hrx_buffer_ref_t & src0_ref,
        const hrx_buffer_ref_t & src1_ref,
        const hrx_buffer_ref_t & dst_ref,
        uint32_t tile_base,   // first tile to dequantize (expert offset)
        uint32_t n_tiles,     // tiles in this slice
        uint32_t k,           // logical in width
        uint32_t rows,        // output rows
        uint32_t cols,        // columns to process
        uint32_t src1_col,    // src1 column offset (elements, multiple of k)
        uint32_t dst_col) {   // dst column offset (elements, multiple of rows)
    ggml_backend_hrx2_device_context * device_context = context->device_context;
    const uint32_t n_tc    = k / GGML_Q4NX_TILE_COLS;
    const uint32_t wg_size = 256;

    ggml_backend_hrx2_trace_event(
        "q4nx_dispatch",
        ggml_backend_hrx2_json_kv("rows", std::to_string(rows)) + "," +
        ggml_backend_hrx2_json_kv("k", std::to_string(k)) + "," +
        ggml_backend_hrx2_json_kv("cols", std::to_string(cols)) + "," +
        ggml_backend_hrx2_json_kv("n_tiles", std::to_string(n_tiles)) + "," +
        ggml_backend_hrx2_json_kv("tile_base", std::to_string(tile_base)) + "," +
        ggml_backend_hrx2_json_kv("n_tc", std::to_string(n_tc)));

    // scratch: dequant out only. The raw tiles are bound IN-PLACE from the
    // weight buffer (round 25m) — no b_raw copy (the ~983 KB/expert device
    // copy on the shared scratch was the per-group serialization + cost).
    const size_t raw_bytes = (size_t) n_tiles * 5120;
    const size_t w_bytes   = (size_t) rows * k * sizeof(float);
    if (!ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_w, &device_context->q4nx_w_cap, w_bytes)) {
        return GGML_STATUS_FAILED;
    }
    hrx_buffer_t b_w = device_context->q4nx_w;
    const size_t raw_base = src0_ref.offset + (size_t) tile_base * 5120;

    // r16x8 dense-prefill fused dequant+mm: 16 rows x 8 cols per workgroup.
    // For dense (cols >= 8, rows % 16 == 0) prefer this over the 1-row/wg
    // tbl kernel: the packed byte stream for a row is read once and feeds 8
    // output columns (tbl re-reads per row-group), and workgroup count drops
    // rows/16 x cols/8 vs rows x cols/8. Identity col mapping only.
    if (cols >= 8 && (cols % 8) == 0 && rows % 16 == 0) {
        const ggml_backend_hrx2_kernel_route * r16x8_route = nullptr;
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_q4nx_fused_f32_r16x8") { r16x8_route = r; break; }
        }
        if (r16x8_route && !getenv("GGML_HRX2_NO_R16X8")) {
            std::vector<ggml_backend_hrx2_config_binding> cfg;
            cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
            cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
            cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
            cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
            cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
            const std::string key = ggml_backend_hrx2_base_cache_key(device_context, r16x8_route) +
                "-q4nx-r16x8-r" + std::to_string(rows) + "-c" + std::to_string(cols) +
                "-k" + std::to_string(k) + "-wg" + std::to_string(wg_size);
            auto * r16x8 = ggml_backend_hrx2_get_provider(device_context, r16x8_route, cfg, key);
            if (r16x8 && r16x8->route.constant_byte_length == 0) {
                hrx_buffer_ref_t bindings[5] = {
                    { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },  // packed
                    { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },  // scales
                    { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },  // zeros
                    { src1_ref.buffer, src1_ref.offset + (size_t) src1_col * sizeof(float), (size_t) k * cols * sizeof(float) },
                    { dst_ref.buffer,  dst_ref.offset  + (size_t) dst_col  * sizeof(float), (size_t) rows * cols * sizeof(float) },
                };
                const uint32_t col_groups = (cols + 7) / 8;
                hrx_dispatch_config_t config = {
                    { rows / 16, col_groups, 1 },
                    { r16x8->export_info.workgroup_size[0] ? r16x8->export_info.workgroup_size[0] : r16x8->route.workgroup_size[0], 1, 1 },
                    0,
                };
                if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                        context->stream, r16x8->executable, r16x8->export_ordinal,
                        &config, nullptr, 0, bindings, 5, HRX_DISPATCH_FLAG_NONE))) {
                    return GGML_STATUS_FAILED;
                }
                return GGML_STATUS_SUCCESS;
            }
        }
    }

    // DENSE prefill (cols >= 8): reuse the fused table-scatter tiled mm
    // (round 23, proven on the MoE grouped path) with IDENTITY column tables —
    // one group containing all cols, src1_cols[c] = c, dst_cols[c] = c,
    // ntokens = cols, nselected = 1. Dequant stays inline (no b_w), weight
    // traffic drops ~8x vs dequant + tiled mm. Falls back to dequant + mm.
    if (cols >= 8) {
        const ggml_backend_hrx2_kernel_route * fused_tbl_route = nullptr;
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_q4nx_fused_tbl_tiled") { fused_tbl_route = r; break; }
        }
        if (fused_tbl_route && !getenv("HRX2_NO_FUSED_TBL")) {
            std::vector<ggml_backend_hrx2_config_binding> ft_cfg;
            ft_cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
            ft_cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
            ft_cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
            ft_cfg.push_back({"@hrx2.shape.ntokens", std::to_string(cols)});
            ft_cfg.push_back({"@hrx2.shape.nselected", std::to_string(1)});
            ft_cfg.push_back({"@hrx2.shape.src1_cols_count", std::to_string(cols)});
            ft_cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
            ft_cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
            const std::string ft_key = ggml_backend_hrx2_base_cache_key(device_context, fused_tbl_route) +
                "-q4nx-ft-r" + std::to_string(rows) + "-c" + std::to_string(cols) +
                "-k" + std::to_string(k) + "-nt" + std::to_string(cols) +
                "-ns1-wg" + std::to_string(wg_size);
            auto * ft = ggml_backend_hrx2_get_provider(device_context, fused_tbl_route, ft_cfg, ft_key);
            if (ft && ft->route.constant_byte_length == 0) {
                const uint32_t cols_padded = (cols + 7) / 8 * 8;
                const size_t tbl_bytes = (size_t) cols_padded * sizeof(int32_t) * 2;
                if (!ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_tbl, &device_context->q4nx_tbl_cap, tbl_bytes)) {
                    return GGML_STATUS_FAILED;
                }
                hrx_buffer_t b_tbl = device_context->q4nx_tbl;
                std::vector<int32_t> s1c_pad(cols_padded, 0);
                std::vector<int32_t> dc_pad(cols_padded, 0);
                for (uint32_t c = 0; c < cols; ++c) {
                    s1c_pad[c] = (int32_t) c;
                    dc_pad[c] = (int32_t) c;
                }
                if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                        context->stream, s1c_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                        b_tbl, 0))) {
                    return GGML_STATUS_FAILED;
                }
                if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                        context->stream, dc_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                        b_tbl, (size_t) cols_padded * sizeof(int32_t)))) {
                    return GGML_STATUS_FAILED;
                }
                // bindings: packed/scales/zeros views of the weight slice (in-place)
                // + full src1 + full dst + tables
                hrx_buffer_ref_t ft_bindings[7] = {
                    { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },
                    { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },
                    { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },
                    { src1_ref.buffer, src1_ref.offset + (size_t) src1_col * sizeof(float), (size_t) k * cols * sizeof(float) },
                    { dst_ref.buffer,  dst_ref.offset  + (size_t) dst_col  * sizeof(float), (size_t) rows * cols * sizeof(float) },
                    { b_tbl,  0, (size_t) cols_padded * sizeof(int32_t) },
                    { b_tbl, (size_t) cols_padded * sizeof(int32_t), (size_t) cols_padded * sizeof(int32_t) },
                };
                const uint32_t ft_col_groups = (cols + 7) / 8;
                hrx_dispatch_config_t ft_config = {
                    { rows, ft_col_groups, 1 },
                    { ft->export_info.workgroup_size[0] ? ft->export_info.workgroup_size[0] : ft->route.workgroup_size[0], 1, 1 },
                    0,
                };
                if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                        context->stream, ft->executable, ft->export_ordinal,
                        &ft_config, nullptr, 0, ft_bindings, 7, HRX_DISPATCH_FLAG_NONE))) {
                    return GGML_STATUS_FAILED;
                }
                return GGML_STATUS_SUCCESS;
            }
            // fall through to dequant + mm if provider unavailable
        }
    }

    // dequant provider (q4nx_dequant_f32 route)
    const ggml_backend_hrx2_kernel_route * deq_route = device_context->q4nx_dequant_routes.front();
    std::vector<ggml_backend_hrx2_config_binding> deq_cfg;
    deq_cfg.push_back({"@hrx2.shape.ncols", std::to_string(k)});
    deq_cfg.push_back({"@hrx2.shape.nrows", std::to_string(rows)});
    deq_cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
    deq_cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
    const std::string deq_key = ggml_backend_hrx2_base_cache_key(device_context, deq_route) +
        "-q4nx-r" + std::to_string(rows) + "-c" + std::to_string(k) +
        "-tc" + std::to_string(n_tc) + "-wg" + std::to_string(wg_size);
    auto * deq = ggml_backend_hrx2_get_provider(device_context, deq_route, deq_cfg, deq_key);
    if (!deq) {
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX dequant provider unavailable\n");
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t deq_bindings[4] = {
        { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },  // packed
        { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },  // scales
        { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },  // zeros
        { b_w,            0, w_bytes },
    };
    hrx_dispatch_config_t deq_config = {
        { (rows * k + wg_size - 1) / wg_size, 1, 1 },
        { wg_size, 1, 1 },
        0,
    };
    if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream, deq->executable, deq->export_ordinal,
            &deq_config, nullptr, 0, deq_bindings, 4, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return ggml_backend_hrx2_dispatch_mul_mat_q4nx_mm(
        context, b_w, w_bytes, src1_ref, dst_ref, k, rows, cols, src1_col, dst_col);
}

// Grouped Q4NX slice: dequant tiles [tile_base, tile_base+n_tiles) of src0
// ONCE, then run a single f32 matmul that scatters the outputs of `cols`
// logical columns into arbitrary src1/dst column positions given by the i32
// tables. Used by MUL_MAT_ID_Q4NX to process every token that selected the
// same expert in one dispatch (prefill: dequant work drops from #tokens to
// #distinct-experts). The mm kernel is hrx2_mul_mat_f32_f32_ggml_tbl_static
// (5 bindings: b_w, full src1, full dst, src1_cols, dst_cols).
// Fused Q4NX dequant+matmul for a single (expert slice, token) pair: reads
// the raw 5120-byte tiles DIRECTLY (no scratch copy, no b_w materialization)
// and dequants inline in the matmul. Bindings: packed/scales/zeros views of
// the expert tensor slice + src1 column + dst column. Used for decode
// (cols=1, every group a single token) where the dequant-write + mm-read of
// 33.5 MB b_w is 2x the weight traffic the fused kernel needs.
static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice_fused(
        ggml_backend_hrx2_context * context,
        const hrx_buffer_ref_t & src0_ref,
        const hrx_buffer_ref_t & src1_ref,
        const hrx_buffer_ref_t & dst_ref,
        uint32_t tile_base,   // first tile (expert offset)
        uint32_t n_tiles,     // tiles in this slice (tpe)
        uint32_t k,
        uint32_t rows,
        uint32_t src1_col,    // src1 column offset (elements, multiple of k)
        uint32_t dst_col) {   // dst column offset (elements, multiple of rows)
    ggml_backend_hrx2_device_context * device_context = context->device_context;
    const uint32_t wg_size = 256;
    const uint32_t n_tc    = k / GGML_Q4NX_TILE_COLS;

    // Prefer the r16 kernel (16 rows per workgroup, contiguous packed reads)
    // for the per-pair (cols == 1) path: it coalesces the packed bytes ~8x
    // better than the per-pair kernel. Falls back to the per-pair kernel if
    // unavailable (e.g. rows not a multiple of 16).
    const ggml_backend_hrx2_kernel_route * fused_route = nullptr;
    if (rows % 16 == 0) {
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_q4nx_fused_f32_r16") { fused_route = r; break; }
        }
    }
    if (!fused_route) {
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_q4nx_fused_f32") { fused_route = r; break; }
        }
    }
    if (!fused_route || getenv("HRX2_NO_FUSED_PAIR")) {
        return GGML_STATUS_FAILED;  // caller falls back to dequant + mm
    }

    std::vector<ggml_backend_hrx2_config_binding> cfg;
    cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
    cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
    cfg.push_back({"@hrx2.shape.cols", std::to_string(1)});
    cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
    cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
    const std::string key = ggml_backend_hrx2_base_cache_key(device_context, fused_route) +
        "-q4nx-fused-r" + std::to_string(rows) + "-c1-k" + std::to_string(k) +
        "-tc" + std::to_string(n_tc) + "-wg" + std::to_string(wg_size);
    auto * fused = ggml_backend_hrx2_get_provider(device_context, fused_route, cfg, key);
    if (!fused) {
        return GGML_STATUS_FAILED;
    }
    if (fused->route.constant_byte_length != 0) {
        return GGML_STATUS_FAILED;
    }

    const size_t slice_bytes = (size_t) n_tiles * 5120;
    const size_t raw_base = src0_ref.offset + (size_t) tile_base * 5120;
    hrx_buffer_ref_t bindings[5] = {
        { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },  // packed
        { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },  // scales
        { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },  // zeros
        { src1_ref.buffer, src1_ref.offset + (size_t) src1_col * sizeof(float), (size_t) k * sizeof(float) },
        { dst_ref.buffer,  dst_ref.offset  + (size_t) dst_col  * sizeof(float), (size_t) rows * sizeof(float) },
    };
    // The r16 kernel computes 16 output rows per workgroup (256 lanes =
    // 16 rows x 16 k-block split); the per-pair kernel is 1 row per
    // workgroup.
    const uint32_t wg_rows = (fused_route->id == "mul_mat_q4nx_fused_f32_r16") ? rows / 16 : rows;
    hrx_dispatch_config_t config = {
        { wg_rows, 1, 1 },
        { fused->export_info.workgroup_size[0] ? fused->export_info.workgroup_size[0] : fused->route.workgroup_size[0], 1, 1 },
        0,
    };
    if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream, fused->executable, fused->export_ordinal,
            &config, nullptr, 0, bindings, 5, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    (void) slice_bytes;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice_grouped(
        ggml_backend_hrx2_context * context,
        const hrx_buffer_ref_t & src0_ref,
        const hrx_buffer_ref_t & src1_ref,
        const hrx_buffer_ref_t & dst_ref,
        uint32_t tile_base,   // first tile to dequantize (expert offset)
        uint32_t n_tiles,     // tiles in this slice
        uint32_t k,           // logical in width
        uint32_t rows,        // output rows
        uint32_t cols,        // group size (number of tokens on this expert)
        uint32_t ntokens,     // total tokens in the MUL_MAT_ID (unused; kept for signature stability)
        uint32_t nselected,   // total selected per token (unused; kept for signature stability)
        uint32_t src1_cols_count, // total src1 columns: ntokens (shared src1) or nselected*ntokens (per-expert src1)
        const int32_t * src1_cols,  // src1 column per group slot (token index, or selected index for per-expert src1)
        const int32_t * dst_cols) { // dst column (i + t*nselected) per group slot

    ggml_backend_hrx2_device_context * device_context = context->device_context;
    const uint32_t n_tc    = k / GGML_Q4NX_TILE_COLS;
    const uint32_t wg_size = 256;

    ggml_backend_hrx2_trace_event(
        "q4nx_dispatch_grouped",
        ggml_backend_hrx2_json_kv("rows", std::to_string(rows)) + "," +
        ggml_backend_hrx2_json_kv("k", std::to_string(k)) + "," +
        ggml_backend_hrx2_json_kv("cols", std::to_string(cols)) + "," +
        ggml_backend_hrx2_json_kv("n_tiles", std::to_string(n_tiles)) + "," +
        ggml_backend_hrx2_json_kv("tile_base", std::to_string(tile_base)));

    // scratch: dequant out only. The raw tiles are bound IN-PLACE from the
    // weight buffer (like slice_fused) — no b_raw copy: the per-expert
    // 983 KB device copy was ~0.13 ms/group on the ~7.5 GB/s scratch path
    // and serialized the groups on the shared scratch (round 25m).
    const size_t raw_bytes = (size_t) n_tiles * 5120;
    const size_t w_bytes   = (size_t) rows * k * sizeof(float);
    if (!ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_w, &device_context->q4nx_w_cap, w_bytes)) {
        return GGML_STATUS_FAILED;
    }
    hrx_buffer_t b_w = device_context->q4nx_w;
    const size_t raw_base = src0_ref.offset + (size_t) tile_base * 5120;

    // PREFER the fused table-scatter tiled mm (round 23): dequant INLINE in
    // the matmul, reading the raw tiles from b_raw directly — no 33.5 MB b_w
    // materialization, no separate dequant pass. Falls back to dequant +
    // tbl-tiled if the route is absent.
    // r16x8t: 16 rows x 8 table-scatter cols per workgroup — same in-place
    // dequant as the fused tbl but rows/16 workgroups (MoE grouped prefill
    // groups are small: rows=1536/2048, cols=2..7 -> tbl launches rows wg,
    // r16x8t launches rows/16). Prefer when rows % 16 == 0.
    if ((rows % 16) == 0) {
        const ggml_backend_hrx2_kernel_route * r16x8t_route = nullptr;
        for (const auto * r : device_context->mul_mat_f32_f32_routes) {
            if (r->id == "mul_mat_q4nx_fused_f32_r16x8t") { r16x8t_route = r; break; }
        }
        if (r16x8t_route && !getenv("GGML_HRX2_NO_R16X8T")) {
            std::vector<ggml_backend_hrx2_config_binding> cfg;
            cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
            cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
            cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
            cfg.push_back({"@hrx2.shape.ntokens", std::to_string(ntokens)});
            cfg.push_back({"@hrx2.shape.nselected", std::to_string(nselected)});
            cfg.push_back({"@hrx2.shape.src1_cols_count", std::to_string(src1_cols_count)});
            cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
            cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
            const std::string key = ggml_backend_hrx2_base_cache_key(device_context, r16x8t_route) +
                "-q4nx-r16x8t-r" + std::to_string(rows) + "-c" + std::to_string(cols) +
                "-k" + std::to_string(k) + "-nt" + std::to_string(ntokens) +
                "-ns" + std::to_string(nselected) + "-wg" + std::to_string(wg_size);
            auto * r16x8t = ggml_backend_hrx2_get_provider(device_context, r16x8t_route, cfg, key);
            if (r16x8t && r16x8t->route.constant_byte_length == 0) {
                const uint32_t cols_padded = (cols + 7) / 8 * 8;
                const size_t tbl_bytes = (size_t) cols_padded * sizeof(int32_t) * 2;
                if (!ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_tbl, &device_context->q4nx_tbl_cap, tbl_bytes)) {
                    return GGML_STATUS_FAILED;
                }
                hrx_buffer_t b_tbl = device_context->q4nx_tbl;
                std::vector<int32_t> s1c_pad(cols_padded, 0);
                std::vector<int32_t> dc_pad(cols_padded, 0);
                for (uint32_t c = 0; c < cols; ++c) {
                    s1c_pad[c] = src1_cols[c];
                    dc_pad[c] = dst_cols[c];
                }
                if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                        context->stream, s1c_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                        b_tbl, 0))) {
                    return GGML_STATUS_FAILED;
                }
                if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                        context->stream, dc_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                        b_tbl, (size_t) cols_padded * sizeof(int32_t)))) {
                    return GGML_STATUS_FAILED;
                }
                hrx_buffer_ref_t bindings[7] = {
                    { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },
                    { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },
                    { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },
                    { src1_ref.buffer, src1_ref.offset, (size_t) k * src1_cols_count * sizeof(float) },
                    { dst_ref.buffer,  dst_ref.offset,  (size_t) rows * nselected * ntokens * sizeof(float) },
                    { b_tbl,  0, (size_t) cols_padded * sizeof(int32_t) },
                    { b_tbl, (size_t) cols_padded * sizeof(int32_t), (size_t) cols_padded * sizeof(int32_t) },
                };
                const uint32_t col_groups = (cols + 7) / 8;
                hrx_dispatch_config_t config = {
                    { rows / 16, col_groups, 1 },
                    { r16x8t->export_info.workgroup_size[0] ? r16x8t->export_info.workgroup_size[0] : r16x8t->route.workgroup_size[0], 1, 1 },
                    0,
                };
                if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                        context->stream, r16x8t->executable, r16x8t->export_ordinal,
                        &config, nullptr, 0, bindings, 7, HRX_DISPATCH_FLAG_NONE))) {
                    return GGML_STATUS_FAILED;
                }
                return GGML_STATUS_SUCCESS;
            }
        }
    }

    const ggml_backend_hrx2_kernel_route * fused_tbl_route = nullptr;
    for (const auto * r : device_context->mul_mat_f32_f32_routes) {
        if (r->id == "mul_mat_q4nx_fused_tbl_tiled") { fused_tbl_route = r; break; }
    }
    if (fused_tbl_route && !getenv("HRX2_NO_FUSED_TBL")) {
        std::vector<ggml_backend_hrx2_config_binding> ft_cfg;
        ft_cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
        ft_cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
        ft_cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
        ft_cfg.push_back({"@hrx2.shape.ntokens", std::to_string(ntokens)});
        ft_cfg.push_back({"@hrx2.shape.nselected", std::to_string(nselected)});
        ft_cfg.push_back({"@hrx2.shape.src1_cols_count", std::to_string(src1_cols_count)});
        ft_cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
        ft_cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
        const std::string ft_key = ggml_backend_hrx2_base_cache_key(device_context, fused_tbl_route) +
            "-q4nx-ft-r" + std::to_string(rows) + "-c" + std::to_string(cols) +
            "-k" + std::to_string(k) + "-nt" + std::to_string(ntokens) +
            "-ns" + std::to_string(nselected) + "-wg" + std::to_string(wg_size);
        auto * ft = ggml_backend_hrx2_get_provider(device_context, fused_tbl_route, ft_cfg, ft_key);
        if (ft && ft->route.constant_byte_length == 0) {
            const uint32_t cols_padded = (cols + 7) / 8 * 8;
            const size_t tbl_bytes = (size_t) cols_padded * sizeof(int32_t) * 2;
            if (!ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_tbl, &device_context->q4nx_tbl_cap, tbl_bytes)) {
                return GGML_STATUS_FAILED;
            }
            hrx_buffer_t b_tbl = device_context->q4nx_tbl;
            std::vector<int32_t> s1c_pad(cols_padded, 0);
            std::vector<int32_t> dc_pad(cols_padded, 0);
            for (uint32_t c = 0; c < cols; ++c) {
                s1c_pad[c] = src1_cols[c];
                dc_pad[c] = dst_cols[c];
            }
            if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                    context->stream, s1c_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                    b_tbl, 0))) {
                return GGML_STATUS_FAILED;
            }
            if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                    context->stream, dc_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                    b_tbl, (size_t) cols_padded * sizeof(int32_t)))) {
                return GGML_STATUS_FAILED;
            }
            // bindings: packed/scales/zeros views of the weight slice (in-place)
            // + full src1 + full dst + tables
            hrx_buffer_ref_t ft_bindings[7] = {
                { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },
                { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },
                { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },
                { src1_ref.buffer, src1_ref.offset, (size_t) k * ntokens * sizeof(float) },
                { dst_ref.buffer,  dst_ref.offset,  (size_t) rows * nselected * ntokens * sizeof(float) },
                { b_tbl,  0, (size_t) cols_padded * sizeof(int32_t) },
                { b_tbl, (size_t) cols_padded * sizeof(int32_t), (size_t) cols_padded * sizeof(int32_t) },
            };
            const uint32_t ft_col_groups = (cols + 7) / 8;
            hrx_dispatch_config_t ft_config = {
                { rows, ft_col_groups, 1 },
                { ft->export_info.workgroup_size[0] ? ft->export_info.workgroup_size[0] : ft->route.workgroup_size[0], 1, 1 },
                0,
            };
            if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                    context->stream, ft->executable, ft->export_ordinal,
                    &ft_config, nullptr, 0, ft_bindings, 7, HRX_DISPATCH_FLAG_NONE))) {
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }
        // fall through to dequant + tbl-tiled if provider unavailable
    }

    // dequant provider (q4nx_dequant_f32 route)
    const ggml_backend_hrx2_kernel_route * deq_route = device_context->q4nx_dequant_routes.front();
    std::vector<ggml_backend_hrx2_config_binding> deq_cfg;
    deq_cfg.push_back({"@hrx2.shape.ncols", std::to_string(k)});
    deq_cfg.push_back({"@hrx2.shape.nrows", std::to_string(rows)});
    deq_cfg.push_back({"@hrx2.tuning.q4nx.workgroup_size", std::to_string(wg_size)});
    deq_cfg.push_back({"@hrx2.tuning.q4nx.n_tile_cols", std::to_string(n_tc)});
    const std::string deq_key = ggml_backend_hrx2_base_cache_key(device_context, deq_route) +
        "-q4nx-r" + std::to_string(rows) + "-c" + std::to_string(k) +
        "-tc" + std::to_string(n_tc) + "-wg" + std::to_string(wg_size);
    auto * deq = ggml_backend_hrx2_get_provider(device_context, deq_route, deq_cfg, deq_key);
    if (!deq) {
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX dequant provider unavailable\n");
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t deq_bindings[4] = {
        { src0_ref.buffer, raw_base + 1024, (size_t) n_tiles * 4096 },  // packed
        { src0_ref.buffer, raw_base,        (size_t) n_tiles * 512  },  // scales
        { src0_ref.buffer, raw_base + 512,  (size_t) n_tiles * 512  },  // zeros
        { b_w,            0, w_bytes },
    };
    hrx_dispatch_config_t deq_config = {
        { (rows * k + wg_size - 1) / wg_size, 1, 1 },
        { wg_size, 1, 1 },
        0,
    };
    if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream, deq->executable, deq->export_ordinal,
            &deq_config, nullptr, 0, deq_bindings, 4, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    // Prefer ONE table-scatter tiled mm per group: b_w x src1[:, src1_cols[*]]
    // -> dst[:, dst_cols[*]] with 8 columns per workgroup, so b_w is read once
    // per 8 output columns (per-slot mms re-read b_w once per token = 32x for
    // a 32-token group). Round 20's tbl kernel failed because index.min/max/rem
    // have no amdgpu lowering (bounds could not be proven -> JIT miscompile /
    // subrange errors) and its use_col_tables condition folded dead. This
    // kernel instead uses TRUTHFUL index.assume predicates with config-derived
    // bounds (src1_cols < ntokens, dst_cols < nselected*ntokens) and the same
    // JIT-constant column guards as the proven tiled kernel; the tables are
    // uploaded with async host->device copies, no host round-trip.
    const ggml_backend_hrx2_kernel_route * tbl_route = nullptr;
    for (const auto * r : device_context->mul_mat_f32_f32_routes) {
        if (r->id == "mul_mat_f32_f32_ggml_tbl_tiled") { tbl_route = r; break; }
    }
    if (tbl_route) {
        // The kernel reads table slots [0, ceil(cols/8)*8) unconditionally
        // (the per-slot guards only protect src1/dst, not the table loads),
        // so pad the tables to the workgroup column grid with zeros.
        const uint32_t cols_padded = (cols + 7) / 8 * 8;
        const size_t tbl_bytes = (size_t) cols_padded * sizeof(int32_t) * 2;
        if (!ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_tbl, &device_context->q4nx_tbl_cap, tbl_bytes)) {
            return GGML_STATUS_FAILED;
        }
        hrx_buffer_t b_tbl = device_context->q4nx_tbl;
        std::vector<int32_t> s1c_pad(cols_padded, 0);
        std::vector<int32_t> dc_pad(cols_padded, 0);
        for (uint32_t c = 0; c < cols; ++c) {
            s1c_pad[c] = src1_cols[c];
            dc_pad[c] = dst_cols[c];
        }
        if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                context->stream, s1c_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                b_tbl, 0))) {
            return GGML_STATUS_FAILED;
        }
        if (!GGML_HRX2_CHECK(hrx_stream_update_buffer(
                context->stream, dc_pad.data(), (size_t) cols_padded * sizeof(int32_t),
                b_tbl, (size_t) cols_padded * sizeof(int32_t)))) {
            return GGML_STATUS_FAILED;
        }

        // tbl-tiled mm provider (mul_mat_f32_f32_ggml_tbl_tiled, 5 bindings)
        std::vector<ggml_backend_hrx2_config_binding> tbl_cfg;
        tbl_cfg.push_back({"@hrx2.shape.k", std::to_string(k)});
        tbl_cfg.push_back({"@hrx2.shape.rows", std::to_string(rows)});
        tbl_cfg.push_back({"@hrx2.shape.cols", std::to_string(cols)});
        tbl_cfg.push_back({"@hrx2.shape.ntokens", std::to_string(ntokens)});
        tbl_cfg.push_back({"@hrx2.shape.nselected", std::to_string(nselected)});
        tbl_cfg.push_back({"@hrx2.shape.src1_cols_count", std::to_string(src1_cols_count)});
        tbl_cfg.push_back({"@hrx2.tuning.workgroup_size", std::to_string(wg_size)});
        const std::string tbl_key = ggml_backend_hrx2_base_cache_key(device_context, tbl_route) +
            "-q4nx-tbl-r" + std::to_string(rows) + "-c" + std::to_string(cols) +
            "-k" + std::to_string(k) + "-nt" + std::to_string(ntokens) +
            "-ns" + std::to_string(nselected) + "-wg" + std::to_string(wg_size);
        auto * tbl = ggml_backend_hrx2_get_provider(device_context, tbl_route, tbl_cfg, tbl_key);
        if (!tbl) {
            GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX tbl-tiled provider unavailable\n");
            return GGML_STATUS_FAILED;
        }
        if (tbl->route.constant_byte_length != 0) {
            GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX tbl-tiled route has unsupported constants\n");
            return GGML_STATUS_FAILED;
        }

        // bindings: b_w, FULL src1, FULL dst, src1_cols table, dst_cols table
        hrx_buffer_ref_t tbl_bindings[5] = {
            { b_w,    0, w_bytes },
            { src1_ref.buffer, src1_ref.offset, (size_t) k * ntokens * sizeof(float) },
            { dst_ref.buffer,  dst_ref.offset,  (size_t) rows * nselected * ntokens * sizeof(float) },
            { b_tbl,  0, (size_t) cols_padded * sizeof(int32_t) },
            { b_tbl, (size_t) cols_padded * sizeof(int32_t), (size_t) cols_padded * sizeof(int32_t) },
        };
        const uint32_t tbl_col_groups = (cols + 7) / 8;
        hrx_dispatch_config_t tbl_config = {
            { rows, tbl_col_groups, 1 },
            { tbl->export_info.workgroup_size[0] ? tbl->export_info.workgroup_size[0] : tbl->route.workgroup_size[0], 1, 1 },
            0,
        };
        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream, tbl->executable, tbl->export_ordinal,
                &tbl_config, nullptr, 0, tbl_bindings, 5, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    // fallback: one proven per-slot mm per group member, all reading the SAME
    // b_w (dequant once per distinct expert; per-slot src1/dst offsets). Kept
    // only if the tbl_tiled route is missing.
    for (uint32_t c = 0; c < cols; ++c) {
        if (ggml_backend_hrx2_dispatch_mul_mat_q4nx_mm(
                context, b_w, w_bytes, src1_ref, dst_ref,
                k, rows, /* cols */ 1,
                (uint32_t) src1_cols[c] * k,
                (uint32_t) dst_cols[c] * rows) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4nx(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t src0_ref = {};
    hrx_buffer_ref_t src1_ref = {};
    hrx_buffer_ref_t dst_ref = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &src0_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &src1_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4NX tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }
    // Dense decode (cols == 1): use the fused per-pair kernel (dequant
    // inline, no b_w materialization) like the MoE decode path; fall back to
    // the dequant+mm slice if the fused route is unavailable.
    if (src1->ne[1] == 1) {
        if (ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice_fused(
                context, src0_ref, src1_ref, dst_ref,
                /* tile_base */ 0,
                /* n_tiles */ (uint32_t) src0->ne[1],
                /* k */ (uint32_t) src1->ne[0],
                /* rows */ (uint32_t) dst->ne[0],
                /* src1_col */ 0,
                /* dst_col */ 0) == GGML_STATUS_SUCCESS) {
            return GGML_STATUS_SUCCESS;
        }
    }
    return ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice(
        context, src0_ref, src1_ref, dst_ref,
        /* tile_base */ 0,
        /* n_tiles */ (uint32_t) src0->ne[1],
        /* k */ (uint32_t) src1->ne[0],
        /* rows */ (uint32_t) dst->ne[0],
        /* cols */ (uint32_t) src1->ne[1],
        /* src1_col */ 0,
        /* dst_col */ 0);
}

static bool ggml_backend_hrx2_supports_mul_mat_id_q4nx_route(
        ggml_backend_hrx2_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    const int64_t k  = src1->ne[0];
    const int64_t tpe = src0->ne[1];
    const int64_t nexperts = src0->ne[2];
    const int64_t nselected = src2->ne[0];
    const int64_t ntokens = op->ne[2];
    bool ok = true;
    auto fail = [&](const char * why) {
        (void) why;
        ok = false;
    };
    if (op->op != GGML_OP_MUL_MAT_ID_Q4NX) fail("op");
    if (!src0 || !src1 || !src2) fail("nullsrc");
    if (op->view_src != nullptr) fail("view_src");
    if (src0->type != GGML_TYPE_Q4NX) fail("src0type");
    if (src1->type != GGML_TYPE_F32) fail("src1type");
    if (src2->type != GGML_TYPE_I32) fail("src2type");
    if (op->type != GGML_TYPE_F32) fail("optype");
    if (k <= 0 || k % GGML_Q4NX_TILE_COLS != 0) fail("k");
    if (src0->ne[0] != GGML_Q4NX_TILE_COLS * GGML_Q4NX_TILE_ROWS) fail("ne0");
    if (tpe <= 0 || tpe % (k / GGML_Q4NX_TILE_COLS) != 0) fail("tpe");
    if (nexperts <= 0 || nselected <= 0 || ntokens <= 0) fail("dims0");
    if (src0->ne[3] != 1 || src1->ne[2] != ntokens || src1->ne[3] != 1) fail("src1dims");
    if (src2->ne[1] != ntokens || src2->ne[2] != 1 || src2->ne[3] != 1) fail("src2dims");
    if (op->ne[1] != nselected || op->ne[3] != 1) fail("opdims");
    if (!ggml_is_contiguous(src0)) fail("contig0");
    if (!ggml_is_contiguous(src1)) fail("contig1");
    if (!ggml_is_contiguous(op)) fail("contigop");
    // src2 (ids) may be a non-contiguous view from argsort/get_rows; the
    // dispatch reads it element-wise via nb[1] stride, so allow views with
    // standard element strides.
    if (src2->nb[0] != sizeof(int32_t) || src2->nb[1] % sizeof(int32_t) != 0) fail("src2nb");
    if (src1->nb[0] != sizeof(float) || src2->nb[0] != sizeof(int32_t)) fail("nb0");
    if (device_context->q4nx_dequant_routes.empty()) fail("nodeq");
    if (device_context->mul_mat_f32_f32_routes.empty()) fail("nomm");
    return ok;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q4nx(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // Q4NX [8192, tpe, n_expert]
    const ggml_tensor * src1 = dst->src[1]; // F32 [k, ntokens]
    const ggml_tensor * src2 = dst->src[2]; // ids [nselected, ntokens]
    hrx_buffer_ref_t src0_ref = {};
    hrx_buffer_ref_t src1_ref = {};
    hrx_buffer_ref_t src2_ref = {};
    hrx_buffer_ref_t dst_ref = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &src0_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &src1_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src2, &src2_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("HRX2: MUL_MAT_ID Q4NX tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    const uint32_t k          = (uint32_t) src1->ne[0];
    const uint32_t tpe        = (uint32_t) src0->ne[1]; // tiles per expert
    const uint32_t n_tc       = k / GGML_Q4NX_TILE_COLS;
    const uint32_t rows       = (tpe / n_tc) * GGML_Q4NX_TILE_ROWS;
    const uint32_t nselected  = (uint32_t) src2->ne[0];
    const uint32_t ntokens    = (uint32_t) dst->ne[2];
    // src1 may be [k, 1, ntokens] (gate/up: shared column per token) or
    // [k, nselected, ntokens] (down: per-expert column per slot).
    const bool per_expert_src1 = src1->ne[1] > 1;
    const uint32_t src1_cols_count = per_expert_src1 ? nselected * ntokens : ntokens;

    // The ids tensor may live in device memory (argsort output on HRX) or be
    // a non-contiguous host view; copy it into a host-visible scratch buffer
    // and read the expert ids from there (row stride = src2->nb[1]).
    ggml_backend_hrx2_device_context * device_context = context->device_context;
    const size_t ids_bytes = (size_t) nselected * (size_t) ntokens * sizeof(int32_t);
    const bool ids_direct = src2->buffer != nullptr &&
        src2->buffer->buft == &device_context->host_buffer_type &&
        src2->data != nullptr;
    if (!ids_direct && !ggml_backend_hrx2_q4nx_scratch_grow(device_context, &device_context->q4nx_ids, &device_context->q4nx_ids_cap, ids_bytes)) {
        return GGML_STATUS_FAILED;
    }
    {
        // ZERO-COPY fast path (round 25i): when src2 lives in the
        // host-coherent buft (CPU argsort wrote the ids into shared GTT),
        // read the ids directly from src2->data — NO device copy, NO stream
        // synchronize. The per-op synchronize was the prefill bottleneck
        // (3812 ms of syncs vs 59 ms of compute on the 30B q4nx pp32); the
        // ids are tiny and the CPU-written GTT is host-coherent, so the read
        // needs no flush. Fall back to the copy+sync path for device ids.
        hrx_buffer_ref_t ids_ref = {};
        if (!ids_direct) {
            if (!ggml_backend_hrx2_tensor_buffer_ref(src2, &ids_ref)) {
                GGML_LOG_ERROR("HRX2: MUL_MAT_ID Q4NX ids tensor not backed by HRX2 buffers\n");
                return GGML_STATUS_FAILED;
            }
            // The ids tensor is [nselected, ntokens] but may be a strided
            // VIEW of the argsort output (nb[1] >> nselected*4). Copy each
            // token row at its real stride into a packed host-visible scratch
            // so the dispatch can read ids as contiguous [nselected, ntokens].
            const size_t ids_src_stride = src2->nb[1];
            const size_t ids_row_bytes  = (size_t) nselected * sizeof(int32_t);
            for (uint32_t t = 0; t < ntokens; ++t) {
                if (!GGML_HRX2_CHECK(hrx_stream_copy_buffer(
                        context->stream, ids_ref.buffer,
                        ids_ref.offset + (size_t) t * ids_src_stride,
                        device_context->q4nx_ids, (size_t) t * ids_row_bytes,
                        ids_row_bytes))) {
                    return GGML_STATUS_FAILED;
                }
            }
            // the copy is async; sync before mapping/reading on the host
            if (!GGML_HRX2_CHECK(hrx_stream_synchronize(context->stream))) {
                return GGML_STATUS_FAILED;
            }
        }
    }
    const int32_t * ids_data = nullptr;
    const uint8_t * ids_host = nullptr;
    if (ids_direct) {
        ids_data = static_cast<const int32_t *>(src2->data);
    } else {
        if (!GGML_HRX2_CHECK(hrx_buffer_map(device_context->q4nx_ids, HRX_MEMORY_ACCESS_READ, 0, ids_bytes, (void**) &ids_host))) {
            return GGML_STATUS_FAILED;
        }
        ids_data = (const int32_t *) ids_host;
    }

    // Prefer the table-scatter path: group every (selected, token) pair by
    // expert and dequant each expert once per graph instead of once per token
    // (prefill: n_dequant_dispatches drops from #tokens to #distinct-experts).
    // Fall back to one slice per pair if the tbl route is unavailable.
    // The grouped path (dequant once per distinct expert + one table-scatter
    // mm per group) is always taken; single-member groups keep the proven
    // per-pair fused path (no table overhead for decode).
    // validate ids and bucket pairs by expert (in (i,t) order per bucket)
    const uint32_t nexperts = (uint32_t) src0->ne[2];
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> groups(nexperts);
    for (uint32_t i = 0; i < nselected; ++i) {
        for (uint32_t t = 0; t < ntokens; ++t) {
            // direct (host-coherent GTT) ids are a strided view: token t at
            // src2->nb[1] bytes; scratch ids are packed [nselected, ntokens].
            const int32_t e = ids_direct
                ? reinterpret_cast<const int32_t *>(
                      reinterpret_cast<const uint8_t *>(ids_data) + (size_t) t * src2->nb[1])[i]
                : ids_data[t * nselected + i];
            if (e < 0 || (uint32_t) e >= nexperts) {
                GGML_LOG_ERROR("HRX2: MUL_MAT_ID Q4NX expert id %d out of range\n", (int) e);
                hrx_buffer_unmap(device_context->q4nx_ids);
                return GGML_STATUS_FAILED;
            }
            groups[(uint32_t) e].emplace_back(i, t);
        }
    }

    // grouped path: one dequant + one table-scatter mm per distinct expert
    {
        std::vector<int32_t> src1_cols;
        std::vector<int32_t> dst_cols;
        src1_cols.reserve(ntokens);
        dst_cols.reserve(ntokens);
        for (uint32_t e = 0; e < nexperts; ++e) {
            const auto & group = groups[e];
            if (group.empty()) continue;
            if (group.size() == 1) {
                // single token: keep the proven per-pair path (no table
                // overhead for decode where every group is size 1)
                const uint32_t i = group[0].first;
                const uint32_t t = group[0].second;
                const uint32_t tile_base = e * tpe;
                const uint32_t src1_col  = (per_expert_src1 ? (i + t * nselected) * k : t * k);
                const uint32_t dst_col   = (i + t * nselected) * rows;
                if (ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice_fused(
                        context, src0_ref, src1_ref, dst_ref,
                        tile_base, tpe, k, rows,
                        src1_col, dst_col) != GGML_STATUS_SUCCESS) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice(
                            context, src0_ref, src1_ref, dst_ref,
                            tile_base, tpe, k, rows, /* cols */ 1,
                            src1_col, dst_col) != GGML_STATUS_SUCCESS) {
                        hrx_buffer_unmap(device_context->q4nx_ids);
                        return GGML_STATUS_FAILED;
                    }
                }
                continue;
            }
            src1_cols.clear();
            dst_cols.clear();
            for (const auto & pr : group) {
                // src1 column: token index (shared) or selected index (per-expert)
                src1_cols.push_back(per_expert_src1 ? (int32_t) (pr.first + pr.second * nselected) : (int32_t) pr.second);
                dst_cols.push_back((int32_t) (pr.first + pr.second * nselected)); // dst col
            }
            const uint32_t tile_base = e * tpe;
            if (ggml_backend_hrx2_dispatch_mul_mat_q4nx_slice_grouped(
                    context, src0_ref, src1_ref, dst_ref,
                    tile_base, tpe, k, rows, (uint32_t) group.size(),
                    ntokens, nselected, src1_cols_count,
                    src1_cols.data(), dst_cols.data()) != GGML_STATUS_SUCCESS) {
                hrx_buffer_unmap(device_context->q4nx_ids);
                return GGML_STATUS_FAILED;
            }
        }
        if (!ids_direct) hrx_buffer_unmap(device_context->q4nx_ids);
        return GGML_STATUS_SUCCESS;
    }

    if (!ids_direct) hrx_buffer_unmap(device_context->q4nx_ids);
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_f32_f32(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT F32/F32 tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_f32_f32_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT F32/F32 shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_f32_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q4_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT F32/F32 route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const int64_t n_batch = (int64_t) src1->ne[2] * src1->ne[3];
        const int64_t n_batch_src0 = (int64_t) src0->ne[2] * src0->ne[3];
        const int64_t inner_src0 = (int64_t) shape.k * shape.rows;
        const int64_t inner_src1 = (int64_t) shape.k * shape.cols;
        const int64_t inner_dst = (int64_t) shape.rows * shape.cols;
        const int64_t src0_repeats = n_batch / (n_batch_src0 > 0 ? n_batch_src0 : 1);
        const int64_t nb2_src0 = (int64_t) src0->nb[2] / (int64_t) sizeof(float);
        const int64_t nb3_src0 = (int64_t) src0->nb[3] / (int64_t) sizeof(float);
        const int64_t nb2_src1 = (int64_t) src1->nb[2] / (int64_t) sizeof(float);
        const int64_t nb3_src1 = (int64_t) src1->nb[3] / (int64_t) sizeof(float);
        const int64_t nb2_dst = (int64_t) dst->nb[2] / (int64_t) sizeof(float);
        const int64_t nb3_dst = (int64_t) dst->nb[3] / (int64_t) sizeof(float);

        for (int64_t b = 0; b < n_batch; ++b) {
            // GQA: src0 batch = b / src0_repeats, src1/dst batch = b
            const int64_t i2 = b % (int64_t) src1->ne[2];
            const int64_t i3 = b / (int64_t) src1->ne[2];
            const int64_t b_src0 = n_batch_src0 > 1 ? b / src0_repeats : 0;
            const int64_t i2_src0 = b_src0 % (int64_t) src0->ne[2];
            const int64_t i3_src0 = b_src0 / (int64_t) src0->ne[2];
            const int64_t off_src0 = i2_src0 * nb2_src0 + i3_src0 * nb3_src0;
            const int64_t off_src1 = i2 * nb2_src1 + i3 * nb3_src1;
            const int64_t off_dst = i2 * nb2_dst + i3 * nb3_dst;
            hrx_buffer_ref_t b_bindings[3] = {
                { bindings[0].buffer, bindings[0].offset + (size_t) off_src0 * sizeof(float), (size_t) inner_src0 * sizeof(float) },
                { bindings[1].buffer, bindings[1].offset + (size_t) off_src1 * sizeof(float), (size_t) inner_src1 * sizeof(float) },
                { bindings[2].buffer, bindings[2].offset + (size_t) off_dst  * sizeof(float), (size_t) inner_dst  * sizeof(float) },
            };
            hrx_dispatch_config_t config = {
                /* .workgroup_count = */ {
                    (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                    (shape.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                    1,
                },
                /* .workgroup_size  = */ {
                    provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                    1,
                    1,
                },
                /* .subgroup_size   = */ 0,
            };

            ggml_backend_hrx2_trace_event(
                "dispatch",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
                ggml_backend_hrx2_json_kv("batch", (uint64_t) b) + "," +
                ggml_backend_hrx2_json_kv("n_batch", (uint64_t) n_batch) + "," +
                ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
                ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
                ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

            if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                    context->stream,
                    provider->executable,
                    provider->export_ordinal,
                    &config,
                    nullptr,
                    0,
                    b_bindings,
                    3,
                    HRX_DISPATCH_FLAG_NONE))) {
                ggml_backend_hrx2_trace_event(
                    "dispatch_failed",
                    ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                    ggml_backend_hrx2_json_kv("family", "mul_mat_f32_f32") + "," +
                    ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                    ggml_backend_hrx2_json_kv("route_id", provider->route.id));
                return GGML_STATUS_FAILED;
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT F32/F32 provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static bool ggml_backend_hrx2_q4_k_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape) {
    return shape.cols > 1 && !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q4_K_Q8_1_PROMPT");
}

static bool ggml_backend_hrx2_q8_0_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape) {
    return shape.cols > 1 && !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q8_0_Q8_1_PROMPT");
}

static bool ggml_backend_hrx2_q5_k_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape) {
    return shape.cols > 1 && !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q5_K_Q8_1_PROMPT");
}

static bool ggml_backend_hrx2_q6_k_q8_1_prompt_enabled(const ggml_backend_hrx2_mul_mat_shape & shape) {
    return shape.cols > 1 && !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q6_K_Q8_1_PROMPT");
}

static bool ggml_backend_hrx2_route_uses_q8_1_rhs(const ggml_backend_hrx2_kernel_route * route) {
    return route && route->id.find("q8_1") != std::string::npos;
}

static bool ggml_backend_hrx2_route_uses_q8_1_x4_rhs(const ggml_backend_hrx2_kernel_route * route) {
    return route && route->id.find("q8_1_x4") != std::string::npos;
}

static bool ggml_backend_hrx2_cont_route_copies_vec4(const ggml_backend_hrx2_kernel_route * route) {
    return route && route->supports_layout == "row_contiguous_src_to_contiguous_dst_vec4";
}

static bool ggml_backend_hrx2_q4_k_q8_1_x4_mmq_enabled() {
    return !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q4_K_Q8_1_X4_MMQ");
}

static bool ggml_backend_hrx2_q5_k_q8_1_x4_prompt_enabled() {
    return !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q5_K_Q8_1_X4_PROMPT");
}

static bool ggml_backend_hrx2_q6_k_q8_1_x4_prompt_enabled() {
    return !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q6_K_Q8_1_X4_PROMPT");
}

static uint32_t ggml_backend_hrx2_provider_workgroup_size_x(const ggml_backend_hrx2_provider * provider) {
    if (!provider) {
        return 1;
    }
    return provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
}

static bool ggml_backend_hrx2_dispatch_quantize_q8_1(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * src,
        bool use_x4,
        hrx_buffer_ref_t * out_q8_1_ref) {
    if (!context || !src || !out_q8_1_ref || context->device_context->quantize_q8_1_routes.empty()) {
        return false;
    }
    if (src->type != GGML_TYPE_F32 || (src->ne[0] % 32) != 0) {
        return false;
    }

    uint32_t ne00 = 0;
    uint32_t ne0 = 0;
    uint32_t ne1 = 0;
    uint32_t ne2 = 0;
    uint32_t ne3 = 0;
    uint32_t s01 = 0;
    uint32_t s02 = 0;
    uint32_t s03 = 0;
    if (!ggml_backend_hrx2_u32(src->ne[0], &ne00) ||
        !ggml_backend_hrx2_u32(src->ne[0], &ne0) ||
        !ggml_backend_hrx2_u32(src->ne[1], &ne1) ||
        !ggml_backend_hrx2_u32(src->ne[2], &ne2) ||
        !ggml_backend_hrx2_u32(src->ne[3], &ne3) ||
        !ggml_backend_hrx2_u32(src->nb[1] / sizeof(float), &s01) ||
        !ggml_backend_hrx2_u32(src->nb[2] / sizeof(float), &s02) ||
        !ggml_backend_hrx2_u32(src->nb[3] / sizeof(float), &s03)) {
        return false;
    }

    const uint32_t blocks = ne0 / 32;
    const uint32_t z_count = ne2 * ne3;
    if (blocks == 0 || blocks > 1024 || ne1 == 0 || ne1 > 512 || z_count == 0 || z_count > 16) {
        return false;
    }
    if (use_x4 && (blocks % 4) != 0) {
        return false;
    }
    const uint32_t dispatch_blocks = use_x4 ? blocks / 4 : blocks;
    const size_t q8_1_size = use_x4 ?
        static_cast<size_t>(ne1) * z_count * (blocks / 4) * 144u :
        static_cast<size_t>(ne1) * z_count * blocks * 36u;
    if (context->q8_1_cached_src == src &&
            context->q8_1_cached_use_x4 == use_x4 &&
            context->q8_1_cached_size == q8_1_size &&
            context->q8_1_cached_ne00 == ne00 &&
            context->q8_1_cached_s01 == s01 &&
            context->q8_1_cached_s02 == s02 &&
            context->q8_1_cached_s03 == s03 &&
            context->q8_1_cached_ne0 == ne0 &&
            context->q8_1_cached_ne1 == ne1 &&
            context->q8_1_cached_ne2 == ne2 &&
            context->q8_1_cached_blocks == blocks &&
            context->q8_1_cached_z_count == z_count) {
        *out_q8_1_ref = context->q8_1_cached_ref;
        ggml_backend_hrx2_trace_event(
            "quantize_cache_hit",
            ggml_backend_hrx2_json_kv("family", "quantize_q8_1_f32") + "," +
            ggml_backend_hrx2_json_kv("blocks", blocks) + "," +
            ggml_backend_hrx2_json_kv("packed_x4", use_x4 ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("ne1", ne1) + "," +
            ggml_backend_hrx2_json_kv("z_count", z_count));
        return true;
    }

    hrx_buffer_ref_t src_ref = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src, &src_ref) ||
        !ggml_backend_hrx2_ensure_device_scratch(context, &context->q8_1_scratch, q8_1_size, out_q8_1_ref)) {
        return false;
    }

    const ggml_backend_hrx2_quantize_q8_1_shape shape = {
        /* .blocks  = */ blocks,
        /* .ne1     = */ ne1,
        /* .z_count = */ z_count,
    };
    const ggml_backend_hrx2_quantize_q8_1_constants constants = {
        /* .ne00 = */ ne00,
        /* .s01  = */ s01,
        /* .s02  = */ s02,
        /* .s03  = */ s03,
        /* .ne0  = */ ne0,
        /* .ne1  = */ ne1,
        /* .ne2  = */ ne2,
    };

    for (const auto * route : context->device_context->quantize_q8_1_routes) {
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) != use_x4) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_quantize_q8_1_plan(context->device_context, route, shape, &plan)) {
            continue;
        }
        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider || provider->route.constant_byte_length != sizeof(constants)) {
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ { dispatch_blocks, ne1, z_count },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };
        hrx_buffer_ref_t bindings[2] = { src_ref, *out_q8_1_ref };
        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "QUANTIZE") + "," +
            ggml_backend_hrx2_json_kv("family", "quantize_q8_1_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("blocks", blocks) + "," +
            ggml_backend_hrx2_json_kv("packed_x4", use_x4 ? 1 : 0) + "," +
            ggml_backend_hrx2_json_kv("ne1", ne1) + "," +
            ggml_backend_hrx2_json_kv("z_count", z_count));

        const bool ok = GGML_HRX2_CHECK(hrx_stream_dispatch(
            context->stream,
            provider->executable,
            provider->export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE));
        if (ok) {
            context->q8_1_cached_src = src;
            context->q8_1_cached_use_x4 = use_x4;
            context->q8_1_cached_size = q8_1_size;
            context->q8_1_cached_ne00 = ne00;
            context->q8_1_cached_s01 = s01;
            context->q8_1_cached_s02 = s02;
            context->q8_1_cached_s03 = s03;
            context->q8_1_cached_ne0 = ne0;
            context->q8_1_cached_ne1 = ne1;
            context->q8_1_cached_ne2 = ne2;
            context->q8_1_cached_blocks = blocks;
            context->q8_1_cached_z_count = z_count;
            context->q8_1_cached_ref = *out_q8_1_ref;
        }
        return ok;
    }

    return false;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT Q4_K tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q4_k_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q4_K shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };

    for (const auto * route : context->device_context->mul_mat_q4_k_routes) {
        const bool use_q8_1_rhs = ggml_backend_hrx2_route_uses_q8_1_rhs(route);
        if (use_q8_1_rhs && !ggml_backend_hrx2_q4_k_q8_1_prompt_enabled(shape)) {
            continue;
        }
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) &&
            !ggml_backend_hrx2_q4_k_q8_1_x4_mmq_enabled()) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q4_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        hrx_buffer_ref_t route_bindings[3] = { bindings[0], bindings[1], bindings[2] };
        if (use_q8_1_rhs) {
            hrx_buffer_ref_t q8_1_ref = {};
            if (!ggml_backend_hrx2_dispatch_quantize_q8_1(
                    context,
                    src1,
                    ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route),
                    &q8_1_ref)) {
                ggml_backend_hrx2_trace_event(
                    "dispatch_failed",
                    ggml_backend_hrx2_json_kv("op", "QUANTIZE") + "," +
                    ggml_backend_hrx2_json_kv("family", "quantize_q8_1_f32") + "," +
                    ggml_backend_hrx2_json_kv("reason", "q8_1_quantize") + "," +
                    ggml_backend_hrx2_json_kv("route_id", provider->route.id));
                continue;
            }
            route_bindings[1] = q8_1_ref;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT Q4_K route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                ggml_backend_hrx2_provider_workgroup_size_x(provider),
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                route_bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q4_K provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4_k_swiglu(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * first,
        const ggml_tensor * second,
        const ggml_tensor * swiglu) {
    ggml_backend_hrx2_mul_mat_q4_k_swiglu_fusion fusion;
    if (!ggml_backend_hrx2_extract_mul_mat_q4_k_swiglu_fusion(first, second, swiglu, &fusion)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "fusion_shape"));
        return GGML_STATUS_FAILED;
    }

    const ggml_tensor * rhs = fusion.x->src[1];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(fusion.x->src[0], &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(fusion.gate->src[0], &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(rhs, &bindings[2]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(swiglu, &bindings[3])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(swiglu)) + "," +
            ggml_backend_hrx2_json_kv("x", ggml_backend_hrx2_tensor_summary(fusion.x)) + "," +
            ggml_backend_hrx2_json_kv("gate", ggml_backend_hrx2_tensor_summary(fusion.gate)));
        GGML_LOG_ERROR("HRX2: fused MUL_MAT Q4_K + SWIGLU tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_q4_k_swiglu_routes) {
        if (route->binding_count != 4) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q4_k_plan(context->device_context, route, fusion.shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GLU") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", fusion.shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", fusion.shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", fusion.shape.cols));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: fused MUL_MAT Q4_K + SWIGLU route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (fusion.shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (fusion.shape.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", fusion.shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", fusion.shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", fusion.shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                4,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "GLU") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: fused MUL_MAT Q4_K + SWIGLU provider is not available for k=%u rows=%u cols=%u\n",
        fusion.shape.k,
        fusion.shape.rows,
        fusion.shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q4_k_packed_swiglu(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * mul_mat,
        const ggml_tensor * swiglu) {
    ggml_backend_hrx2_mul_mat_q4_k_packed_swiglu_fusion fusion;
    if (!ggml_backend_hrx2_extract_mul_mat_q4_k_packed_swiglu_fusion(mul_mat, swiglu, &fusion)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "packed_fusion_shape"));
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(mul_mat->src[0], &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(mul_mat->src[1], &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(swiglu, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(swiglu)) + "," +
            ggml_backend_hrx2_json_kv("mul_mat", ggml_backend_hrx2_tensor_summary(mul_mat)));
        GGML_LOG_ERROR("HRX2: fused packed MUL_MAT Q4_K + SWIGLU tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_q4_k_swiglu_routes) {
        if (route->binding_count != 3) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q4_k_plan(context->device_context, route, fusion.shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "GLU") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", fusion.shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", fusion.shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", fusion.shape.cols));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: fused packed MUL_MAT Q4_K + SWIGLU route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (fusion.shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (fusion.shape.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "GLU") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", fusion.shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", fusion.shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", fusion.shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "GLU") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q4_k_swiglu_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: fused packed MUL_MAT Q4_K + SWIGLU provider is not available for k=%u rows=%u cols=%u\n",
        fusion.shape.k,
        fusion.shape.rows,
        fusion.shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst,
        ggml_type src0_type,
        const char * family,
        const char * type_label,
        const std::vector<const ggml_backend_hrx2_kernel_route *> & routes) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[3])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)) + "," +
            ggml_backend_hrx2_json_kv("src2", ggml_backend_hrx2_tensor_summary(src2)));
        GGML_LOG_ERROR("HRX2: MUL_MAT_ID %s tensor is not backed by HRX2 buffers\n", type_label);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_id_shape shape;
    if (!ggml_backend_hrx2_mul_mat_id_k_shape(dst, src0_type, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)) + "," +
            ggml_backend_hrx2_json_kv("src2", ggml_backend_hrx2_tensor_summary(src2)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT_ID %s shape during dispatch\n", type_label);
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_id_q4_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
                ggml_backend_hrx2_json_kv("family", family) + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
                ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
                ggml_backend_hrx2_json_kv("ntokens", shape.ntokens));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT_ID %s route %s has unsupported constant byte length %u\n",
                type_label,
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (shape.nselected + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                shape.ntokens,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
            ggml_backend_hrx2_json_kv("family", family) + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("nexperts", shape.nexperts) + "," +
            ggml_backend_hrx2_json_kv("nselected", shape.nselected) + "," +
            ggml_backend_hrx2_json_kv("ntokens", shape.ntokens) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_z", config.workgroup_count[2]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                4,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT_ID") + "," +
                ggml_backend_hrx2_json_kv("family", family) + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: MUL_MAT_ID %s provider is not available for k=%u rows=%u nexperts=%u nselected=%u ntokens=%u\n",
        type_label,
        shape.k,
        shape.rows,
        shape.nexperts,
        shape.nselected,
        shape.ntokens);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q4_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    return ggml_backend_hrx2_dispatch_mul_mat_id_k(
        context, dst, GGML_TYPE_Q4_K, "mul_mat_id_q4_k_f32", "Q4_K", context->device_context->mul_mat_id_q4_k_routes);
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q5_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    return ggml_backend_hrx2_dispatch_mul_mat_id_k(
        context, dst, GGML_TYPE_Q5_K, "mul_mat_id_q5_k_f32", "Q5_K", context->device_context->mul_mat_id_q5_k_routes);
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_id_q6_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    return ggml_backend_hrx2_dispatch_mul_mat_id_k(
        context, dst, GGML_TYPE_Q6_K, "mul_mat_id_q6_k_f32", "Q6_K", context->device_context->mul_mat_id_q6_k_routes);
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q6_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT Q6_K tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q6_k_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q6_K shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };
    for (const auto * route : context->device_context->mul_mat_q6_k_routes) {
        const bool use_q8_1_rhs = ggml_backend_hrx2_route_uses_q8_1_rhs(route);
        if (use_q8_1_rhs && !ggml_backend_hrx2_q6_k_q8_1_prompt_enabled(shape)) {
            continue;
        }
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) &&
            !ggml_backend_hrx2_q6_k_q8_1_x4_prompt_enabled()) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q6_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        hrx_buffer_ref_t route_bindings[3] = { bindings[0], bindings[1], bindings[2] };
        if (use_q8_1_rhs) {
            hrx_buffer_ref_t q8_1_ref = {};
            if (!ggml_backend_hrx2_dispatch_quantize_q8_1(
                    context,
                    src1,
                    ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route),
                    &q8_1_ref)) {
                ggml_backend_hrx2_trace_event(
                    "dispatch_failed",
                    ggml_backend_hrx2_json_kv("op", "QUANTIZE") + "," +
                    ggml_backend_hrx2_json_kv("family", "quantize_q8_1_f32") + "," +
                    ggml_backend_hrx2_json_kv("reason", "q8_1_quantize") + "," +
                    ggml_backend_hrx2_json_kv("route_id", provider->route.id));
                continue;
            }
            route_bindings[1] = q8_1_ref;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT Q6_K route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint32_t workgroup_size_x = ggml_backend_hrx2_provider_workgroup_size_x(provider);
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                workgroup_size_x,
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                route_bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q6_k_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q6_K provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_q5_k(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT Q5_K tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_shape shape;
    if (!ggml_backend_hrx2_mul_mat_q5_k_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT Q5_K shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_constants constants = {
        /* .k    = */ static_cast<uint32_t>(src0->ne[0]),
        /* .rows = */ static_cast<uint32_t>(src0->ne[1]),
        /* .cols = */ static_cast<uint32_t>(src1->ne[1]),
    };

    for (const auto * route : context->device_context->mul_mat_q5_k_routes) {
        const bool use_q8_1_rhs = ggml_backend_hrx2_route_uses_q8_1_rhs(route);
        if (use_q8_1_rhs && !ggml_backend_hrx2_q5_k_q8_1_prompt_enabled(shape)) {
            continue;
        }
        if (ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route) &&
            !ggml_backend_hrx2_q5_k_q8_1_x4_prompt_enabled()) {
            continue;
        }
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_q5_k_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols));
            continue;
        }

        hrx_buffer_ref_t route_bindings[3] = { bindings[0], bindings[1], bindings[2] };
        if (use_q8_1_rhs) {
            hrx_buffer_ref_t q8_1_ref = {};
            if (!ggml_backend_hrx2_dispatch_quantize_q8_1(
                    context,
                    src1,
                    ggml_backend_hrx2_route_uses_q8_1_x4_rhs(route),
                    &q8_1_ref)) {
                ggml_backend_hrx2_trace_event(
                    "dispatch_failed",
                    ggml_backend_hrx2_json_kv("op", "QUANTIZE") + "," +
                    ggml_backend_hrx2_json_kv("family", "quantize_q8_1_f32") + "," +
                    ggml_backend_hrx2_json_kv("reason", "q8_1_quantize") + "," +
                    ggml_backend_hrx2_json_kv("route_id", provider->route.id));
                continue;
            }
            route_bindings[1] = q8_1_ref;
        }

        const void * constant_data = nullptr;
        size_t constant_size = 0;
        if (provider->route.constant_byte_length == sizeof(constants)) {
            constant_data = &constants;
            constant_size = sizeof(constants);
        } else if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT Q5_K route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint32_t workgroup_size_x = ggml_backend_hrx2_provider_workgroup_size_x(provider);
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (constants.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (constants.cols + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                workgroup_size_x,
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                constant_data,
                constant_size,
                route_bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("family", "mul_mat_q5_k_f32") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: MUL_MAT Q5_K provider is not available for k=%u rows=%u cols=%u\n", shape.k, shape.rows, shape.cols);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_f16_f32(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: MUL_MAT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_f16_shape shape;
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_shape(dst, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("dst", ggml_backend_hrx2_tensor_summary(dst)) + "," +
            ggml_backend_hrx2_json_kv("src0", ggml_backend_hrx2_tensor_summary(src0)) + "," +
            ggml_backend_hrx2_json_kv("src1", ggml_backend_hrx2_tensor_summary(src1)));
        GGML_LOG_ERROR("HRX2: invalid MUL_MAT F16/F32 shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_f16_f32_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_f16_f32_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
                ggml_backend_hrx2_json_kv("dst_ne2", shape.dst_ne2) + "," +
                ggml_backend_hrx2_json_kv("dst_ne3", shape.dst_ne3));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: MUL_MAT route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (shape.cols * shape.dst_ne2 * shape.dst_ne3 + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("dst_ne2", shape.dst_ne2) + "," +
            ggml_backend_hrx2_json_kv("dst_ne3", shape.dst_ne3) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: MUL_MAT F16/F32 provider is not available for k=%u rows=%u cols=%u dst_ne2=%u dst_ne3=%u\n",
        shape.k,
        shape.rows,
        shape.cols,
        shape.dst_ne2,
        shape.dst_ne3);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_mul_mat_f16_f32_cont(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * mul_mat,
        const ggml_tensor * permute,
        const ggml_tensor * cont) {
    const ggml_tensor * src0 = mul_mat->src[0];
    const ggml_tensor * src1 = mul_mat->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(cont, &bindings[2])) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_CONT") + "," +
            ggml_backend_hrx2_json_kv("reason", "buffer_ref") + "," +
            ggml_backend_hrx2_json_kv("mul_mat", ggml_backend_hrx2_tensor_summary(mul_mat)) + "," +
            ggml_backend_hrx2_json_kv("permute", ggml_backend_hrx2_tensor_summary(permute)) + "," +
            ggml_backend_hrx2_json_kv("cont", ggml_backend_hrx2_tensor_summary(cont)));
        GGML_LOG_ERROR("HRX2: fused MUL_MAT F16/F32 + CONT tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_mul_mat_f16_shape shape;
    if (!ggml_backend_hrx2_extract_mul_mat_f16_f32_cont_fusion(mul_mat, permute, cont, &shape)) {
        ggml_backend_hrx2_trace_event(
            "dispatch_failed",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_CONT") + "," +
            ggml_backend_hrx2_json_kv("reason", "shape") + "," +
            ggml_backend_hrx2_json_kv("mul_mat", ggml_backend_hrx2_tensor_summary(mul_mat)) + "," +
            ggml_backend_hrx2_json_kv("permute", ggml_backend_hrx2_tensor_summary(permute)) + "," +
            ggml_backend_hrx2_json_kv("cont", ggml_backend_hrx2_tensor_summary(cont)));
        GGML_LOG_ERROR("HRX2: invalid fused MUL_MAT F16/F32 + CONT shape during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    for (const auto * route : context->device_context->mul_mat_f16_f32_cont_routes) {
        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_mul_mat_f16_f32_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT_CONT") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("k", shape.k) + "," +
                ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
                ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
                ggml_backend_hrx2_json_kv("dst_ne2", shape.dst_ne2) + "," +
                ggml_backend_hrx2_json_kv("dst_ne3", shape.dst_ne3));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: fused MUL_MAT F16/F32 + CONT route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                (shape.rows + provider->route.rows_per_workgroup - 1) / provider->route.rows_per_workgroup,
                (shape.cols * shape.dst_ne2 * shape.dst_ne3 + provider->route.cols_per_workgroup - 1) / provider->route.cols_per_workgroup,
                1,
            },
            /* .workgroup_size  = */ {
                provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0],
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "MUL_MAT_CONT") + "," +
            ggml_backend_hrx2_json_kv("fusion", "F16_KQV_CONT") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("k", shape.k) + "," +
            ggml_backend_hrx2_json_kv("rows", shape.rows) + "," +
            ggml_backend_hrx2_json_kv("cols", shape.cols) + "," +
            ggml_backend_hrx2_json_kv("dst_ne2", shape.dst_ne2) + "," +
            ggml_backend_hrx2_json_kv("dst_ne3", shape.dst_ne3) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroups_y", config.workgroup_count[1]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            ggml_backend_hrx2_trace_event(
                "dispatch_failed",
                ggml_backend_hrx2_json_kv("op", "MUL_MAT_CONT") + "," +
                ggml_backend_hrx2_json_kv("reason", "hrx_stream_dispatch") + "," +
                ggml_backend_hrx2_json_kv("route_id", provider->route.id));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR(
        "HRX2: fused MUL_MAT F16/F32 + CONT provider is not available for k=%u rows=%u cols=%u dst_ne2=%u dst_ne3=%u\n",
        shape.k,
        shape.rows,
        shape.cols,
        shape.dst_ne2,
        shape.dst_ne3);
    return GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx2_dispatch_set_rows_host_fallback(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst,
        const ggml_backend_hrx2_set_rows_shape & shape) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t src0_ref = {};
    hrx_buffer_ref_t src1_ref = {};
    hrx_buffer_ref_t dst_ref = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &src0_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &src1_ref) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("HRX2: SET_ROWS host fallback tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    if (!GGML_HRX2_CHECK(hrx_stream_synchronize(context->stream))) {
        return GGML_STATUS_FAILED;
    }

    std::vector<uint8_t> src0_host(src0_ref.length);
    std::vector<uint8_t> src1_host(src1_ref.length);
    std::vector<uint8_t> dst_host(dst_ref.length);

    if (!GGML_HRX2_CHECK(hrx_synchronous_d2h(
            context->device_context->device,
            src0_ref.buffer,
            src0_ref.offset,
            src0_host.data(),
            src0_host.size())) ||
        !GGML_HRX2_CHECK(hrx_synchronous_d2h(
            context->device_context->device,
            src1_ref.buffer,
            src1_ref.offset,
            src1_host.data(),
            src1_host.size())) ||
        !GGML_HRX2_CHECK(hrx_synchronous_d2h(
            context->device_context->device,
            dst_ref.buffer,
            dst_ref.offset,
            dst_host.data(),
            dst_host.size()))) {
        return GGML_STATUS_FAILED;
    }

    const uint8_t * src0_base = src0_host.data();
    const uint8_t * src1_base = src1_host.data();
    uint8_t * dst_base = dst_host.data();

    for (uint32_t i3 = 0; i3 < shape.ne03; ++i3) {
        const uint32_t i12 = shape.ne12 == 0 ? 0 : i3 % shape.ne12;
        for (uint32_t i2 = 0; i2 < shape.ne02; ++i2) {
            const uint32_t i11 = shape.ne11 == 0 ? 0 : i2 % shape.ne11;
            for (uint32_t i = 0; i < shape.nr; ++i) {
                const size_t idx_offset =
                    static_cast<size_t>(i) * src1->nb[0] +
                    static_cast<size_t>(i11) * src1->nb[1] +
                    static_cast<size_t>(i12) * src1->nb[2];
                if (idx_offset + sizeof(int64_t) > src1_host.size()) {
                    GGML_LOG_ERROR("HRX2: SET_ROWS host fallback index offset is out of bounds\n");
                    return GGML_STATUS_FAILED;
                }
                const int64_t row = *reinterpret_cast<const int64_t *>(src1_base + idx_offset);
                if (row < 0 || row >= static_cast<int64_t>(shape.ne1)) {
                    continue;
                }

                for (uint32_t i0 = 0; i0 < shape.nc; ++i0) {
                    const size_t src0_offset =
                        static_cast<size_t>(i0) * sizeof(float) +
                        static_cast<size_t>(i) * src0->nb[1] +
                        static_cast<size_t>(i2) * src0->nb[2] +
                        static_cast<size_t>(i3) * src0->nb[3];
                    if (src0_offset + sizeof(float) > src0_host.size()) {
                        GGML_LOG_ERROR("HRX2: SET_ROWS host fallback source offset is out of bounds\n");
                        return GGML_STATUS_FAILED;
                    }
                    const float value = *reinterpret_cast<const float *>(src0_base + src0_offset);
                    const size_t dst_offset =
                        static_cast<size_t>(i0) * ggml_type_size(dst->type) +
                        static_cast<size_t>(row) * dst->nb[1] +
                        static_cast<size_t>(i2) * dst->nb[2] +
                        static_cast<size_t>(i3) * dst->nb[3];
                    if (dst->type == GGML_TYPE_F16) {
                        if (dst_offset + sizeof(ggml_fp16_t) > dst_host.size()) {
                            GGML_LOG_ERROR("HRX2: SET_ROWS host fallback destination offset is out of bounds\n");
                            return GGML_STATUS_FAILED;
                        }
                        *reinterpret_cast<ggml_fp16_t *>(dst_base + dst_offset) = GGML_FP32_TO_FP16(value);
                    } else if (dst->type == GGML_TYPE_F32) {
                        if (dst_offset + sizeof(float) > dst_host.size()) {
                            GGML_LOG_ERROR("HRX2: SET_ROWS host fallback destination offset is out of bounds\n");
                            return GGML_STATUS_FAILED;
                        }
                        *reinterpret_cast<float *>(dst_base + dst_offset) = value;
                    } else {
                        GGML_LOG_ERROR("HRX2: SET_ROWS host fallback destination type is unsupported\n");
                        return GGML_STATUS_FAILED;
                    }
                }
            }
        }
    }

    ggml_backend_hrx2_trace_event(
        "dispatch",
        ggml_backend_hrx2_json_kv("op", "SET_ROWS") + "," +
        ggml_backend_hrx2_json_kv("route_id", std::string("host_fallback_set_rows_f32_") + ggml_type_name(dst->type)) + "," +
        ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
        ggml_backend_hrx2_json_kv("nc", shape.nc) + "," +
        ggml_backend_hrx2_json_kv("nr", shape.nr) + "," +
        ggml_backend_hrx2_json_kv("ne02", shape.ne02) + "," +
        ggml_backend_hrx2_json_kv("ne03", shape.ne03));

    if (!GGML_HRX2_CHECK(hrx_synchronous_h2d(
            context->device_context->device,
            dst_host.data(),
            dst_ref.buffer,
            dst_ref.offset,
            dst_host.size()))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx2_dispatch_set_rows(
        ggml_backend_hrx2_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx2_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx2_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("HRX2: SET_ROWS tensor is not backed by HRX2 buffers\n");
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx2_set_rows_shape shape;
    if (!ggml_backend_hrx2_extract_set_rows_shape(dst, &shape)) {
        GGML_LOG_ERROR("HRX2: invalid SET_ROWS shape/type/layout during dispatch\n");
        return GGML_STATUS_FAILED;
    }

    if (ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_SET_ROWS_LOOM")) {
        return ggml_backend_hrx2_dispatch_set_rows_host_fallback(context, dst, shape);
    }

    for (const auto * route : context->device_context->set_rows_routes) {
        if ((dst->type == GGML_TYPE_F16 && route->export_name != "hrx2_set_rows_f32_f16") ||
            (dst->type == GGML_TYPE_F32 && route->export_name != "hrx2_set_rows_f32_f32")) {
            continue;
        }

        ggml_backend_hrx2_provider_plan plan;
        if (!ggml_backend_hrx2_make_set_rows_plan(context->device_context, route, shape, &plan)) {
            continue;
        }

        const auto * provider = ggml_backend_hrx2_get_provider(
            context->device_context,
            plan.route,
            plan.config_bindings,
            plan.cache_key);
        if (!provider) {
            ggml_backend_hrx2_trace_event(
                "provider_unavailable",
                ggml_backend_hrx2_json_kv("op", "SET_ROWS") + "," +
                ggml_backend_hrx2_json_kv("route_id", plan.route->id) + "," +
                ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
                ggml_backend_hrx2_json_kv("cache_key", plan.cache_key) + "," +
                ggml_backend_hrx2_json_kv("nc", shape.nc) + "," +
                ggml_backend_hrx2_json_kv("nr", shape.nr));
            continue;
        }

        if (provider->route.constant_byte_length != 0) {
            GGML_LOG_ERROR(
                "HRX2: SET_ROWS route %s has unsupported constant byte length %u\n",
                provider->route.id.c_str(),
                provider->route.constant_byte_length);
            continue;
        }

        const uint64_t total =
            static_cast<uint64_t>(shape.nc) *
            static_cast<uint64_t>(shape.nr) *
            static_cast<uint64_t>(shape.ne02) *
            static_cast<uint64_t>(shape.ne03);
        const uint32_t workgroup_size =
            provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : provider->route.workgroup_size[0];
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size  = */ {
                workgroup_size,
                1,
                1,
            },
            /* .subgroup_size   = */ 0,
        };

        ggml_backend_hrx2_trace_event(
            "dispatch",
            ggml_backend_hrx2_json_kv("op", "SET_ROWS") + "," +
            ggml_backend_hrx2_json_kv("route_id", provider->route.id) + "," +
            ggml_backend_hrx2_json_kv("target_key", context->device_context->architecture) + "," +
            ggml_backend_hrx2_json_kv("cache_key", provider->cache_key) + "," +
            ggml_backend_hrx2_json_kv("nc", shape.nc) + "," +
            ggml_backend_hrx2_json_kv("nr", shape.nr) + "," +
            ggml_backend_hrx2_json_kv("ne02", shape.ne02) + "," +
            ggml_backend_hrx2_json_kv("ne03", shape.ne03) + "," +
            ggml_backend_hrx2_json_kv("workgroups_x", config.workgroup_count[0]) + "," +
            ggml_backend_hrx2_json_kv("workgroup_size_x", config.workgroup_size[0]));

        if (!GGML_HRX2_CHECK(hrx_stream_dispatch(
                context->stream,
                provider->executable,
                provider->export_ordinal,
                &config,
                nullptr,
                0,
                bindings,
                3,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    GGML_LOG_ERROR("HRX2: SET_ROWS provider is not available for nc=%u nr=%u dst=%s\n",
            shape.nc, shape.nr, ggml_type_name(dst->type));
    return ggml_backend_hrx2_dispatch_set_rows_host_fallback(context, dst, shape);
}

static const char * ggml_backend_hrx2_get_name(ggml_backend_t backend) {
    return ggml_backend_hrx2_get_context(backend)->name.c_str();
}

static void ggml_backend_hrx2_free(ggml_backend_t backend) {
    auto * context = ggml_backend_hrx2_get_context(backend);
    if (context->stream) {
        (void) GGML_HRX2_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx2_unregister_stream(context->device_context, context->stream);
        hrx_stream_release(context->stream);
    }
    ggml_backend_hrx2_release_scratch_buffers(context);
    ggml_backend_hrx2_release_device_scratch(context->q8_1_scratch);
    ggml_backend_hrx2_release_device_scratch(context->route_scratch);
    delete context;
    delete backend;
}

static void ggml_backend_hrx2_synchronize(ggml_backend_t backend) {
    auto * context = ggml_backend_hrx2_get_context(backend);
    if (context->stream) {
        (void) GGML_HRX2_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx2_recycle_scratch_buffers(context);
        ggml_backend_hrx2_release_retired_device_scratch(context->q8_1_scratch);
        ggml_backend_hrx2_release_retired_device_scratch(context->route_scratch);
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        if (auto * arena = ggml_backend_hrx2_find_staging_arena_locked(context->device_context, context->stream)) {
            ggml_backend_hrx2_reset_staging_arena_locked(*arena);
        }
    }
}

struct ggml_backend_hrx2_active_graph_guard {
    ggml_backend_hrx2_context * context = nullptr;
    ggml_backend_hrx2_context * previous_context = nullptr;
    const ggml_tensor * previous_node = nullptr;

    explicit ggml_backend_hrx2_active_graph_guard(ggml_backend_hrx2_context * context)
        : context(context),
          previous_context(g_hrx2_active_graph_context),
          previous_node(g_hrx2_active_graph_node) {
        ggml_backend_hrx2_begin_submit_batch(context);
        g_hrx2_active_graph_context = context;
        g_hrx2_active_graph_node = nullptr;
    }

    ~ggml_backend_hrx2_active_graph_guard() {
        if (context) {
            context->last_total_mul_mat_bytes = context->total_mul_mat_bytes;
            ggml_backend_hrx2_trace_event(
                "submit_batch_graph_end",
                ggml_backend_hrx2_json_kv("total_mul_mat_bytes", context->last_total_mul_mat_bytes) + "," +
                ggml_backend_hrx2_json_kv("submit_flushes", context->submit_flush_count));
        }
        g_hrx2_active_graph_context = previous_context;
        g_hrx2_active_graph_node = previous_node;
    }
};

static enum ggml_status ggml_backend_hrx2_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = ggml_backend_hrx2_get_context(backend);
    if (!ggml_backend_hrx2_sync_graph_entry_streams(context->device_context, context->stream)) {
        return GGML_STATUS_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        context->device_context->active_stream = context->stream;
    }
    context->q8_1_cached_src = nullptr;
    context->q8_1_cached_ref = {};
    ggml_backend_hrx2_active_graph_guard active_graph_guard(context);
    // per-op timing (env HRX2_OPTIME): accumulate wall time per op type
    // across all graphs in the process; print for the large graph only.
    // With HRX2_OPTIME_SYNC=1 additionally drains the stream after every
    // node so the per-op deltas include GPU execution (the normal async
    // enqueue-only deltas are ~0 once the stream is warm).
    static std::map<int, double> optime_accum;
    static std::map<int, int64_t> optime_count;
    const bool optime_enabled = getenv("HRX2_OPTIME") != nullptr;
    const bool optime_sync = getenv("HRX2_OPTIME_SYNC") != nullptr;
    const auto optime_t0 = std::chrono::steady_clock::now();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        g_hrx2_active_graph_node = node;
        if (ggml_backend_hrx2_trace_graph_enabled()) {
            std::fprintf(
                stderr,
                "HRX2 graph node=%d op=%s type=%s name=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                "] src0=%s src1=%s src2=%s src3=%s src4=%s src5=%s flags=0x%x\n",
                i,
                ggml_op_desc(node),
                ggml_type_name(node->type),
                node->name,
                node->ne[0],
                node->ne[1],
                node->ne[2],
                node->ne[3],
                node->src[0] ? node->src[0]->name : "",
                node->src[1] ? node->src[1]->name : "",
                node->src[2] ? node->src[2]->name : "",
                node->src[3] ? node->src[3]->name : "",
                node->src[4] ? node->src[4]->name : "",
                node->src[5] ? node->src[5]->name : "",
                static_cast<unsigned>(node->flags));
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_ADD_RMS_NORM_MUL_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            node->op == GGML_OP_ADD &&
            cgraph->nodes[i + 1]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            ggml_backend_hrx2_supports_add_rms_norm_mul_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1],
                cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 })) {
            if (ggml_backend_hrx2_dispatch_add_rms_norm_mul(
                    context,
                    node,
                    cgraph->nodes[i + 1],
                    cgraph->nodes[i + 2]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_RMS_NORM_MUL_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            node->op == GGML_OP_RMS_NORM &&
            ggml_backend_hrx2_supports_rms_norm_mul_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, { i + 1 })) {
            if (ggml_backend_hrx2_dispatch_rms_norm_mul(
                    context,
                    node,
                    cgraph->nodes[i + 1]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 1;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q4K_SWIGLU_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            node->op == GGML_OP_MUL_MAT &&
            ggml_backend_hrx2_supports_mul_mat_q4_k_packed_swiglu_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_GLU }, { i + 1 })) {
            if (ggml_backend_hrx2_dispatch_mul_mat_q4_k_packed_swiglu(
                    context,
                    node,
                    cgraph->nodes[i + 1]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 1;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_Q4K_SWIGLU_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            node->op == GGML_OP_MUL_MAT &&
            ggml_backend_hrx2_supports_mul_mat_q4_k_swiglu_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1],
                cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { i + 2 })) {
            if (ggml_backend_hrx2_dispatch_mul_mat_q4_k_swiglu(
                    context,
                    node,
                    cgraph->nodes[i + 1],
                    cgraph->nodes[i + 2]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_ROPE_SET_ROWS_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            node->op == GGML_OP_ROPE &&
            cgraph->nodes[i + 1]->op == GGML_OP_VIEW &&
            cgraph->nodes[i + 2]->op == GGML_OP_SET_ROWS &&
            ggml_backend_hrx2_supports_rope_set_rows_route(
                context->device_context,
                node,
                cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 2 })) {
            if (ggml_backend_hrx2_dispatch_rope_set_rows(
                    context,
                    node,
                    cgraph->nodes[i + 2]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_ROPE_SET_ROWS_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            node->op == GGML_OP_ROPE &&
            ggml_backend_hrx2_supports_rope_set_rows_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1]) &&
            ggml_backend_hrx2_can_fuse_rope_set_rows(cgraph, i)) {
            if (ggml_backend_hrx2_dispatch_rope_set_rows(
                    context,
                    node,
                    cgraph->nodes[i + 1]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 1;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_CONT_SET_ROWS_FUSION") &&
            i + 3 < cgraph->n_nodes &&
            node->op == GGML_OP_CONT &&
            cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE &&
            cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE &&
            cgraph->nodes[i + 3]->op == GGML_OP_SET_ROWS &&
            ggml_backend_hrx2_supports_cont_set_rows_route(
                context->device_context,
                node,
                cgraph->nodes[i + 3]) &&
            ggml_backend_hrx2_can_fuse_cont_set_rows(cgraph, i, i + 3)) {
            if (ggml_backend_hrx2_dispatch_cont_set_rows(
                    context,
                    node,
                    cgraph->nodes[i + 3]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 3;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_CONT_SET_ROWS_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            node->op == GGML_OP_CONT &&
            cgraph->nodes[i + 1]->op == GGML_OP_SET_ROWS &&
            ggml_backend_hrx2_supports_cont_set_rows_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1]) &&
            ggml_backend_hrx2_can_fuse_cont_set_rows(cgraph, i, i + 1)) {
            if (ggml_backend_hrx2_dispatch_cont_set_rows(
                    context,
                    node,
                    cgraph->nodes[i + 1]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 1;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_F16_FA0_ATTENTION_FUSION") &&
            i + 4 < cgraph->n_nodes &&
            node->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 1]->op == GGML_OP_SOFT_MAX &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 3]->op == GGML_OP_PERMUTE &&
            cgraph->nodes[i + 4]->op == GGML_OP_CONT &&
            ggml_backend_hrx2_supports_flash_attn_fa0_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1],
                cgraph->nodes[i + 2],
                cgraph->nodes[i + 3],
                cgraph->nodes[i + 4]) &&
            ggml_can_fuse_subgraph(
                cgraph,
                i,
                { GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT, GGML_OP_PERMUTE, GGML_OP_CONT },
                { i + 4 })) {
            if (ggml_backend_hrx2_dispatch_flash_attn_fa0(
                    context,
                    node,
                    cgraph->nodes[i + 1],
                    cgraph->nodes[i + 2],
                    cgraph->nodes[i + 3],
                    cgraph->nodes[i + 4]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 4;
            continue;
        }
        if (ggml_backend_hrx2_fusion_enabled() &&
            !ggml_backend_hrx2_env_enabled("GGML_HRX2_DISABLE_F16_KQV_CONT_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            node->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 1]->op == GGML_OP_PERMUTE &&
            cgraph->nodes[i + 2]->op == GGML_OP_CONT &&
            ggml_backend_hrx2_supports_mul_mat_f16_f32_cont_route(
                context->device_context,
                node,
                cgraph->nodes[i + 1],
                cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_PERMUTE, GGML_OP_CONT }, { i + 2 })) {
            if (ggml_backend_hrx2_dispatch_mul_mat_f16_f32_cont(
                    context,
                    node,
                    cgraph->nodes[i + 1],
                    cgraph->nodes[i + 2]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_SCALE:
            case GGML_OP_CLAMP:
                if (!ggml_backend_hrx2_supports_pointwise_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported pointwise shape/type/layout: dst=%s src0=%s src1=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_pointwise(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SUM_ROWS:
                if (!ggml_backend_hrx2_supports_sum_rows_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported SUM_ROWS shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_sum_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GET_ROWS:
                if (ggml_backend_hrx2_supports_get_rows_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_get_rows(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_get_rows_quantized_route(
                        context->device_context, node, GGML_TYPE_Q8_0, context->device_context->get_rows_q8_0_routes)) {
                    if (ggml_backend_hrx2_dispatch_get_rows_quantized(
                            context,
                            node,
                            GGML_TYPE_Q8_0,
                            "get_rows_q8_0_f32",
                            "Q8_0",
                            context->device_context->get_rows_q8_0_routes) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_get_rows_quantized_route(
                        context->device_context, node, GGML_TYPE_Q4_K, context->device_context->get_rows_q4_k_routes)) {
                    if (ggml_backend_hrx2_dispatch_get_rows_quantized(
                            context,
                            node,
                            GGML_TYPE_Q4_K,
                            "get_rows_q4_k_f32",
                            "Q4_K",
                            context->device_context->get_rows_q4_k_routes) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_get_rows_quantized_route(
                        context->device_context, node, GGML_TYPE_Q5_K, context->device_context->get_rows_q5_k_routes)) {
                    if (ggml_backend_hrx2_dispatch_get_rows_quantized(
                            context,
                            node,
                            GGML_TYPE_Q5_K,
                            "get_rows_q5_k_f32",
                            "Q5_K",
                            context->device_context->get_rows_q5_k_routes) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_get_rows_quantized_route(
                        context->device_context, node, GGML_TYPE_Q6_K, context->device_context->get_rows_q6_k_routes)) {
                    if (ggml_backend_hrx2_dispatch_get_rows_quantized(
                            context,
                            node,
                            GGML_TYPE_Q6_K,
                            "get_rows_q6_k_f32",
                            "Q6_K",
                            context->device_context->get_rows_q6_k_routes) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (!ggml_backend_hrx2_supports_get_rows_moe_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported GET_ROWS shape/type/layout: dst=%s src0=%s src1=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_get_rows_moe(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ARGSORT:
                if (!ggml_backend_hrx2_supports_argsort_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported ARGSORT shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_argsort(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ROPE:
                if (!ggml_backend_hrx2_supports_rope_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported ROPE shape/type/layout: dst=%s src0=%s src1=%s src2=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[2]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_rope(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SOFT_MAX:
                if (!ggml_backend_hrx2_supports_soft_max_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported SOFT_MAX shape/type/layout: dst=%s src0=%s src1=%s src2=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[2]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_soft_max(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CONT:
                if (!ggml_backend_hrx2_supports_cont_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported CONT shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_cont(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CPY:
                if (!ggml_backend_hrx2_supports_cpy(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported CPY shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_cpy(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GLU:
                if (!ggml_backend_hrx2_supports_swiglu_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported GLU shape/type/layout: dst=%s src0=%s src1=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_swiglu(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_RMS_NORM:
                if (!ggml_backend_hrx2_supports_rms_norm_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported RMS_NORM shape/type/layout: dst=%s src0=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_rms_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT_Q4NX:
                if (ggml_backend_hrx2_supports_mul_mat_q4nx_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q4nx(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT_Q4NX shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT_ID_Q4NX:
                if (ggml_backend_hrx2_supports_mul_mat_id_q4nx_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q4nx(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT_ID_Q4NX shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (ggml_backend_hrx2_supports_mul_mat_q4nx_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q4nx(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q8_0_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q8_0(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q4_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q4_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q5_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q5_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_q6_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_q6_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_f32_f32_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_f32_f32(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_f16_f32_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_f16_f32(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                if (ggml_backend_hrx2_supports_mul_mat_id_q4_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q4_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_id_q5_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q5_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (ggml_backend_hrx2_supports_mul_mat_id_q6_k_route(context->device_context, node)) {
                    if (ggml_backend_hrx2_dispatch_mul_mat_id_q6_k(context, node) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                {
                    GGML_LOG_ERROR("HRX2: unsupported MUL_MAT_ID shape/type/layout\n");
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SET_ROWS:
                if (!ggml_backend_hrx2_supports_set_rows_route(context->device_context, node)) {
                    GGML_LOG_ERROR("HRX2: unsupported SET_ROWS shape/type/layout: dst=%s src0=%s src1=%s src2=%s\n",
                            ggml_backend_hrx2_tensor_summary(node).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[0]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[1]).c_str(),
                            ggml_backend_hrx2_tensor_summary(node->src[2]).c_str());
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx2_dispatch_set_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("HRX2: unsupported op %s\n", ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
        if (optime_enabled) {
            auto t1 = std::chrono::steady_clock::now();
            optime_accum[(int) node->op] += std::chrono::duration<double, std::milli>(t1 - t0).count();
            optime_count[(int) node->op] += 1;
            t0 = t1;
        }
        if (optime_sync) {
            // drain the stream so the GPU execution of THIS node is charged
            // to it: accumulate the flush+wait duration into the same op.
            if (!GGML_HRX2_CHECK(hrx_stream_flush(context->stream)) ||
                !GGML_HRX2_CHECK(hrx_stream_wait(context->stream))) {
                return GGML_STATUS_FAILED;
            }
            auto t2 = std::chrono::steady_clock::now();
            optime_accum[(int) node->op] += std::chrono::duration<double, std::milli>(t2 - t0).count();
            t0 = t2;
        }
    }
    if (optime_enabled && cgraph->n_nodes > 5) {
        for (const auto & kv : optime_accum) {
            fprintf(stderr, "HRX2_OPTIME: op=%-22s count=%-5lld total_ms=%8.2f avg_ms=%8.3f\n",
                ggml_op_name((enum ggml_op) kv.first), (long long) optime_count[kv.first], kv.second,
                kv.second / optime_count[kv.first]);
        }
    }
    if (ggml_backend_hrx2_env_enabled("GGML_HRX2_ASYNC_GRAPH_COMPUTE")) {
        if (!GGML_HRX2_CHECK(hrx_stream_flush(context->stream))) {
            return GGML_STATUS_FAILED;
        }
    } else {
        ggml_backend_hrx2_synchronize(backend);
    }
    if (optime_enabled) {
        // per-graph wall time INCLUDING the end-of-graph sync: this is what
        // the caller actually waits for (async enqueue deltas are ~0 once
        // warm, so the per-op print above alone understates GPU time).
        auto t3 = std::chrono::steady_clock::now();
        fprintf(stderr, "HRX2_OPTIME: graph_total_ms=%8.2f nodes=%d\n",
                std::chrono::duration<double, std::milli>(t3 - optime_t0).count(), cgraph->n_nodes);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx2_i = {
    /* .get_name           = */ ggml_backend_hrx2_get_name,
    /* .free               = */ ggml_backend_hrx2_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_hrx2_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hrx2_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static ggml_guid_t ggml_backend_hrx2_guid(void) {
    static ggml_guid guid = { 0x82, 0x48, 0x52, 0x58, 0x32, 0x2d, 0x4c, 0x4f, 0x4f, 0x4d, 0x2d, 0x4a, 0x49, 0x54, 0x00, 0x01 };
    return &guid;
}

static const char * ggml_backend_hrx2_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_hrx2_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_hrx2_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_hrx2_get_device_context(dev)->description.c_str();
}

static void ggml_backend_hrx2_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_hrx2_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx2_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx2_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_hrx2_device_get_name(dev);
    props->description = ggml_backend_hrx2_device_get_description(dev);
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    ggml_backend_hrx2_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_hrx2_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_hrx2_get_device_context(dev);
    hrx_stream_t stream = nullptr;
    if (!GGML_HRX2_CHECK(hrx_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx2_context {
        /* .device_context = */ device_context,
        /* .stream         = */ stream,
        /* .name           = */ device_context->name,
        /* .last_total_mul_mat_bytes = */ 0,
        /* .submitted_dispatches     = */ 0,
        /* .mul_mat_bytes            = */ 0,
        /* .total_mul_mat_bytes      = */ 0,
        /* .mul_mat_bytes_per_submit = */ 0,
        /* .submit_count             = */ 0,
        /* .submit_flush_count       = */ 0,
        /* .submit_last_node         = */ nullptr,
        /* .scratch_buffers          = */ {},
        /* .q8_1_scratch             = */ {},
        /* .route_scratch            = */ {},
    };
    if (!context) {
        hrx_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_hrx2_guid(),
        /* .iface   = */ ggml_backend_hrx2_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        hrx_stream_release(stream);
        delete context;
        return nullptr;
    }
    ggml_backend_hrx2_register_stream(device_context, stream);
    return backend;
}

static bool ggml_backend_hrx2_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // ZERO-COPY HYBRID (round 25i): with activations in the host-visible buft,
    // the CPU<->HRX2 shuttling that made the old hybrid net-negative is GONE —
    // every op reads/writes the same GTT memory, no copies. So claim the full
    // hybrid op set again: Q4NX mms + all F32 pointwise/norm/rope/etc. routes.
    auto * devctx = ggml_backend_hrx2_get_device_context(dev);
    // zero-copy: weights must live in the DEVICE-LOCAL buft (the default
    // buft is host-visible for activations). Reject Q4NX weight ops whose
    // src0 sits in the host buft so the loader picks the device-local extra.
    if ((op->op == GGML_OP_MUL_MAT_Q4NX || op->op == GGML_OP_MUL_MAT_ID_Q4NX) &&
        op->src[0] && op->src[0]->buffer &&
        op->src[0]->buffer->buft == &devctx->host_buffer_type) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT_Q4NX:
            return ggml_backend_hrx2_supports_mul_mat_q4nx_route(devctx, op);
        case GGML_OP_MUL_MAT_ID_Q4NX:
            return ggml_backend_hrx2_supports_mul_mat_id_q4nx_route(devctx, op);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;   // metadata-only, no data movement
        case GGML_OP_RMS_NORM:
            return ggml_backend_hrx2_supports_rms_norm_route(devctx, op);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
            return ggml_backend_hrx2_supports_pointwise_route(devctx, op);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx2_supports_sum_rows_route(devctx, op);
        case GGML_OP_ROPE:
            return ggml_backend_hrx2_supports_rope_route(devctx, op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx2_supports_soft_max_route(devctx, op);
        case GGML_OP_CONT:
            return ggml_backend_hrx2_supports_cont_route(devctx, op);
        case GGML_OP_CPY:
            return ggml_backend_hrx2_supports_cpy(devctx, op);
        case GGML_OP_SET_ROWS:
            return ggml_backend_hrx2_supports_set_rows_route(devctx, op);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx2_supports_argsort_route(devctx, op);
        case GGML_OP_GLU:
            return ggml_backend_hrx2_supports_swiglu_route(devctx, op);
        case GGML_OP_GET_ROWS:
            // embedding lookup: HRX2 get_rows_f32 route (CPU would have to
            // read/write shared GTT -> per-dispatch coherency tax)
            return ggml_backend_hrx2_supports_get_rows_route(devctx, op);
        case GGML_OP_MUL_MAT:
            // attention KQ^T/kqv mms: F16 src0 (KV cache) x F32 src1 (q/kq),
            // strided GQA batched views -> existing mul_mat_f16_f32_batched kernel
            return ggml_backend_hrx2_supports_mul_mat_f16_f32_route(devctx, op);
        default:
            return false;
    }
}

static bool ggml_backend_hrx2_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || !buft->context) {
        return false;
    }
    return ggml_backend_hrx2_get_buft_context(buft)->device_context == ggml_backend_hrx2_get_device_context(dev);
}

static const ggml_backend_device_i ggml_backend_hrx2_device_i = {
    /* .get_name             = */ ggml_backend_hrx2_device_get_name,
    /* .get_description      = */ ggml_backend_hrx2_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx2_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx2_device_get_type,
    /* .get_props            = */ ggml_backend_hrx2_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx2_device_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hrx2_device_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx2_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx2_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_backend_hrx2_reg_context * ggml_backend_hrx2_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_hrx2_reg_context *>(reg->context);
}

static const char * ggml_backend_hrx2_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX2_NAME;
}

static size_t ggml_backend_hrx2_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_hrx2_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_hrx2_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static ggml_backend_buffer_type_t * ggml_backend_hrx2_dev_get_extra_bufts(ggml_backend_dev_t device) {
    auto * device_context = ggml_backend_hrx2_get_device_context(device);
    // one extra buft: the device-local weight buft (the DEFAULT is host-visible
    // for zero-copy compute; weights must stay device-local for NPU reads).
    static ggml_backend_buffer_type_t extra[] = { nullptr, nullptr };
    extra[0] = &device_context->weight_buffer_type;
    return extra;
}

static void * ggml_backend_hrx2_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_hrx2_dev_get_extra_bufts;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx2_reg_i = {
    /* .get_name         = */ ggml_backend_hrx2_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx2_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx2_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx2_reg_get_proc_address,
};

static std::unique_ptr<ggml_backend_hrx2_reg_context> ggml_backend_hrx2_create_reg_context() {
    auto context = std::make_unique<ggml_backend_hrx2_reg_context>();

    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        GGML_HRX2_CHECK(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_HRX2_CHECK(hrx_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);
    for (int i = 0; i < device_count; ++i) {
        hrx_device_t device = nullptr;
        if (!GGML_HRX2_CHECK(hrx_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        hrx_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_hrx2_device_context>();
        device_context->device = device;
        device_context->name = std::string(GGML_HRX2_NAME) + std::to_string(i);
        device_context->description = ggml_backend_hrx2_device_description(device);
        device_context->architecture = ggml_backend_hrx2_device_string_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
        device_context->memory_total = ggml_backend_hrx2_total_memory(device);
        if (!GGML_HRX2_CHECK(hrx_stream_create(device_context->device, 0, &device_context->transfer_stream))) {
            hrx_device_release(device);
            continue;
        }
        ggml_backend_hrx2_register_stream(device_context.get(), device_context->transfer_stream);
        device_context->catalog = ggml_backend_hrx2_load_catalog();
        if (device_context->catalog) {
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "rms_norm_f32",
                "RMS_NORM",
                &device_context->rms_norm_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "rms_norm_mul_f32",
                "MUL",
                &device_context->rms_norm_mul_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "add_rms_norm_mul_f32",
                "MUL",
                &device_context->add_rms_norm_mul_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q8_0_f32",
                "MUL_MAT",
                &device_context->mul_mat_q8_0_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "quantize_q8_1_f32",
                "QUANTIZE",
                &device_context->quantize_q8_1_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_f32_f32",
                "MUL_MAT",
                &device_context->mul_mat_f32_f32_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "q4nx_dequant_f32",
                "Q4NX_DEQUANT",
                &device_context->q4nx_dequant_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q4_k_f32",
                "MUL_MAT",
                &device_context->mul_mat_q4_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q4_k_swiglu_f32",
                "GLU",
                &device_context->mul_mat_q4_k_swiglu_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_id_q4_k_f32",
                "MUL_MAT_ID",
                &device_context->mul_mat_id_q4_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q5_k_f32",
                "MUL_MAT",
                &device_context->mul_mat_q5_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_id_q5_k_f32",
                "MUL_MAT_ID",
                &device_context->mul_mat_id_q5_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_q6_k_f32",
                "MUL_MAT",
                &device_context->mul_mat_q6_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_id_q6_k_f32",
                "MUL_MAT_ID",
                &device_context->mul_mat_id_q6_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_f16_f32_batched",
                "MUL_MAT",
                &device_context->mul_mat_f16_f32_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_mat_f16_f32_batched_cont",
                "CONT",
                &device_context->mul_mat_f16_f32_cont_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "flash_attn_fa0_f32_f16",
                "CONT",
                &device_context->flash_attn_fa0_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "copy_f32_f16",
                "CPY",
                &device_context->copy_f32_f16_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "cont_f32",
                "CONT",
                &device_context->cont_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "cont_set_rows_f32",
                "SET_ROWS",
                &device_context->cont_set_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "swiglu_f32",
                "GLU",
                &device_context->swiglu_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "set_rows_f32",
                "SET_ROWS",
                &device_context->set_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "rope_set_rows_f32",
                "SET_ROWS",
                &device_context->rope_set_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "add_f32",
                "ADD",
                &device_context->add_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "mul_f32",
                "MUL",
                &device_context->mul_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "div_f32",
                "DIV",
                &device_context->div_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "scale_f32",
                "SCALE",
                &device_context->scale_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "clamp_f32",
                "CLAMP",
                &device_context->clamp_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "sum_rows_f32",
                "SUM_ROWS",
                &device_context->sum_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_f32",
                "GET_ROWS",
                &device_context->get_rows_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_q8_0_f32",
                "GET_ROWS",
                &device_context->get_rows_q8_0_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_q4_k_f32",
                "GET_ROWS",
                &device_context->get_rows_q4_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_q5_k_f32",
                "GET_ROWS",
                &device_context->get_rows_q5_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_q6_k_f32",
                "GET_ROWS",
                &device_context->get_rows_q6_k_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "get_rows_moe_weights_f32",
                "GET_ROWS",
                &device_context->get_rows_moe_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "argsort_f32_i32",
                "ARGSORT",
                &device_context->argsort_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                nullptr,
                "ROPE",
                &device_context->rope_routes);
            ggml_backend_hrx2_catalog_find_routes(
                *device_context->catalog,
                "soft_max_f32",
                "SOFT_MAX",
                &device_context->soft_max_routes);
            const auto route_less = [](const ggml_backend_hrx2_kernel_route * lhs, const ggml_backend_hrx2_kernel_route * rhs) {
                if (lhs->priority != rhs->priority) {
                    return lhs->priority > rhs->priority;
                }
                return lhs->id < rhs->id;
            };
            std::sort(
                device_context->rms_norm_routes.begin(),
                device_context->rms_norm_routes.end(),
                route_less);
            std::sort(
                device_context->rms_norm_mul_routes.begin(),
                device_context->rms_norm_mul_routes.end(),
                route_less);
            std::sort(
                device_context->add_rms_norm_mul_routes.begin(),
                device_context->add_rms_norm_mul_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q8_0_routes.begin(),
                device_context->mul_mat_q8_0_routes.end(),
                route_less);
            std::sort(
                device_context->quantize_q8_1_routes.begin(),
                device_context->quantize_q8_1_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q4_k_routes.begin(),
                device_context->mul_mat_q4_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q4_k_swiglu_routes.begin(),
                device_context->mul_mat_q4_k_swiglu_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_id_q4_k_routes.begin(),
                device_context->mul_mat_id_q4_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q5_k_routes.begin(),
                device_context->mul_mat_q5_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_id_q5_k_routes.begin(),
                device_context->mul_mat_id_q5_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_q6_k_routes.begin(),
                device_context->mul_mat_q6_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_id_q6_k_routes.begin(),
                device_context->mul_mat_id_q6_k_routes.end(),
                route_less);
            std::sort(
                device_context->mul_mat_f16_f32_routes.begin(),
                device_context->mul_mat_f16_f32_routes.end(),
                route_less);
            std::sort(
                device_context->flash_attn_fa0_routes.begin(),
                device_context->flash_attn_fa0_routes.end(),
                route_less);
            std::sort(
                device_context->copy_f32_f16_routes.begin(),
                device_context->copy_f32_f16_routes.end(),
                route_less);
            std::sort(
                device_context->cont_routes.begin(),
                device_context->cont_routes.end(),
                route_less);
            std::sort(
                device_context->cont_set_rows_routes.begin(),
                device_context->cont_set_rows_routes.end(),
                route_less);
            std::sort(
                device_context->swiglu_routes.begin(),
                device_context->swiglu_routes.end(),
                route_less);
            std::sort(
                device_context->set_rows_routes.begin(),
                device_context->set_rows_routes.end(),
                route_less);
            std::sort(
                device_context->rope_set_rows_routes.begin(),
                device_context->rope_set_rows_routes.end(),
                route_less);
            std::sort(
                device_context->add_routes.begin(),
                device_context->add_routes.end(),
                route_less);
            std::sort(
                device_context->mul_routes.begin(),
                device_context->mul_routes.end(),
                route_less);
            std::sort(
                device_context->div_routes.begin(),
                device_context->div_routes.end(),
                route_less);
            std::sort(
                device_context->scale_routes.begin(),
                device_context->scale_routes.end(),
                route_less);
            std::sort(
                device_context->clamp_routes.begin(),
                device_context->clamp_routes.end(),
                route_less);
            std::sort(
                device_context->sum_rows_routes.begin(),
                device_context->sum_rows_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_routes.begin(),
                device_context->get_rows_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_q8_0_routes.begin(),
                device_context->get_rows_q8_0_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_q4_k_routes.begin(),
                device_context->get_rows_q4_k_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_q5_k_routes.begin(),
                device_context->get_rows_q5_k_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_q6_k_routes.begin(),
                device_context->get_rows_q6_k_routes.end(),
                route_less);
            std::sort(
                device_context->get_rows_moe_routes.begin(),
                device_context->get_rows_moe_routes.end(),
                route_less);
            std::sort(
                device_context->argsort_routes.begin(),
                device_context->argsort_routes.end(),
                route_less);
            std::sort(
                device_context->rope_routes.begin(),
                device_context->rope_routes.end(),
                route_less);
            std::sort(
                device_context->soft_max_routes.begin(),
                device_context->soft_max_routes.end(),
                route_less);
        }
        device_context->buffer_type_context = {
            /* .device_context = */ device_context.get(),
            /* .name           = */ device_context->name,
        };
        device_context->buffer_type = {
            /* .iface   = */ ggml_backend_hrx2_buffer_type_i,
            /* .device  = */ nullptr,
            /* .context = */ &device_context->buffer_type_context,
        };
        device_context->host_buffer_type_context = {
            /* .device_context = */ device_context.get(),
            /* .name           = */ device_context->name + "_host",
        };
        device_context->host_buffer_type = {
            /* .iface   = */ ggml_backend_hrx2_host_buffer_type_i,
            /* .device  = */ nullptr,
            /* .context = */ &device_context->host_buffer_type_context,
        };
        device_context->weight_buffer_type_context = {
            /* .device_context = */ device_context.get(),
            /* .name           = */ device_context->name + "_w",
        };
        device_context->weight_buffer_type = {
            /* .iface   = */ ggml_backend_hrx2_buffer_type_i,
            /* .device  = */ nullptr,
            /* .context = */ &device_context->weight_buffer_type_context,
        };

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_hrx2_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
        context->device_contexts.back()->buffer_type.device = &context->devices.back();
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_hrx2_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid HRX2 device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_hrx2(ggml_backend_t backend) {
    return backend != nullptr && backend->guid == ggml_backend_hrx2_guid();
}

int ggml_backend_hrx2_get_device_count(void) {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_hrx2_reg()));
}

void ggml_backend_hrx2_get_device_description(int device, char * description, size_t description_size) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        if (description_size) {
            description[0] = '\0';
        }
        return;
    }
    const char * desc = ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, device));
    std::snprintf(description, description_size, "%s", desc ? desc : "");
}

void ggml_backend_hrx2_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        *free = 0;
        *total = 0;
        return;
    }
    ggml_backend_dev_memory(ggml_backend_reg_dev_get(reg, device), free, total);
}

ggml_backend_buffer_type_t ggml_backend_hrx2_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx2_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_hrx2_reg(void) {
    static std::unique_ptr<ggml_backend_hrx2_reg_context> context = ggml_backend_hrx2_create_reg_context();
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_hrx2_reg_i,
        /* .context     = */ context.get(),
    };
    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx2_reg)
