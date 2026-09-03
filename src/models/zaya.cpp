// zaya.cpp — Zaya (1bit-MONSTER) hybrid CCA-attention + top-1 MoE model.
// Ported from the stale HRX2 fork (bong-water-water-bong/llama.cpp hrx-v2 @ 8df3330)
// onto the AMD ggml-hrx base (round-28). Every layer has BOTH CCA attention and
// MoE (HF ZayaDecoderLayer). GGUF: zaya-q4nx-c43.gguf (GGML_TYPE_Q4NX at 43).
#include "models.h"

#include <cmath>
#include <stdexcept>

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,            hparams.ssm_d_conv);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    if (hparams.f_norm_rms_eps == 0.0f) {
        hparams.f_norm_rms_eps = 1e-5f; // zaya reference default
    }
    // Zaya: all 40 layers are hybrid (CCA + MoE); no separate recurrent layers array.
    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), 0);
}

void llama_model_zaya::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hd = hparams.n_embd_head_k();
    const int64_t n_embd_q = n_head * hd;              // q rows (8 x 128 = 1024)
    const int64_t n_embd_k = n_head_kv * hd;           // k rows (2 x 128 = 256)
    const int64_t n_qk     = n_embd_q + n_embd_k;      // conv width
    const int64_t n_ff_x    = hparams.n_ff(0); // feed_forward_length 2048
    const int64_t n_ff_exp = hparams.n_ff_exp;         // router MLP width (256)

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }
    // model-level input scale/bias (applied to embeddings)
    input_hidden_states_scale = create_tensor(tn(LLM_TENSOR_INPUT_HIDDEN_STATES_SCALE, "weight"), { n_embd }, TENSOR_NOT_REQUIRED);
    input_hidden_states_bias  = create_tensor(tn(LLM_TENSOR_INPUT_HIDDEN_STATES_SCALE, "bias"),   { n_embd }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        llama_layer & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.post_attn_norm = create_tensor(tn(LLM_TENSOR_POST_ATTN_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);

        // post-attention residual scaling
        layer.res_scale_hs    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.res_scale_hs_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS, "bias", i),   { n_embd }, TENSOR_NOT_REQUIRED);
        layer.res_scale_res   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES, "bias", i),   { n_embd }, TENSOR_NOT_REQUIRED);

        // CCA attention projections
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_q }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k }, 0);
        layer.cca_val_proj1 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ1, "weight", i), { n_embd, n_embd_k / 2 }, 0);
        layer.cca_val_proj2 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ2, "weight", i), { n_embd, n_embd_k / 2 }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_q, n_embd }, 0);

        // conv-qk (2-tap depthwise) + grouped conv + k scale
        const int64_t d_conv = hparams.ssm_d_conv ? hparams.ssm_d_conv : 2;
        layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), { d_conv, n_qk }, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i),   { n_qk }, TENSOR_NOT_REQUIRED);
        const int64_t n_groups = n_head + n_head_kv;
        layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i), { d_conv, n_qk / n_groups, n_qk }, 0);
        layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias", i),   { n_qk }, 0);
        layer.cca_k_scale    = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE, "weight", i), { n_head_kv }, 0);

        // post-MLP residual scaling
        layer.res_scale_hs_mlp    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_MLP, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.res_scale_hs_mlp_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_MLP, "bias", i),   { n_embd }, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_mlp   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_MLP, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_mlp_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_MLP, "bias", i),   { n_embd }, TENSOR_NOT_REQUIRED);

        // 17-slot router stack (16 experts + skip): down(inp) -> EDA -> MLP x2 -> biases
        layer.ffn_gate_inp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_ff_exp }, 0);
        layer.ffn_gate_inp_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "bias", i),   { n_ff_exp }, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_ff_exp }, TENSOR_NOT_REQUIRED);
        layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_ff_exp, n_ff_exp }, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i),   { n_ff_exp }, TENSOR_NOT_REQUIRED);
        layer.zaya_router_mlp2   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "weight", i), { n_ff_exp, n_ff_exp }, 0);
        layer.zaya_router_mlp2_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2, "bias", i),   { n_ff_exp }, TENSOR_NOT_REQUIRED);
        layer.zaya_router_mlp4   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP4, "weight", i), { n_ff_exp, n_expert + 1 }, 0);
        layer.zaya_router_biases   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_BIASES, "weight", i), { n_expert + 1 }, TENSOR_NOT_REQUIRED);
        layer.zaya_router_eda_scale = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_EDA_SCALE, "weight", i), { n_ff_exp }, TENSOR_NOT_REQUIRED);

        // stacked MoE experts (gate_up fused, down)
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), { n_embd, n_ff_x * 2, n_expert }, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),    { n_ff_x, n_embd, n_expert }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_zaya::build_arch_graph(const llm_graph_params & params) const {
    throw std::runtime_error("zaya graph build not yet ported (round-28 T2 step 2)");
    (void) params;
}
