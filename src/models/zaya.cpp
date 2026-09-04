// zaya.cpp — Zaya (1bit-MONSTER) hybrid CCA-attention + top-1 MoE model.
// Ported from the stale HRX2 fork (bong-water-water-bong/llama.cpp hrx-v2 @ 8df3330)
// onto the AMD ggml-hrx base (round-28). Every layer has BOTH CCA attention and
// MoE (HF ZayaDecoderLayer). GGUF: zaya-q4nx-c43.gguf (GGML_TYPE_Q4NX at 43).
//
// Architecture (engine reference: engine/npu/src/zaya_decode.cpp + zaya_cca_attn_cpu.h):
//   per layer: residual = h
//     cur  = rmsnorm(residual)                         (attn_norm)
//     attn = CCA(cur): q/k projections -> 2-tap depthwise conv (ssm_conv1d,
//            recurrent conv_state) + grouped conv (cca_conv_grp) over q||k,
//            qk mean/l2 norm + rope + GQA flash attn, v = val_proj1(cur) ||
//            val_proj2(prev_hs delay)
//     residual = res_scale_hs(attn) + res_scale_res(residual)
//     cur = rmsnorm(residual)                          (post_attn_norm)
//     moe = MoE(cur, prev_router): router down_proj -> EDA(prev router) ->
//            RMSNorm -> GELU MLP x2 -> softmax -> top-1 of 17 slots (last=skip);
//            stacked ffn_gate_up_exps/ffn_down_exps
//     h   = res_scale_hs_mlp(moe) + res_scale_res_mlp(residual)
//   model-level input_hidden_states_scale/bias applied to the embeddings.
#include "models.h"

#include "ggml.h"
#include "llama-memory-recurrent.h"

#include <cmath>

static struct ggml_tensor * ggml_conv_1d_grouped(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   s0,
        int                   p0,
        int                   d0,
        int                   groups) {
    GGML_ASSERT(groups > 0);

    const int64_t OC   = a->ne[2];
    const int64_t IC_G = a->ne[1];
    const int64_t IC   = b->ne[1];

    GGML_ASSERT(IC % groups == 0);
    GGML_ASSERT(OC % groups == 0);
    GGML_ASSERT(IC_G == IC / groups);

    if (groups == 1) {
        return ggml_conv_1d(ctx, a, b, s0, p0, d0);
    }
    if (groups == IC && groups == OC) {
        return ggml_conv_1d_dw(ctx, a, b, s0, p0, d0);
    }

    const int64_t OC_G = OC / groups;

    struct ggml_tensor * result = NULL;

    for (int g = 0; g < groups; g++) {
        struct ggml_tensor * a_g = ggml_view_3d(ctx, a,
            a->ne[0], IC_G, OC_G,
            a->nb[1], a->nb[2],
            g * OC_G * a->nb[2]);

        struct ggml_tensor * b_g = ggml_view_3d(ctx, b,
            b->ne[0], IC_G, b->ne[2],
            b->nb[1], b->nb[2],
            g * IC_G * b->nb[1]);

        struct ggml_tensor * out_g = ggml_conv_1d(ctx, a_g, b_g, s0, p0, d0);

        if (result == NULL) {
            result = out_g;
        } else {
            result = ggml_concat(ctx, result, out_g, 1);
        }
    }

    return result;
}

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,            hparams.ssm_d_conv);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    if (hparams.f_norm_rms_eps == 0.0f) {
        hparams.f_norm_rms_eps = 1e-5f; // zaya reference default
    }
    // Zaya: all 40 layers are hybrid (CCA + MoE); no separate recurrent layers array.
    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), 0);
    // recurrent conv-state cache: conv_state (2*(n_q+n_k)) + prev_hs (n_embd)
    const int64_t hd = hparams.n_embd_head_k();
    hparams.zaya_n_embd_s = (uint32_t)(2*(int64_t)(hparams.n_head() + hparams.n_head_kv())*hd + hparams.n_embd);
    hparams.n_rot_full = (uint32_t)(hd / 2); // rope.dimension_count = 64 (partial rotary 0.5)
    hparams.n_rot_swa  = (uint32_t)(hd / 2);
    hparams.ssm_d_inner = (uint32_t)(2*(int64_t)(hparams.n_head() + hparams.n_head_kv())*hd + hparams.n_embd);
    hparams.ssm_d_state = 1;
    hparams.ssm_n_group = 0;
}

void llama_model_zaya::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hd = hparams.n_embd_head_k();
    const int64_t n_embd_q = n_head * hd;              // q rows (8 x 128 = 1024)
    const int64_t n_embd_k = n_head_kv * hd;           // k rows (2 x 128 = 256)
    const int64_t n_qk     = n_embd_q + n_embd_k;      // conv width
    const int64_t n_ff_x   = hparams.n_ff(0);          // feed_forward_length 2048
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

llama_model_zaya::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_expert    = hparams.n_expert;
    const int64_t n_seqs      = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_tokens % n_seqs == 0);

    const int64_t n_seq_tokens = n_tokens / n_seqs;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "embd", -1);

    if (model.input_hidden_states_scale != nullptr) {
        if (model.input_hidden_states_bias != nullptr) {
            inpL = ggml_add(ctx0, inpL, model.input_hidden_states_bias);
        }
        inpL = ggml_mul(ctx0, inpL, model.input_hidden_states_scale);
        cb(inpL, "input_hs_scaled", -1);
    }

    auto * inp = build_inp_mem_hybrid();
    auto * inp_recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * prev_router = nullptr;

    const auto apply_res_scale = [&](ggml_tensor * x, ggml_tensor * scale, ggml_tensor * bias, const char * name, int il) {
        if (scale == nullptr) {
            return x;
        }
        if (bias != nullptr) {
            x = ggml_add(ctx0, x, bias);
        }
        x = ggml_mul(ctx0, x, scale);
        cb(x, name, il);
        return x;
    };

    const int64_t n_rot      = hparams.n_rot();
    const int64_t n_ctx_orig = cparams.n_ctx_orig_yarn;
    fprintf(stderr, "[zaya] dbg: n_rot=%lld head_k=%d n_head=%d n_head_kv=%d ssm_d_conv=%u n_ff_exp=%u n_expert=%u n_embd_s=%u\n",
        (long long)n_rot, (int)hparams.n_embd_head_k(), (int)hparams.n_head(), (int)hparams.n_head_kv(),
        hparams.ssm_d_conv, hparams.n_ff_exp, hparams.n_expert, hparams.n_embd_s());

    const int n_layer_run = getenv("ZAYA_1LAYER") ? 1 : n_layer;
    for (int il = 0; il < n_layer_run; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head    = hparams.n_head();
        const int64_t n_head_kv = hparams.n_head_kv();
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_gqa     = n_head / n_head_kv;

        ggml_tensor * residual = inpL;

        cur = build_norm(residual, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "input_norm", il);

        // ===== CCA attention (every layer) =====

        const int64_t conv_state_size = 2*n_qk;
        const int64_t cca_state_size  = conv_state_size + n_embd;
        GGML_ASSERT((int64_t) hparams.n_embd_s() == cca_state_size);

        ggml_tensor * cca_state_all = inp_recr->mctx->get_s_l(il);
        ggml_tensor * cca_state     = build_rs(inp_recr, cca_state_all, hparams.n_embd_s(), n_seqs);
        cb(cca_state, "cca_state", il);

        ggml_tensor * conv_state = ggml_view_3d(ctx0, cca_state, 2, n_qk, n_seqs,
                2*ggml_element_size(cca_state),
                cca_state->nb[1],
                0);
        cb(conv_state, "cca_conv_state", il);

        ggml_tensor * prev_hs = ggml_view_2d(ctx0, cca_state, n_embd, n_seqs,
                cca_state->nb[1],
                conv_state_size*ggml_element_size(cca_state));
        cb(prev_hs, "cca_prev_hs", il);

        ggml_tensor * Qraw = ggml_mul_mat(ctx0, layer.wq, cur);
        cb(Qraw, "Qraw", il);
        ggml_tensor * Kraw = ggml_mul_mat(ctx0, layer.wk, cur);
        cb(Kraw, "Kraw", il);

        ggml_tensor * cur_state_src = ggml_cont(ctx0, cur);
        ggml_tensor * cur_seq = ggml_reshape_3d(ctx0, cur_state_src, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * hs_d = ggml_reshape_3d(ctx0, ggml_cont(ctx0, prev_hs), n_embd, 1, n_seqs);
        if (n_seq_tokens > 1) {
            ggml_tensor * cur_shift = ggml_view_3d(ctx0, cur_seq, n_embd, n_seq_tokens - 1, n_seqs,
                    cur_seq->nb[1],
                    cur_seq->nb[2],
                    0);
            hs_d = ggml_concat(ctx0, hs_d, cur_shift, 1);
        }
        hs_d = ggml_reshape_2d(ctx0, ggml_cont(ctx0, hs_d), n_embd, n_tokens);
        cb(hs_d, "cca_hs_d", il);

        ggml_tensor * V1 = ggml_mul_mat(ctx0, layer.cca_val_proj1, cur);
        cb(V1, "V1", il);
        ggml_tensor * V2 = ggml_mul_mat(ctx0, layer.cca_val_proj2, hs_d);
        cb(V2, "V2", il);
        ggml_tensor * Vcur = ggml_concat(ctx0, V1, V2, 0);
        cb(Vcur, "Vcur", il);

        ggml_tensor * QKraw = ggml_concat(ctx0, Qraw, Kraw, 0);
        cb(QKraw, "QKraw", il);

        ggml_tensor * Qpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Qraw), n_embd_head, n_head, n_tokens);
        ggml_tensor * Kpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Kraw), n_embd_head, n_head_kv, n_tokens);

        ggml_tensor * Kpre_grouped = ggml_reshape_4d(ctx0, Kpre, n_embd_head, 1, n_head_kv, n_tokens);
        Kpre_grouped = ggml_repeat_4d(ctx0, Kpre_grouped, n_embd_head, n_gqa, n_head_kv, n_tokens);
        ggml_tensor * Kpre_rep = ggml_reshape_3d(ctx0, Kpre_grouped, n_embd_head, n_head, n_tokens);
        ggml_tensor * qk_mean_q = ggml_scale(ctx0, ggml_add(ctx0, Qpre, Kpre_rep), 0.5f);
        cb(qk_mean_q, "qk_mean_q", il);

        ggml_tensor * Qgroup = ggml_reshape_4d(ctx0, Qpre, n_embd_head, n_gqa, n_head_kv, n_tokens);
        Qgroup = ggml_permute(ctx0, Qgroup, 1, 0, 2, 3);
        Qgroup = ggml_cont(ctx0, Qgroup);
        ggml_tensor * Qmean = ggml_mean(ctx0, Qgroup);
        Qmean = ggml_reshape_3d(ctx0, Qmean, n_embd_head, n_head_kv, n_tokens);
        ggml_tensor * qk_mean_k = ggml_scale(ctx0, ggml_add(ctx0, Qmean, Kpre), 0.5f);
        cb(qk_mean_k, "qk_mean_k", il);

        ggml_tensor * QKraw_t = ggml_cont(ctx0, ggml_transpose(ctx0, QKraw));
        QKraw_t = ggml_reshape_3d(ctx0, QKraw_t, n_seq_tokens, n_qk, n_seqs);

        ggml_tensor * conv_input = ggml_concat(ctx0, conv_state, QKraw_t, 0);
        cb(conv_input, "cca_conv_input", il);

        ggml_tensor * last_conv_states = ggml_view_3d(ctx0, conv_input, 2, n_qk, n_seqs,
                conv_input->nb[1],
                conv_input->nb[2],
                n_seq_tokens*conv_input->nb[0]);
        cb(last_conv_states, "cca_last_conv_states", il);

        const auto kv_head = inp_recr->mctx->get_head();
        ggml_tensor * conv_state_update_target = ggml_view_2d(ctx0, cca_state_all, conv_state_size, n_seqs,
                cca_state_all->nb[1],
                kv_head*cca_state_size*ggml_element_size(cca_state_all));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, conv_state_update_target));

        ggml_tensor * last_hs = ggml_view_2d(ctx0, cur_seq, n_embd, n_seqs,
                cur_seq->nb[2],
                (n_seq_tokens - 1)*cur_seq->nb[1]);
        ggml_tensor * prev_hs_update_target = ggml_view_2d(ctx0, cca_state_all, n_embd, n_seqs,
                cca_state_all->nb[1],
                (kv_head*cca_state_size + conv_state_size)*ggml_element_size(cca_state_all));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_hs, prev_hs_update_target));

        ggml_tensor * conv_dw = layer.ssm_conv1d;
        if (conv_dw->type != GGML_TYPE_F32) {
            conv_dw = ggml_cont(ctx0, ggml_cast(ctx0, conv_dw, GGML_TYPE_F32));
        }
        ggml_tensor * QK = ggml_ssm_conv(ctx0, conv_input, conv_dw);
        QK = ggml_cont(ctx0, ggml_permute(ctx0, QK, 1, 0, 2, 3));
        if (layer.ssm_conv1d_b) {
            QK = ggml_add(ctx0, QK, ggml_reshape_3d(ctx0, layer.ssm_conv1d_b, 1, n_qk, 1));
        }
        cb(QK, "QK_dw", il);

        ggml_tensor * conv_grp = layer.cca_conv_grp;
        if (conv_grp->type != GGML_TYPE_F16) {
            conv_grp = ggml_cont(ctx0, ggml_cast(ctx0, conv_grp, GGML_TYPE_F16));
        }
        QK = ggml_conv_1d_grouped(ctx0, conv_grp, QK, 1, 0, 1, n_groups);
        QK = ggml_add(ctx0, QK, ggml_reshape_3d(ctx0, layer.cca_conv_grp_b, 1, n_qk, 1));
        cb(QK, "QK_grp", il);

        QK = ggml_cont(ctx0, ggml_permute(ctx0, QK, 1, 0, 2, 3));
        QK = ggml_reshape_2d(ctx0, QK, n_qk, n_tokens);

        ggml_tensor * Q_conv = ggml_view_2d(ctx0, QK, n_embd_q, n_tokens, QK->nb[1], 0);
        ggml_tensor * K_conv = ggml_view_2d(ctx0, QK, n_embd_k, n_tokens, QK->nb[1], n_embd_q*ggml_element_size(QK));

        ggml_tensor * Qcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Q_conv), n_embd_head, n_head, n_tokens);
        ggml_tensor * Kcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, K_conv), n_embd_head, n_head_kv, n_tokens);

        Qcur = ggml_add(ctx0, Qcur, qk_mean_q);
        Kcur = ggml_add(ctx0, Kcur, qk_mean_k);

        Qcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Qcur, 1e-12f), sqrtf((float) n_embd_head));
        Kcur = ggml_scale(ctx0, ggml_l2_norm(ctx0, Kcur, 1e-12f), sqrtf((float) n_embd_head));
        Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.cca_k_scale, 1, n_head_kv, 1));
        cb(Qcur, "Qcur_pre_rope", il);
        cb(Kcur, "Kcur_pre_rope", il);

        ggml_tensor * rope_factors = nullptr; // no yarn factors for zaya
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                n_rot, GGML_ROPE_TYPE_NEOX, n_ctx_orig, 5000000.0f, 1.0f,
                1.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                n_rot, GGML_ROPE_TYPE_NEOX, n_ctx_orig, 5000000.0f, 1.0f,
                1.0f, 1.0f, 0.0f, 0.0f);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        Vcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Vcur), n_embd_head, n_head_kv, n_tokens);

        cur = build_attn(inp->get_attn(), layer.wo, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
            1.0f / sqrtf((float) n_embd_head), il);
        cb(cur, "attn_out", il);

        // ---- post-attention residual scale ----
        ggml_tensor * hs_scaled = apply_res_scale(cur, layer.res_scale_hs, layer.res_scale_hs_b, "res_scale_hs", il);
        ggml_tensor * res_scaled = apply_res_scale(residual, layer.res_scale_res, layer.res_scale_res_b, "res_scale_res", il);
        residual = ggml_add(ctx0, hs_scaled, res_scaled);
        cb(residual, "residual_post_attn", il);

        // ---- post-attention layernorm ----
        cur = build_norm(residual, layer.post_attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        // ===== MoE (every layer) =====

        ggml_tensor * router_h = ggml_mul_mat(ctx0, layer.ffn_gate_inp, cur);
        if (layer.ffn_gate_inp_b) {
            router_h = ggml_add(ctx0, router_h, layer.ffn_gate_inp_b);
        }
        cb(router_h, "router_down", il);

        if (prev_router != nullptr && layer.zaya_router_eda_scale != nullptr) {
            router_h = ggml_add(ctx0, router_h, ggml_mul(ctx0, prev_router, layer.zaya_router_eda_scale));
            cb(router_h, "router_eda", il);
        }

        prev_router = router_h;

        router_h = build_norm(router_h, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(router_h, "router_norm", il);

        router_h = ggml_mul_mat(ctx0, layer.ffn_gate, router_h);
        if (layer.ffn_gate_b) {
            router_h = ggml_add(ctx0, router_h, layer.ffn_gate_b);
        }
        router_h = ggml_gelu(ctx0, router_h);
        cb(router_h, "router_mlp0", il);

        router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp2, router_h);
        if (layer.zaya_router_mlp2_b) {
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp2_b);
        }
        router_h = ggml_gelu(ctx0, router_h);
        cb(router_h, "router_mlp2", il);

        router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp4, router_h);
        cb(router_h, "router_logits", il);

        router_h = ggml_soft_max(ctx0, router_h);
        cb(router_h, "router_probs", il);

        ggml_tensor * gate_probs = ggml_cont(ctx0,
                ggml_view_2d(ctx0, router_h, n_expert, n_tokens, router_h->nb[1], 0));
        cb(gate_probs, "gate_probs", il);

        ggml_tensor * expert_biases = nullptr;
        if (layer.zaya_router_biases != nullptr) {
            expert_biases = ggml_view_1d(ctx0, layer.zaya_router_biases, n_expert, 0);
        }

        cur = build_moe_ffn(cur,
            /* gate_inp */        nullptr,
            /* gate_inp_b */      nullptr,
            /* up_exps */         nullptr,
            /* up_exps_b */       nullptr,
            /* gate_exps */       nullptr,
            /* gate_exps_b */     nullptr,
            /* down_exps */       layer.ffn_down_exps,
            /* down_exps_b */     nullptr,
            /* exp_probs_b */     expert_biases,
            /* n_expert */        n_expert,
            /* n_expert_used */   hparams.n_expert_used,
            /* type_op */         LLM_FFN_SILU,
            /* norm_w */          false,
            /* w_scale */         1.0f,
            /* gating_op */       LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
            /* il */              il,
            /* probs_in */        gate_probs,
            /* gate_up_exps */    layer.ffn_gate_up_exps);
        cb(cur, "moe_out", il);

        // ---- post-MLP residual scale ----
        hs_scaled = apply_res_scale(cur, layer.res_scale_hs_mlp, layer.res_scale_hs_mlp_b, "res_scale_hs_mlp", il);
        res_scaled = apply_res_scale(residual, layer.res_scale_res_mlp, layer.res_scale_res_mlp_b, "res_scale_res_mlp", il);
        inpL = ggml_add(ctx0, hs_scaled, res_scaled);
        cb(inpL, "layer_out", il);
    }

    cur = inpL;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    cur = ggml_cont(ctx0, ggml_cast(ctx0, cur, GGML_TYPE_F32));
    cb(cur, "result_output_fp32", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
