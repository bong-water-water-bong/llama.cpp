#include "models.h"

#include "ggml.h"
#include "llama-memory-recurrent.h"

#include <cmath>

// Zaya (1bit-MONSTER / Zyphra) — 8B-era hybrid: layers alternate CCA
// attention (even) / MoE (odd) with a running residual and per-layer
// post-attention / post-mlp residual scaling. Adapted from upstream
// llama.cpp PR #23112 (Zaya1 base) to the 8B-era layout:
//   - 40 layers, even = CCA attention, odd = MoE (top-1 over 16 experts +
//     skip expert)
//   - per-layer res_scale_hs/res_scale_res (post-attention, even) and
//     res_scale_hs_mlp/res_scale_res_mlp (post-mlp, odd)
//   - model-level input_hidden_states_scale/bias applied to the embeddings
//   - conv_qk is a 2-tap depthwise conv (ssm_conv1d) + a grouped conv
//     (cca_conv_grp); the 2-tap conv needs a recurrent state (conv_state |
//     prev_hs), sized n_embd_s = 2*n_qk + n_embd.
//   - experts are PRE-STACKED (ffn_gate_up_exps / ffn_down_exps)
//   - router: down_proj -> EDA -> RMSNorm -> MLP(GELU)x2 -> softmax ->
//     top-1 over n_expert+1 slots (last = skip expert)
//
// Engine reference: engine/npu/src/zaya_decode.cpp + zaya_cca_attn_cpu.h +
// zaya_moe_cpu.h (CPU reference ports, corr-verified vs the GPU kernels).

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

llm_build_zaya::llm_build_zaya(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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

    if (model.zaya_input_hs_scale != nullptr) {
        if (model.zaya_input_hs_bias != nullptr) {
            inpL = ggml_add(ctx0, inpL, model.zaya_input_hs_bias);
        }
        inpL = ggml_mul(ctx0, inpL, model.zaya_input_hs_scale);
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

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_gqa     = n_head / n_head_kv;

        // Zaya 8B (HF ZayaDecoderLayer): EVERY layer runs BOTH blocks.
        //   residual = h (layer input, fp32 stream)
        //   cur = input_layernorm(residual)                       (attn_norm)
        //   attn = CCA(cur)
        //   residual = (attn+pa_hsb)*pa_hss + (residual+pa_rsb)*pa_rss   (res_scale_hs/res)
        //   cur = post_attention_layernorm(residual)              (post_attn_norm)
        //   moe = MoE(cur, prev_router)
        //   h   = (moe+pm_hsb)*pm_hss + (residual+pm_rsb)*pm_rss         (res_scale_hs_mlp/res_mlp)
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

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        Vcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Vcur), n_embd_head, n_head_kv, n_tokens);

        cur = build_attn(inp->get_attn(), layer.wo, nullptr,
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
            /* gating_op */       LLAMA_EXPERT_GATING_FUNC_TYPE_NONE,
            /* il */              il,
            /* probs_in */        gate_probs,
            /* gate_up_exps */    layer.ffn_gate_up_exps,
            /* gate_up_exps_b */  nullptr,
            /* up_exps_s */       nullptr,
            /* gate_exps_s */     nullptr,
            /* down_exps_s */     nullptr);
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
