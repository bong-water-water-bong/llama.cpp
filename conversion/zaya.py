from __future__ import annotations

import logging
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf

logger = logging.getLogger("hf-to-gguf")


@ModelBase.register("ZayaModel", "ZayaForCausalLM")
class ZayaModel(TextModel):
    """Zaya-1 model with Compressed Convolutional Attention and MoE"""
    model_arch = gguf.MODEL_ARCH.ZAYA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Buffer for accumulating expert weights per layer
        self._experts: dict[int, dict[str, Tensor]] | None = {}
        # Track emitted tensor names to avoid duplicates
        self._emitted: set[str] = set()
        # Pre-load tokenizer to know the vocab count for embedding trimming
        self._tokenizer_vocab_size: int | None = None
        try:
            from gguf.vocab import LlamaHfVocab
            self._tokenizer_vocab_size = LlamaHfVocab(self.dir_model).vocab_size
        except Exception:
            pass

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Use the LARGER of tokenizer vocab size and config vocab_size.
        # Truncating would lose token IDs and cause garbled model output.
        hparam_vocab = self.hparams.get("vocab_size", 0)
        tokenizer_vocab = self._tokenizer_vocab_size or 0
        vocab_size = max(hparam_vocab, tokenizer_vocab)
        self.gguf_writer.add_vocab_size(vocab_size)

        # n_ff = ffn_hidden_size / 2 (SwiGLU halves the intermediate)
        n_ff = self.hparams.get("ffn_hidden_size", 4096) // 2
        self.gguf_writer.add_feed_forward_length(n_ff)

        # ssm_d_conv = conv_qk kernel size (cca_time0 = first depthwise conv kernel)
        cca_time0 = self.hparams.get("cca_time0", 2)
        self.gguf_writer.add_ssm_conv_kernel(cca_time0)

        # partial_rotary_factor -> n_rot
        head_dim = self.hparams.get("head_dim", 128)
        partial_rotary = self.hparams.get("partial_rotary_factor", 0.5)
        self.gguf_writer.add_rope_dimension_count(int(partial_rotary * head_dim))

        # RoPE freq_base from rope_parameters.hybrid.rope_theta (default 10000)
        rope_params = self.hparams.get("rope_parameters", {})
        hybrid_params = rope_params.get("hybrid", {}) if isinstance(rope_params, dict) else {}
        rope_theta = hybrid_params.get("rope_theta", 10000.0)
        self.gguf_writer.add_rope_freq_base(rope_theta)

        # MoE params
        n_expert = self.find_hparam(["num_experts"])
        self.gguf_writer.add_expert_count(n_expert)
        n_expert_used = self.find_hparam(["moe_router_topk", "num_experts_per_tok"], optional=True) or 1
        self.gguf_writer.add_expert_used_count(n_expert_used)

        # Router MLP hidden size (zaya_mlp_expansion)
        n_ff_exp = self.hparams.get("zaya_mlp_expansion", 256)
        self.gguf_writer.add_expert_feed_forward_length(n_ff_exp)

    def _map_cca(self, name: str, data_torch: Tensor, bid: int) -> Iterable[tuple[str, Tensor]]:
        # Original naming (old Zaya checkpoints)
        if "linear_q" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), data_torch
        elif "linear_k" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), data_torch
        elif "val_proj1" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_VAL_PROJ1, bid), data_torch
        elif "val_proj2" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_VAL_PROJ2, bid), data_torch
        elif "o_proj" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), data_torch
        elif "conv_qk.0" in name and name.endswith(".weight"):
            data_torch = data_torch.squeeze(1).contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), data_torch
        elif "conv_qk.0" in name and name.endswith(".bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".bias"), data_torch
        elif "conv_qk.1" in name and name.endswith(".weight"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_CONV_GRP, bid), data_torch
        elif "conv_qk.1" in name and name.endswith(".bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_CONV_GRP, bid, suffix=".bias"), data_torch
        elif "temp" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_K_SCALE, bid), data_torch
        # New naming (Zyphra ZAYA1-8B checkpoint)
        elif name.endswith("qkv_proj.q_proj.weight") or name.endswith("qkv_proj.q_proj.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), data_torch
        elif name.endswith("qkv_proj.k_proj.weight") or name.endswith("qkv_proj.k_proj.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), data_torch
        elif name.endswith("qkv_proj.v_proj_current.weight") or name.endswith("qkv_proj.v_proj_current.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_VAL_PROJ1, bid), data_torch
        elif name.endswith("qkv_proj.v_proj_delayed.weight") or name.endswith("qkv_proj.v_proj_delayed.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_VAL_PROJ2, bid), data_torch
        elif name.endswith("qkv_proj.conv_qk_depthwise.weight"):
            data_torch = data_torch.squeeze(1).contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), data_torch
        elif name.endswith("qkv_proj.conv_qk_depthwise.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".bias"), data_torch
        elif name.endswith("qkv_proj.conv_qk_grouped.weight"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_CONV_GRP, bid), data_torch
        elif name.endswith("qkv_proj.conv_qk_grouped.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.CCA_CONV_GRP, bid, suffix=".bias"), data_torch

    def _map_router(self, name: str, data_torch: Tensor, bid: int) -> Iterable[tuple[str, Tensor]]:
        # Original naming (old Zaya checkpoints)
        if "down_proj.weight" in name and "router_mlp" not in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid), data_torch
        elif "down_proj.bias" in name and "router_mlp" not in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid, suffix=".bias"), data_torch
        elif "rmsnorm_eda" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_NORM, bid), data_torch
        elif "router_mlp.0.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch
        elif "router_mlp.0.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid, suffix=".bias"), data_torch
        elif "router_mlp.2.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2, bid), data_torch
        elif "router_mlp.2.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2, bid, suffix=".bias"), data_torch
        elif "router_mlp.4.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP4, bid), data_torch
        elif "balancing_biases" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_BIASES, bid), data_torch
        elif "router_states_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_EDA_SCALE, bid), data_torch
        # New naming (Zyphra ZAYA1-8B checkpoint)
        elif "gate.down_proj.weight" in name or name.endswith("gate.down_proj.bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid), data_torch
        elif "router_mlp.fc1.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), data_torch
        elif "router_mlp.fc1.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid, suffix=".bias"), data_torch
        elif "router_mlp.fc2.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2, bid), data_torch
        elif "router_mlp.fc2.bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2, bid, suffix=".bias"), data_torch
        elif "router_mlp.out_proj.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP4, bid), data_torch
        elif "router_mlp.norm.weight" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_NORM, bid), data_torch

    def _map_res_scale(self, name: str, data_torch: Tensor, bid: int) -> Iterable[tuple[str, Tensor]]:
        if "hidden_states_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS, bid), data_torch
        elif "hidden_states_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS, bid, suffix=".bias"), data_torch
        elif "residual_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES, bid), data_torch
        elif "residual_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES, bid, suffix=".bias"), data_torch

    def _map_final_res_scale(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        if "hidden_states_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS_FINAL), data_torch
        elif "hidden_states_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_HS_FINAL, suffix=".bias"), data_torch
        elif "residual_scale" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES_FINAL), data_torch
        elif "residual_bias" in name:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.RES_SCALE_RES_FINAL, suffix=".bias"), data_torch

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Common tensors
        if name == "model.embed_tokens.weight":
            # Pad embedding to full model vocab size if tokenizer is smaller.
            # Truncation would lose token IDs and cause garbled output.
            if self._tokenizer_vocab_size is not None and data_torch.shape[0] > self._tokenizer_vocab_size:
                # There are extra tokens in the model that the tokenizer doesn't know
                # about. Keep them (don't truncate) — they'll be unused during tokenization
                # but needed for correct LM head / embedding tensor shape.
                pass
            elif self._tokenizer_vocab_size is not None and data_torch.shape[0] < self.hparams.get("vocab_size", 0):
                # Tokenizer is larger than the raw embedding tensor. Pad with zeros.
                target = max(data_torch.shape[0], self.hparams.get("vocab_size", data_torch.shape[0]))
                if target > data_torch.shape[0]:
                    pad = torch.zeros(target - data_torch.shape[0], data_torch.shape[1], dtype=data_torch.dtype)
                    data_torch = torch.cat([data_torch, pad], dim=0)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch
            return
        if name == "model.final_norm.weight" or name == "model.norm.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch
            return
        # Skip final residual scale for now
        if name.startswith("model.input_hidden_states_"):
            return
        if name.startswith("model.res_scale."):
            yield from self._map_final_res_scale(name, data_torch)
            return

        # Block-level tensors
        if bid is not None:
            # CCA attention tensors
            if "self_attn" in name:
                yield from self._map_cca(name, data_torch, bid)
                return

            # Router tensors (including mlp.gate tensors without 'router' in name)
            if "mlp.gate" in name and "experts" not in name:
                yield from self._map_router(name, data_torch, bid)
                return
            if "router" in name:
                yield from self._map_router(name, data_torch, bid)
                return

            # Input norm (accept both input_norm and input_layernorm)
            if "input_norm" in name or "input_layernorm" in name:
                yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid), data_torch
                return
            # Post-attention layernorm — second RMSNorm before MoE in every layer
            if "post_attention_layernorm" in name:
                yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM_2, bid), data_torch
                return

            # Residual scaling — emit post-mlp scales with .mlp suffix
            if "residual_scale" in name or "post_attention_res_scale" in name or "post_mlp_res_scale" in name or "post_attention_residual_scale" in name or "post_mlp_residual_scale" in name:
                is_mlp = "post_mlp" in name
                for tensor_name, data in self._map_res_scale(name, data_torch, bid):
                    if is_mlp:
                        # Add .mlp suffix: blk.0.res_scale_hs.weight -> blk.0.res_scale_hs.mlp.weight
                        parts = tensor_name.rsplit('.', 1)
                        tensor_name = parts[0] + '.mlp.' + parts[1]
                    if tensor_name not in self._emitted:
                        self._emitted.add(tensor_name)
                        yield tensor_name, data
                return

            # Expert tensors — already pre-stacked in new checkpoints ([n_expert, ...])
            # or per-expert in old checkpoints (need stacking)
            if "experts" in name and ("zaya_block" in name or "mlp.experts" in name):
                assert bid is not None
                # New format: pre-stacked tensors like mlp.experts.down_proj [16, 2048, 2048]
                if "mlp.experts" in name:
                    if name.endswith("down_proj"):
                        yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid), data_torch
                    elif name.endswith("gate_up_proj"):
                        yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, bid), data_torch
                    return
                # Old format: per-expert tensors, need stacking (existing logic)
                if self._experts is None:
                    self._experts = {}
                if bid not in self._experts:
                    self._experts[bid] = {}
                self._experts[bid][name] = data_torch
                n_expert = self.find_hparam(["num_experts"])
                expected = n_expert * 2
                if len(self._experts[bid]) >= expected:
                    for w_name, gguf_tensor, permute_dims in [
                        ("linear_fc1", gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, None),
                        ("linear_fc2", gguf.MODEL_TENSOR.FFN_DOWN_EXP, None),
                    ]:
                        datas: list[Tensor] = []
                        for xid in range(n_expert):
                            ename = f"model.layers.{bid}.zaya_block.experts.local_experts.{xid}.{w_name}.weight"
                            datas.append(self._experts[bid][ename])
                            del self._experts[bid][ename]
                        data_torch_stacked = torch.stack(datas, dim=0)
                        if permute_dims is not None:
                            data_torch_stacked = data_torch_stacked.permute(*permute_dims)
                        yield self.format_tensor_name(gguf_tensor, bid), data_torch_stacked
                    del self._experts[bid]
                return

        # Fallback for any remaining tensors: use tensor_mapping
        try:
            yield from super().modify_tensors(data_torch, name, bid)
        except ValueError as e:
            if "Can not map tensor" in str(e):
                logger.warning(f"Skipping unmapped tensor: {name}")
            else:
                raise

    def set_vocab(self):
        from gguf.vocab import LlamaHfVocab

        vocab = LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) >= vocab.vocab_size, f"tokenizer has {len(tokens)} tokens, expected at least {vocab.vocab_size}"

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts:
            unprocessed = [k for d in self._experts.values() for k in d.keys()]
            if unprocessed:
                raise ValueError(f"Unprocessed expert tensors: {unprocessed}")
