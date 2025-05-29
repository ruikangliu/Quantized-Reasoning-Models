# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""
import os
import re
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2MLP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer,
                                              is_pp_missing_parameter, maybe_prefix)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)

from vllm_custom.model_executor.layers.quantization.utils.fake_quant_utils import ActivationQuantizer
from vllm_custom.model_executor.layers.quantization.utils.flatquant_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix, InvSingleTransMatrix, InvDecomposeTransMatrix
from vllm_custom.model_executor.layers.quantization.utils.flatquant_utils import get_decompose_dim

logger = init_logger(__name__)


class Qwen3MLP(Qwen2MLP):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        fake_quant_config: dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config,
            prefix
        )
        tp_size = get_tensor_model_parallel_world_size()

        # fake quantization for activations
        self.up_gate_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
                                                lac=True, groupsize=-1, clip_ratio=None)
        self.down_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
                                            lac=True, groupsize=-1, clip_ratio=None)
        
        # online trans
        if fake_quant_config["direct_inv"]:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix
        if fake_quant_config["a_bits"] < 16 or fake_quant_config["w_bits"] < 16:
            up_dim_left, up_dim_right = get_decompose_dim(hidden_size)
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=False)
            self.up_gate_trans.to_eval_mode()
            down_dim_left, down_dim_right = get_decompose_dim(intermediate_size // tp_size)
            self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=False)
            self.down_trans.to_eval_mode()
        else:
            self.up_gate_trans, self.down_trans = None, None

    def forward(self, x):
        if self.up_gate_trans is not None:
            x = self.up_gate_trans(x)
        x = self.up_gate_quant(x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        if self.down_trans is not None:
            x = self.down_trans(x)
        x = self.down_quant(x)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 fake_quant_config: dict = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # fake quantization for activations
        self.qkv_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
                                             lac=True, groupsize=-1, clip_ratio=None)
        self.o_quant = ActivationQuantizer(bits=fake_quant_config["a_bits"], sym=not fake_quant_config["a_asym"],
                                            lac=True, groupsize=-1, clip_ratio=None)
        self.k_cache_quant = ActivationQuantizer(bits=fake_quant_config["k_bits"], sym=not fake_quant_config["k_asym"],
                                                lac=True, groupsize=fake_quant_config["k_groupsize"], clip_ratio=None)
        self.v_cache_quant = ActivationQuantizer(bits=fake_quant_config["v_bits"], sym=not fake_quant_config["v_asym"],
                                                lac=True, groupsize=fake_quant_config["v_groupsize"], clip_ratio=None)
        
        # online trans
        if fake_quant_config["direct_inv"]:
            SingleTransMatrix, DecomposeTransMatrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if fake_quant_config["a_bits"] < 16 or fake_quant_config["w_bits"] < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(hidden_size)
            self.ln_trans = DecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=False)
            self.ln_trans.to_eval_mode()
            self.o_trans = SingleTransMatrix(self.num_heads)
            self.o_trans.to_eval_mode()
        else:
            self.ln_trans, self.o_trans = None, None
        if fake_quant_config["k_bits"] < 16:
            self.kcache_trans = SingleTransMatrix(self.head_dim)
            self.kcache_trans.to_eval_mode()
        else:
            self.kcache_trans = None
        if fake_quant_config["v_bits"] < 16 and self.o_trans is None:
            self.vcache_trans = SingleTransMatrix(self.head_dim)
            self.vcache_trans.to_eval_mode()
        else:
            self.vcache_trans = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        hidden_states = self.qkv_quant(hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm.forward_native(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm.forward_native(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        if self.kcache_trans:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        k = self.k_cache_quant(k)
        v = self.v_cache_quant(v)
        attn_output = self.attn(q, k, v)
        if self.o_trans is not None:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
        elif self.vcache_trans is not None:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)).reshape(init_shape)
        attn_output = self.o_quant(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            fake_quant_config=config.fake_quant_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fake_quant_config=config.fake_quant_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3Model(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=Qwen3DecoderLayer)
        
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # load transform matrices & act clip
        rank = torch.distributed.get_rank()
        flat_parameters = torch.load(os.path.join(self.config.name_or_path, "flat_matrices.pth"), weights_only=True)
        for i in range(self.start_layer, self.end_layer):
            flat_param = flat_parameters[i]
            for name in list(flat_param.keys()):
                # substitute names
                if "self_attn.q_proj.act_quantizer" in name \
                    or "self_attn.k_proj.act_quantizer" in name \
                    or "self_attn.v_proj.act_quantizer" in name:
                    flat_param[re.sub("self_attn.*.act_quantizer", "self_attn.qkv_quant", name)] = flat_param[name]
                    del flat_param[name]
                if "mlp.up_proj.act_quantizer" in name \
                    or "mlp.gate_proj.act_quantizer" in name:
                    flat_param[re.sub("mlp.*.act_quantizer", "mlp.up_gate_quant", name)] = flat_param[name]
                    del flat_param[name]
                if "self_attn.k_cache_quantizer" in name:
                    flat_param[re.sub("self_attn.k_cache_quantizer", "self_attn.k_cache_quant", name)] = flat_param[name]
                    del flat_param[name]
                if "self_attn.v_cache_quantizer" in name:
                    flat_param[re.sub("self_attn.v_cache_quantizer", "self_attn.v_cache_quant", name)] = flat_param[name]
                    del flat_param[name]
                if "self_attn.o_proj.act_quantizer" in name:
                    flat_param[re.sub("self_attn.o_proj.act_quantizer", "self_attn.o_quant", name)] = flat_param[name][rank].unsqueeze(0)
                    del flat_param[name]
                if "mlp.down_proj.act_quantizer" in name:
                    flat_param[re.sub("mlp.down_proj.act_quantizer", "mlp.down_quant", name)] = flat_param[name][rank].unsqueeze(0)
                    del flat_param[name]
                if f"mlp.down_trans.trans_list.{rank}" in name:
                    flat_param[name.replace(f"mlp.down_trans.trans_list.{rank}", "mlp.down_trans")] = flat_param[name]
                    del flat_param[name]
                if f"self_attn.o_trans.trans_list.{rank}" in name:
                    flat_param[name.replace(f"self_attn.o_trans.trans_list.{rank}", "self_attn.o_trans")] = flat_param[name]
                    del flat_param[name]
            loaded_results = self.layers[i].load_state_dict(flat_param, strict=False)
            matched_keys = set(flat_param.keys()) - set(loaded_results.unexpected_keys)
            for key in matched_keys:
                loaded_params.add(f"layers.{i}.{key}")

        return loaded_params


class Qwen3FlatQuantForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
