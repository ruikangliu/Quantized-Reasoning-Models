import os
import logging

import torch
import transformers


LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
QWEN2_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN2_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
QWEN3_MODEL = transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM
QWEN3_LAYER = transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif isinstance(model, QWEN3_MODEL):
        return QWEN3_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, QWEN2_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, QWEN3_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    elif isinstance(model, QWEN2_MODEL):
        return model.model.layers
    elif isinstance(model, QWEN3_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, seqlen, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = seqlen
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_qwen2(model_name, seqlen, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError:
        logging.error("Qwen2 model is not available in this version of 'transformers'. Please update the library.")
        raise ImportError("Qwen2 model is not available. Ensure you're using a compatible version of the 'transformers' library.")

    config = transformers.Qwen2Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = Qwen2ForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.bfloat16,    # TODO
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = seqlen
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')

    return model


def get_qwen3(model_name, seqlen, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    try:
        from transformers import Qwen3ForCausalLM
    except ImportError:
        logging.error("Qwen3 model is not available in this version of 'transformers'. Please update the library.")
        raise ImportError("Qwen3 model is not available. Ensure you're using a compatible version of the 'transformers' library.")

    config = transformers.Qwen3Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = Qwen3ForCausalLM.from_pretrained(model_name,
                                                          torch_dtype=torch.bfloat16,    # TODO
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = seqlen
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')

    return model


def get_model(
    model_name, seqlen=2048, hf_token=None
):
    if 'llama' in model_name.lower():
        return get_llama(model_name, seqlen, hf_token)
    elif "qwen3" in model_name.lower():
        return get_qwen3(model_name, seqlen, hf_token)
    elif 'QwQ' in model_name or "qwen" in model_name.lower():
        return get_qwen2(model_name, seqlen, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN2_MODEL):
        model_type = QWEN2_MODEL
    elif isinstance(model, QWEN3_MODEL):
        model_type = QWEN3_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type


def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == QWEN2_MODEL:
        return [model.model.embed_tokens]
    elif model_type == QWEN3_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == QWEN2_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == QWEN3_MODEL:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    elif model_type == QWEN2_MODEL:
        return model.lm_head
    elif model_type == QWEN3_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm)
    elif model_type == QWEN3_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm


def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == QWEN2_MODEL:
        return model.config.intermediate_size
    elif model_type == QWEN3_MODEL:
        return model.config.intermediate_size
    else:
        raise ValueError(f'Unknown model type {model_type}')


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
