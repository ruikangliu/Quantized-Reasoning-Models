import copy
import functools
import types

import torch

from ..utils import model_utils
from ..utils import utils


@torch.no_grad()
def get_acts(model, dataset, num_tokens=None, type="linear", keep_seqlen_dim=False, return_abs=True):
    # linear -> input activations of linear layers
    # block -> output activations of Transformer layers
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    device = next(model.parameters()).device

    if type == "linear":
        target_layer_types = [
            torch.nn.Linear,
        ]
    elif type == "block":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        target_layer_types = [
            LlamaDecoderLayer,
            Qwen2DecoderLayer,
            Qwen3DecoderLayer,
        ]
    else:
        raise NotImplementedError

    abs_acts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        if not keep_seqlen_dim:
            tensor = tensor.view(-1, hidden_dim)
        if return_abs:
            tensor = tensor.abs()
        tensor = tensor.detach().cpu()
        if name in abs_acts:
            abs_acts[name] = torch.cat([abs_acts[name], tensor], dim=0)
        else:
            abs_acts[name] = tensor

    def stat_act_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        if type == "linear":
            stat_tensor(name, x)
            if "k_proj" in name or "v_proj" in name:
                stat_tensor(f"{name}.out", y)
        elif type == "block":
            stat_tensor(name, y)

    hooks = []
    for name, m in model.named_modules():
        if any([isinstance(m, target_layer_type) for target_layer_type in target_layer_types]) and not "lm_head" in name:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_act_hook, name=name))
            )

    i = 0
    tokens = 0
    while tokens < num_tokens:
        max_length = num_tokens - tokens
        input_ids = dataset[i][0][:, :max_length].to(device)
        model(input_ids, output_attentions=False)
        i += 1
        tokens += input_ids.shape[1]

    for h in hooks:
        h.remove()

    model.config.use_cache = use_cache
    return abs_acts


@torch.no_grad()
def get_attn(model, dataset, num_tokens):
    model.eval()
    device = next(model.parameters()).device

    input_ids = dataset[0][0][:, :num_tokens].to(device)
    attns = model(input_ids, output_attentions=True).attentions
    attns = [attn.mean(dim=1).squeeze().float().cpu().numpy() for attn in attns]

    return attns


def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_wrapper_after_function_call_in_method(module, method_name, function_name, wrapper_fn):
    '''
    This function adds a wrapper after the output of a function call in the method named `method_name`. 
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    '''

    original_method = getattr(module, method_name).__func__
    method_globals = dict(original_method.__globals__)
    wrapper = wrapper_fn(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper


class QKRoPEWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.func = func

    def forward(self, *args, **kwargs):
        self.kcache_before_rope = args[1].clone().detach().cpu()
        q, k = self.func(*args, **kwargs)
        self.kcache_after_rope = k.clone().detach().cpu()
        return q, k


def add_qk_rope_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    attr_name = f"{function_name}_qk_rope_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = add_wrapper_after_function_call_in_method(module, "forward",
                                                        function_name, functools.partial(QKRoPEWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)


@torch.no_grad()
def get_kcache(model, dataset, num_tokens=None, keep_seqlen_dim=False):
    model.eval()

    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        add_qk_rope_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config)

    utils.distribute_model(model)
    device = next(model.parameters()).device

    kcache_before_rope_dict, kcache_after_rope_dict = {}, {}
    i = 0
    tokens = 0
    while tokens < num_tokens:
        max_length = num_tokens - tokens
        input_ids = dataset[i][0][:, :max_length].to(device)
        model(input_ids, output_attentions=False)
        i += 1
        tokens += input_ids.shape[1]

        for name, module in model.named_modules():
            if isinstance(module, QKRoPEWrapper):
                layer = int(name.split(".")[2])
                bsz, num_heads, k_len, head_dim = module.kcache_before_rope.shape
                if keep_seqlen_dim:
                    kcache_before_rope = module.kcache_before_rope.permute(0, 2, 1, 3).reshape(bsz, k_len, -1)
                    kcache_after_rope = module.kcache_after_rope.permute(0, 2, 1, 3).reshape(bsz, k_len, -1)
                else:
                    kcache_before_rope = module.kcache_before_rope.permute(0, 2, 1, 3).reshape(bsz*k_len, -1)
                    kcache_after_rope = module.kcache_after_rope.permute(0, 2, 1, 3).reshape(bsz*k_len, -1)
                if name in kcache_before_rope_dict:
                    kcache_before_rope_dict[name] = torch.cat([kcache_before_rope_dict[name], kcache_before_rope], dim=0)
                    kcache_after_rope_dict[name] = torch.cat([kcache_after_rope_dict[name], kcache_after_rope], dim=0)
                else:
                    kcache_before_rope_dict[name] = kcache_before_rope
                    kcache_after_rope_dict[name] = kcache_after_rope

    # compute K cache before bias
    kcache_before_bias_dict = {}
    for name, kcache_before_rope in kcache_before_rope_dict.items():
        layer_idx = int(name.split(".")[2])
        bias = model.model.layers[layer_idx].self_attn.k_proj.bias.to(kcache_before_rope)
        kcache_before_bias_dict[name] = kcache_before_rope - bias

    # return absolute values
    kcache_before_rope_dict = {k:v.abs() for k, v in kcache_before_rope_dict.items()}
    kcache_after_rope_dict = {k:v.abs() for k, v in kcache_after_rope_dict.items()}
    kcache_before_bias_dict = {k:v.abs() for k, v in kcache_before_bias_dict.items()}

    return kcache_before_bias_dict, kcache_before_rope_dict, kcache_after_rope_dict

