import os

import transformers

from ..utils import utils
from ..utils import model_utils
from ..utils import data_utils
from ..utils import quant_utils
from ..utils import gptq_utils
from ..utils import hadamard_utils
from . import rotation_utils


def main():
    args = utils.parser_gen()

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.seqlen, args.hf_token)
    model.eval()

    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_dim = model.config.intermediate_size // args.tp
                had_K, K = hadamard_utils.get_hadK(had_dim)
                qlayers[name].had_dim = had_dim
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
            if 'o_proj' in name and args.tp == 1:   # For TP > 1, use head-wise Hadamard transform
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16:
        if not args.w_rtn: # GPTQ Weight Quantization
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)

    if args.save_qmodel_path:
        os.makedirs(args.save_qmodel_path, exist_ok=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
        tokenizer.save_pretrained(args.save_qmodel_path)
        # load model weights
        state_dict = model.state_dict()
        for name in list(state_dict.keys()):
            if ".module" in name or "quantizer" in name or "had" in name:
                del state_dict[name]
        model_transformers = model_utils.get_model(args.model, args.seqlen, args.hf_token)
        model_transformers.load_state_dict(state_dict, strict=True)
        model_transformers.save_pretrained(args.save_qmodel_path)
        if args.rotate:
            if isinstance(model, transformers.Qwen2ForCausalLM):
                model.config.architectures = ["Qwen2QuaRotForCausalLM"]
            elif isinstance(model, transformers.Qwen3ForCausalLM):
                model.config.architectures = ["Qwen3QuaRotForCausalLM"]
            elif isinstance(model, transformers.LlamaForCausalLM):
                model.config.architectures = ["LlamaQuaRotForCausalLM"]
            else:
                raise NotImplementedError
        elif args.k_bits < 16 or args.v_bits < 16:
            if isinstance(model, transformers.Qwen2ForCausalLM):
                model.config.architectures = ["Qwen2QuaRotKVForCausalLM"]
            elif isinstance(model, transformers.Qwen3ForCausalLM):
                model.config.architectures = ["Qwen3QuaRotKVForCausalLM"]
            elif isinstance(model, transformers.LlamaForCausalLM):
                model.config.architectures = ["LlamaQuaRotKVForCausalLM"]
            else:
                raise NotImplementedError
        model.config.fake_quant_config = {
            "tp": args.tp,
            "w_bits": args.w_bits,
            "w_clip": args.w_clip,
            "a_bits": args.a_bits,
            "a_asym": args.a_asym,
            "a_clip_ratio": args.a_clip_ratio,
            "k_bits": args.k_bits,
            "k_asym": args.k_asym,
            "k_groupsize": args.k_groupsize,
            "k_clip_ratio": args.k_clip_ratio,
            "v_bits": args.v_bits,
            "v_asym": args.v_asym,
            "v_groupsize": args.v_groupsize,
            "v_clip_ratio": args.v_clip_ratio,
        }
        model.config.save_pretrained(args.save_qmodel_path)
        print(f"Model saved at {args.save_qmodel_path}.")


if __name__ == '__main__':
    main()
