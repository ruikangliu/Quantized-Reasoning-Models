import os

import torch
import transformers

from ..utils import utils
from ..utils import model_utils
from ..utils import gptq_utils
from ..utils import data_utils
from . import smoothquant_utils


def main():
    args = utils.parser_gen()

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.seqlen, args.hf_token)
    model.eval()

    # get SQ scales from Pile dataset
    act_scales_path = os.path.join(f"{args.save_qmodel_path}/act_scales.pt")
    if not os.path.exists(act_scales_path):
        utils.distribute_model(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
        act_scales = smoothquant_utils.get_act_scales(
            args, model, tokenizer, args.cal_dataset, num_samples=args.nsamples, seq_len=args.seqlen
        )
        os.makedirs(args.save_qmodel_path, exist_ok=True)
        torch.save(act_scales, act_scales_path)
        print(f"SQ scales saved at {act_scales_path}")
        
        model = model_utils.get_model(args.model, args.seqlen, args.hf_token)
        model.eval()
        utils.cleanup_memory()
    else:
        act_scales = torch.load(act_scales_path, weights_only=True)

    # merge scales
    smoothquant_utils.smooth_lm(model, act_scales, alpha=args.smooth_alpha)

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
        model.save_pretrained(args.save_qmodel_path)
        if isinstance(model, transformers.Qwen2ForCausalLM):
            model.config.architectures = ["Qwen2FakeQuantizedForCausalLM"]
        elif isinstance(model, transformers.Qwen3ForCausalLM):
            model.config.architectures = ["Qwen3FakeQuantizedForCausalLM"]
        elif isinstance(model, transformers.LlamaForCausalLM):
            model.config.architectures = ["LlamaFakeQuantizedForCausalLM"]
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
