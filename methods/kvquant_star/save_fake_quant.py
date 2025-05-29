import os

import torch
import transformers

from ..utils import utils
from ..utils import model_utils
from . import rope_utils


def main():
    args = utils.parser_gen()

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.seqlen, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    
    # get min&max of pre-RoPE K -> #layers x {"scale": [dim, 1], "zero": [dim, 1]}
    pre_rope_k_scale_zero = rope_utils.get_pre_rope_k_stats(args, model, tokenizer, args.cal_dataset,
                                                            num_samples=args.nsamples, seq_len=args.seqlen)

    if args.save_qmodel_path:
        os.makedirs(args.save_qmodel_path, exist_ok=True)
        tokenizer.save_pretrained(args.save_qmodel_path)
        # load model weights
        state_dict = model.state_dict()
        for name in list(state_dict.keys()):
            if ".module" in name:
                del state_dict[name]
        model_transformers = model_utils.get_model(args.model, args.seqlen, args.hf_token)
        model_transformers.load_state_dict(state_dict, strict=True)
        model_transformers.save_pretrained(args.save_qmodel_path)
        if isinstance(model, transformers.Qwen2ForCausalLM):
            model.config.architectures = ["Qwen2KVQuantStarForCausalLM"]
        elif isinstance(model, transformers.Qwen3ForCausalLM):
            model.config.architectures = ["Qwen3KVQuantStarForCausalLM"]
        elif isinstance(model, transformers.LlamaForCausalLM):
            model.config.architectures = ["LlamaKVQuantStarForCausalLM"]
        else:
            raise NotImplementedError
        torch.save(pre_rope_k_scale_zero, f"{args.save_qmodel_path}/pre_rope_k_scale_zero.pth")
        model.config.fake_quant_config = {
            "tp": args.tp,
            "k_bits": args.k_bits,
            "k_asym": args.k_asym,
            "k_scale_path": f"{args.save_qmodel_path}/pre_rope_k_scale_zero.pth",
            "k_pre_bias": args.k_pre_bias,
            "v_bits": args.v_bits,
            "v_asym": args.v_asym,
            "v_groupsize": args.v_groupsize,
        }
        model.config.save_pretrained(args.save_qmodel_path)
        print(f"Model saved at {args.save_qmodel_path}.")


if __name__ == '__main__':
    main()
