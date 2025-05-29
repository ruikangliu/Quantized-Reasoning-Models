import os

import torch
import transformers

from ..utils import data_utils
from .flatquant import utils
from .flatquant import args_utils
from .flatquant import model_utils
from .flatquant import eval_utils
from .flatquant import train_utils
from .flatquant import flat_utils
from . import gptq_utils


def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.seqlen, args.hf_token)
    model.eval()

    # get calibration data
    trainloader = data_utils.get_loaders(
        args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    logger.info("Finished loading training data.")

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            if args.resume_training:
                _, start_layer_idx = flat_utils.resume_training(args, model)
            else:
                start_layer_idx = 0
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger, start_layer_idx=start_layer_idx)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(model)
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    if args.save_qmodel_path:
        os.makedirs(args.save_qmodel_path, exist_ok=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
        tokenizer.save_pretrained(args.save_qmodel_path)
        # save flatquant params
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_matrices.pth"), weights_only=True)
        flat_matrices_path = os.path.join(args.save_qmodel_path, f"flat_matrices.pth")
        torch.save(flat_parameters, flat_matrices_path)
        # load model weights
        state_dict = model.state_dict()
        for name in list(state_dict.keys()):
            if "clip_factor" in name or "trans" in name or "quantizer" in name:
                del state_dict[name]
            elif ".linear" in name:
                state_dict[name.replace(".linear", "")] = state_dict[name]
                del state_dict[name]
        model_transformers, _ = model_utils.get_model(args.model, args.seqlen, args.hf_token)
        model_transformers.load_state_dict(state_dict, strict=True)
        model_transformers.save_pretrained(args.save_qmodel_path)
        if isinstance(model, transformers.Qwen2ForCausalLM):
            model.config.architectures = ["Qwen2FlatQuantForCausalLM"]
        elif isinstance(model, transformers.Qwen3ForCausalLM):
            model.config.architectures = ["Qwen3FlatQuantForCausalLM"]
        elif isinstance(model, transformers.LlamaForCausalLM):
            model.config.architectures = ["LlamaFlatQuantForCausalLM"]
        else:
            raise NotImplementedError
        model.config.fake_quant_config = {
            "tp": args.tp,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "a_asym": args.a_asym,
            "k_bits": args.k_bits,
            "k_asym": args.k_asym,
            "k_groupsize": args.k_groupsize,
            "v_bits": args.v_bits,
            "v_asym": args.v_asym,
            "v_groupsize": args.v_groupsize,
            "lwc": args.lwc,
            "lac": args.lac,
            "direct_inv": args.direct_inv
        }
        model.config.save_pretrained(args.save_qmodel_path)
        print(f"Model saved at {args.save_qmodel_path}.")

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    # Evaluating PPL
    for eval_dataset in ["wikitext2"]:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
                eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                hf_token=args.hf_token,
                eval_mode=True
            )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)


if __name__ == '__main__':
    main()
