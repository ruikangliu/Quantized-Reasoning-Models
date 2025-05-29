import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import transformers


def read_jsonl(file_name):
    with open(file_name, mode='r') as reader:
        data = json.load(reader)
    return data


def get_num_steps(generated_text, step_token="\n\n"):
    return len(generated_text.split(step_token))


def get_num_tokens(tokenizer, generated_text):
    return len(tokenizer.encode(generated_text))


def get_length_stats(data, tokenizer=None):
    num_steps = []
    num_tokens = []
    for item in data:
        num_steps.append(get_num_steps(item["generated_text"]))
        if tokenizer is not None:
            num_tokens.append(get_num_tokens(tokenizer, item["generated_text"]))
    avg_steps = sum(num_steps) / len(num_steps)
    if tokenizer is not None:
        avg_tokens = sum(num_tokens) / len(num_tokens)
    else:
        avg_tokens = None
    return avg_steps, avg_tokens


def get_accuracy_stats(data):
    metric_name = list(data[0]["metrics"].keys())[0]
    correct_count = sum(1 for item in data if item["metrics"][metric_name])
    total_count = len(data)
    acc = correct_count / total_count * 100
    return acc


def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', type=str, default="acc", choices=["acc", "length"],
                        help='Compute model accuracy or response length.')
    parser.add_argument('--output_dir', type=str, default="./outputs/inference",
                        help='Path to the inference results.')
    parser.add_argument('--models', type=str, nargs='+',
                        default=[
                            "DeepSeek-R1-Distill-Qwen-1.5B",
                            "DeepSeek-R1-Distill-Qwen-7B",
                            "DeepSeek-R1-Distill-Qwen-14B",
                            "DeepSeek-R1-Distill-Qwen-32B",
                            "QwQ-32B",
                            "DeepSeek-R1-Distill-Llama-8B",
                            "DeepSeek-R1-Distill-Llama-70B",
                        ],
                        help='Evaluated models.')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=[
                            "", "awq-w4g128", "gptq-w4g128", "awq-w3g128", "gptq-w3g128",
                            "kvquant_star-kv4", "quarot-kv4", "kvquant_star-kv3", "quarot-kv3",
                            "smoothquant-w8a8kv8", "quarot-w8a8kv8", "flatquant-w8a8kv8", "mxfp4-w4a4kv4", "quarot-w4a4kv4", "flatquant-w4a4kv4",
                        ],
                        help='Evaluated quantization methods.')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=["AIME-120", "AIME-90", "AIME-2025", "MATH-500", "GSM8K", "GPQA-Diamond", "LiveCodeBench"],
                        help='Evaluation datasets.')
    parser.add_argument('--seeds', type=str, nargs='+',
                        default=["42", "43", "44"],
                        help='Evaluation seeds. Statisticas are averaged over different seeds.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_gen()

    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # quantized_model -> dataset -> seed -> stat

    # read evaluation results from json file
    for model in tqdm(args.models):
        tp = 8 if "70B" in model else 4
        if "QwQ" in model:
            modelzoo_dir = "./modelzoo/QwQ" 
        elif "Qwen3" in model:
            modelzoo_dir = "./modelzoo/Qwen3" 
        else:
            modelzoo_dir = "./modelzoo/DeepSeek-R1"
        tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(modelzoo_dir, model))
        for method in args.methods:
            if method == '':
                quantized_model = model
            elif 'quantized' in method:
                quantized_model = "-".join([model, method])
            else:
                quantized_model = "-".join([model, method, f"tp{tp}"])
            for seed in args.seeds:
                full_name = "-".join([quantized_model, f"seed{seed}"])
                for dataset in args.datasets:
                    file_path = os.path.join(args.output_dir, full_name, f"{dataset}.jsonl")
                    if os.path.exists(file_path):
                        data = read_jsonl(file_path)
                        if args.stats == "acc":
                            stats[quantized_model][dataset][seed] = get_accuracy_stats(data)
                        elif args.stats == "length":
                            stats[quantized_model][dataset][seed] = get_length_stats(data)[0]
                            # stats[quantized_model][dataset][seed] = get_length_stats(data, tokenizer)[1]  # very slow
                    else:
                        stats[quantized_model][dataset][seed] = None

    # compute stats for AIME-120 and Avg.
    for quantized_model in stats.keys():
        for seed in args.seeds:
            if stats[quantized_model]["AIME-90"][seed] is not None and \
                stats[quantized_model]["AIME-2025"][seed] is not None:
                stats[quantized_model]["AIME-120"][seed] = 0.75 * stats[quantized_model]["AIME-90"][seed] + 0.25 * stats[quantized_model]["AIME-2025"][seed]
            else:
                stats[quantized_model]["AIME-120"][seed] = None

            avg_stat = [stats[quantized_model][dataset][seed] for dataset in stats[quantized_model].keys() if not dataset in ["AIME-90", "AIME-2025", "Avg."]]
            if not None in avg_stat:
                stats[quantized_model]["Avg."][seed] = sum(avg_stat) / len(avg_stat)
            else:
                stats[quantized_model]["Avg."][seed] = None
        del stats[quantized_model]["AIME-90"]
        del stats[quantized_model]["AIME-2025"]

    # print avg stat & std
    stats_avg = defaultdict(lambda: defaultdict(int))  # quantized_model -> dataset -> stat
    stats_std = defaultdict(lambda: defaultdict(int))  # quantized_model -> dataset -> std
    quantized_models = stats.keys()
    for quantized_model in quantized_models:
        datasets = stats[quantized_model].keys()
        for dataset in datasets:
            seed_stat = [stats[quantized_model][dataset][seed] for seed in args.seeds]
            if not None in seed_stat:
                stats_avg[quantized_model][dataset] = np.mean(seed_stat)
                stats_std[quantized_model][dataset] = np.std(seed_stat, ddof=1)
            else:
                stats_avg[quantized_model][dataset] = None
                stats_std[quantized_model][dataset] = None

    # print Markdown table (stat)
    head_str = "Acc" if args.stats == "acc" else "#Steps/Tokens"
    print(f"| Quantized Model ({head_str}) | " + " | ".join(datasets) + " |")
    print("|------------------|" + "|".join([":---:"] * len(datasets)) + "|")
    for quantized_model in quantized_models:
        row = [quantized_model]
        for dataset in datasets:
            if stats_avg[quantized_model][dataset] is not None:
                row.append(f"{stats_avg[quantized_model][dataset]:.2f}")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")
    print()

    # print Markdown table (stat std)
    print(f"| Quantized Model ({head_str} Std) | " + " | ".join(datasets) + " |")
    print("|------------------|" + "|".join([":---:"] * len(datasets)) + "|")
    for quantized_model in quantized_models:
        row = [quantized_model]
        for dataset in datasets:
            if stats_std[quantized_model][dataset] is not None:
                row.append(f"{stats_std[quantized_model][dataset]:.2f}")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")
    print()
