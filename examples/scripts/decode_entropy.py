import argparse
import json
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-1.7B"

# Example:
# uv run python examples/scripts/decode_entropy.py
#
# Tensor shape suffixes:
# B: batch size
# L: sequence length
# V: vocabulary size


def entropy_from_logits(logits_V: torch.Tensor) -> torch.Tensor:
    log_probs_V = torch.nn.functional.log_softmax(logits_V, dim=-1)
    probs_V = log_probs_V.exp()
    return -(probs_V * log_probs_V).sum(dim=-1)


def print_entropy_histogram(entropy_L: list[float], token_text_L: list[str], bins: int = 40, num_samples: int = 5) -> None:
    if not entropy_L:
        print("entropy histogram: no generated tokens")
        return

    min_entropy = min(entropy_L)
    max_entropy = max(entropy_L)
    if min_entropy == max_entropy:
        print("entropy histogram:")
        samples = ", ".join(json.dumps(token) for token in token_text_L[:num_samples])
        print(f"[{min_entropy:.4f}, {max_entropy:.4f}] {len(entropy_L)} {samples}")
        return

    bin_width = (max_entropy - min_entropy) / bins
    counts = [0] * bins
    token_samples_by_bin = [[] for _ in range(bins)]
    for entropy, token_text in zip(entropy_L, token_text_L):
        bin_index = min(int((entropy - min_entropy) / bin_width), bins - 1)
        counts[bin_index] += 1
        if len(token_samples_by_bin[bin_index]) < num_samples:
            token_samples_by_bin[bin_index].append(token_text)

    print("entropy histogram:")
    for bin_index, count in enumerate(counts):
        start = min_entropy + bin_index * bin_width
        end = start + bin_width
        samples = ", ".join(json.dumps(token) for token in token_samples_by_bin[bin_index])
        print(f"[{start:7.4f}, {end:7.4f}) {count} {samples}")


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": question + ' Let\'s think step by step and output the final answer after "####".',
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one GSM8K example and print per-token decode entropy from Qwen logits."
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model id.")
    parser.add_argument("--dataset", default="openai/gsm8k", help="Hugging Face dataset id.")
    parser.add_argument("--dataset-config", default="main", help="Hugging Face dataset config name.")
    parser.add_argument("--split", default="train", help="Dataset split to sample from.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selecting the GSM8K example.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of decode tokens.")
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample decode tokens instead of greedy decoding. Entropy is still computed from full logits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    example_index = rng.randrange(len(dataset))
    example = dataset[example_index]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda")
    model.eval()

    prompt = build_prompt(tokenizer, example["question"])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_token_id_BL = inputs["input_ids"].to("cuda")
    attention_mask_BL = inputs["attention_mask"].to("cuda")

    print(f"model: {args.model}")
    print(f"dataset: {args.dataset} {args.dataset_config} {args.split}[{example_index}]")
    print(f"question: {example['question']}")
    print()
    print("decode tokens:")

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_token_id_BL,
            attention_mask=attention_mask_BL,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_token_id_L = output.sequences[0, input_token_id_BL.shape[-1] :]
    entropy_L = []
    token_text_L = []
    for step, (token_id, logits_BV) in enumerate(zip(generated_token_id_L, output.scores), start=1):
        logits_V = logits_BV[0].float()
        entropy = entropy_from_logits(logits_V).item()
        entropy_L.append(entropy)
        token = tokenizer.decode([token_id.item()])
        token_text_L.append(token)
        # print(f"{step:04d}\ttoken={json.dumps(token)}\tentropy={entropy:.6f}")

    print_entropy_histogram(entropy_L, token_text_L)


if __name__ == "__main__":
    main()
