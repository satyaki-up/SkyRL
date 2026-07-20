"""Distill reasoning traces without teacher logits.

This script:
1. Loads a teacher model (default: Qwen/Qwen3-8B).
2. Builds math prompts from Hugging Face math datasets.
3. Generates teacher reasoning traces.
4. Fine-tunes a student model (default: Qwen/Qwen2-0.5B-Instruct) on those traces.
5. Evaluates the distilled student on GSM8K.

Example:
    WANDB_API_KEY=... uv run --extra dev python examples/scripts/distill_without_logits.py --num-train-examples 20 --max-train-steps 10 --eval-limit 500

Defaults are sized for a single A100/H100 GSM8K distillation run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from math_verify import parse as math_verify_parse
    from math_verify import verify as math_verify

    HAS_MATH_VERIFY = True
except ImportError:
    math_verify_parse = None
    math_verify = None
    HAS_MATH_VERIFY = False


TEACHER_MODEL = "Qwen/Qwen3-8B"
STUDENT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
ANSWER_RE = re.compile(r"####\s*(-?(?:\d+)(?:\.\d+)?)")


@dataclass
class Example:
    question: str
    answer: str | None
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    parser.add_argument("--student-model", default=STUDENT_MODEL)
    parser.add_argument("--output-dir", default="outputs/distill_without_logits")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-train-examples", type=int, default=7473)
    parser.add_argument("--gsm8k-frac", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--reuse-traces", action="store_true")

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-train-steps", type=int, default=1869)

    parser.add_argument("--eval-limit", type=int, default=1319)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        help="Optional transformers attention backend. Use flash_attention_2/3 only if installed.",
    )

    parser.add_argument("--wandb-run-name", default="qwen3-8b-to-qwen2-0.5b-gsm8k")
    return parser.parse_args()


def setup_wandb(args: argparse.Namespace):
    if not os.environ.get("WANDB_API_KEY"):
        return None

    import wandb

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="distill", name=args.wandb_run_name, config=vars(args))
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.define_metric("data/*", step_metric="global_step")
    return run


def extract_gsm8k_answer(answer: str) -> str | None:
    match = ANSWER_RE.search(answer)
    if not match:
        return None
    return normalize_number(match.group(1))


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.replace(",", "").strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1]
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return str(number)


def extract_model_answer(text: str) -> str | None:
    if "####" in text:
        return normalize_number(text.split("####")[-1])
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    return normalize_number(text)


def verify_answer(gold: str | None, prediction: str) -> bool:
    if gold is None:
        return False

    if HAS_MATH_VERIFY:
        try:
            gold_parsed = math_verify_parse(str(gold))
            prediction_parsed = math_verify_parse(prediction)
            return bool(math_verify(gold_parsed, prediction_parsed))
        except Exception:
            pass

    return extract_model_answer(prediction) == normalize_number(gold)


def load_math_examples(num_examples: int, gsm8k_frac: float, seed: int) -> list[Example]:
    rng = random.Random(seed)
    gsm_count = min(num_examples, max(0, int(num_examples * gsm8k_frac)))
    other_count = max(0, num_examples - gsm_count)

    examples: list[Example] = []

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_indices = rng.sample(range(len(gsm)), k=min(gsm_count, len(gsm)))
    for idx in gsm_indices:
        row = gsm[idx]
        examples.append(
            Example(
                question=row["question"],
                answer=extract_gsm8k_answer(row["answer"]),
                source="openai/gsm8k",
            )
        )

    if other_count:
        try:
            math_ds = load_dataset("hendrycks/competition_math", split="train")
            math_indices = rng.sample(range(len(math_ds)), k=min(other_count, len(math_ds)))
            for idx in math_indices:
                row = math_ds[idx]
                examples.append(
                    Example(
                        question=row["problem"],
                        answer=extract_model_answer(row.get("solution", "")),
                        source="hendrycks/competition_math",
                    )
                )
        except Exception as exc:
            print(f"Warning: could not load hendrycks/competition_math ({exc}); using GSM8K only.")

    rng.shuffle(examples)
    return examples[:num_examples]


def make_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Solve the problem step by step. End your response with the final answer in the format #### <answer>."
            ),
        }
    ]


def apply_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    kwargs = {}
    # Qwen3 tokenizers accept this; older Qwen2 tokenizers may not.
    if "Qwen3" in getattr(tokenizer, "name_or_path", ""):
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    except (TypeError, ValueError):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except (TypeError, ValueError):
            rendered = ""
            for message in messages:
                rendered += f"{message['role'].title()}: {message['content']}\n"
            if add_generation_prompt:
                rendered += "Assistant:"
            return rendered


def model_kwargs(args: argparse.Namespace, dtype: torch.dtype) -> dict:
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation is not None:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


@torch.inference_mode()
def generate_teacher_traces(args: argparse.Namespace, trace_path: Path) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map="auto",
        **model_kwargs(args, dtype),
    )
    model.eval()

    examples = load_math_examples(args.num_train_examples, args.gsm8k_frac, args.seed)
    rows: list[dict] = []

    for start in tqdm(range(0, len(examples), args.teacher_batch_size), desc="Generating teacher traces"):
        batch = examples[start : start + args.teacher_batch_size]
        prompts = [apply_chat(tokenizer, make_prompt(ex.question), add_generation_prompt=True) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        for ex, completion in zip(batch, completions):
            completion = completion.strip()
            if not completion:
                continue
            rows.append(
                {
                    "messages": make_prompt(ex.question) + [{"role": "assistant", "content": completion}],
                    "question": ex.question,
                    "teacher_answer": extract_model_answer(completion),
                    "gold_answer": ex.answer,
                    "source": ex.source,
                }
            )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def load_or_generate_traces(args: argparse.Namespace, trace_path: Path) -> list[dict]:
    if args.reuse_traces and trace_path.exists():
        with trace_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    return generate_teacher_traces(args, trace_path)


def tokenize_distill_row(row: dict, tokenizer, max_length: int) -> dict | None:
    prompt_text = apply_chat(tokenizer, row["messages"][:-1], add_generation_prompt=True)
    full_text = apply_chat(tokenizer, row["messages"], add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        response_start = max(0, len(prompt_ids) - overflow)
    else:
        response_start = len(prompt_ids)

    labels = [-100] * len(full_ids)
    response_start = min(response_start, len(full_ids))
    for i in range(response_start, len(full_ids)):
        labels[i] = full_ids[i]

    if all(label == -100 for label in labels):
        return None

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class DistillCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        for feat in features:
            pad = max_len - len(feat["input_ids"])
            input_ids.append([self.tokenizer.pad_token_id] * pad + feat["input_ids"])
            attention_mask.append([0] * pad + feat["attention_mask"])
            labels.append([-100] * pad + feat["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_student(
    args: argparse.Namespace, traces: list[dict], wandb_run
) -> tuple[AutoModelForCausalLM, AutoTokenizer, int, dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = [tokenize_distill_row(row, tokenizer, args.max_length) for row in traces]
    tokenized = [row for row in tokenized if row is not None]
    if not tokenized:
        raise RuntimeError("No usable distillation traces after tokenization.")

    dataset = Dataset.from_list(tokenized)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=DistillCollator(tokenizer),
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        **model_kwargs(args, dtype),
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.max_train_steps)

    last_eval_metrics = evaluate_gsm8k(args, model, tokenizer, wandb_run, step=0)
    model.train()

    micro_step = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_train_steps, desc="Training student")

    while optimizer_step < args.max_train_steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                metrics = {
                    "train/loss": loss.detach().float().item() * args.grad_accum_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": float(grad_norm),
                }
                if wandb_run is not None:
                    wandb_run.log({"global_step": optimizer_step, **metrics})
                progress.update(1)

                if args.eval_interval > 0 and optimizer_step % args.eval_interval == 0:
                    last_eval_metrics = evaluate_gsm8k(args, model, tokenizer, wandb_run, step=optimizer_step)
                    model.train()

                if optimizer_step >= args.max_train_steps:
                    break
        else:
            continue
        break

    progress.close()
    if args.save_model:
        save_dir = Path(args.output_dir) / "student"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    return model, tokenizer, optimizer_step, last_eval_metrics


def iter_gsm8k_eval(limit: int | None) -> Iterable[Example]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    for row in ds:
        yield Example(question=row["question"], answer=extract_gsm8k_answer(row["answer"]), source="openai/gsm8k")


@torch.inference_mode()
def evaluate_gsm8k(
    args: argparse.Namespace,
    model,
    tokenizer,
    wandb_run,
    step: int | None = None,
) -> dict[str, float]:
    model.eval()
    examples = list(iter_gsm8k_eval(args.eval_limit))
    correct = 0
    total = 0

    for start in tqdm(range(0, len(examples), args.eval_batch_size), desc="Evaluating GSM8K"):
        batch = examples[start : start + args.eval_batch_size]
        prompts = [apply_chat(tokenizer, make_prompt(ex.question), add_generation_prompt=True) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.eval_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        for ex, completion in zip(batch, completions):
            correct += int(verify_answer(ex.answer, completion))
            total += 1

    metrics = {"eval/gsm8k_accuracy": correct / max(total, 1), "eval/gsm8k_total": float(total)}
    if wandb_run is not None:
        wandb_run.log({"global_step": step, **metrics})
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_run = setup_wandb(args)
    if not HAS_MATH_VERIFY:
        print(
            "Warning: math-verify is not installed; falling back to simple numeric answer matching. "
            "Install with: pip install 'math-verify[antlr4_13_2]'"
        )
    trace_path = output_dir / "teacher_traces.jsonl"

    traces = load_or_generate_traces(args, trace_path)
    if wandb_run is not None:
        wandb_run.log({"global_step": 0, "data/num_traces": len(traces)})

    model, tokenizer, final_step, last_eval_metrics = train_student(args, traces, wandb_run)
    if args.eval_interval <= 0 or final_step % args.eval_interval != 0:
        metrics = evaluate_gsm8k(args, model, tokenizer, wandb_run, step=final_step)
    else:
        metrics = last_eval_metrics
    print(json.dumps(metrics, indent=2))

    with (output_dir / "eval_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
