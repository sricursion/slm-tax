"""GRPO (RLVR) on top of an SFT checkpoint, with a dense verifiable reward.

Run on a GPU node AFTER SFT (the base model can't produce the format cold).
Deps: pip install torch transformers trl peft datasets (+ vllm optional).

The reward is fully verifiable (slm_tax.score.reward): dense (process + by_line
+ chain) with a sparse strict bonus, so the group advantage has variance even
when no sampled completion is fully correct — avoiding the RLVR sparse-reward
collapse (all-wrong group -> zero gradient) that kills weak-base GRPO.

Examples:
  # continue from an SFT LoRA adapter (merged into base, fresh GRPO LoRA)
  python scripts/train_grpo.py --sft-adapter runs/pilot5-1.7b --base Qwen/Qwen3-1.7B \
      --data data/train_20k_v5.jsonl --limit 4000 --out runs/pilot5-grpo

  # from a full (merged) SFT model
  python scripts/train_grpo.py --model runs/qwen3-4b-sft --data data/train_100k_v5.jsonl \
      --out runs/qwen3-4b-grpo --num-generations 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Heavy deps (datasets/transformers/trl) are imported lazily inside the
# functions that need them, so the verifiable reward wrapper stays importable
# and unit-testable on a CPU box without the training stack.
from slm_tax.render import PROMPT_TEMPLATE  # noqa: E402
from slm_tax.score import reward as score_reward  # noqa: E402


def build_prompt(tokenizer, input_json: str) -> str:
    """Same templating as SFT/eval (thinking disabled) so train/RL/eval match."""
    user = PROMPT_TEMPLATE.format(input_data=input_json)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_rl_dataset(path: str, tokenizer, limit: int | None):
    """RL needs only the prompt + the oracle line values (to score rollouts).
    The oracle answer travels as a JSON string column so it survives batching."""
    from datasets import Dataset

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(
                {"prompt": build_prompt(tokenizer, r["input_json"]),
                 "lines_json": json.dumps(r["lines"])}
            )
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def _completion_text(comp) -> str:
    """TRL may hand back a string or a list of message dicts."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        return comp[0].get("content", "")
    return str(comp)


def make_reward(component: str):
    """Build a TRL reward_func for one component of score.reward.

    TRL calls reward_func(prompts=..., completions=..., **columns) where the
    dataset's extra columns (here `lines_json`) arrive as parallel lists.
    """

    def reward_func(completions, lines_json=None, **kwargs):
        scores = []
        n = len(completions)
        lj = lines_json if lines_json is not None else [None] * n
        for comp, oracle in zip(completions, lj):
            try:
                lines = json.loads(oracle) if oracle else {}
                scores.append(float(score_reward(_completion_text(comp), lines)[component]))
            except Exception:
                scores.append(0.0)
        return scores

    reward_func.__name__ = f"reward_{component}"
    return reward_func


def load_model(args):
    """Return a model ready for GRPO. If an SFT LoRA adapter is given, merge it
    into the base first so GRPO learns a fresh adapter on the SFT-tuned weights."""
    import torch

    if args.sft_adapter:
        from peft import AutoPeftModelForCausalLM

        print(f"merging SFT adapter {args.sft_adapter} into base for GRPO init")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.sft_adapter, torch_dtype=torch.bfloat16
        )
        return model.merge_and_unload()
    return args.model  # a model id / merged model dir; GRPOTrainer loads it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="merged model id/dir (no adapter)")
    ap.add_argument("--sft-adapter", default=None, help="SFT LoRA dir to merge as init")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B", help="base for --sft-adapter tokenizer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=4000, help="RL prompts (RL needs far fewer than SFT)")
    ap.add_argument("--reward", default="shaped", help="component of score.reward to optimize")
    ap.add_argument("--num-generations", type=int, default=8, help="rollouts per prompt (group size)")
    ap.add_argument("--max-completion-length", type=int, default=3072)
    ap.add_argument("--max-prompt-length", type=int, default=7168)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bsz", type=int, default=8, help="prompts per device step (x num_generations rollouts)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--use-vllm", action="store_true", help="fast rollouts via vLLM (if available)")
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 = no KL, common for RLVR)")
    args = ap.parse_args()

    if not args.model and not args.sft_adapter:
        sys.exit("provide --model (merged) or --sft-adapter (LoRA to merge)")
    # TRL requires the effective batch (bsz * accum, x num_processes) to be
    # divisible by num_generations — a common GRPO startup crash.
    if (args.bsz * args.accum) % args.num_generations != 0:
        sys.exit(
            f"bsz*accum ({args.bsz * args.accum}) must be divisible by "
            f"num_generations ({args.num_generations}); adjust --bsz/--accum"
        )

    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok_src = args.base if args.sft_adapter else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_rl_dataset(args.data, tokenizer, args.limit)
    print(f"RL prompts: {len(dataset)} | reward: {args.reward} | group size: {args.num_generations}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        num_train_epochs=args.epochs,
        beta=args.beta,  # KL to the SFT reference; 0.0 is common for verifiable RL
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        use_vllm=args.use_vllm,
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=load_model(args),
        reward_funcs=[make_reward(args.reward)],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
