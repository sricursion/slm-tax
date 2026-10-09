"""SFT a small Qwen on oracle-labeled tax returns.

Run on a GPU node (not the dev box). Deps beyond this repo:
    pip install torch transformers trl peft datasets

Examples:
  # L40 pilot: LoRA on Qwen3-1.7B, 20k examples
  python scripts/train_sft.py --model Qwen/Qwen3-1.7B --data data/train_20k.jsonl \
      --lora --epochs 2 --out runs/pilot-1.7b-lora

  # H100 headline: full fine-tune Qwen3-4B, 100k examples
  python scripts/train_sft.py --model Qwen/Qwen3-4B --data data/train_100k.jsonl \
      --epochs 2 --bsz 4 --accum 8 --lr 1e-5 --out runs/qwen3-4b-full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import Dataset
from transformers import AutoTokenizer

from slm_tax.render import PROMPT_TEMPLATE


def load_rows(path: str, tokenizer, curriculum: bool = False) -> Dataset:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if curriculum:
        # Literature-backed easy->hard ordering (two-stage/DMT-style). Only
        # meaningful without shuffling — pass shuffle=False in the trainer or
        # rely on the curriculum being an early-epoch bias.
        rows.sort(key=lambda r: r.get("tier", 0))
    prompts, completions = [], []
    for row in rows:
        user_prompt = PROMPT_TEMPLATE.format(input_data=row["input_json"])
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(templated)
        completions.append(row["target"] + tokenizer.eos_token)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bsz", type=int, default=2, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    # NOTE: --curriculum sorts easy->hard, but HF Trainer shuffles the train set
    # by default; honoring the order needs a SequentialSampler override, so
    # verify on the node before relying on it (pending experiment P6).
    ap.add_argument("--curriculum", action="store_true", help="sort train data easy->hard by tier")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42, help="training seed (E2 replicas)")
    args = ap.parse_args()

    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = load_rows(args.data, tokenizer, curriculum=args.curriculum)
    eval_ds = load_rows(args.val_data, tokenizer) if args.val_data else None
    print(f"train examples: {len(train_ds)}")

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    # TRL >=1.x removed warmup_ratio from SFTConfig; convert to steps there.
    import inspect
    import math

    _sig = set(inspect.signature(SFTConfig.__init__).parameters)
    if "warmup_ratio" in _sig:
        warmup_kw = {"warmup_ratio": args.warmup}
    else:
        steps_per_epoch = math.ceil(len(train_ds) / (args.bsz * args.accum))
        total_steps = math.ceil(steps_per_epoch * args.epochs)
        warmup_kw = {"warmup_steps": max(1, round(args.warmup * total_steps))}
        print(f"[compat] warmup_ratio unsupported -> warmup_steps={warmup_kw['warmup_steps']} (of {total_steps})")
    config = SFTConfig(
        output_dir=args.out,
        max_length=args.seq_len,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        **warmup_kw,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        completion_only_loss=True,  # loss on the return, not the prompt
        packing=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
