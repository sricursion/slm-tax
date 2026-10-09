"""Evaluate a trained model on TaxCalcBench TY24 and/or a synthetic held-out set.

Run on a GPU node with vLLM:
    pip install vllm

Examples:
  # benchmark (needs external/tax-calc-bench cloned)
  python scripts/eval_model.py --model runs/qwen3-4b-full --benchmark

  # synthetic held-out set (no external clone needed)
  python scripts/eval_model.py --model runs/qwen3-4b-full --synthetic data/val_2k.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm_tax.render import PROMPT_TEMPLATE  # noqa: E402
from slm_tax.score import score_output  # noqa: E402


def build_prompts(input_jsons: list[str], tokenizer) -> list[str]:
    out = []
    for ij in input_jsons:
        user_prompt = PROMPT_TEMPLATE.format(input_data=ij)
        out.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return out


def _bootstrap_ci(values: list[float], iters: int = 5000) -> tuple[float, float]:
    """95% bootstrap CI for a mean — essential on a 50-case benchmark where
    49/50 vs 47/50 is not a real difference."""
    import random

    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(iters):
        means.append(sum(values[random.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def summarize(name: str, results: list[dict]) -> dict:
    n = len(results)
    strict_vals = [1.0 if r["strict"] else 0.0 for r in results]
    lo, hi = _bootstrap_ci(strict_vals)
    s = {
        "eval": name,
        "n": n,
        "strict": sum(strict_vals) / n,
        "strict_ci95": [lo, hi],
        "lenient": sum(r["lenient"] for r in results) / n,
        "by_line": sum(r["by_line"] for r in results) / n,
        "by_line_lenient": sum(r["by_line_lenient"] for r in results) / n,
    }
    print(
        f"===== {name} (n={n}) =====\n"
        f"strict:  {s['strict']:.1%}  (95% CI {lo:.1%}-{hi:.1%})\n"
        f"lenient: {s['lenient']:.1%}\nby line: {s['by_line']:.1%}\n"
        f"by line lenient: {s['by_line_lenient']:.1%}"
    )
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--synthetic", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2_048)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf",
                    help="hf works everywhere (incl. ROCm + LoRA adapters); vllm is faster")
    ap.add_argument("--batch-size", type=int, default=8, help="hf backend batch size")
    ap.add_argument("--limit", type=int, default=None, help="cap synthetic examples")
    ap.add_argument("--save-texts", type=str, default=None, help="dir to dump raw generations")
    ap.add_argument("--guided", action="store_true",
                    help="vllm only: constrain the form section to the 19-line grammar "
                    "(free trace, constrained form). Falls back to unconstrained if the "
                    "vLLM structured-output API is unavailable.")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer source if the model dir lacks one (e.g. mid-training checkpoints)")
    ap.add_argument("--num-samples", type=int, default=1,
                    help="test-time scaling (E4): sample N at temp 0.7, per-line majority vote")
    ap.add_argument("--token-budget", type=int, default=500_000,
                    help="hf backend: max sum of (prompt+max_new) tokens per "
                    "batch; long-prompt batches shrink automatically")
    ap.add_argument("--no-stop", action="store_true",
                    help="hf backend: skip per-step stop-string checking (a large "
                    "per-step overhead at big batch sizes); always generate "
                    "max-new-tokens and truncate at the terminator afterwards")
    args = ap.parse_args()
    multi = args.num_samples > 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, max_model_len=10_240, dtype="bfloat16")
        sp_kwargs = dict(
            temperature=0.7 if multi else 0.0,
            max_tokens=args.max_new_tokens,
            n=args.num_samples,
        )
        if args.guided:
            # Defensive: vLLM's structured-output API has changed across
            # versions. Try the known variants; fall back to unconstrained (with
            # the terminator stop) rather than crash the run.
            from slm_tax.score import form_constraint_regex

            regex = form_constraint_regex()
            try:
                from vllm.sampling_params import GuidedDecodingParams

                sp_kwargs["guided_decoding"] = GuidedDecodingParams(regex=regex)
                print("[guided] using GuidedDecodingParams(regex=...)")
            except Exception:
                try:
                    from vllm.sampling_params import StructuredOutputsParams

                    sp_kwargs["structured_outputs"] = StructuredOutputsParams(regex=regex)
                    print("[guided] using StructuredOutputsParams(regex=...)")
                except Exception:
                    print("[guided] structured-output API unavailable; unconstrained")
                    sp_kwargs["stop"] = ["--- END OF RETURN ---"]
        else:
            sp_kwargs["stop"] = ["--- END OF RETURN ---"]
        sampling = SamplingParams(**sp_kwargs)

        def generate_all(prompts: list[str]) -> list[str]:
            outs = llm.generate(prompts, sampling)
            if multi:
                from slm_tax.score import per_line_majority

                return [per_line_majority([c.text for c in o.outputs]) for o in outs]
            return [o.outputs[0].text for o in outs]
    else:
        import torch

        # LoRA adapter dirs load via peft; merged/full models via transformers
        if (Path(args.model) / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            model = model.merge_and_unload()
        else:
            from transformers import AutoModelForCausalLM

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, torch_dtype=torch.bfloat16, device_map="cuda",
                    attn_implementation="sdpa",
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, torch_dtype=torch.bfloat16, device_map="cuda"
                )
        model.eval()
        tokenizer.padding_side = "left"

        def generate_all(prompts: list[str]) -> list[str]:
            from slm_tax.score import per_line_majority

            # Length-sorted batching: decode cost is dominated by KV reads at
            # the batch's max context, so mixed-length batches pay the longest
            # prompt's price for every sequence. Sort, batch, then restore the
            # original order.
            order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
            sorted_prompts = [prompts[i] for i in order]
            # Greedy token-budget batches: KV memory scales with
            # batch_max_len x batch_size, so fixed-size batches OOM exactly
            # when sorting concentrates the longest prompts together. Budget
            # counts (prompt_chars/3 + max_new) as a token estimate.
            batches = []
            cur: list[str] = []
            cur_max = 0
            for sp_ in sorted_prompts:
                est = len(sp_) // 3 + args.max_new_tokens
                m_ = max(cur_max, est)
                if cur and (m_ * (len(cur) + 1) > args.token_budget
                            or len(cur) >= args.batch_size):
                    batches.append(cur)
                    cur, cur_max = [], 0
                    m_ = est
                cur.append(sp_)
                cur_max = m_
            if cur:
                batches.append(cur)
            texts_sorted = []
            texts = texts_sorted
            done_ct = 0
            for batch in batches:
                enc = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=multi,
                    temperature=0.7 if multi else None,
                    num_return_sequences=args.num_samples,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if not args.no_stop:
                    gen_kwargs["stop_strings"] = ["--- END OF RETURN ---"]
                    gen_kwargs["tokenizer"] = tokenizer
                else:
                    # Token-ID early stop: batch ends once every sequence has
                    # emitted the terminator. Tensor compare on the last k
                    # tokens per step — no string decode (that per-step decode
                    # is why stop_strings was slow). Uses the in-context token
                    # form (newline + terminator, minus the leading token) so
                    # the match fires despite newline merging; a missed match
                    # just falls back to running to max_new_tokens.
                    from transformers import StoppingCriteria, StoppingCriteriaList

                    ctx_ids = tokenizer("\n--- END OF RETURN ---",
                                        add_special_tokens=False).input_ids[1:]

                    class _TermStop(StoppingCriteria):
                        def __init__(self, term_ids):
                            self.t = torch.tensor(term_ids)
                            self.done = None

                        def __call__(self, input_ids, scores, **kwargs):
                            k = self.t.shape[0]
                            if self.done is None or self.done.shape[0] != input_ids.shape[0]:
                                self.done = torch.zeros(
                                    input_ids.shape[0], dtype=torch.bool,
                                    device=input_ids.device,
                                )
                            if input_ids.shape[1] >= k:
                                tail = input_ids[:, -k:]
                                self.done |= (tail == self.t.to(input_ids.device)).all(dim=1)
                            return bool(self.done.all())

                    gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_TermStop(ctx_ids)])
                with torch.no_grad():
                    out = model.generate(**enc, **gen_kwargs)
                # out is (batch*num_samples, seq); regroup per prompt
                for j in range(len(batch)):
                    samples = []
                    for k in range(args.num_samples):
                        row = out[j * args.num_samples + k][enc.input_ids.shape[1]:]
                        samples.append(
                            tokenizer.decode(row, skip_special_tokens=True).split(
                                "--- END OF RETURN ---"
                            )[0]
                        )
                    texts.append(per_line_majority(samples) if multi else samples[0])
                done_ct += len(batch)
                print(f"  generated {done_ct}/{len(prompts)}", flush=True)
            unsorted = [None] * len(prompts)
            for pos, idx in enumerate(order):
                unsorted[idx] = texts_sorted[pos]
            return unsorted

    summaries = []

    if args.benchmark:
        sys.path.insert(0, str(ROOT / "external" / "tax-calc-bench"))
        from tax_calc_bench.tax_return_evaluator import TaxReturnEvaluator

        from slm_tax.tcb import TY24_TEST_DATA, load_all_cases

        cases = load_all_cases()
        input_jsons = []
        for case in cases:
            raw = json.loads(
                (TY24_TEST_DATA / case["name"] / "input.json").read_text(encoding="utf-8")
            )["input"]
            # Match training serialization exactly (compact separators, same
            # {"input": ...} envelope as render.build_input_json).
            input_jsons.append(json.dumps({"input": raw}, separators=(",", ":")))
        prompts = build_prompts(input_jsons, tokenizer)
        texts = generate_all(prompts)
        if args.save_texts:
            td = Path(args.save_texts); td.mkdir(parents=True, exist_ok=True)
            for case_, text_ in zip(cases, texts):
                (td / f"{case_['name']}.md").write_text(text_, encoding="utf-8")
        evaluator = TaxReturnEvaluator()
        results = []
        for case, text in zip(cases, texts):
            r = evaluator.evaluate(text, case["expected_xml"], tax_year="ty24")
            results.append(
                {
                    "name": case["name"],
                    "strict": r.strictly_correct_return,
                    "lenient": r.lenient_correct_return,
                    "by_line": r.correct_by_line_score,
                    "by_line_lenient": r.lenient_correct_by_line_score,
                }
            )
            mark = "OK    " if r.strictly_correct_return else "MISS  "
            print(f"[{mark}] {case['name']} (by-line {r.correct_by_line_score:.0%})")
        summaries.append(summarize("TaxCalcBench TY24", results))

    if args.synthetic:
        rows = [json.loads(line) for line in open(args.synthetic, encoding="utf-8")]
        if args.limit:
            rows = rows[: args.limit]
        prompts = build_prompts([r["input_json"] for r in rows], tokenizer)
        texts = generate_all(prompts)
        if args.save_texts:
            td = Path(args.save_texts); td.mkdir(parents=True, exist_ok=True)
            for row_, text_ in zip(rows, texts):
                (td / f"{row_['id']}.md").write_text(text_, encoding="utf-8")
        results = [
            score_output(text, row["lines"]) for row, text in zip(rows, texts)
        ]
        summaries.append(summarize(f"synthetic {Path(args.synthetic).name}", results))

    out_path = Path(args.out) if args.out else Path(args.model) / "eval_results.json"
    out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nresults: {out_path}")


if __name__ == "__main__":
    main()
