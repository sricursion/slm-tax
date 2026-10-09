"""Run open-weight model baselines on TaxCalcBench TY24 via the Fireworks API.

Uses TaxCalcBench's own prompt template and evaluator so results are directly
comparable to the published leaderboard. Frontier numbers are cited from the
leaderboard; this fills in the open-weight rows.

Usage:
  uv run python scripts/run_baseline.py --model accounts/fireworks/models/deepseek-v3p1 --smoke
  uv run python scripts/run_baseline.py --model accounts/fireworks/models/kimi-k2-instruct
  uv run python scripts/run_baseline.py --list-models          # see what's served

Requires FIREWORKS_API_KEY in the environment or a .env file at the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tax-calc-bench"))

from tax_calc_bench.tax_return_evaluator import TaxReturnEvaluator  # noqa: E402
from tax_calc_bench.ty24_prompt import TAX_RETURN_GENERATION_PROMPT  # noqa: E402

from slm_tax.tcb import load_all_cases  # noqa: E402

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODELS_URL = "https://api.fireworks.ai/inference/v1/models"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("FIREWORKS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("FIREWORKS_API_KEY not found (set env var or put it in .env)")
    return key


def build_prompt(case: dict) -> str:
    input_json = json.dumps({"input": {"return_header": case["raw_header"], "return_data": case["raw_data"]}}, indent=1)
    return TAX_RETURN_GENERATION_PROMPT.format(
        tax_year="2024", tool_use_hint="", input_data=input_json
    )


def load_raw_cases() -> list[dict]:
    """Cases with the original wrapped JSON (what the benchmark sends models)."""
    cases = load_all_cases()
    for case in cases:
        raw = json.loads(
            (
                ROOT / "external" / "tax-calc-bench" / "tax_calc_bench" / "ty24"
                / "test_data" / case["name"] / "input.json"
            ).read_text(encoding="utf-8")
        )["input"]
        case["raw_header"] = raw.get("return_header", {})
        case["raw_data"] = raw.get("return_data", {})
    return cases


def call_model(key: str, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    backoff = 2.0
    for attempt in range(6):
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        if resp.status_code == 200:
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            # reasoning models: strip <think> blocks / separate reasoning field
            return THINK_RE.sub("", content).strip()
        if resp.status_code in (429, 500, 502, 503, 520):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        raise RuntimeError(f"Fireworks {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("exhausted retries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str)
    ap.add_argument("--smoke", action="store_true", help="run only the first case")
    ap.add_argument("--test-name", type=str)
    ap.add_argument("--max-tokens", type=int, default=16_000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    key = load_api_key()
    if args.list_models:
        resp = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        for m in sorted(x["id"] for x in resp.json().get("data", [])):
            print(m)
        return
    if not args.model:
        sys.exit("--model required (or --list-models)")

    cases = load_raw_cases()
    if args.test_name:
        cases = [c for c in cases if args.test_name in c["name"]]
    if args.smoke:
        cases = cases[:1]

    model_slug = args.model.rsplit("/", 1)[-1]
    out_dir = ROOT / "reports" / "baselines" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator = TaxReturnEvaluator()

    def run_case(case: dict) -> dict:
        out_file = out_dir / f"{case['name']}.md"
        if out_file.exists():
            output = out_file.read_text(encoding="utf-8")
        else:
            try:
                output = call_model(key, args.model, build_prompt(case), args.max_tokens, args.temperature)
                out_file.write_text(output, encoding="utf-8")
            except Exception as e:
                return {"name": case["name"], "error": str(e)}
        result = evaluator.evaluate(output, case["expected_xml"], tax_year="ty24")
        return {
            "name": case["name"],
            "strict": result.strictly_correct_return,
            "lenient": result.lenient_correct_return,
            "by_line": result.correct_by_line_score,
            "by_line_lenient": result.lenient_correct_by_line_score,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run_case, cases))
    dt = time.time() - t0

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    for r in results:
        if "error" in r:
            print(f"[ERROR ] {r['name']}: {r['error'][:120]}")
        else:
            mark = "OK    " if r["strict"] else ("LENIENT" if r["lenient"] else "MISS  ")
            print(f"[{mark}] {r['name']} (by-line {r['by_line']:.0%})")

    if ok:
        n = len(ok)
        print(f"\n===== {args.model} on TaxCalcBench TY24 ({n} cases, {dt:.0f}s) =====")
        print(f"strict returns:   {sum(r['strict'] for r in ok)}/{n} = {sum(r['strict'] for r in ok) / n:.1%}")
        print(f"lenient returns:  {sum(r['lenient'] for r in ok)}/{n} = {sum(r['lenient'] for r in ok) / n:.1%}")
        print(f"by line:          {sum(r['by_line'] for r in ok) / n:.1%}")
        print(f"by line lenient:  {sum(r['by_line_lenient'] for r in ok) / n:.1%}")
        print(f"errors: {len(errs)}")
        summary = {
            "model": args.model, "cases": n, "errors": len(errs),
            "strict": sum(r["strict"] for r in ok) / n,
            "lenient": sum(r["lenient"] for r in ok) / n,
            "by_line": sum(r["by_line"] for r in ok) / n,
            "by_line_lenient": sum(r["by_line_lenient"] for r in ok) / n,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "results": results,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
