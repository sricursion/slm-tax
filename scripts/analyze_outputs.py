"""Failure taxonomy for model generations — run BEFORE spending GPU on a rerun.

Given a directory of model outputs (one <case>.md per benchmark case, produced
by eval_model.py --save-texts), classify every failure quantitatively:
truncation (model won't stop), internal trace-arithmetic errors, and the first
form line that diverges from benchmark truth. This is the cheap diagnostic that
replaces guessing at what to fix next.

    python scripts/analyze_outputs.py artifacts/pilot2-gen-texts
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tax-calc-bench"))

from lxml import etree  # noqa: E402

from slm_tax.score import parse_line, parse_money  # noqa: E402
from slm_tax.tcb import load_all_cases  # noqa: E402
from tax_calc_bench.tax_return_evaluator import LINES_TO_XPATH_VALUES  # noqa: E402


def expected_vals(xml: str) -> dict:
    tree = etree.fromstring(xml.encode())
    out = {}
    for label, xp in LINES_TO_XPATH_VALUES.items():
        hits = tree.xpath(xp)
        out[label] = float(hits[0].text) if hits and hits[0].text else 0.0
    return out


def trace_arith_errors(text: str) -> tuple[int, int]:
    """Count the model's own arithmetic mistakes inside its scratchpad."""
    errs = checks = 0
    for m in re.finditer(r"=\s*([\d,]+(?:\s*\+\s*[\d,]+)+)\s*=\s*(-?[\d,]+)", text):
        parts = [parse_money(p) for p in m.group(1).split("+")]
        checks += 1
        errs += abs(sum(parts) - parse_money(m.group(2))) > 0.5
    for m in re.finditer(r"=\s*([\d,]+)\s*-\s*([\d,]+)\s*=\s*(-?[\d,]+)", text):
        a, b, z = (parse_money(m.group(i)) for i in (1, 2, 3))
        checks += 1
        errs += abs((a - b) - z) > 0.5
    return errs, checks


def main() -> None:
    gen_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "pilot2-gen-texts"
    cases = {c["name"]: c for c in load_all_cases()}
    rows = []
    for name, case in cases.items():
        f = gen_dir / f"{name}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        exp = expected_vals(case["expected_xml"])
        # incomplete = the last scored form line (37) never appeared, whether
        # from runaway continuation OR premature stop at the terminator.
        trunc = "Line 37:" not in text
        ramble = bool(re.search(r"\nForm \d{3,4}[A-Z]|Part [IVX]{3,}", text))
        aerr, achk = trace_arith_errors(text)
        first_wrong = None
        nwrong = 0
        for label in LINES_TO_XPATH_VALUES:
            if abs(parse_line(text, label) - exp[label]) > 0.5:
                nwrong += 1
                first_wrong = first_wrong or label.split(":")[0]
        rows.append((name, trunc, ramble, aerr, achk, nwrong, first_wrong))

    n = len(rows)
    if not n:
        sys.exit(f"no outputs found in {gen_dir}")
    print(f"=== FAILURE TAXONOMY: {gen_dir} ({n} cases) ===\n")
    print(f"truncated (won't stop):   {sum(r[1] for r in rows)}/{n}")
    print(f"rambled (invented forms): {sum(r[2] for r in rows)}/{n}")
    coh = [r for r in rows if not r[1] and not r[2]]
    print(f"coherent & complete:      {len(coh)}/{n}")
    if coh:
        print(f"  fully correct:          {sum(1 for r in coh if r[5] == 0)}/{len(coh)}")
        print(f"  avg wrong lines:        {sum(r[5] for r in coh) / len(coh):.1f}/19")
    ae, ac = sum(r[3] for r in rows), sum(r[4] for r in rows)
    print(f"trace-arithmetic errors:  {ae}/{ac} ({ae / max(1, ac):.1%})")
    print("\nfirst divergent line (coherent cases):")
    for line, c in Counter(r[6] for r in coh if r[6]).most_common():
        print(f"  {line}: {c}")

    # per-line hit rate across ALL cases — shows which strict-critical-chain
    # lines the model actually nails (income/AGI/deduction/taxable/tax/...).
    chain = {"Line 1a", "Line 9", "Line 11", "Line 12", "Line 15", "Line 16",
             "Line 24", "Line 25d", "Line 33", "Line 34", "Line 37"}
    hit = Counter()
    for name, case in cases.items():
        f = gen_dir / f"{name}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        exp = expected_vals(case["expected_xml"])
        for label in LINES_TO_XPATH_VALUES:
            if abs(parse_line(text, label) - exp[label]) <= 0.5:
                hit[label.split(":")[0]] += 1
    print("\nper-line hit rate (* = strict-critical chain):")
    for label in LINES_TO_XPATH_VALUES:
        tag = label.split(":")[0]
        mark = "*" if tag in chain else " "
        print(f" {mark}{tag:9s} {hit[tag] / n:5.0%}")


if __name__ == "__main__":
    main()
