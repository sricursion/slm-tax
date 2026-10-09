"""Paired statistics over saved benchmark generations (E5).

McNemar exact test on per-case strict correctness between two runs whose
generations were saved with --save-texts. On n=51 the discordant-pair count
is small, so the exact binomial form is required (chi-square approximations
are invalid here).

Usage:
  python scripts/stats.py --a artifacts/run11_s1000-bench-texts --b artifacts/headline-4b-gen-texts
  python scripts/stats.py --a <dir> --b <dir> --label-a run11 --label-b run8
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tax-calc-bench"))


def strict_by_case(gen_dir: Path) -> dict[str, bool]:
    from lxml import etree

    from slm_tax.score import parse_line
    from slm_tax.tcb import load_all_cases
    from tax_calc_bench.tax_return_evaluator import LINES_TO_XPATH_VALUES

    out: dict[str, bool] = {}
    for case in load_all_cases():
        f = gen_dir / f"{case['name']}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        tree = etree.fromstring(case["expected_xml"].encode())
        ok = True
        for line, xp in LINES_TO_XPATH_VALUES.items():
            nodes = tree.xpath(xp)
            want = float(nodes[0].text) if nodes and nodes[0].text else 0.0
            if abs(parse_line(text, line) - want) >= 0.5:
                ok = False
                break
        out[case["name"]] = ok
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="gen-texts dir for run A")
    ap.add_argument("--b", required=True, help="gen-texts dir for run B")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    la = args.label_a or Path(args.a).name
    lb = args.label_b or Path(args.b).name
    sa = strict_by_case(Path(args.a))
    sb = strict_by_case(Path(args.b))
    common = sorted(set(sa) & set(sb))
    if not common:
        raise SystemExit("no common cases between the two dirs")

    both = a_only = b_only = neither = 0
    a_only_names, b_only_names = [], []
    for name in common:
        if sa[name] and sb[name]:
            both += 1
        elif sa[name] and not sb[name]:
            a_only += 1
            a_only_names.append(name)
        elif sb[name] and not sa[name]:
            b_only += 1
            b_only_names.append(name)
        else:
            neither += 1

    p = mcnemar_exact(a_only, b_only)
    n = len(common)
    print(f"n = {n} common cases")
    print(f"{la}: {sum(sa[c] for c in common)}/{n} strict   {lb}: {sum(sb[c] for c in common)}/{n} strict")
    print(f"contingency: both {both} | {la}-only {a_only} | {lb}-only {b_only} | neither {neither}")
    print(f"McNemar exact two-sided p = {p:.4f}")
    if a_only_names:
        print(f"  {la}-only wins: {', '.join(a_only_names[:8])}")
    if b_only_names:
        print(f"  {lb}-only wins: {', '.join(b_only_names[:8])}")


if __name__ == "__main__":
    main()
