"""Verify every arithmetic claim in the oracle traces actually computes.

A trace line like "= 17,700 + 10,360 = 28,060" or "20% of 42,027 = 8,405"
teaches the model a relationship; if the stated result doesn't equal the
computation, the trace teaches WRONG math (this caught a QBID income-limit
bug that showed AGI where it meant taxable-income-before-QBI). Run over a
large sample; any inconsistency is a training-signal bug to fix.

    python scripts/check_trace_consistency.py [n]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm_tax.engine import Return  # noqa: E402
from slm_tax.generator import generate_case  # noqa: E402
from slm_tax.trace import build_trace  # noqa: E402


def _n(s: str) -> float:
    return float(s.replace(",", ""))


# Non-capturing money token so embedding it doesn't shift group indices.
MM = r"-?[\d,]+(?:\.\d+)?"
NUM = re.compile(MM)


def check_line(line: str) -> list[str]:
    problems = []
    # A + B (+ C ...) = S
    for m in re.finditer(rf"({MM})(?:\s*\+\s*{MM})+\s*=\s*({MM})", line):
        nums = [_n(x) for x in NUM.findall(m.group(0))]
        *parts, total = nums
        if abs(sum(parts) - total) > 1.0:
            problems.append(f"sum: {m.group(0)} ({sum(parts)} != {total})")
    # A - B = D  (guard against matching inside a longer number/date)
    for m in re.finditer(rf"(?<![\d,\-])({MM})\s*-\s*({MM})\s*=\s*({MM})", line):
        a, b, d = _n(m.group(1)), _n(m.group(2)), _n(m.group(3))
        if abs((a - b) - d) > 1.0:
            problems.append(f"sub: {m.group(0)} ({a - b} != {d})")
    # P% of B = R
    for m in re.finditer(rf"(\d+(?:\.\d+)?)%\s+of\s+({MM})\s*=\s*({MM})", line):
        p, b, r = float(m.group(1)), _n(m.group(2)), _n(m.group(3))
        if abs(p / 100 * b - r) > 1.5:
            problems.append(f"pct: {m.group(0)} ({p / 100 * b:.2f} != {r})")
    return problems


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    total_lines = total_bad = 0
    examples = []
    for seed in range(1000, 1000 + n):
        ret = Return(generate_case(seed))
        ret.compute()
        for line in build_trace(ret).splitlines():
            if "=" not in line:
                continue
            total_lines += 1
            probs = check_line(line)
            if probs:
                total_bad += 1
                if len(examples) < 12:
                    examples.append((seed, line.strip(), probs))
    print(f"checked {total_lines} arithmetic trace lines across {n} cases")
    print(f"inconsistent lines: {total_bad}")
    for seed, line, probs in examples:
        print(f"  seed {seed}: {line}")
        for p in probs:
            print(f"      -> {p}")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
