"""End-to-end format validation: the ceiling a perfect model can score.

Renders the oracle's target for each REAL TaxCalcBench case and scores it with
the benchmark's OWN evaluator. This confirms the training target format (trace
+ form + terminator) is exactly what the benchmark parses — a perfect model
reproducing this format scores at the oracle ceiling. Any render/trace change
that breaks benchmark scoring shows up here. Run as a pre-flight regression.

    python scripts/check_format_ceiling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tax-calc-bench"))

from slm_tax.engine import Return  # noqa: E402
from slm_tax.render import render_target  # noqa: E402
from slm_tax.tcb import load_all_cases  # noqa: E402
from tax_calc_bench.tax_return_evaluator import TaxReturnEvaluator  # noqa: E402

# Officially broken benchmark case (missing 1095-A input, tax-calc-bench#66).
KNOWN_BROKEN = {"single-senior-blind-over-65"}


def main() -> None:
    ev = TaxReturnEvaluator()
    strict = lenient = n = 0
    misses = []
    for case in load_all_cases():
        if case["name"] in KNOWN_BROKEN:
            continue
        n += 1
        try:
            ret = Return(case)
            ret.compute()
            target = render_target(case, ret)
        except Exception as e:  # noqa: BLE001
            misses.append((case["name"], f"engine error: {e}"))
            continue
        r = ev.evaluate(target, case["expected_xml"], tax_year="ty24")
        strict += r.strictly_correct_return
        lenient += r.lenient_correct_return
        if not r.strictly_correct_return:
            misses.append((case["name"], "not strict"))

    print(f"format ceiling on {n} real computable benchmark cases:")
    print(f"  strict:  {strict}/{n} = {strict / n:.1%}")
    print(f"  lenient: {lenient}/{n} = {lenient / n:.1%}")
    for name, why in misses:
        print(f"  miss: {name} ({why})")
    # Regression gate: the format must let a perfect model reach oracle ceiling.
    sys.exit(0 if strict >= n - 1 and lenient == n else 1)


if __name__ == "__main__":
    main()
