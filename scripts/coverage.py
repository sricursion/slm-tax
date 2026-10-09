"""Train-vs-benchmark feature coverage — run BEFORE spending GPU on a rerun.

Compares the structural feature distribution of generated training cases against
the 51 TaxCalcBench cases. Configurations the benchmark exercises but the
generator rarely produces are exactly the ones a trained model cannot learn.
This is the check that caught the 4-box / no-W-2 / 3-business-Schedule-C gaps.

    python scripts/coverage.py [n_train_samples]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm_tax.engine import Return  # noqa: E402
from slm_tax.generator import generate_case  # noqa: E402
from slm_tax.tcb import load_all_cases  # noqa: E402


def features(ret: Return, case: dict) -> set:
    f = {f"status:{ret.status}"}
    boxes = int(ret.tp_age >= 65) + int(ret.tp_blind)
    if ret.joint:
        boxes += int((ret.sp_age or 0) >= 65) + int(ret.sp_blind)
    f.add(f"age_blind_boxes:{boxes}")
    f.add(f"n_w2:{min(len(case['data'].get('w2') or []), 3)}")
    f.add(f"n_sched_c:{min(len(ret.qbi_businesses), 3)}")
    f.add(f"n_deps:{min(len(ret.dependents), 3)}")
    d = ret.detail
    for cond, tag in [
        (ret.ss_benefits, "has_ss"),
        (ret.ss_benefits and d["6b"] == 0, "ss_nontaxable"),
        (ret.interest, "has_interest"),
        (ret.ordinary_div, "has_dividends"),
        (ret.st_gain or ret.lt_gain or ret.cap_gain_distr, "has_capgains"),
        (d["13"], "has_qbid"),
        (d["27"], "has_eitc"),
        (d["19"], "has_ctc"),
        (ret.unemployment, "has_unemployment"),
        (ret.pension_taxable, "has_pension"),
        (ret.excess_ss, "has_excess_ss"),
    ]:
        if cond:
            f.add(tag)
    return f


def distribution(cases) -> tuple[Counter, int]:
    c = Counter()
    n = 0
    for case in cases:
        try:
            ret = Return(case)
            ret.compute()
            for feat in features(ret, case):
                c[feat] += 1
            n += 1
        except Exception:
            pass
    return c, n


def main() -> None:
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    bench, nb = distribution(load_all_cases())
    train, nt = distribution(generate_case(s) for s in range(1000, 1000 + n_train))

    print(f"{'feature':24s} {'bench%':>7s} {'train%':>7s}  flag")
    print("-" * 52)
    for feat in sorted(set(bench) | set(train)):
        bp, tp = bench[feat] / nb * 100, train[feat] / nt * 100
        flag = ""
        if bp >= 8 and tp < bp * 0.4:
            flag = " <<< UNDER-COVERED"
        print(f"{feat:24s} {bp:6.1f}% {tp:6.1f}%{flag}")


if __name__ == "__main__":
    main()
