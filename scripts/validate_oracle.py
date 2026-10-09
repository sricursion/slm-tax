"""Validate the PolicyEngine-based oracle against all TaxCalcBench TY24 cases.

Compares the oracle's 19 scored Form 1040 lines against each case's expected
output.xml using the same XPaths and leniency rules as TaxCalcBench itself.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_tax.tcb import load_all_cases  # noqa: E402

# Officially broken benchmark case: input is missing the Form 1095-A data its
# expected output was computed with (tax-calc-bench GitHub issue #66).
KNOWN_BROKEN = {"single-senior-blind-over-65"}

XPATHS = {
    "WagesAmt": "/Return/ReturnData/IRS1040/WagesAmt",
    "TotalIncomeAmt": "/Return/ReturnData/IRS1040/TotalIncomeAmt",
    "TotalAdjustmentsAmt": "/Return/ReturnData/IRS1040/TotalAdjustmentsAmt",
    "AdjustedGrossIncomeAmt": "/Return/ReturnData/IRS1040/AdjustedGrossIncomeAmt",
    "TotalItemizedOrStandardDedAmt": "/Return/ReturnData/IRS1040/TotalItemizedOrStandardDedAmt",
    "TaxableIncomeAmt": "/Return/ReturnData/IRS1040/TaxableIncomeAmt",
    "TaxAmt": "/Return/ReturnData/IRS1040/TaxAmt",
    "CTCODCAmt": "/Return/ReturnData/IRS1040/CTCODCAmt",
    "TotalTaxAmt": "/Return/ReturnData/IRS1040/TotalTaxAmt",
    "WithholdingTaxAmt": "/Return/ReturnData/IRS1040/WithholdingTaxAmt",
    "EstimatedTaxPaymentsAmt": "/Return/ReturnData/IRS1040/EstimatedTaxPaymentsAmt",
    "EarnedIncomeCreditAmt": "/Return/ReturnData/IRS1040/EarnedIncomeCreditAmt",
    "AdditionalChildTaxCreditAmt": "/Return/ReturnData/IRS1040/AdditionalChildTaxCreditAmt",
    "RefundableAmerOppCreditAmt": "/Return/ReturnData/IRS1040/RefundableAmerOppCreditAmt",
    "RefundableCreditsAmt": "/Return/ReturnData/IRS1040/RefundableCreditsAmt",
    "TotalPaymentsAmt": "/Return/ReturnData/IRS1040/TotalPaymentsAmt",
    "OverpaidAmt": "/Return/ReturnData/IRS1040/OverpaidAmt",
    "RefundAmt": "/Return/ReturnData/IRS1040/RefundAmt",
    "OwedAmt": "/Return/ReturnData/IRS1040/OwedAmt",
}


def expected_values(xml_text: str) -> dict[str, float]:
    tree = etree.fromstring(xml_text.encode("utf-8"))
    out = {}
    for tag, xpath in XPATHS.items():
        hits = tree.xpath(xpath)
        out[tag] = float(hits[0].text) if hits and hits[0].text else 0.0
    return out


def main() -> None:
    from slm_tax.engine import compute_lines

    only = sys.argv[1] if len(sys.argv) > 1 else None
    cases = load_all_cases()
    if only:
        cases = [c for c in cases if only in c["name"]]

    results = []
    strict_ok = lenient_ok = 0
    line_total = line_ok = 0
    for case in cases:
        exp = expected_values(case["expected_xml"])
        row = {"name": case["name"], "lines": {}, "error": None}
        try:
            got = compute_lines(case)
        except Exception:
            row["error"] = traceback.format_exc(limit=3)
            results.append(row)
            print(f"[ERROR ] {case['name']}")
            continue
        n_ok = 0
        for tag in XPATHS:
            e, g = exp[tag], float(got[tag])
            match = e == g
            row["lines"][tag] = {"expected": e, "got": g, "ok": match}
            line_total += 1
            if match:
                line_ok += 1
                n_ok += 1
        strict = n_ok == len(XPATHS)
        lenient = all(
            abs(v["expected"] - v["got"]) <= 5 for v in row["lines"].values()
        )
        strict_ok += strict
        lenient_ok += lenient
        row["strict"], row["lenient"] = strict, lenient
        results.append(row)
        status = "OK    " if strict else ("LENIENT" if lenient else "MISS  ")
        bad = [
            f"{t}(exp {v['expected']:.0f} got {v['got']:.0f})"
            for t, v in row["lines"].items()
            if not v["ok"]
        ]
        print(f"[{status}] {case['name']}" + (f" -> {', '.join(bad[:6])}" if bad else ""))

    n = len(cases)
    n_err = sum(1 for r in results if r["error"])
    fixable = [r for r in results if r["name"] not in KNOWN_BROKEN]
    f_strict = sum(1 for r in fixable if r.get("strict"))
    print("\n===== ORACLE AGREEMENT vs TaxCalcBench TY24 =====")
    print(f"cases:              {n}")
    print(f"errors (crashed):   {n_err}")
    print(f"strict returns:     {strict_ok}/{n} = {strict_ok / n:.1%}")
    print(f"lenient returns:    {lenient_ok}/{n} = {lenient_ok / n:.1%}")
    if line_total:
        print(f"lines correct:      {line_ok}/{line_total} = {line_ok / line_total:.1%}")
    if fixable and len(fixable) != n:
        print(
            f"strict (excl. known-broken case): {f_strict}/{len(fixable)}"
            f" = {f_strict / len(fixable):.1%}"
        )

    out_path = Path(__file__).resolve().parents[1] / "reports" / "oracle_agreement.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
