"""Standalone scorer for model outputs against oracle line labels.

Replicates TaxCalcBench's parsing/scoring exactly (same line labels, same
last-|-field parsing, same strict/lenient rules) but scores against our
oracle's `lines` dict instead of expected XML — usable for synthetic held-out
sets without the benchmark clone.
"""

from __future__ import annotations

# Benchmark line label -> our lines-dict tag (mirrors LINES_TO_XPATH_VALUES)
LABEL_TO_TAG = {
    "Line 1a: Total amount from Form(s) W-2, box 1": "WagesAmt",
    "Line 9: Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income": "TotalIncomeAmt",
    "Line 10: Adjustments to income from Schedule 1, line 26": "TotalAdjustmentsAmt",
    "Line 11: Subtract line 10 from line 9. This is your adjusted gross income": "AdjustedGrossIncomeAmt",
    "Line 12: Standard deduction or itemized deductions (from Schedule A)": "TotalItemizedOrStandardDedAmt",
    "Line 15: Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income": "TaxableIncomeAmt",
    "Line 16: Tax": "TaxAmt",
    "Line 19: Child tax credit or credit for other dependents from Schedule 8812": "CTCODCAmt",
    "Line 24: Add lines 22 and 23. This is your total tax": "TotalTaxAmt",
    "Line 25d: Add lines 25a through 25c": "WithholdingTaxAmt",
    "Line 26: 2024 estimated tax payments and amount applied from 2023 return": "EstimatedTaxPaymentsAmt",
    "Line 27: Earned income credit (EIC)": "EarnedIncomeCreditAmt",
    "Line 28: Additional child tax credit from Schedule 8812": "AdditionalChildTaxCreditAmt",
    "Line 29: American opportunity credit from Form 8863, line 8": "RefundableAmerOppCreditAmt",
    "Line 32: Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits": "RefundableCreditsAmt",
    "Line 33: Add lines 25d, 26, and 32. These are your total payments": "TotalPaymentsAmt",
    "Line 34: If line 33 is more than line 24, subtract line 24 from line 33. This is the amount you overpaid": "OverpaidAmt",
    "Line 35a: Amount of line 34 you want refunded to you.": "RefundAmt",
    "Line 37: Subtract line 33 from line 24. This is the amount you owe": "OwedAmt",
}


def parse_money(s: str) -> float:
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return 0.0
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_line(generated: str, label: str) -> float:
    """Identical to TaxCalcBench parse_generated_value."""
    if label not in generated:
        return 0.0
    rest = generated.split(label)[1].split("\n")[0]
    if "|" in rest:
        return parse_money(rest.split("|")[-1].strip())
    return 0.0


def score_output(generated: str, lines: dict) -> dict:
    """Score one model output against oracle lines (same metrics as benchmark)."""
    correct = lenient = 0
    per_line = {}
    for label, tag in LABEL_TO_TAG.items():
        got = parse_line(generated, label)
        want = float(lines[tag])
        ok = got == want
        ok_len = abs(got - want) <= 5
        correct += ok
        lenient += ok_len
        per_line[tag] = {"got": got, "want": want, "ok": ok}
    n = len(LABEL_TO_TAG)
    return {
        "strict": correct == n,
        "lenient": lenient == n,
        "by_line": correct / n,
        "by_line_lenient": lenient / n,
        "per_line": per_line,
    }


import re as _re

# Arithmetic-claim patterns for the process (trace-step) reward. The trace's own
# "A + B = C" / "A - B = D" / "P% of B = R" claims are checkable without the
# oracle, giving a DENSE signal even when the final answer is wrong — which the
# RLVR literature (sparse-reward collapse: all-wrong group -> zero gradient)
# says is essential for a weak base model. Process supervision also outperforms
# outcome-only for multi-step reasoning (Lightman et al.; StepORLM).
_MM = r"-?[\d,]+(?:\.\d+)?"


def _f2n(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def trace_self_consistency(generated: str) -> float:
    """Fraction of the model's own arithmetic claims that actually compute."""
    good = total = 0
    for line in generated.splitlines():
        for m in _re.finditer(rf"({_MM})(?:\s*\+\s*{_MM})+\s*=\s*({_MM})", line):
            nums = [_f2n(x) for x in _re.findall(_MM, m.group(0))]
            *parts, tot = nums
            total += 1
            good += abs(sum(parts) - tot) <= 1.0
        for m in _re.finditer(rf"(?<![\d,\-])({_MM})\s*-\s*({_MM})\s*=\s*({_MM})", line):
            a, b, d = _f2n(m.group(1)), _f2n(m.group(2)), _f2n(m.group(3))
            total += 1
            good += abs((a - b) - d) <= 1.0
        for m in _re.finditer(rf"(\d+(?:\.\d+)?)%\s+of\s+({_MM})\s*=\s*({_MM})", line):
            p, b, r = float(m.group(1)), _f2n(m.group(2)), _f2n(m.group(3))
            total += 1
            good += abs(p / 100 * b - r) <= 1.5
    return good / total if total else 0.0


def reward(generated: str, lines: dict) -> dict:
    """Shaped reward for RLVR (RL with verifiable rewards).

    Given a model generation and the oracle's correct line values, returns a
    dict of reward signals derived from the SAME exact-match rules the benchmark
    scores with — so the RL objective is the eval metric, not a proxy:

      by_line   : fraction of the 19 scored lines exactly correct (dense; good
                  for credit assignment early in training)
      strict    : 1.0 iff every scored line is exactly correct (the true goal)
      chain     : fraction of the ~8 always-nonzero strict-critical lines
                  correct (income->AGI->deduction->taxable->tax->total tax->
                  payments->owe) — the lines that actually decide correctness
      process   : fraction of the model's OWN trace arithmetic that computes
                  (dense, has signal even at 0 strict — avoids sparse collapse)
      shaped    : 0.4*process + 0.4*by_line + 0.2*chain + 1.0 bonus if strict
                  (dense everywhere, with a sparse spike at full correctness)
    """
    s = score_output(generated, lines)
    chain_tags = {
        "WagesAmt", "TotalIncomeAmt", "AdjustedGrossIncomeAmt",
        "TotalItemizedOrStandardDedAmt", "TaxableIncomeAmt", "TaxAmt",
        "TotalTaxAmt", "TotalPaymentsAmt", "OwedAmt",
    }
    chain = [v["ok"] for tag, v in s["per_line"].items() if tag in chain_tags]
    chain_r = sum(chain) / len(chain) if chain else 0.0
    process = trace_self_consistency(generated)
    strict = 1.0 if s["strict"] else 0.0
    return {
        "by_line": s["by_line"],
        "strict": strict,
        "chain": chain_r,
        "process": process,
        "shaped": 0.4 * process + 0.4 * s["by_line"] + 0.2 * chain_r + strict,
    }


def form_constraint_regex() -> str:
    """Regex enforcing free reasoning then the 19 scored form lines (each with an
    integer value) then the terminator — for vLLM guided_regex decoding.

    Design: constrain ONLY the form, never the trace, to avoid the "alignment
    tax" of over-constraining reasoning. Verified to match oracle compact-form
    outputs on all 50 computable benchmark cases. Below ~4B params, extraction
    accuracy is dominated by structured-output-compliance failures, so this
    guarantees compliance and lets the model spend capacity on the values.
    """
    import re as _re2

    parts = [_re2.escape(lab) + r" \| [^|\n]* \| -?\d+" for lab in LABEL_TO_TAG]
    return r"(?s).*?" + r".*?".join(parts) + r".*?--- END OF RETURN ---"


def best_of_n_consistency(generations: list[str]) -> str:
    """Answer-free Best-of-N: pick the sample whose OWN arithmetic is most
    self-consistent. A verifier that needs no ground truth (usable at eval).
    Verifiers beat plain self-consistency at lower N (test-time-scaling lit)."""
    if not generations:
        return ""
    return max(generations, key=trace_self_consistency)


def per_line_majority(generations: list[str]) -> str:
    """Structured self-consistency: vote each of the 19 scored lines
    independently across N samples, then emit a consensus return. Because the
    lines are independent, this assembles a correct return even when no single
    sample is fully correct — directly countering the cascade failure mode.
    Returns a scoreable string (the 19 winning label|value lines)."""
    from collections import Counter

    if not generations:
        return ""
    lines_out = []
    for label in LABEL_TO_TAG:
        votes = Counter()
        for g in generations:
            # only count samples that actually contain the line
            if label in g:
                votes[parse_line(g, label)] += 1
        val = votes.most_common(1)[0][0] if votes else 0.0
        lines_out.append(f"{label} |  | {int(val)}")
    return "\n".join(lines_out)
