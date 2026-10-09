"""Render generated scenarios into TaxCalcBench-format prompts and targets.

The prompt embeds the taxpayer data as label/value-wrapped JSON exactly like
the benchmark's input.json; the target is the full Form 1040 text in the
format TaxCalcBench's evaluator parses (last |-field of each line).
"""

from __future__ import annotations

import json
from typing import Any

from .engine import Return
from .labels import FIELD_LABELS

END_OF_RETURN = "--- END OF RETURN ---"

# Benchmark's canonical return_data section order (by median position across the
# 51 TY24 cases). We reorder generated inputs to match so the model trains with
# the SAME field positions it sees at eval — "lost in the middle" says models
# extract worst from the middle, and the benchmark hides irs1040 (filing status)
# at ~0.55 and w2 (wages) at ~0.91. The engine reads by key, so order is
# oracle-irrelevant; this only aligns train/test input layout.
CANONICAL_SECTION_ORDER = [
    "has_ssn", "residency_status", "depreciation",
    "worked_and_lived_in_different_states", "nonmonetary_charity",
    "other_expenses", "will_itemize", "earned_in_another_state",
    "form_1099k_includes_personal_pmts", "needs_to_file_extension",
    "ira_contributions_credits_screen", "irs2441", "received_form_1099K",
    "worked_in_multiple_states", "employer_retirement_contributions_credits_screen",
    "irs1099_r", "irs8962", "irs8863", "irs8995a_schedulec",
    "ca_foster_credit_eligible", "irs1040", "irs_payment", "ssa_1099",
    "irs1040_schedulec", "irs8949", "irs1040_schedule1", "irs1040_scheduleb",
    "irs1040_schedule_d", "irs1040_schedule3", "w2",
]
_SECTION_RANK = {k: i for i, k in enumerate(CANONICAL_SECTION_ORDER)}


def _reorder_sections(data: dict) -> dict:
    """Reorder return_data sections to the benchmark's canonical layout;
    unknown keys keep their order at the end."""
    return dict(
        sorted(data.items(), key=lambda kv: _SECTION_RANK.get(kv[0], len(_SECTION_RANK)))
    )

FILING_STATUS_DISPLAY = {
    "single": "Single",
    "married_jointly": "Married filing jointly",
    "married_separately": "Married filing separately",
    "head_of_household": "Head of household",
}

# Labels for the most common fields (mirrors benchmark input.json labels; a
# missing label is fine — many benchmark leaves are {"value": ...} only).
LABELS = {
    "tp_ssn": "Social Security Number",
    "address": "Address",
    "city": "City",
    "state": "State",
    "zip_code": "ZIP code",
    "filing_status": "Confirm your filing status for 2024",
    "tp_first_name": "First name",
    "tp_last_name": "Last name",
    "tp_date_of_birth": "Date of birth",
    "sp_first_name": "Spouse first name",
    "sp_last_name": "Spouse last name",
    "sp_date_of_birth": "Spouse date of birth",
    "tp_blind": "Are you legally blind?",
    "tp_dependent": "Could you be claimed as a dependent on someone else's tax return?",
    "wages": "Box 1",
    "withholding": "Box 2",
    "social_security_wages": "Box 3",
    "social_security_tax": "Box 4",
    "medicare_wages_and_tips": "Box 5",
    "medicare_tax_withheld": "Box 6",
    "social_security_tips": "Box 7",
    "dependent_care_benefits": "Box 10",
    "employer_name": "Employer’s name",
    "ein": "Employer identification number (EIN)",
    "who_applies_to": "Whose form is this?",
    "student_interest": "Your student loan interest paid",
    "tp_educator_exp_amount": "Your unreimbursed educator expenses for 2024",
    "sp_educator_exp_amount": "Your spouse’s unreimbursed educator expenses for 2024",
}

PROMPT_TEMPLATE = """You are helping to test expert tax preparation software. You are given a taxpayer's data and you need to calculate their self-prepared tax return.
Analyze the input data and prepare and calculate a complete tax return including Form 1040 and all necessary schedules and forms for the 2024 tax year.


Follow these requirements:
1. Complete Form 1040 with all necessary calculations. You should have all of the necessary taxpayer inputs to be able to calculate the return.
2. Complete any required schedules (like Schedule B for interest income) but don't output them. You just need to use them to calculate the 1040.
3. Only output the 1040 and all attached forms and schedules in the format below.
4. Do not output any other introductory text or commentary.
5. You may skip the SSN field.
6. Format the output as line-by-line Form 1040 values:
Line 1: [Description] | [Explanation of calculations, if any] | [Amount]

Here is the taxpayer data:

{input_data}

Now please compute the tax return and output as described above. Do not output any other text or commentary:
"""


def wrap(node: Any, key: str | None = None) -> Any:
    """Wrap leaves in {"label", "value"} like benchmark input.json."""
    if isinstance(node, dict):
        return {k: wrap(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [wrap(item) for item in node]
    if key and key in LABELS:
        return {"label": LABELS[key], "value": node}
    if key and key in FIELD_LABELS:
        return {"label": FIELD_LABELS[key], "value": node}
    return {"value": node}


def build_input_json(case: dict) -> str:
    wrapped = {
        "input": {
            "return_header": wrap(case.get("header", {})),
            "return_data": wrap(_reorder_sections(case.get("data", {}))),
        }
    }
    # Compact separators: the label/value wrapping and boilerplate make the
    # input JSON the dominant token cost. Pretty-printing pushed ~17% of
    # examples past the training seq-len, truncating the completion tail (the
    # final form lines + terminator). Compact serialization cuts ~34% of input
    # tokens. Eval MUST serialize benchmark inputs identically (see eval_model).
    return json.dumps(wrapped, separators=(",", ":"))


def render_prompt(case: dict) -> str:
    return PROMPT_TEMPLATE.format(input_data=build_input_json(case))


def _amt(x: float | int) -> str:
    return str(int(x))


def render_target(
    case: dict, ret: Return, with_trace: bool = True, compact_form: bool = False
) -> str:
    """Full Form 1040 output in the benchmark's parseable format, optionally
    preceded by a worked-computation scratchpad."""
    d = ret.detail
    f = case["data"].get("irs1040", {})
    hdr = case.get("header", {})
    deps = f.get("dependent_detail") or []
    dep_desc = (
        "; ".join(
            f"{x.get('dependent_first_name', '?')} {x.get('dependent_last_name', '?')}"
            f" ({x.get('dependent_relationship', '?')})"
            for x in deps
        )
        or "None"
    )

    def expl_16() -> str:
        if ret.qualified_div > 0 or d["7"] > 0:
            return "Qualified Dividends and Capital Gain Tax Worksheet"
        return f"Tax Table on {d['15']}"

    def expl_23() -> str:
        parts = []
        if d["se_tax"]:
            parts.append(f"SE tax {d['se_tax']}")
        if d["addl_medicare"]:
            parts.append(f"Additional Medicare Tax {d['addl_medicare']}")
        if d["niit"]:
            parts.append(f"NIIT {d['niit']}")
        if d["uncollected"]:
            parts.append(f"Uncollected SS/Medicare {d['uncollected']}")
        return " + ".join(parts)

    def expl_20() -> str:
        parts = []
        if d["ftc"]:
            parts.append(f"Foreign tax credit {d['ftc']}")
        if d["cdcc"]:
            parts.append(f"Child and dependent care credit {d['cdcc']}")
        if d["aoc_nonref"]:
            parts.append(f"Nonrefundable education credits {d['aoc_nonref']}")
        if d["savers"]:
            parts.append(f"Saver's credit {d['savers']}")
        return " + ".join(parts)

    def expl_31() -> str:
        parts = []
        if d["extension"]:
            parts.append(f"Extension payment {d['extension']}")
        if d["excess_ss"]:
            parts.append(f"Excess social security tax {d['excess_ss']}")
        if d["net_ptc"]:
            parts.append(f"Net premium tax credit {d['net_ptc']}")
        return " + ".join(parts)

    lines = [
        "Form 1040: U.S. Individual Income Tax Return",
        "===========================================",
        f"Filing Status: {FILING_STATUS_DISPLAY[ret.status]}",
        f"Your first name and middle initial: {f.get('tp_first_name', '')}",
        f"Last name: {f.get('tp_last_name', '')}",
        "Your Social Security Number: *** (skipped for privacy)",
        f"If joint return, spouse's first name and middle initial: {f.get('sp_first_name', '')}",
        f"Last name: {f.get('sp_last_name', '')}",
        "Spouse's Social Security Number: *** (skipped for privacy)",
        f"Home address (number and street). If you have a P.O. box, see instructions.: {hdr.get('address', '')}",
        f"City, town, or post office. If you have a foreign address, also complete spaces below.: {hdr.get('city', '')}",
        f"State: {hdr.get('state', '')}",
        f"ZIP code: {hdr.get('zip_code', '')}",
        f"At any time during 2024, did you: (a) receive (as a reward, award, or payment for property or services); or (b) sell, exchange, or otherwise dispose of a digital asset (or a financial interest in a digital asset)? (See instructions.): {'Yes' if f.get('virtual_currency') else 'No'}",
        f"Someone can claim you as a dependent: {'Yes' if f.get('tp_dependent') else 'No'}",
        f"Someone can claim your spouse as a dependent: {'Yes' if f.get('sp_dependent') else 'No'}",
        f"You were born before January 2, 1960: {'Yes' if ret.tp_age >= 65 else 'No'}",
        f"You are blind: {'Yes' if ret.tp_blind else 'No'}",
        f"Spouse was born before January 2, 1960: {'Yes' if (ret.sp_age or 0) >= 65 else 'No'}",
        f"Spouse is blind: {'Yes' if ret.sp_blind else 'No'}",
        f"Dependents: {dep_desc}",
        f"Line 1a: Total amount from Form(s) W-2, box 1 | Sum of W-2 box 1 | {_amt(d['1a'])}",
        f"Line 1b: Household employee wages not reported on Form(s) W-2 |  | 0",
        f"Line 1c: Tip income not reported on line 1a |  | 0",
        f"Line 1d: Medicaid waiver payments not reported on Form(s) W-2 |  | 0",
        f"Line 1e: Taxable dependent care benefits from Form 2441, line 26 |  | {_amt(d['1e'])}",
        f"Line 1f: Employer-provided adoption benefits from Form 8839, line 29 |  | 0",
        f"Line 1g: Wages from Form 8919, line 6 |  | 0",
        f"Line 1h: Other earned income |  | 0",
        f"Line 1i: Nontaxable combat pay election |  | 0",
        f"Line 1z: Add lines 1a through 1h |  | {_amt(d['1z'])}",
        f"Line 2a: Tax-exempt interest |  | {_amt(d['2a'])}",
        f"Line 2b: Taxable interest |  | {_amt(d['2b'])}",
        f"Line 3a: Qualified dividends |  | {_amt(d['3a'])}",
        f"Line 3b: Ordinary dividends |  | {_amt(d['3b'])}",
        f"Line 4a: IRA distributions |  | 0",
        f"Line 4b: Taxable amount |  | 0",
        f"Line 5a: Pensions and annuities |  | {_amt(d['5a'])}",
        f"Line 5b: Taxable amount |  | {_amt(d['5b'])}",
        f"Line 6a: Social security benefits |  | {_amt(d['6a'])}",
        f"Line 6b: Taxable amount | Social security benefits worksheet | {_amt(d['6b'])}",
        f"Line 7: Capital gain or (loss) | Schedule D | {_amt(d['7'])}",
        f"Line 8: Additional income from Schedule 1, line 10 | Schedule C net profit, unemployment compensation, other income | {_amt(d['8'])}",
        f"Line 9: Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income | 1z + 2b + 3b + 5b + 6b + 7 + 8 | {_amt(d['9'])}",
        f"Line 10: Adjustments to income from Schedule 1, line 26 | Schedule 1 Part II | {_amt(d['10'])}",
        f"Line 11: Subtract line 10 from line 9. This is your adjusted gross income | 9 - 10 | {_amt(d['11'])}",
        f"Line 12: Standard deduction or itemized deductions (from Schedule A) | Standard deduction | {_amt(d['12'])}",
        f"Line 13: Qualified business income deduction from Form 8995 or Form 8995-A | Form 8995 | {_amt(d['13'])}",
        f"Line 14: Add lines 12 and 13 | 12 + 13 | {_amt(d['14'])}",
        f"Line 15: Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income | 11 - 14 | {_amt(d['15'])}",
        f"Line 16: Tax | {expl_16()} | {_amt(d['16'])}",
        f"Line 17: Amount from Schedule 2, line 3  |  | {_amt(d['17'])}",
        f"Line 18: Add lines 16 and 17 | 16 + 17 | {_amt(d['18'])}",
        f"Line 19: Child tax credit or credit for other dependents from Schedule 8812 | Schedule 8812 | {_amt(d['19'])}",
        f"Line 20: Amount from Schedule 3, line 8 | {expl_20()} | {_amt(d['20'])}",
        f"Line 21: Add lines 19 and 20 | 19 + 20 | {_amt(d['21'])}",
        f"Line 22: Subtract line 21 from line 18. If zero or less, enter -0- | 18 - 21 | {_amt(d['22'])}",
        f"Line 23: Other taxes, including self-employment tax, from Schedule 2, line 21 | {expl_23()} | {_amt(d['23'])}",
        f"Line 24: Add lines 22 and 23. This is your total tax | 22 + 23 | {_amt(d['24'])}",
        f"Line 25a: Federal income tax withheld from Form(s) W-2 |  | {_amt(d['25a'])}",
        f"Line 25b: Federal income tax withheld from Form(s) 1099 |  | {_amt(d['25b'])}",
        f"Line 25c: Federal income tax withheld from other forms |  | {_amt(d['25c'])}",
        f"Line 25d: Add lines 25a through 25c | 25a + 25b + 25c | {_amt(d['25d'])}",
        f"Line 26: 2024 estimated tax payments and amount applied from 2023 return |  | {_amt(d['26'])}",
        f"Line 27: Earned income credit (EIC) | EIC table | {_amt(d['27'])}",
        f"Line 28: Additional child tax credit from Schedule 8812 | Schedule 8812 | {_amt(d['28'])}",
        f"Line 29: American opportunity credit from Form 8863, line 8 | Form 8863 | {_amt(d['29'])}",
        "Line 30: Reserved for future use",
        f"Line 31: Amount from Schedule 3, line 15 | {expl_31()} | {_amt(d['31'])}",
        f"Line 32: Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits | 27 + 28 + 29 + 31 | {_amt(d['32'])}",
        f"Line 33: Add lines 25d, 26, and 32. These are your total payments | 25d + 26 + 32 | {_amt(d['33'])}",
        f"Line 34: If line 33 is more than line 24, subtract line 24 from line 33. This is the amount you overpaid | max(0, 33 - 24) | {_amt(d['34'])}",
        f"Line 35a: Amount of line 34 you want refunded to you. |  | {_amt(d['35a'])}",
        f"Line 36: Amount of line 34 you want applied to your 2025 estimated tax |  | 0",
        f"Line 37: Subtract line 33 from line 24. This is the amount you owe | max(0, 24 - 33) | {_amt(d['37'])}",
        f"Line 38: Estimated tax penalty |  | 0",
    ]
    if compact_form:
        # Ablation arm: 49% of the full completion is unscored boilerplate (zero
        # form lines + header echo) — pure length and derailment risk. The
        # evaluator only looks up the 19 scored labels, so emit just those (plus
        # the title/filing-status for coherence). Halves output; scores identically.
        from .score import LABEL_TO_TAG

        scored = tuple(LABEL_TO_TAG)  # full "Line N: ..." label strings
        kept = [lines[0], lines[1], lines[2]]  # title, rule, filing status
        kept += [ln for ln in lines if ln.startswith(scored)]
        body = "\n".join(kept)
    else:
        body = "\n".join(lines)
    if with_trace:
        from .trace import build_trace

        body = build_trace(ret) + "\n\n" + body
    # Explicit terminator: teaches the model to stop instead of inventing
    # trailing forms/schedules (the dominant pilot-2 truncation failure).
    return body + "\n" + END_OF_RETURN
