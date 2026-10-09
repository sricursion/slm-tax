"""Worked-computation traces ("show your work") for training targets.

Emits a CALCULATION NOTES block decomposing the hard arithmetic into explicit
steps so a small model can chain token-level operations instead of computing
multi-step sums implicitly. Every derived number mirrors the engine's own
formulas and is asserted equal to the oracle's value — a trace can never
contain numbers that disagree with the target form.

Trace lines deliberately avoid the exact benchmark label phrasings (e.g.
"Line 16: Tax") because TaxCalcBench's parser reads from the first occurrence
of each label string.
"""

from __future__ import annotations

import math

from .engine import (
    CTC_PER_CHILD,
    CTC_PHASEOUT_START,
    EITC,
    EITC_INVESTMENT_LIMIT,
    MAX_EMPLOYEE_SS_TAX,
    ODC_PER_DEP,
    SS_WAGE_BASE,
    STD_DEDUCTION,
    ADDL_STD,
    Return,
)
from .irs_math import CG_BRACKETS, RATE_SCHEDULES, rnd, schedule_tax, tax_table
from .tcb import num


def _f(x: float) -> str:
    # No thousands separators: Qwen tokenizes numbers digit-by-digit, so a comma
    # inserts a noise token that breaks the digit stream arithmetic relies on,
    # and it would mismatch the comma-free form values the model must output.
    return f"{int(round(x))}" if float(x).is_integer() else f"{x:.2f}"


def _summed(values: list[float]) -> str:
    """'a + b + c = total' when >1 value, else just the single value."""
    vals = [v for v in values if v]
    if len(vals) > 1:
        return f"{' + '.join(_f(v) for v in vals)} = {_f(sum(vals))}"
    return _f(sum(vals))


def _bracket_steps(amount: float, status: str, out: list[str], indent: str = "    ") -> None:
    """Write the piecewise rate-schedule computation bracket by bracket.

    Pilot-5 evidence: line 16 (tax) hit only 25% and was the first divergence in
    17/51 cases because the trace STATED the tax in one step, forcing the model
    to do a multi-bracket piecewise sum implicitly (e.g. it produced 12305 for
    MFJ/105800 where the answer is 13382). The model is 99.2% accurate on
    trace arithmetic of the "P% of N = R" shape, so writing each bracket turns
    its weakest step into its strongest.
    """
    lower = 0.0
    parts = []
    for top, rate in RATE_SCHEDULES[status]:
        if amount <= lower:
            break
        taxed = min(amount, top) - lower
        part = taxed * rate
        parts.append(part)
        out.append(f"{indent}{rate:.0%} of {_f(taxed)} = {_f(part)}")
        lower = top
    if len(parts) > 1:
        out.append(f"{indent}sum: {' + '.join(_f(p) for p in parts)} = {_f(sum(parts))}")


def _tax_table_steps(ti: float, status: str, out: list[str], prefix: str = "tax") -> int:
    ti = max(0.0, ti)
    if ti >= 100_000:
        val = rnd(schedule_tax(ti, status))
        out.append(f"  {prefix}: income {_f(ti)} >= 100000 -> rate schedule brackets:")
        _bracket_steps(ti, status, out)
        out.append(f"    rounded: {_f(val)}")
        return val
    if ti < 5:
        out.append(f"  {prefix}: income {_f(ti)} < 5 -> 0")
        return 0
    if ti < 15:
        mid = 10.0
    elif ti < 25:
        mid = 20.0
    elif ti < 3_000:
        mid = math.floor(ti / 25) * 25 + 12.5
    else:
        mid = math.floor(ti / 50) * 50 + 25.0
    val = rnd(schedule_tax(mid, status))
    out.append(f"  {prefix}: tax table row for {_f(ti)} -> midpoint {_f(mid)}; brackets:")
    _bracket_steps(mid, status, out)
    out.append(f"    rounded: {_f(val)}")
    return val


def build_trace(ret: Return) -> str:
    d = ret.detail
    t: list[str] = ["CALCULATION NOTES", "================="]

    # --- income assembly (show extraction+sum so the model learns to read the
    #     input, not just to state a total) --------------------------------------
    if ret.wages_box1:
        box1 = [num(w.get("wages")) for w in ret.data.get("w2") or []]
        t.append(f"* W-2 box 1: {_summed(box1)}")
    if d["1e"]:
        t.append(f"* taxable dependent care benefits = {_f(d['1e'])}; wages total {_f(d['1z'])}")
    if ret.interest:
        amts = []
        for i in (ret.data.get("irs1040_scheduleb", {}) or {}).get("irs1099_int") or []:
            amts.append(
                num(i.get("interest_1099int_interest"))
                + num(i.get("interest_1099int_us_saving_bond_treasury_obligations"))
            )
        t.append(f"* taxable interest: {_summed(amts)}")
    if ret.ordinary_div:
        divs = [
            num(dv.get("dividend_1099div_ordinary_div"))
            for dv in (ret.data.get("irs1040_scheduleb", {}) or {}).get("irs1099_div") or []
        ]
        t.append(
            f"* ordinary dividends: {_summed(divs)}"
            f" (qualified portion {_f(d['3a'])})"
        )
    if ret.pension_taxable:
        t.append(f"* pensions: gross {_f(d['5a'])}, taxable {_f(d['5b'])}")

    # Schedule C — derive gross and expenses from their itemized components.
    # Pilot-6: line 8 (Schedule 1) was wrong in 67% of total-income failures
    # because gross/expenses were stated as single numbers the model had to
    # extract-and-sum implicitly. Same fix that took line 1a from 49% -> 98%.
    if ret.qbi_businesses:
        for i, b in enumerate(ret.qbi_businesses, 1):
            inc = b.get("income_items") or []
            exp = b.get("expense_items") or []
            if len(inc) > 1:
                t.append(
                    f"* Schedule C business {i} ({b['pid']}) income:"
                    f" {' + '.join(_f(v) for _, v in inc)} = {_f(b['gross'])}"
                )
            if len(exp) > 1:
                t.append(
                    f"  expenses: {' + '.join(_f(v) for _, v in exp)} = {_f(b['expenses'])}"
                )
            t.append(
                f"* Schedule C business {i} ({b['pid']}): gross receipts {_f(b['gross'])}"
                f" - expenses {_f(b['expenses'])} = net profit {_f(b['profit'])}"
            )
        # Always emit the aggregate (even for one business) so the SE-tax step
        # always has an explicit "combined" antecedent to reference.
        profits = " + ".join(_f(b["profit"]) for b in ret.qbi_businesses)
        t.append(
            f"* combined self-employment net profit = {profits}"
            f" = {_f(sum(ret.se_profit.values()))}"
        )

    # SE tax per person (mirror of _se_tax)
    se_tax_total = 0
    se_ald_total = 0
    for pid in ("tp", "sp"):
        profit = ret.se_profit[pid]
        base = profit * 0.9235
        if base >= 400:
            ss_taxable = max(0.0, min(base, SS_WAGE_BASE - min(ret.ss_wages[pid], SS_WAGE_BASE)))
            ss_part = rnd(0.124 * ss_taxable)
            med_part = rnd(0.029 * base)
            tax = ss_part + med_part
            ald = rnd(tax / 2.0)
            se_tax_total += tax
            se_ald_total += ald
            # Say "combined net profit" explicitly: SE tax is computed ONCE on
            # the aggregate, never per business. The 4B model over-generalized
            # the per-business Schedule C lines and emitted "SE tax (business
            # 1/2/3)", summing separately-rounded halves and corrupting line 10.
            t.append(
                f"* SE tax ({pid}) on combined net profit {_f(profit)}:"
                f" {_f(profit)} x 0.9235 = {_f(base)};"
                f" SS 12.4% of {_f(ss_taxable)} = {_f(ss_part)};"
                f" Medicare 2.9% = {_f(med_part)}; SE tax = {_f(tax)}; half = {_f(ald)}"
            )
    assert se_tax_total == d["se_tax"], (se_tax_total, d["se_tax"])

    # Social security benefits — derive taxability (Pub 915 worksheet)
    if ret.ss_benefits:
        prov = ret._ss_provisional
        b1, b2 = ret._ss_base1, ret._ss_base2
        t.append(
            f"* social security benefits {_f(d['6a'])}; provisional income"
            f" (other income + half of benefits) = {_f(prov)}"
        )
        if prov <= b1:
            t.append(
                f"  provisional {_f(prov)} <= threshold {_f(b1)} -> 0 taxable"
            )
        else:
            band = "over upper threshold" if prov > b2 else "between thresholds"
            t.append(
                f"  provisional {_f(prov)} {band} ({_f(b1)}/{_f(b2)})"
                f" -> taxable portion = {_f(d['6b'])}"
            )

    # Capital gains
    if ret.st_gain or ret.lt_gain or ret.cap_gain_distr:
        net = ret.st_gain + ret.lt_gain + ret.cap_gain_distr
        t.append(
            f"* capital: short-term {_f(ret.st_gain)}, long-term {_f(ret.lt_gain)},"
            f" distributions {_f(ret.cap_gain_distr)}; net = {_f(net)} -> reported {_f(d['7'])}"
        )
    if ret.unemployment:
        t.append(f"* unemployment compensation = {_f(ret.unemployment)}")
    if ret.ak_pfd:
        t.append(f"* Alaska PFD = {_f(ret.ak_pfd)}")

    # Schedule 1 additional income (line 8) — derive before it enters line 9
    if d["8"]:
        s1 = []
        se_total = sum(ret.se_profit.values())
        if se_total:
            s1.append(f"Schedule C {_f(se_total)}")
        if ret.unemployment:
            s1.append(f"unemployment {_f(ret.unemployment)}")
        if ret.ak_pfd:
            s1.append(f"Alaska PFD {_f(ret.ak_pfd)}")
        # always show line 8, even with a single component: it is the #1 source
        # of total-income errors and the model must see it as its own step.
        t.append(f"* additional income (Schedule 1) = {' + '.join(s1)} = {_f(d['8'])}")

    # Total income and adjustments
    t.append(
        f"* total income = {_f(d['1z'])} + {_f(d['2b'])} + {_f(d['3b'])} + {_f(d['5b'])}"
        f" + {_f(d['6b'])} + {_f(d['7'])} + {_f(d['8'])} = {_f(d['9'])}"
    )
    adj_parts = []
    if se_ald_total:
        adj_parts.append(f"half SE tax {_f(se_ald_total)}")
    if d["se_health_ald"]:
        adj_parts.append(f"SE health insurance {_f(d['se_health_ald'])}")
    if d["educator_ald"]:
        adj_parts.append(f"educator {_f(d['educator_ald'])}")
    if d["student_loan_ald"]:
        adj_parts.append(f"student loan interest {_f(d['student_loan_ald'])}")
    if d["early_withdrawal_ald"]:
        adj_parts.append(f"early-withdrawal penalty {_f(d['early_withdrawal_ald'])}")
    # ALWAYS emit, including the zero case. Previously this line was omitted
    # when adjustments were 0, so the model never saw "adjustments = 0" and
    # hallucinated one on complex returns (e.g. invented "other deductions
    # 2815", cascading through AGI -> taxable income -> tax -> credits).
    t.append(f"* adjustments ({', '.join(adj_parts) if adj_parts else 'none'}) = {_f(d['10'])}")
    t.append(f"* AGI = {_f(d['9'])} - {_f(d['10'])} = {_f(d['11'])}")

    # Standard deduction
    base_std = STD_DEDUCTION[ret.status]
    boxes = int(ret.tp_age >= 65) + int(ret.tp_blind)
    if ret.joint:
        boxes += int((ret.sp_age or 0) >= 65) + int(ret.sp_blind)
    if boxes or d["12"] != base_std:
        t.append(
            f"* standard deduction: base {_f(base_std)}"
            + (f" + {boxes} x {_f(ADDL_STD[ret.status])} (65+/blind)" if boxes else "")
            + f" = {_f(d['12'])}"
        )
    else:
        t.append(f"* standard deduction = {_f(d['12'])}")
    if d["13"]:
        # Form 8995: lesser of 20% of QBI and 20% of (taxable income - net
        # capital gains). Derive both so the model computes rather than guesses.
        qbi_20 = 0.2 * ret._qbid_total
        t.append(
            f"* QBI deduction: 20% of QBI {_f(ret._qbid_total)} = {_f(qbi_20)};"
            f" income limit = 20% of {_f(ret._qbid_ti_net_cg)}"
            f" (taxable income before QBI minus net capital gains)"
            f" = {_f(ret._qbid_income_limit)}; lesser = {_f(d['13'])}"
        )
        t.append(f"* total deductions = {_f(d['12'])} + {_f(d['13'])} = {_f(d['14'])}")
    raw_ti = d["11"] - d["14"]
    if raw_ti < 0:
        # IRS: "if zero or less, enter -0-" — show the floor so the trace never
        # teaches that a smaller-minus-larger subtraction equals 0.
        t.append(
            f"* taxable income = {_f(d['11'])} - {_f(d['14'])} = {_f(raw_ti)},"
            f" not less than 0 -> 0"
        )
    else:
        t.append(f"* taxable income = {_f(d['11'])} - {_f(d['14'])} = {_f(d['15'])}")

    # Tax (line 16 arithmetic, mirrored)
    ti = float(d["15"])
    schd_15 = ret.lt_gain + ret.cap_gain_distr
    schd_16 = ret.st_gain + ret.lt_gain + ret.cap_gain_distr
    net_cg_pref = (
        max(0.0, min(schd_15, schd_16)) if schd_16 > 0 or ret.cap_gain_distr else 0.0
    )
    if (ret.qualified_div > 0 or net_cg_pref > 0) and ti > 0:
        qd = ret.qualified_div
        l3 = min(qd + net_cg_pref, ti)
        l5 = ti - l3
        t0_, t15_ = CG_BRACKETS[ret.status]
        l7 = min(ti, t0_)
        l8 = min(l5, l7)
        l9 = l7 - l8
        l10 = min(ti, l3)
        l12 = l10 - l9
        l14 = min(ti, t15_)
        l15v = l5 + l9
        l16v = max(0.0, l14 - l15v)
        l17 = min(l12, l16v)
        t.append(
            f"* qualified div/cap-gain worksheet: preferential {_f(l3)}"
            f" (qualified div {_f(qd)} + net LT gain {_f(net_cg_pref)}, capped at income),"
            f" ordinary {_f(l5)}"
        )
        computed = _tax_table_steps(l5, ret.status, t, prefix="ordinary part")
        if l9:
            t.append(f"  0% bracket covers {_f(l9)}")
        if l17:
            t.append(f"  15% bracket: {_f(l17)} x 0.15 = {_f(l17 * 0.15)}")
        l20 = l10 - (l9 + l17)
        if l20 > 0:
            t.append(f"  20% bracket: {_f(l20)} x 0.20 = {_f(l20 * 0.20)}")
        total = min(rnd(l17 * 0.15 + max(0.0, l20) * 0.20 + computed), tax_table(ti, ret.status))
        t.append(f"  income tax = {_f(total)}")
        assert total == d["16"], (total, d["16"])
    elif ti > 0:
        computed = _tax_table_steps(ti, ret.status, t, prefix="income tax")
        assert computed == d["16"], (computed, d["16"])
    else:
        t.append("* taxable income is zero -> income tax 0")

    if d["17"]:
        t.append(f"* excess advance premium tax credit repayment = {_f(d['17'])}")

    # Credits — derive the worksheet-based ones before summing Schedule 3
    if d["cdcc"]:
        t.append(
            f"* child/dependent care credit: eligible expenses {_f(ret._cdcc_expenses)}"
            f" x rate {ret._cdcc_rate:.0%} = {_f(d['cdcc'])}"
        )
    if d["aoc_nonref"] or d["29"]:
        total_edu = ret._aoc + ret._llc
        t.append(
            f"* education credits (Form 8863): American Opportunity {_f(ret._aoc)}"
            f" + Lifetime Learning {_f(ret._llc)} = {_f(total_edu)}"
            + (f"; refundable 40% = {_f(d['29'])}" if d["29"] else "")
        )
    if d["20"]:
        parts = []
        for key, label in [("ftc", "foreign tax credit"), ("cdcc", "dependent care credit"),
                           ("aoc_nonref", "education (nonrefundable)"), ("savers", "saver's credit")]:
            if d[key]:
                parts.append(f"{label} {_f(d[key])}")
        t.append(f"* Schedule 3 nonrefundable credits: {' + '.join(parts)} = {_f(d['20'])}")

    ctc_kids, odc = d["ctc_kids"], d["odc_count"]
    if ret.dependents:
        # DERIVE each dependent's classification. Previously the trace only
        # stated the counts, so the model had to apply the age/relationship/
        # SSN rules implicitly and routinely miscounted (e.g. "3 children + 3
        # other" vs truth "3 + 2", and "3 qualifying children" vs 4 for EITC).
        for dep in ret.dependents:
            if not ret.tp_dependent and dep["qc"] and dep["age"] < 17 and dep["ssn_ok"]:
                cls = "child tax credit (under 17, qualifying child, valid SSN)"
            elif not ret.tp_dependent:
                why = "17 or older" if dep["age"] >= 17 else (
                    "not a qualifying child" if not dep["qc"] else "no valid SSN"
                )
                cls = f"other dependent ({why})"
            else:
                cls = "no credit (taxpayer is claimed as a dependent)"
            eitc_ok = dep["qc"] and dep["ssn_ok"] and not ret.eic_not_allowed
            t.append(
                f"  dependent age {dep['age']}, months lived {int(dep['months'])}"
                f"{', student' if dep['student'] else ''} -> {cls}"
                f"; EITC qualifying child: {'yes' if eitc_ok else 'no'}"
            )
        t.append(
            f"  totals: {ctc_kids} child-tax-credit children,"
            f" {odc} other dependents,"
            f" {sum(1 for x in ret.dependents if x['qc'] and x['ssn_ok'])} EITC qualifying children"
        )
    if ctc_kids or odc:
        potential = CTC_PER_CHILD * ctc_kids + ODC_PER_DEP * odc
        agi = float(d.get("_agi", d["11"]))
        po = CTC_PHASEOUT_START.get(ret.status, 200_000.0)
        note = f"* child credit: {ctc_kids} qualifying children x 2,000 + {odc} other dependents x 500 = {_f(potential)}"
        if agi > po:
            over = math.ceil((agi - po) / 1_000.0) * 1_000
            potential = max(0.0, potential - 0.05 * over)
            note += f"; phase-out 5% of {_f(over)} -> {_f(potential)}"
        limit = max(0, d["18"] - d["20"])
        note += f"; limited by tax {_f(limit)} -> nonrefundable {_f(d['19'])}"
        t.append(note)
        if d["28"]:
            t.append(
                f"* additional child tax credit: min(remaining credit, 1,700/child,"
                f" 15% of earned income over 2,500) = {_f(d['28'])}"
            )

    # Other taxes
    if d["23"]:
        parts = []
        for key, label in [("se_tax", "SE tax"), ("addl_medicare", "additional Medicare"),
                           ("niit", "net investment income tax"), ("uncollected", "uncollected SS/Medicare")]:
            if d[key]:
                parts.append(f"{label} {_f(d[key])}")
        t.append(f"* other taxes: {' + '.join(parts)} = {_f(d['23'])}")
    t.append(f"* total tax = {_f(d['22'])} + {_f(d['23'])} = {_f(d['24'])}")

    # EITC (mirror of _eitc_amount). Emit whenever the credit is nonzero OR
    # there are EITC-qualifying children — the dependents block above names
    # those children, and leaving the credit unresolved afterwards is the same
    # prime-then-omit pattern that made the model hallucinate adjustments.
    # When the answer is 0, DERIVE the disqualifier explicitly.
    kids = sum(1 for x in ret.dependents if x["qc"] and x["ssn_ok"])
    if d["27"] or kids:
        earned = sum(ret.earned.values())
        agi = float(d.get("_agi", d["11"]))
        invest = ret.interest + ret.tax_exempt_interest + ret.ordinary_div + max(
            0.0, ret.st_gain + ret.lt_gain + ret.cap_gain_distr
        )
        zero_reason = None
        if ret.eic_not_allowed:
            zero_reason = "EIC not allowed for this taxpayer"
        elif ret.tp_dependent:
            zero_reason = "taxpayer can be claimed as a dependent"
        elif ret.status == "married_separately":
            zero_reason = "filing status is married filing separately"
        elif invest > EITC_INVESTMENT_LIMIT:
            zero_reason = (
                f"investment income {_f(invest)} > limit {_f(EITC_INVESTMENT_LIMIT)}"
            )
        elif earned <= 0:
            zero_reason = "no earned income"
        if zero_reason is not None:
            t.append(f"* earned income credit: {zero_reason} -> 0")
            assert d["27"] == 0, (zero_reason, d["27"])
            _emit_eitc = False
        else:
            _emit_eitc = True
    else:
        _emit_eitc = False
    if _emit_eitc:
        rate_in, max_credit, rate_out, po_s, po_j = EITC[min(kids, 3)]
        po_start = po_j if ret.joint else po_s

        def mid(x: float) -> float:
            if x < 1:
                return 0.0
            if x < 25:
                return 12.5
            return math.floor(x / 50) * 50 + 25.0

        def f(x: float) -> float:
            m = mid(x)
            return max(0.0, min(max_credit, rate_in * m) - max(0.0, (m - po_start) * rate_out))

        def steps(x: float, label: str) -> float:
            """Derive the EITC table value: phase-in %, cap, phase-out."""
            m = mid(x)
            phase_in = rate_in * m
            capped = min(max_credit, phase_in)
            reduction = max(0.0, (m - po_start) * rate_out)
            val_ = max(0.0, capped - reduction)
            t.append(
                f"  {label} {_f(x)} -> row midpoint {_f(m)};"
                f" {rate_in:.2%} of {_f(m)} = {_f(phase_in)}"
                + (f"; capped at max {_f(max_credit)}" if phase_in > max_credit else "")
                + (
                    f"; over phase-out start {_f(po_start)} by {_f(m - po_start)},"
                    f" reduce {rate_out:.2%} = {_f(reduction)}"
                    if reduction > 0
                    else ""
                )
                + f" -> {_f(val_)}"
            )
            return val_

        t.append(f"* earned income credit ({kids} qualifying children):")
        val = steps(earned, "earned income")
        if agi > po_start:
            v_agi = steps(agi, "AGI")
            val = min(val, v_agi)
            t.append(f"  smaller of the two = {_f(val)}")
        assert rnd(val) == d["27"], (rnd(val), d["27"])

    # Payments
    wh_parts = [f"W-2 withholding {_f(d['25a'])}"]
    if d["25b"]:
        wh_parts.append(f"1099 withholding {_f(d['25b'])}")
    if d["25c"]:
        wh_parts.append(f"other (Form 8959) {_f(d['25c'])}")
    t.append(f"* payments withheld: {' + '.join(wh_parts)} = {_f(d['25d'])}")
    if d["26"]:
        t.append(f"* estimated payments applied = {_f(d['26'])}")
    if d["31"]:
        parts = []
        if d["extension"]:
            parts.append(f"extension payment {_f(d['extension'])}")
        if d["excess_ss"]:
            parts.append(f"excess social security {_f(d['excess_ss'])}")
        if d["net_ptc"]:
            parts.append(f"net premium tax credit {_f(d['net_ptc'])}")
        t.append(f"* other refundable amounts: {' + '.join(parts)} = {_f(d['31'])}")
    t.append(
        f"* total payments = {_f(d['25d'])} + {_f(d['26'])} + {_f(d['32'])} = {_f(d['33'])}"
    )
    if d["34"]:
        t.append(f"* overpaid: {_f(d['33'])} - {_f(d['24'])} = {_f(d['34'])} refund")
    else:
        t.append(f"* amount owed: {_f(d['24'])} - {_f(d['33'])} = {_f(d['37'])}")

    return "\n".join(t)
