"""Compute the 19 TaxCalcBench-scored Form 1040 lines for a TY24 case.

Strategy: PolicyEngine-US supplies the hard tax semantics (AGI, deductions,
taxable income, credits, SE tax); this module supplies the parts a
microsimulation engine doesn't do 1:1 with the form — input echoes
(withholding, estimated payments), the IRS tax-table lookup for line 16, the
QDCGT worksheet, Schedule C net-profit assembly from raw expense fields, and
line arithmetic (lines 33/34/35a/37).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from policyengine_us import Simulation

from .irs_math import (
    HOME_OFFICE_RATE,
    HOME_OFFICE_SQFT_CAP,
    MAX_EMPLOYEE_SS_TAX_2024,
    STANDARD_MILEAGE_RATE_2024,
    qdcgt_worksheet,
    rnd,
    tax_table,
)
from .tcb import num

YEAR = 2024
P = str(YEAR)

FILING_STATUS_MAP = {
    "single": "SINGLE",
    "married_jointly": "JOINT",
    "married_separately": "SEPARATE",
    "head_of_household": "HEAD_OF_HOUSEHOLD",
    "qualifying_widow": "SURVIVING_SPOUSE",
}

# W-2 box 12 codes for uncollected social security / Medicare tax (Schedule 2
# line 13) and elective deferrals that count for the saver's credit.
UNCOLLECTED_TAX_CODES = {"A", "B", "M", "N"}
SAVERS_CREDIT_CODES = {"D", "E", "F", "G", "H", "S", "AA", "BB", "EE"}


def _age_at_year_end(dob_str: str) -> int:
    dob = dt.date.fromisoformat(dob_str)
    # IRS convention: born Jan 1 counts as having the birthday in the prior
    # year (age as of Jan 1 of the following year).
    ref = dt.date(YEAR + 1, 1, 1)
    age = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    return age


def _who(entry: dict) -> str:
    w = entry.get("who_applies_to", "taxpayer")
    return {"taxpayer": "tp", "spouse": "sp"}.get(w, w)


def _schedule_c_net_profit(sc: dict) -> float:
    """Net profit per Schedule C from raw input fields."""
    income = num(sc.get("gross_receipts_cash"))
    for nec in sc.get("irs1099_nec") or []:
        income += num(nec.get("nec_comp"))
    for k in sc.get("irs1099_k") or []:
        income += num(k.get("k_comp"))
    for m in sc.get("irs1099_misc") or []:
        income += num(m.get("other_income")) + num(m.get("nonqual_deferred_comp"))
    income += num(sc.get("other_income"))
    income -= num(sc.get("returns_allowances"))

    expense_fields = [
        "advertising", "commissions_fees", "contract_labor", "depletion",
        "employee_benefit", "insurance", "legal_professional",
        "machinery_equip_rent", "other_rent", "mortgage_interest",
        "other_interest", "office_expense", "pension_psp",
        "repairs_maintenance", "supplies", "tax_licenses", "travel",
        "utilities", "wages_expense",
    ]
    expenses = sum(num(sc.get(f)) for f in expense_fields)
    expenses += 0.5 * num(sc.get("meal_entertainment"))  # 50% limit
    for oe in sc.get("other_expense_detail") or []:
        expenses += num(oe.get("other_expense_detail_amt"))
    for v in sc.get("vehicle_info_group") or []:
        if v.get("vehicle_standard_mileage", True):
            expenses += STANDARD_MILEAGE_RATE_2024 * num(v.get("vehicle_business_miles"))
        expenses += (
            num(v.get("car_truck_expense_parking"))
            + num(v.get("car_truck_expense_tolls"))
            + num(v.get("car_truck_expense_interest"))
            + num(v.get("car_truck_expense_tax"))
        )
    biz_area = num(sc.get("business_home_area"))
    if biz_area > 0:
        expenses += HOME_OFFICE_RATE * min(biz_area, HOME_OFFICE_SQFT_CAP)
    return income - expenses


def build(case: dict) -> tuple[dict, dict]:
    """Build (policyengine_situation, echo) from a flattened TCB case."""
    data = case["data"]
    header = case["header"]
    f1040 = data.get("irs1040", {})
    sched1 = data.get("irs1040_schedule1", {})
    sched3 = data.get("irs1040_schedule3", {})

    filing_status_raw = f1040.get("filing_status", "single")
    is_joint = filing_status_raw == "married_jointly"

    people: dict[str, dict] = {}
    echo: dict[str, Any] = {
        "filing_status": filing_status_raw,
        "warnings": [],
    }

    # --- people -----------------------------------------------------------
    people["tp"] = {
        "age": {P: _age_at_year_end(f1040["tp_date_of_birth"])},
        "is_blind": {P: bool(f1040.get("tp_blind"))},
        "is_tax_unit_dependent": {P: False},
    }
    if is_joint:
        people["sp"] = {
            "age": {P: _age_at_year_end(f1040["sp_date_of_birth"])},
            "is_blind": {P: bool(f1040.get("sp_blind"))},
            "is_tax_unit_dependent": {P: False},
        }

    dep_ids = []
    for i, dep in enumerate(f1040.get("dependent_detail") or []):
        pid = f"dep{i}"
        dep_ids.append(pid)
        people[pid] = {
            "age": {P: _age_at_year_end(dep["dependent_date_of_birth"])},
            "is_tax_unit_dependent": {P: True},
        }
        if dep.get("dependent_disabled"):
            people[pid]["is_permanently_and_totally_disabled"] = {P: True}
        if dep.get("dependent_student_for_5_plus_months"):
            people[pid]["is_full_time_student"] = {P: True}
        ssn_type = str(dep.get("dependent_ssn_itin_atin") or "ssn").lower()
        if ssn_type != "ssn":
            people[pid]["ssn_card_type"] = {P: "NONE" if ssn_type == "itin" else "NON_CITIZEN"}
        elif dep.get("dependent_ssn_employment_valid") is False:
            people[pid]["ssn_card_type"] = {P: "NON_CITIZEN"}

    # --- W-2 wages ---------------------------------------------------------
    wages: dict[str, float] = {}
    ss_tax_by_person: dict[str, list[float]] = {}
    withholding = 0.0
    uncollected_tax = 0.0
    savers_contrib: dict[str, float] = {}
    dependent_care_benefits = 0.0
    for w2 in data.get("w2") or []:
        pid = _who(w2)
        if pid not in people:
            echo["warnings"].append(f"W-2 for unknown person {pid}")
            pid = "tp"
        wages[pid] = wages.get(pid, 0.0) + num(w2.get("wages"))
        withholding += num(w2.get("withholding"))
        ss_tax_by_person.setdefault(pid, []).append(num(w2.get("social_security_tax")))
        dependent_care_benefits += num(w2.get("dependent_care_benefits"))
        for grp in w2.get("employers_use_grp") or []:
            code = str(grp.get("box_12_code", "")).upper()
            amt = num(grp.get("box_12_amount"))
            if code in UNCOLLECTED_TAX_CODES:
                uncollected_tax += amt
            elif code in SAVERS_CREDIT_CODES:
                savers_contrib[pid] = savers_contrib.get(pid, 0.0) + amt
    for pid, amt in wages.items():
        people[pid]["employment_income"] = {P: amt}
    for pid, amt in savers_contrib.items():
        people[pid]["savers_credit_qualified_contributions"] = {P: amt}
    echo["wages_box1_total"] = sum(wages.values())
    echo["dependent_care_benefits"] = dependent_care_benefits

    # Excess social security tax withheld (2+ employers, per person)
    excess_ss = 0.0
    for pid, taxes in ss_tax_by_person.items():
        if len(taxes) >= 2:
            excess_ss += max(0.0, sum(taxes) - MAX_EMPLOYEE_SS_TAX_2024)
    echo["excess_ss"] = excess_ss
    echo["uncollected_tax"] = uncollected_tax

    # --- Schedule C / self-employment income --------------------------------
    se_income: dict[str, float] = {}
    for sc in data.get("irs1040_schedulec") or []:
        pid = _who(sc)
        if pid not in people:
            pid = "tp"
        profit = _schedule_c_net_profit(sc)
        se_income[pid] = se_income.get(pid, 0.0) + profit
        for nec in sc.get("irs1099_nec") or []:
            withholding += num(nec.get("nec_fed_wh"))
        for k in sc.get("irs1099_k") or []:
            withholding += num(k.get("k_fed_wh"))
        for m in sc.get("irs1099_misc") or []:
            withholding += num(m.get("misc_fed_wh"))
    for pid, amt in se_income.items():
        people[pid]["self_employment_income"] = {P: amt}

    # --- Schedule B: interest and dividends ---------------------------------
    taxable_interest: dict[str, float] = {}
    exempt_interest: dict[str, float] = {}
    qualified_div: dict[str, float] = {}
    ordinary_div: dict[str, float] = {}
    cap_gain_distr = 0.0
    schedb = data.get("irs1040_scheduleb", {})
    for i1099 in schedb.get("irs1099_int") or []:
        pid = _who({"who_applies_to": i1099.get("interest_1099int_who_applies_to", "taxpayer")})
        if pid not in people:
            pid = "tp"
        taxable_interest[pid] = (
            taxable_interest.get(pid, 0.0)
            + num(i1099.get("interest_1099int_interest"))
            + num(i1099.get("interest_1099int_us_saving_bond_treasury_obligations"))
        )
        exempt_interest[pid] = exempt_interest.get(pid, 0.0) + num(
            i1099.get("interest_1099int_tax_exempt")
        )
        withholding += num(i1099.get("interest_1099int_fed_inc_tax_withheld"))
    for d1099 in schedb.get("irs1099_div") or []:
        pid = _who({"who_applies_to": d1099.get("dividend_1099div_who_applies_to", "taxpayer")})
        if pid not in people:
            pid = "tp"
        ordinary = num(d1099.get("dividend_1099div_ordinary_div"))
        qualified = num(d1099.get("dividend_1099div_qualified_div"))
        ordinary_div[pid] = ordinary_div.get(pid, 0.0) + ordinary
        qualified_div[pid] = qualified_div.get(pid, 0.0) + qualified
        cap_gain_distr += num(d1099.get("dividend_1099div_capital_gain_distr"))
        withholding += num(d1099.get("dividend_1099div_federal_inc_tax_withheld"))
    for pid in taxable_interest:
        people[pid]["taxable_interest_income"] = {P: taxable_interest[pid]}
    for pid in exempt_interest:
        if exempt_interest[pid]:
            people[pid]["tax_exempt_interest_income"] = {P: exempt_interest[pid]}
    for pid in ordinary_div:
        people[pid]["qualified_dividend_income"] = {P: qualified_div.get(pid, 0.0)}
        people[pid]["non_qualified_dividend_income"] = {
            P: ordinary_div[pid] - qualified_div.get(pid, 0.0)
        }
    echo["qualified_dividends"] = sum(qualified_div.values())

    # --- Capital gains: Form 8949 / Schedule D -------------------------------
    st_gain = 0.0
    lt_gain = 0.0
    for grp in (data.get("irs8949") or {}).get("irs1099_b_grp") or []:
        for b in grp.get("irs1099_b") or []:
            gain = num(b.get("proceeds")) - num(b.get("cost_basis"))
            gain += num(b.get("wash_sale_loss"))  # disallowed loss added back
            gain += num(b.get("accrued_market_discount")) * 0  # goes to interest; rare
            term = str(b.get("input_sale_term", "")).lower()
            if "short" in term:
                st_gain += gain
            else:
                lt_gain += gain
            withholding += num(b.get("fed_inc_tax_wh"))
    schedd = data.get("irs1040_schedule_d", {})
    st_gain -= abs(num(schedd.get("st_capital_loss_carryover_input")))
    lt_gain -= abs(num(schedd.get("lt_capital_loss_carryover_input")))
    if st_gain or lt_gain:
        people["tp"]["short_term_capital_gains"] = {P: st_gain}
        people["tp"]["long_term_capital_gains"] = {P: lt_gain}
    if cap_gain_distr:
        people["tp"]["capital_gain_distributions"] = {P: cap_gain_distr}
    echo["st_gain"], echo["lt_gain"], echo["cap_gain_distr"] = st_gain, lt_gain, cap_gain_distr

    # --- Retirement and other income ----------------------------------------
    for r in data.get("irs1099_r") or []:
        pid = _who(r)
        if pid not in people:
            pid = "tp"
        gross = num(r.get("gross_distribution_amt_input"))
        taxable = num(r.get("taxable_amt_input"))
        people[pid]["pension_income"] = {P: gross}
        people[pid]["taxable_pension_income"] = {P: taxable}
        withholding += num(r.get("federal_income_tax_withheld_amt_input"))
    for grp in (data.get("ssa_1099") or {}).get("ssa_1099_grp") or []:
        pid = _who({"who_applies_to": grp.get("ssa_who_applies_to", "taxpayer")})
        if pid not in people:
            pid = "tp"
        people[pid]["social_security_retirement"] = {
            P: people[pid].get("social_security_retirement", {}).get(P, 0.0)
            + num(grp.get("ssa_net_benefits"))
        }
        withholding += num(grp.get("ssa_federal_withholding"))
    for g in sched1.get("irs1099_g") or []:
        pid = _who({"who_applies_to": g.get("g_who_applies_to", "taxpayer")})
        if pid not in people:
            pid = "tp"
        comp = num(g.get("g_unemployment_comp")) - num(g.get("g_amount_repaid_current_year"))
        people[pid]["unemployment_compensation"] = {
            P: people[pid].get("unemployment_compensation", {}).get(P, 0.0) + comp
        }
        withholding += num(g.get("g_fed_wh"))

    # --- Adjustments ---------------------------------------------------------
    if sched1.get("qualified_educator"):
        people["tp"]["educator_expense"] = {P: num(sched1.get("tp_educator_exp_amount"))}
        if is_joint:
            people["sp"]["educator_expense"] = {P: num(sched1.get("sp_educator_exp_amount"))}
    if sched1.get("paid_student_loan_interest"):
        people["tp"]["student_loan_interest"] = {P: num(sched1.get("student_interest"))}

    # --- Child and dependent care (Form 2441) --------------------------------
    irs2441 = data.get("irs2441", {})
    for qp in irs2441.get("qualifying_person_grp") or []:
        person_ref = qp.get("person")
        expenses = num(qp.get("qualifying_expenses_paid"))
        # person ref is an index or name; map to dep ids in order
        idx = None
        if isinstance(person_ref, (int, float)):
            idx = int(person_ref)
        target = dep_ids[idx] if idx is not None and idx < len(dep_ids) else (dep_ids[0] if dep_ids else None)
        if target and expenses:
            people[target]["childcare_expenses"] = {P: expenses}

    # --- Education credits (Form 8863) ---------------------------------------
    irs8863 = data.get("irs8863", {})
    for i, sg in enumerate(irs8863.get("student_educational_instn_grp") or []):
        expenses = sum(
            num(inst.get("qualified_expenses"))
            for inst in sg.get("educational_institution_group") or []
        )
        student_ref = sg.get("student")
        target = None
        if isinstance(student_ref, str) and student_ref in ("taxpayer", "spouse"):
            target = "tp" if student_ref == "taxpayer" else "sp"
        elif dep_ids:
            target = dep_ids[min(i, len(dep_ids) - 1)]
        else:
            target = "tp"
        if expenses and target:
            people[target]["qualified_tuition_expenses"] = {P: expenses}
            people[target]["is_eligible_for_american_opportunity_credit"] = {P: True}

    # --- Alaska PFD (explicitly zero unless in input) -------------------------
    pfd = num(sched1.get("alaska_permanent_dividend_tp_input"))
    people["tp"]["ak_permanent_fund_dividend"] = {P: pfd}
    for pid in people:
        if pid != "tp":
            people[pid]["ak_permanent_fund_dividend"] = {P: 0.0}

    # --- payments (echo only) -------------------------------------------------
    est = sum(num(f1040.get(f"estimated_tax_payment_{i}")) for i in (1, 2, 3, 4))
    est += num(f1040.get("applied_from_prior_year"))
    echo["estimated_payments"] = est
    echo["extension_payment"] = (
        num(sched3.get("extension_payment")) if sched3.get("requested_extension") else 0.0
    )
    echo["withholding"] = withholding
    echo["eic_not_allowed"] = bool(f1040.get("eic_not_allowed"))
    echo["tp_is_dependent"] = bool(f1040.get("tp_dependent"))

    # --- assemble situation -----------------------------------------------------
    members = list(people.keys())
    situation = {
        "people": people,
        "tax_units": {
            "tax_unit": {
                "members": members,
                "filing_status": {P: FILING_STATUS_MAP[filing_status_raw]},
            }
        },
        "families": {"family": {"members": members}},
        "spm_units": {"spm_unit": {"members": members}},
        "households": {
            "household": {
                "members": members,
                "state_name": {P: header.get("state", "TX")},
            }
        },
    }
    if is_joint:
        situation["marital_units"] = {"marital_unit": {"members": ["tp", "sp"]}}
    return situation, echo


def _tu(sim: Simulation, var: str) -> float:
    return float(sim.calculate(var, YEAR)[0])


def compute_lines(case: dict) -> dict[str, int]:
    """Return the 19 scored Form 1040 values keyed by MeF XML tag name."""
    situation, echo = build(case)
    sim = Simulation(situation=situation)
    status = echo["filing_status"]

    agi = _tu(sim, "adjusted_gross_income")
    adjustments = _tu(sim, "above_the_line_deductions")
    total_income = agi + adjustments

    itemizes = bool(_tu(sim, "tax_unit_itemizes"))
    deduction = (
        _tu(sim, "itemized_taxable_income_deductions")
        if itemizes
        else _tu(sim, "standard_deduction")
    )
    qbid = _tu(sim, "qualified_business_income_deduction")
    taxable_income = max(0.0, rnd(total_income) - rnd(adjustments) - rnd(deduction) - rnd(qbid))

    # Line 16: tax-table / QDCGT worksheet
    qd = echo["qualified_dividends"]
    st, lt, cgd = echo["st_gain"], echo["lt_gain"], echo["cap_gain_distr"]
    # Net capital gain taxed preferentially: min(Sch D 15, 16) floored at 0
    schd_15 = lt + cgd
    schd_16 = st + lt + cgd
    net_cg = max(0.0, min(schd_15, schd_16))
    if qd > 0 or net_cg > 0:
        tax16 = qdcgt_worksheet(taxable_income, qd, net_cg, status)
    else:
        tax16 = tax_table(taxable_income, status)

    # Nonrefundable credits
    ctc_nonref = _tu(sim, "non_refundable_ctc")
    cdcc = _tu(sim, "capped_cdcc")
    education_nonref = _tu(sim, "non_refundable_american_opportunity_credit") + _tu(
        sim, "lifetime_learning_credit"
    )
    savers = _tu(sim, "savers_credit")
    elderly_disabled = _tu(sim, "elderly_disabled_credit")
    amt = _tu(sim, "alternative_minimum_tax")

    line17 = rnd(amt)
    line18 = tax16 + line17
    sch3_nonref = rnd(cdcc) + rnd(education_nonref) + rnd(savers) + rnd(elderly_disabled)
    line19 = rnd(ctc_nonref)
    line21 = line19 + sch3_nonref
    line22 = max(0, line18 - line21)

    se_tax = _tu(sim, "self_employment_tax")
    addl_medicare = _tu(sim, "additional_medicare_tax")
    niit = _tu(sim, "net_investment_income_tax")
    line23 = rnd(se_tax) + rnd(addl_medicare) + rnd(niit) + rnd(echo["uncollected_tax"])
    line24 = line22 + line23

    eitc = 0.0 if echo["eic_not_allowed"] else _tu(sim, "eitc")
    actc = _tu(sim, "refundable_ctc")
    aoc_ref = _tu(sim, "refundable_american_opportunity_credit")
    line31 = rnd(echo["extension_payment"]) + rnd(echo["excess_ss"])
    line32 = rnd(eitc) + rnd(actc) + rnd(aoc_ref) + line31

    line25d = rnd(echo["withholding"])
    line26 = rnd(echo["estimated_payments"])
    line33 = line25d + line26 + line32
    overpaid = max(0, line33 - line24)
    owed = max(0, line24 - line33)

    return {
        "WagesAmt": rnd(echo["wages_box1_total"]),
        "TotalIncomeAmt": rnd(total_income),
        "TotalAdjustmentsAmt": rnd(adjustments),
        "AdjustedGrossIncomeAmt": rnd(agi),
        "TotalItemizedOrStandardDedAmt": rnd(deduction),
        "TaxableIncomeAmt": int(taxable_income),
        "TaxAmt": tax16,
        "CTCODCAmt": line19,
        "TotalTaxAmt": line24,
        "WithholdingTaxAmt": line25d,
        "EstimatedTaxPaymentsAmt": line26,
        "EarnedIncomeCreditAmt": rnd(eitc),
        "AdditionalChildTaxCreditAmt": rnd(actc),
        "RefundableAmerOppCreditAmt": rnd(aoc_ref),
        "RefundableCreditsAmt": line32,
        "TotalPaymentsAmt": line33,
        "OverpaidAmt": overpaid,
        "RefundAmt": overpaid,
        "OwedAmt": owed,
    }
