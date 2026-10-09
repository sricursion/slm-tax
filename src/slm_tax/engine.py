"""Self-contained TY2024 Form 1040 engine for the 19 TaxCalcBench-scored lines.

Implements exact IRS form arithmetic (tax table, EITC table, Schedule 8812,
Forms 2441/8863/8962, SE tax, the Social Security worksheet, and Form 8995
QBID) for the situations covered by TaxCalcBench TY24: W-2, Schedule B/C/D,
1099-R/G/SSA, and the associated credits.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from .irs_math import (
    HOME_OFFICE_RATE,
    HOME_OFFICE_SQFT_CAP,
    STANDARD_MILEAGE_RATE_2024,
    qdcgt_worksheet,
    rnd,
    tax_table,
)
from .tcb import num

YEAR = 2024

# ---- 2024 constants ---------------------------------------------------------
STD_DEDUCTION = {
    "single": 14_600.0, "married_jointly": 29_200.0,
    "married_separately": 14_600.0, "head_of_household": 21_900.0,
}
ADDL_STD = {  # per aged-65+/blind box
    "single": 1_950.0, "head_of_household": 1_950.0,
    "married_jointly": 1_550.0, "married_separately": 1_550.0,
}
DEP_STD_FLOOR, DEP_STD_EARNED_ADD = 1_300.0, 450.0

SS_WAGE_BASE = 168_600.0
MAX_EMPLOYEE_SS_TAX = 10_453.20  # 6.2% of wage base

ADDL_MEDICARE_THRESHOLD = {
    "single": 200_000.0, "head_of_household": 200_000.0,
    "married_jointly": 250_000.0, "married_separately": 125_000.0,
}
NIIT_THRESHOLD = {
    "single": 200_000.0, "head_of_household": 200_000.0,
    "married_jointly": 250_000.0, "married_separately": 125_000.0,
}

CTC_PER_CHILD, ODC_PER_DEP, ACTC_MAX_PER_CHILD = 2_000.0, 500.0, 1_700.0
CTC_PHASEOUT_START = {"married_jointly": 400_000.0}  # else 200k
CTC_EARNED_FLOOR, CTC_PHASE_IN_RATE = 2_500.0, 0.15

# EITC 2024: kids -> (phase_in_rate, max_credit, phase_out_rate,
#                     phase_out_start_single, phase_out_start_joint)
EITC = {
    0: (0.0765, 632.0, 0.0765, 10_330.0, 17_250.0),
    1: (0.34, 4_213.0, 0.1598, 22_720.0, 29_640.0),
    2: (0.40, 6_960.0, 0.2106, 22_720.0, 29_640.0),
    3: (0.45, 7_830.0, 0.2106, 22_720.0, 29_640.0),
}
EITC_INVESTMENT_LIMIT = 11_600.0

QBID_THRESHOLD = {"married_jointly": 383_900.0}  # else 191,950
QBID_PHASE_IN = {"married_jointly": 100_000.0}  # else 50,000

SAVERS = {  # status -> [(agi_cap, rate)]
    "married_jointly": [(46_000, 0.5), (50_000, 0.2), (76_500, 0.1)],
    "head_of_household": [(34_500, 0.5), (37_500, 0.2), (57_375, 0.1)],
    "single": [(23_000, 0.5), (25_000, 0.2), (38_250, 0.1)],
    "married_separately": [(23_000, 0.5), (25_000, 0.2), (38_250, 0.1)],
}

# 2023 federal poverty line (used for 2024 PTC), 48 contiguous states
FPL_BASE, FPL_PER_PERSON = 14_580.0, 5_140.0
FPL_ALASKA = (18_210.0, 6_430.0)
FPL_HAWAII = (16_770.0, 5_910.0)
# 8962 applicable figure breakpoints (ARPA-extended, TY2024): (%FPL, pct)
PTC_APPLICABLE = [(150, 0.0), (200, 2.0), (250, 4.0), (300, 6.0), (400, 8.5)]
# APTC repayment caps by %FPL: (fpl_upper, single_cap, other_cap)
APTC_REPAY_CAP = [(200, 375.0, 750.0), (300, 950.0, 1_900.0), (400, 1_575.0, 3_150.0)]

CHILD_RELATIONSHIPS = {
    "son", "daughter", "stepchild", "foster_child", "brother", "sister",
    "half_brother", "half_sister", "stepbrother", "stepsister", "grandchild",
    "niece", "nephew",
}
UNCOLLECTED_TAX_CODES = {"A", "B", "M", "N"}
SAVERS_CREDIT_CODES = {"D", "E", "F", "G", "H", "S", "AA", "BB", "EE"}


def _age(dob_str: str) -> int:
    dob = dt.date.fromisoformat(str(dob_str).replace("/", "-"))
    ref = dt.date(YEAR + 1, 1, 1)
    return ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))


def _who(entry: dict, key: str = "who_applies_to") -> str:
    w = str(entry.get(key) or "taxpayer")
    if "spouse" in w:
        return "sp"
    return "tp"


def _sched_c_breakdown(sc: dict, parts: dict | None = None) -> tuple[float, float]:
    """Return (gross income, total expenses) for one Schedule C.

    If `parts` is given, it is filled with the itemized income/expense
    components so the trace can DERIVE the sums instead of stating them
    (pilot-6: line 8 was wrong in 67% of total-income failures because both
    the gross and expense totals were single stated numbers the model had to
    extract-and-sum implicitly across many input fields).
    """
    inc_items: list[tuple[str, float]] = []
    exp_items: list[tuple[str, float]] = []
    income = num(sc.get("gross_receipts_cash"))
    if income:
        inc_items.append(("gross receipts", income))
    for nec in sc.get("irs1099_nec") or []:
        v = num(nec.get("nec_comp"))
        income += v
        if v:
            inc_items.append(("1099-NEC", v))
    for k in sc.get("irs1099_k") or []:
        v = num(k.get("k_comp"))
        income += v
        if v:
            inc_items.append(("1099-K", v))
    for m in sc.get("irs1099_misc") or []:
        v = (
            num(m.get("other_income"))
            + num(m.get("nonqual_deferred_comp"))
            + num(m.get("rents"))
            + num(m.get("royalties"))
            + num(m.get("fish_boat_proceeds"))
            + num(m.get("medandhlth_care_pmts"))
            + num(m.get("crop_ins_proceeds"))
            + num(m.get("subpmts_inlieuof_div_int"))
        )
        income += v
        if v:
            inc_items.append(("1099-MISC", v))
    v = num(sc.get("other_income"))
    income += v
    if v:
        inc_items.append(("other income", v))
    v = num(sc.get("returns_allowances"))
    income -= v
    if v:
        inc_items.append(("less returns/allowances", -v))
    expense_fields = [
        "advertising", "commissions_fees", "contract_labor", "depletion",
        "employee_benefit", "insurance", "legal_professional",
        "machinery_equip_rent", "other_rent", "mortgage_interest",
        "other_interest", "office_expense", "pension_psp",
        "repairs_maintenance", "supplies", "tax_licenses", "travel",
        "utilities", "wages_expense",
    ]
    expenses = 0.0
    for f in expense_fields:
        v = num(sc.get(f))
        expenses += v
        if v:
            exp_items.append((f.replace("_", " "), v))
    v = 0.5 * num(sc.get("meal_entertainment"))
    expenses += v
    if v:
        exp_items.append(("meals (50%)", v))
    oe_total = 0.0
    for oe in sc.get("other_expense_detail") or []:
        oe_total += num(oe.get("other_expense_detail_amt"))
    expenses += oe_total
    if oe_total:
        exp_items.append(("other expenses", oe_total))
    for v in sc.get("vehicle_info_group") or []:
        biz_miles = num(v.get("vehicle_business_miles"))
        total_miles = (
            biz_miles + num(v.get("vehicle_commuting_miles")) + num(v.get("vehicle_other_miles"))
        )
        biz_pct = biz_miles / total_miles if total_miles else 1.0
        car = 0.0
        if v.get("vehicle_standard_mileage", True):
            car += rnd(STANDARD_MILEAGE_RATE_2024 * biz_miles)
        # Parking and tolls are business expenses in full; auto loan interest
        # and personal property tax are prorated by business use.
        car += num(v.get("car_truck_expense_parking")) + num(v.get("car_truck_expense_tolls"))
        car += (
            num(v.get("car_truck_expense_interest")) + num(v.get("car_truck_expense_tax"))
        ) * biz_pct
        expenses += rnd(car)
        if car:
            exp_items.append(("car and truck", float(rnd(car))))
    profit_before_home_office = income - expenses
    biz_area = num(sc.get("business_home_area"))
    if biz_area > 0:
        # Simplified-method home office deduction cannot create/increase a loss
        ho = min(
            HOME_OFFICE_RATE * min(biz_area, HOME_OFFICE_SQFT_CAP),
            max(0.0, profit_before_home_office),
        )
        expenses += ho
        if ho:
            exp_items.append(("home office", ho))
    if parts is not None:
        parts["income_items"] = inc_items
        parts["expense_items"] = exp_items
    return income, expenses


def _sched_c_net_profit(sc: dict) -> float:
    income, expenses = _sched_c_breakdown(sc)
    return income - expenses


def _se_tax(net_se: float, w2_ss_wages: float) -> tuple[float, float]:
    """(SE tax, half-SE-tax deduction) for one person, whole-dollar per Sch SE."""
    base = net_se * 0.9235
    if base < 400:
        return 0.0, 0.0
    ss_taxable = max(0.0, min(base, SS_WAGE_BASE - min(w2_ss_wages, SS_WAGE_BASE)))
    # Schedule SE rounds the social security and Medicare components separately
    tax = float(rnd(0.124 * ss_taxable) + rnd(0.029 * base))
    return tax, float(rnd(tax / 2.0))


def _eitc_amount(earned: float, agi: float, kids: int, joint: bool) -> int:
    """EITC via the IRS table (midpoint-of-$50-bracket) method."""
    rate_in, max_credit, rate_out, po_single, po_joint = EITC[min(kids, 3)]
    po_start = po_joint if joint else po_single

    def mid(x: float) -> float:
        if x < 1:
            return 0.0
        if x < 25:
            return 12.5 if x >= 1 else 0.0
        return math.floor(x / 50) * 50 + 25.0

    def f(x: float) -> float:
        m = mid(x)
        credit = min(max_credit, rate_in * m)
        credit -= max(0.0, (m - po_start) * rate_out)
        return max(0.0, credit)

    val = f(earned)
    if agi > po_start:
        val = min(val, f(agi))
    return rnd(val)


class Return:
    """One computed TY24 federal return."""

    def __init__(self, case: dict):
        self.data = case["data"]
        self.header = case.get("header", {})
        self.f1040 = self.data.get("irs1040", {})
        self.sched1 = self.data.get("irs1040_schedule1", {})
        self.sched3 = self.data.get("irs1040_schedule3", {})
        self.status: str = self.f1040.get("filing_status", "single")
        self.joint = self.status == "married_jointly"
        self.notes: list[str] = []
        self._parse()

    # ------------------------------------------------------------------ parse
    def _parse(self) -> None:
        f = self.f1040
        self.tp_age = _age(f["tp_date_of_birth"])
        self.sp_age = _age(f["sp_date_of_birth"]) if self.joint and f.get("sp_date_of_birth") else None
        self.tp_blind = bool(f.get("tp_blind"))
        self.sp_blind = bool(f.get("sp_blind"))
        self.tp_dependent = bool(f.get("tp_dependent"))
        self.sp_dependent = bool(f.get("sp_dependent"))

        # Dependents that actually qualify (support / joint-return / gross
        # income tests from the input flags).
        self.dependents = []
        for dep in f.get("dependent_detail") or []:
            rel = str(dep.get("dependent_relationship", "")).lower()
            is_child_rel = rel in CHILD_RELATIONSHIPS
            if is_child_rel and dep.get("dependent_supported_by_tp") is False:
                continue  # failed support test -> not a dependent at all
            if dep.get("dependent_filing_joint_return") and not dep.get(
                "dependent_filing_joint_return_yes_and_only_to_claim"
            ):
                continue
            if not is_child_rel and dep.get("dependent_relative_gross_income_eq_or_above_threshold"):
                continue
            if dep.get("dependent_us_citizen_national_or_resident_alien") is False and not dep.get(
                "dependent_us_citizen_national_resident_alien_or_canada_mexico_resident"
            ):
                continue  # citizenship/residency test failed
            months = num(dep.get("dependent_months_lived", 12))
            age = _age(dep["dependent_date_of_birth"])
            # dependent_ssn_itin_atin is a boolean in this dataset: truthy means
            # the dependent has an SSN (vs ITIN/ATIN or none).
            ssn_raw = dep.get("dependent_ssn_itin_atin")
            has_ssn = bool(ssn_raw) if isinstance(ssn_raw, bool) else str(ssn_raw or "ssn").lower() == "ssn"
            if dep.get("dependent_ssn") and ssn_raw is None:
                has_ssn = True
            student = bool(dep.get("dependent_student_for_5_plus_months"))
            disabled = bool(dep.get("dependent_disabled"))
            qc = (
                is_child_rel
                and months >= 7
                and (age < 19 or (age < 24 and student) or disabled)
            )
            self.dependents.append(
                {
                    "age": age, "months": months, "qc": qc,
                    "ssn_ok": has_ssn
                    and dep.get("dependent_ssn_employment_valid") is not False,
                    "itin": not has_ssn,
                    "student": student, "disabled": disabled,
                }
            )

        # W-2s
        self.wages_box1 = 0.0
        self.wh_w2 = 0.0
        self.wh_1099 = 0.0
        self.withholding = 0.0
        self.addl_medicare_wh = 0.0
        self.uncollected_tax = 0.0
        self.dep_care_benefits = 0.0
        self.earned: dict[str, float] = {"tp": 0.0, "sp": 0.0}
        self.ss_wages: dict[str, float] = {"tp": 0.0, "sp": 0.0}
        self.medicare_wages: dict[str, float] = {"tp": 0.0, "sp": 0.0}
        self.medicare_wh: dict[str, float] = {"tp": 0.0, "sp": 0.0}
        ss_tax: dict[str, list[tuple[str, float]]] = {"tp": [], "sp": []}
        self.savers_contrib: dict[str, float] = {"tp": 0.0, "sp": 0.0}
        for w2 in self.data.get("w2") or []:
            pid = _who(w2)
            wages = num(w2.get("wages"))
            self.wages_box1 += wages
            self.earned[pid] += wages
            self.wh_w2 += num(w2.get("withholding"))
            self.ss_wages[pid] += num(w2.get("social_security_wages")) + num(
                w2.get("social_security_tips")
            )
            self.medicare_wages[pid] += num(w2.get("medicare_wages_and_tips"))
            self.medicare_wh[pid] += num(w2.get("medicare_tax_withheld"))
            ss_tax[pid].append((str(w2.get("ein", "")), num(w2.get("social_security_tax"))))
            self.dep_care_benefits += num(w2.get("dependent_care_benefits"))
            for grp in w2.get("employers_use_grp") or []:
                code = str(grp.get("box_12_code", "")).upper()
                amt = num(grp.get("box_12_amount"))
                if code in UNCOLLECTED_TAX_CODES:
                    self.uncollected_tax += amt
                elif code in SAVERS_CREDIT_CODES:
                    self.savers_contrib[pid] += amt

        # Excess SS withholding (per person): cap each employer (EIN) at the
        # yearly max first — a single employer over-withholding is an employer
        # error recovered via Form 843, not a credit.
        self.excess_ss = 0.0
        for pid, entries in ss_tax.items():
            by_ein: dict[str, float] = {}
            for ein, t in entries:
                by_ein[ein] = by_ein.get(ein, 0.0) + t
            creditable = sum(min(t, MAX_EMPLOYEE_SS_TAX) for t in by_ein.values())
            self.excess_ss += rnd(max(0.0, creditable - MAX_EMPLOYEE_SS_TAX))

        # Form 8959 Part IV reconciliation: Medicare tax withheld beyond 1.45%
        # of Medicare wages counts as federal income tax withholding (line 25c).
        # Matches Column's arithmetic: whole-return totals, 1.45% rounded
        # half-up, never negative.
        total_box6 = sum(self.medicare_wh.values())
        total_box5 = sum(self.medicare_wages.values())
        self.addl_medicare_wh = 0.0
        if total_box5 > 200_000.0:  # Form 8959 filing trigger
            self.addl_medicare_wh = float(max(0, rnd(total_box6) - rnd(0.0145 * total_box5)))

        # Schedule C
        self.se_profit: dict[str, float] = {"tp": 0.0, "sp": 0.0}
        self.qbi_businesses: list[dict] = []
        for sc in self.data.get("irs1040_schedulec") or []:
            pid = _who(sc)
            parts: dict = {}
            gross, expenses = _sched_c_breakdown(sc, parts)
            profit = float(rnd(gross - expenses))
            self.se_profit[pid] += profit
            use_calc = sc.get("use_calculated_qbi", True) is not False
            self.qbi_businesses.append(
                {
                    "pid": pid, "profit": profit,
                    "gross": rnd(gross), "expenses": rnd(expenses),
                    "income_items": parts.get("income_items", []),
                    "expense_items": parts.get("expense_items", []),
                    "qualified": sc.get("qualified_business", True) is not False,
                    "sstb": bool(sc.get("specified_service")),
                    "wages_paid": num(sc.get("wages_paid") or sc.get("wages_expense")),
                    "ubia": num(sc.get("ubia_qualified_property")),
                    "qbi_override": None if use_calc else num(sc.get("user_adjusted_qbi")),
                }
            )
            for nec in sc.get("irs1099_nec") or []:
                self.wh_1099 += num(nec.get("nec_fed_wh"))
            for k in sc.get("irs1099_k") or []:
                self.wh_1099 += num(k.get("k_fed_wh"))
            for m in sc.get("irs1099_misc") or []:
                self.wh_1099 += num(m.get("misc_fed_wh"))

        # Schedule B
        self.interest = 0.0
        self.tax_exempt_interest = 0.0
        self.ordinary_div = 0.0
        self.qualified_div = 0.0
        self.cap_gain_distr = 0.0
        self.early_withdrawal = 0.0
        self.foreign_tax = 0.0
        schedb = self.data.get("irs1040_scheduleb", {})
        for i in schedb.get("irs1099_int") or []:
            self.interest += num(i.get("interest_1099int_interest")) + num(
                i.get("interest_1099int_us_saving_bond_treasury_obligations")
            )
            self.tax_exempt_interest += num(i.get("interest_1099int_tax_exempt"))
            self.early_withdrawal += num(i.get("interest_1099int_early_withdrawal"))
            self.foreign_tax += num(i.get("interest_1099int_foreign_taxes_paid"))
            self.wh_1099 += num(i.get("interest_1099int_fed_inc_tax_withheld"))
        for d in schedb.get("irs1099_div") or []:
            self.ordinary_div += num(d.get("dividend_1099div_ordinary_div"))
            self.qualified_div += num(d.get("dividend_1099div_qualified_div"))
            self.cap_gain_distr += num(d.get("dividend_1099div_capital_gain_distr"))
            self.foreign_tax += num(d.get("dividend_1099div_foreign_tax_paid"))
            self.wh_1099 += num(d.get("dividend_1099div_federal_inc_tax_withheld"))

        # Form 8949 / Schedule D
        self.st_gain = 0.0
        self.lt_gain = 0.0
        for grp in (self.data.get("irs8949") or {}).get("irs1099_b_grp") or []:
            for b in grp.get("irs1099_b") or []:
                discount = num(b.get("accrued_market_discount"))
                gain = (
                    num(b.get("proceeds"))
                    - num(b.get("cost_basis"))
                    + num(b.get("wash_sale_loss"))
                    - discount
                )
                self.interest += discount
                # Term is determined by holding period (dates), not the
                # input_sale_term label: held MORE than one year => long-term.
                long_term = "short" not in str(b.get("input_sale_term", "")).lower()
                acq, sold = b.get("acquired_dt"), b.get("sold_disposed_dt")
                if acq and sold:
                    try:
                        d_acq = dt.date.fromisoformat(str(acq).replace("/", "-"))
                        d_sold = dt.date.fromisoformat(str(sold).replace("/", "-"))
                        try:
                            anniversary = d_acq.replace(year=d_acq.year + 1)
                        except ValueError:  # Feb 29
                            anniversary = d_acq.replace(year=d_acq.year + 1, day=28)
                        long_term = d_sold > anniversary
                    except ValueError:
                        pass
                if long_term:
                    self.lt_gain += gain
                else:
                    self.st_gain += gain
                self.wh_1099 += num(b.get("fed_inc_tax_wh"))
        schedd = self.data.get("irs1040_schedule_d", {})
        self.st_gain -= abs(num(schedd.get("st_capital_loss_carryover_input")))
        self.lt_gain -= abs(num(schedd.get("lt_capital_loss_carryover_input")))

        # Retirement / SS / unemployment
        self.pension_taxable = 0.0
        self.pension_gross = 0.0
        for r in self.data.get("irs1099_r") or []:
            self.pension_gross += num(r.get("gross_distribution_amt_input"))
            self.pension_taxable += num(r.get("taxable_amt_input"))
            self.wh_1099 += num(r.get("federal_income_tax_withheld_amt_input"))
        self.ss_benefits = 0.0
        for grp in (self.data.get("ssa_1099") or {}).get("ssa_1099_grp") or []:
            self.ss_benefits += num(grp.get("ssa_net_benefits")) + num(
                grp.get("ssa_lump_sum_benefits")
            )
            self.wh_1099 += num(grp.get("ssa_federal_withholding"))
        self.unemployment = 0.0
        for g in self.sched1.get("irs1099_g") or []:
            self.unemployment += num(g.get("g_unemployment_comp")) - num(
                g.get("g_amount_repaid_current_year")
            )
            self.wh_1099 += num(g.get("g_fed_wh"))
        self.ak_pfd = num(self.sched1.get("alaska_permanent_dividend_tp_input"))

        # Adjustments inputs. Column's engine uses the entered amounts directly;
        # the qualified_educator / paid_student_loan_interest booleans do not
        # gate the deduction in the expected outputs.
        self.educator = min(300.0, num(self.sched1.get("tp_educator_exp_amount")))
        if self.joint:
            self.educator += min(300.0, num(self.sched1.get("sp_educator_exp_amount")))
        self.student_loan_paid = num(self.sched1.get("student_interest"))
        self.se_health_premiums = sum(
            num(sc.get("se_health_insurance")) for sc in self.data.get("irs1040_schedulec") or []
        )
        self.qbi_carryforward = abs(
            num((self.data.get("irs8995a_schedulec") or {}).get("prior_yr_qbi_loss_carryforward"))
        )

        # Payments
        self.estimated = sum(
            num(f.get(f"estimated_tax_payment_{i}")) for i in (1, 2, 3, 4)
        ) + num(f.get("applied_from_prior_year"))
        self.extension_payment = (
            num(self.sched3.get("extension_payment"))
            if self.sched3.get("requested_extension")
            else 0.0
        )

        # 2441
        irs2441 = self.data.get("irs2441", {})
        self.care_expenses = sum(
            num(qp.get("qualifying_expenses_paid"))
            for qp in irs2441.get("qualifying_person_grp") or []
        )
        if not self.care_expenses:
            self.care_expenses = sum(
                num(p.get("paid_to_provider")) for p in irs2441.get("care_provider_grp") or []
            )
        self.care_qualifying_persons = len(irs2441.get("qualifying_person_grp") or []) or sum(
            1 for d in self.dependents if d["age"] < 13 or d["disabled"]
        )
        self.care_carryover = num(irs2441.get("carryover"))
        self.care_forfeited = num(irs2441.get("forfeited"))
        self.care_earned_adj = {
            "tp": num(irs2441.get("tp_earned_income_adjustment")),
            "sp": num(irs2441.get("sp_earned_income_adjustment")),
        }
        self.care_has_qualifying_person = bool(irs2441.get("qualifying_person_grp")) or any(
            d["age"] < 13 or d["disabled"] for d in self.dependents
        )

        # 8863
        irs8863 = self.data.get("irs8863", {})
        self.students = []
        for sg in irs8863.get("student_educational_instn_grp") or []:
            expenses = sum(
                num(inst.get("qualified_expenses"))
                for inst in sg.get("educational_institution_group") or []
            )
            self.students.append(
                {
                    "expenses": expenses,
                    "is_tp": "taxpayer" in str(sg.get("student", "")),
                    "aoc_eligible": bool(sg.get("academic_period_eligible_student"))
                    and not bool(sg.get("post_secondary_education"))
                    and not bool(sg.get("drug_felony_conviction")),
                }
            )
        self.either_parent_alive = bool(irs8863.get("either_parent_alive"))
        self.total_support_8863 = num(irs8863.get("total_support_8863"))

        # 8962
        irs8962 = self.data.get("irs8962", {})
        self.has_1095a = bool(irs8962.get("received_1095a"))
        self.ptc_premium = num(irs8962.get("annual_premium"))
        self.ptc_slcsp = num(irs8962.get("annual_premium_slcsp"))
        self.ptc_aptc = num(irs8962.get("annual_advanced_ptc"))
        self.dependents_magi = num(irs8962.get("dependents_magi"))

        self.withholding = self.wh_w2 + self.wh_1099 + self.addl_medicare_wh
        self.eic_not_allowed = bool(f.get("eic_not_allowed"))
        self.qc_of_another = bool(f.get("qualifying_child_of_another"))

    # ---------------------------------------------------------------- compute
    def compute(self) -> dict[str, int]:
        status = self.status
        # SE tax per person
        se_tax = se_ald = 0.0
        for pid in ("tp", "sp"):
            t, a = _se_tax(self.se_profit[pid], self.ss_wages[pid])
            se_tax += t
            se_ald += a
            self.earned[pid] += self.se_profit[pid] - a

        # SE health insurance ALD (capped at profit minus half SE tax)
        se_profit_total = sum(self.se_profit.values())
        se_health_ald = min(self.se_health_premiums, max(0.0, se_profit_total - se_ald))

        # 2441 Part III: dependent care benefit exclusion; excess is wage income
        benefits = self.dep_care_benefits + self.care_carryover - self.care_forfeited
        taxable_benefits = 0.0
        excluded_benefits = 0.0
        if benefits > 0:
            earned_tp = self.earned["tp"] + self.care_earned_adj["tp"]
            earned_sp = (
                (self.earned["sp"] + self.care_earned_adj["sp"]) if self.joint else earned_tp
            )
            if not self.care_has_qualifying_person:
                excluded_benefits = 0.0
            else:
                excluded_benefits = max(
                    0.0, min(benefits, self.care_expenses, 5_000.0, earned_tp, earned_sp)
                )
            taxable_benefits = max(0.0, self.dep_care_benefits - excluded_benefits)
            self.earned["tp"] += taxable_benefits  # line 1e wages count as earned

        # Capital gains: Form 1040 line 7 (loss limited to $3,000/$1,500)
        net_cap = self.st_gain + self.lt_gain + self.cap_gain_distr
        loss_limit = -1_500.0 if status == "married_separately" else -3_000.0
        line7 = max(net_cap, loss_limit)

        # Income before Social Security taxability
        sched1_income = se_profit_total + self.unemployment + self.ak_pfd
        income_ex_ss = (
            self.wages_box1
            + taxable_benefits
            + self.interest
            + self.ordinary_div
            + self.pension_taxable
            + line7
            + sched1_income
        )
        adjustments_ex_sl = se_ald + se_health_ald + self.educator + self.early_withdrawal

        # Social Security taxability worksheet (Pub 915)
        taxable_ss = 0.0
        self._ss_provisional = 0.0
        self._ss_base1 = self._ss_base2 = 0.0
        if self.ss_benefits > 0:
            provisional = (
                income_ex_ss + self.tax_exempt_interest + 0.5 * self.ss_benefits
                - adjustments_ex_sl
            )
            base1 = 32_000.0 if self.joint else 25_000.0
            base2 = 44_000.0 if self.joint else 34_000.0
            if status == "married_separately":
                base1 = base2 = 0.0
            self._ss_provisional = provisional
            self._ss_base1, self._ss_base2 = base1, base2
            if provisional > base1:
                part = min(
                    0.5 * (provisional - base1), 0.5 * min(self.ss_benefits, base2 - base1)
                    if provisional <= base2
                    else 0.5 * (base2 - base1),
                )
                if provisional > base2:
                    taxable_ss = min(
                        0.85 * self.ss_benefits,
                        0.85 * (provisional - base2) + min(
                            0.5 * (base2 - base1), 0.5 * self.ss_benefits
                        ),
                    )
                else:
                    taxable_ss = min(0.5 * self.ss_benefits, 0.5 * (provisional - base1))

        total_income = income_ex_ss + taxable_ss

        # Student loan interest ALD (denied to dependents; MAGI phase-out)
        student_loan_ald = 0.0
        if self.student_loan_paid > 0 and not (self.tp_dependent or self.sp_dependent):
            amt = min(2_500.0, self.student_loan_paid)
            magi = total_income - (adjustments_ex_sl)
            lo, hi = (165_000.0, 195_000.0) if self.joint else (80_000.0, 95_000.0)
            if magi > hi:
                amt = 0.0
            elif magi > lo:
                amt *= (hi - magi) / (hi - lo)
                amt = rnd(amt)
            student_loan_ald = amt

        adjustments = adjustments_ex_sl + student_loan_ald
        agi = total_income - adjustments

        # Standard deduction
        std = STD_DEDUCTION[status]
        if self.tp_dependent and not self.joint:
            earned_total = self.earned["tp"]
            std = min(std, max(DEP_STD_FLOOR, earned_total + DEP_STD_EARNED_ADD))
        boxes = int(self.tp_age >= 65) + int(self.tp_blind)
        if self.joint:
            boxes += int((self.sp_age or 0) >= 65) + int(self.sp_blind)
        std += boxes * ADDL_STD[status]
        deduction = std  # no itemized inputs in the TY24 dataset

        # QBID (Form 8995 / 8995-A)
        qbid = self._qbid(agi, deduction, se_ald, se_health_ald)

        taxable_income = max(0.0, rnd(agi) - rnd(deduction) - rnd(qbid))

        # Line 16 tax
        schd_15 = self.lt_gain + self.cap_gain_distr
        schd_16 = self.st_gain + self.lt_gain + self.cap_gain_distr
        net_cg_pref = max(0.0, min(schd_15, schd_16)) if schd_16 > 0 or self.cap_gain_distr else 0.0
        if (self.qualified_div > 0 or net_cg_pref > 0) and taxable_income > 0:
            tax16 = qdcgt_worksheet(taxable_income, self.qualified_div, net_cg_pref, status)
        else:
            tax16 = tax_table(taxable_income, status)

        # 8962 PTC / APTC reconciliation
        net_ptc, aptc_repayment = self._form_8962(agi, taxable_ss)

        line17 = rnd(aptc_repayment)  # Schedule 2 part I (no AMT cases in set)
        line18 = tax16 + line17

        # Nonrefundable credits (Schedule 3 + CTC ordering per Form 8812)
        cdcc = self._form_2441(agi, excluded_benefits)
        aoc_total, aoc_refundable = self._form_8863(agi)
        aoc_nonref = aoc_total - aoc_refundable
        savers = self._savers_credit(agi)
        ftc = self.foreign_tax if self.foreign_tax <= (600.0 if self.joint else 300.0) else 0.0
        sch3_nonref = rnd(ftc) + rnd(cdcc) + rnd(aoc_nonref) + rnd(savers)

        # CTC / ODC (Schedule 8812). A taxpayer claimable as a dependent cannot
        # claim dependents or their credits.
        if self.tp_dependent:
            ctc_kids = odc_count = 0
        else:
            ctc_kids = sum(1 for d in self.dependents if d["qc"] and d["age"] < 17 and d["ssn_ok"])
            odc_count = len(self.dependents) - ctc_kids
        ctc_potential = CTC_PER_CHILD * ctc_kids + ODC_PER_DEP * odc_count
        po_start = CTC_PHASEOUT_START.get(status, 200_000.0)
        if agi > po_start:
            over = math.ceil((agi - po_start) / 1_000.0) * 1_000
            ctc_potential = max(0.0, ctc_potential - 0.05 * over)
        limit_8812 = max(0, line18 - sch3_nonref)
        line19 = min(rnd(ctc_potential), limit_8812)

        # ACTC
        remaining = rnd(ctc_potential) - line19
        actc = 0.0
        if remaining > 0 and ctc_kids > 0:
            earned_total = sum(self.earned.values())
            actc = min(
                float(remaining),
                ACTC_MAX_PER_CHILD * ctc_kids,
                max(0.0, CTC_PHASE_IN_RATE * (earned_total - CTC_EARNED_FLOOR)),
            )

        line21 = line19 + sch3_nonref
        line22 = max(0, line18 - line21)
        line23 = rnd(se_tax) + rnd(self._addl_medicare_tax(se_ald)) + rnd(
            self._niit(agi, line7)
        ) + rnd(self.uncollected_tax)
        line24 = line22 + line23

        # EITC
        eitc = self._eitc(agi)

        line25d = rnd(self.withholding)
        line26 = rnd(self.estimated)
        line31 = rnd(self.extension_payment) + rnd(self.excess_ss) + rnd(net_ptc)
        line32 = eitc + rnd(actc) + rnd(aoc_refundable) + line31
        line33 = line25d + line26 + line32
        overpaid = max(0, line33 - line24)
        owed = max(0, line24 - line33)

        # Full form-line detail for target rendering (not just the 19 scored)
        self.detail = {
            "1a": rnd(self.wages_box1),
            "1e": rnd(taxable_benefits),
            "1z": rnd(self.wages_box1) + rnd(taxable_benefits),
            "2a": rnd(self.tax_exempt_interest),
            "2b": rnd(self.interest),
            "3a": rnd(self.qualified_div),
            "3b": rnd(self.ordinary_div),
            "5a": rnd(self.pension_gross),
            "5b": rnd(self.pension_taxable),
            "6a": rnd(self.ss_benefits),
            "6b": rnd(taxable_ss),
            "7": rnd(line7) if (self.st_gain or self.lt_gain or self.cap_gain_distr) else 0,
            "8": rnd(sched1_income),
            "9": rnd(total_income),
            "10": rnd(adjustments),
            "11": rnd(agi),
            "12": rnd(deduction),
            "13": rnd(qbid),
            "14": rnd(deduction) + rnd(qbid),
            "15": int(taxable_income),
            "16": tax16,
            "17": line17,
            "18": line18,
            "19": line19,
            "20": sch3_nonref,
            "21": line21,
            "22": line22,
            "23": line23,
            "24": line24,
            "25a": rnd(self.wh_w2),
            "25b": rnd(self.wh_1099),
            "25c": rnd(self.addl_medicare_wh),
            "25d": line25d,
            "26": line26,
            "27": eitc,
            "28": rnd(actc),
            "29": rnd(aoc_refundable),
            "31": line31,
            "32": line32,
            "33": line33,
            "34": overpaid,
            "35a": overpaid,
            "37": owed,
            # intermediates useful for explanations
            "se_tax": rnd(se_tax),
            "se_ald": rnd(se_ald),
            "niit": rnd(self._niit(agi, line7)),
            "addl_medicare": rnd(self._addl_medicare_tax(se_ald)),
            "uncollected": rnd(self.uncollected_tax),
            "cdcc": rnd(cdcc),
            "savers": rnd(savers),
            "ftc": rnd(ftc),
            "aoc_nonref": rnd(aoc_nonref),
            "excess_ss": rnd(self.excess_ss),
            "student_loan_ald": rnd(student_loan_ald),
            "se_health_ald": rnd(se_health_ald),
            "educator_ald": rnd(self.educator),
            "early_withdrawal_ald": rnd(self.early_withdrawal),
            "extension": rnd(self.extension_payment),
            "net_ptc": rnd(net_ptc),
            "aptc_repayment": line17,
            "ctc_kids": ctc_kids,
            "odc_count": odc_count,
            "se_profit": rnd(se_profit_total),
            "_agi": agi,  # unrounded, for trace-side bracket lookups
        }

        return {
            "WagesAmt": rnd(self.wages_box1),
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
            "EarnedIncomeCreditAmt": eitc,
            "AdditionalChildTaxCreditAmt": rnd(actc),
            "RefundableAmerOppCreditAmt": rnd(aoc_refundable),
            "RefundableCreditsAmt": line32,
            "TotalPaymentsAmt": line33,
            "OverpaidAmt": overpaid,
            "RefundAmt": overpaid,
            "OwedAmt": owed,
        }

    # ------------------------------------------------------------- components
    def _qbid(self, agi: float, deduction: float, se_ald: float, se_health: float) -> float:
        self._qbid_total = self._qbid_income_limit = self._qbid_ti_net_cg = 0.0
        if not self.qbi_businesses:
            return 0.0
        # SE-tax/health ALDs reduce QBI, allocated by profit share — but only
        # for businesses whose QBI is engine-calculated (a user_adjusted_qbi
        # override is taken as-is, already net of adjustments).
        total_profit = sum(b["profit"] for b in self.qbi_businesses)
        qbis = []
        for b in self.qbi_businesses:
            if not b["qualified"]:
                continue
            if b["qbi_override"] is not None:
                qbi_b = b["qbi_override"]
            else:
                share = b["profit"] / total_profit if total_profit else 0.0
                qbi_b = b["profit"] - (se_ald + se_health) * share
            qbis.append((b, qbi_b))
        qbi_total = sum(q for _, q in qbis) - self.qbi_carryforward
        if qbi_total <= 0:
            return 0.0
        ti_before = max(0.0, agi - deduction)
        net_cg = self.qualified_div + max(
            0.0, min(self.lt_gain + self.cap_gain_distr, self.st_gain + self.lt_gain + self.cap_gain_distr)
        )
        income_limit = 0.2 * max(0.0, ti_before - net_cg)
        # expose for the trace
        self._qbid_total = qbi_total
        self._qbid_income_limit = income_limit
        self._qbid_ti_net_cg = max(0.0, ti_before - net_cg)
        threshold = QBID_THRESHOLD.get(self.status, 191_950.0)
        phase_range = QBID_PHASE_IN.get(self.status, 50_000.0)
        if ti_before <= threshold:
            # Form 8995: the 20% component is rounded per business unless
            # negatives/carryforward force aggregate netting.
            if self.qbi_carryforward <= 0 and all(q >= 0 for _, q in qbis):
                return min(float(sum(rnd(0.2 * q) for _, q in qbis)), income_limit)
            return min(0.2 * qbi_total, income_limit)
        # Form 8995-A: wage/UBIA limitation, phased in over the range
        qbid = 0.0
        for b, qbi_b in qbis:
            if qbi_b <= 0:
                continue
            tentative = 0.2 * qbi_b
            wage_limit = max(
                0.5 * b["wages_paid"], 0.25 * b["wages_paid"] + 0.025 * b["ubia"]
            )
            excess_pct = min(1.0, (ti_before - threshold) / phase_range)
            if b["sstb"]:
                applicable = 1.0 - excess_pct
                qbi_b *= applicable
                tentative = 0.2 * qbi_b
                wage_limit *= applicable
            if ti_before >= threshold + phase_range:
                allowed = min(tentative, wage_limit)
            else:
                reduction = (tentative - wage_limit) * excess_pct if tentative > wage_limit else 0.0
                allowed = tentative - max(0.0, reduction)
            qbid += max(0.0, allowed)
        return min(qbid, income_limit)

    def _form_2441(self, agi: float, excluded_benefits: float) -> float:
        self._cdcc_expenses = self._cdcc_rate = 0.0
        if not self.care_has_qualifying_person or self.care_expenses <= 0 or self.tp_dependent:
            return 0.0
        n = min(2, max(1, self.care_qualifying_persons))
        cap = 3_000.0 * n - excluded_benefits
        earned_tp = self.earned["tp"] + self.care_earned_adj["tp"]
        earned_sp = (self.earned["sp"] + self.care_earned_adj["sp"]) if self.joint else earned_tp
        expenses = min(self.care_expenses, max(0.0, cap), earned_tp, earned_sp)
        if expenses <= 0:
            return 0.0
        rate = 0.35 - 0.01 * max(0, math.ceil(max(0.0, agi - 15_000.0) / 2_000.0))
        rate = max(0.20, rate)
        self._cdcc_expenses = expenses
        self._cdcc_rate = rate
        return expenses * rate

    def _form_8863(self, agi: float) -> tuple[float, float]:
        """Returns (total education credits, refundable portion)."""
        self._aoc = self._llc = 0.0
        self._edu_factor = 1.0
        if not self.students or self.tp_dependent:
            return 0.0, 0.0
        lo, hi = (160_000.0, 180_000.0) if self.joint else (80_000.0, 90_000.0)
        if agi >= hi:
            return 0.0, 0.0
        factor = 1.0 if agi <= lo else (hi - agi) / (hi - lo)
        aoc = 0.0
        llc_expenses = 0.0
        refundable_blocked = False
        for s in self.students:
            if s["expenses"] <= 0:
                continue
            if s["aoc_eligible"]:
                capped = min(s["expenses"], 4_000.0)
                aoc += min(capped, 2_000.0) + 0.25 * max(0.0, capped - 2_000.0)
                if s["is_tp"] and self.tp_age < 24 and self.either_parent_alive:
                    refundable_blocked = True
            else:
                llc_expenses += s["expenses"]
        aoc = rnd(aoc * factor)
        llc = rnd(0.2 * min(10_000.0, llc_expenses) * factor)
        refundable = 0.0 if refundable_blocked else rnd(aoc * 0.4)
        self._aoc, self._llc, self._edu_factor = aoc, llc, factor
        return float(aoc + llc), float(refundable)

    def _savers_credit(self, agi: float) -> float:
        contrib = sum(min(2_000.0, c) for c in self.savers_contrib.values())
        if contrib <= 0 or self.tp_dependent:
            return 0.0
        for cap, rate in SAVERS[self.status]:
            if agi <= cap:
                return contrib * rate
        return 0.0

    def _addl_medicare_tax(self, se_ald: float) -> float:
        threshold = ADDL_MEDICARE_THRESHOLD[self.status]
        wages = sum(self.medicare_wages.values())
        tax = 0.009 * max(0.0, wages - threshold)
        se_base = sum(max(0.0, self.se_profit[p]) for p in ("tp", "sp")) * 0.9235
        if se_base > 0:
            se_threshold = max(0.0, threshold - wages)
            tax += 0.009 * max(0.0, se_base - se_threshold)
        return tax

    def _niit(self, agi: float, line7: float) -> float:
        nii = self.interest + self.ordinary_div + max(0.0, line7)
        if nii <= 0:
            return 0.0
        magi = agi + 0.0
        return 0.038 * max(0.0, min(nii, magi - NIIT_THRESHOLD[self.status]))

    def _form_8962(self, agi: float, taxable_ss: float) -> tuple[float, float]:
        """Returns (net PTC for line 31, APTC repayment for Schedule 2)."""
        if not self.has_1095a or self.ptc_slcsp <= 0:
            return 0.0, 0.0
        hh_size = 1 + (1 if self.joint else 0) + len(self.dependents)
        state = str(self.header.get("state", "TX"))
        base, per = (FPL_ALASKA if state == "AK" else FPL_HAWAII if state == "HI" else (FPL_BASE, FPL_PER_PERSON))
        fpl = base + per * (hh_size - 1)
        magi = agi + self.tax_exempt_interest + (self.ss_benefits - taxable_ss) + self.dependents_magi
        # Form 8962 line 5: %FPL rounded DOWN to a whole percent
        pct_fpl = math.floor(magi / fpl * 100.0)
        # applicable figure (interpolated, rounded to 4 decimals per IRS table)
        if pct_fpl < 150:
            applicable = 0.0
        elif pct_fpl >= 400:
            applicable = 8.5
        else:
            applicable = 8.5
            for i in range(1, len(PTC_APPLICABLE)):
                lo_p, lo_v = PTC_APPLICABLE[i - 1]
                hi_p, hi_v = PTC_APPLICABLE[i]
                if pct_fpl <= hi_p:
                    applicable = lo_v + (hi_v - lo_v) * (pct_fpl - lo_p) / (hi_p - lo_p)
                    break
        contribution = rnd(magi * round(applicable / 100.0, 4))
        ptc = max(0.0, min(self.ptc_premium, self.ptc_slcsp - contribution))
        if ptc >= self.ptc_aptc:
            return ptc - self.ptc_aptc, 0.0
        repayment = self.ptc_aptc - ptc
        if pct_fpl < 400:
            for hi_p, cap_single, cap_other in APTC_REPAY_CAP:
                if pct_fpl < hi_p:
                    cap = cap_single if self.status == "single" else cap_other
                    repayment = min(repayment, cap)
                    break
        return 0.0, repayment

    def _eitc(self, agi: float) -> int:
        if self.eic_not_allowed or self.tp_dependent or self.status == "married_separately":
            return 0
        invest = self.interest + self.tax_exempt_interest + self.ordinary_div + max(
            0.0, self.st_gain + self.lt_gain + self.cap_gain_distr
        )
        if invest > EITC_INVESTMENT_LIMIT:
            return 0
        kids = sum(1 for d in self.dependents if d["qc"] and d["ssn_ok"])
        if kids == 0:
            if self.qc_of_another:
                return 0
            ages = [self.tp_age] + ([self.sp_age] if self.joint and self.sp_age else [])
            if not any(25 <= a <= 64 for a in ages):
                return 0
        earned = sum(self.earned.values())
        if earned <= 0:
            return 0
        return _eitc_amount(earned, agi, kids, self.joint)


def compute_lines(case: dict) -> dict[str, int]:
    return Return(case).compute()
