"""Synthetic taxpayer scenario generator.

Produces cases in the same flattened structure the engine consumes (and that
render.py wraps into TaxCalcBench-format input JSON). Sampling is stratified
by income profile with deliberate mass near rule boundaries (EITC brackets,
CTC/QBID/NIIT phase-outs, the social security wage base, age 65/17/19/24
edges) so a trained model sees the discontinuities, not just the interiors.
"""

from __future__ import annotations

import random

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Carlos", "Maria", "Wei", "Ana",
    "Mohammed", "Fatima", "Sanjay", "Priya", "Dmitri", "Elena", "Kwame", "Ayo",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Chen", "Patel",
    "Kim", "Nguyen", "Ali", "Okafor", "Ivanov", "Kowalski", "Larsen", "Silva",
]
EMPLOYERS = [
    "Acme Corp", "Globex Inc", "Initech LLC", "Umbrella Co", "Stark Industries",
    "Wayne Enterprises", "Hooli", "Pied Piper", "Vandelay Industries",
    "Dunder Mifflin", "Cyberdyne Systems", "Wonka Industries", "Oceanic Air",
    "Prestige Worldwide", "Bluth Company", "Sterling Cooper", "Soylent Corp",
]
NO_TAX_STATES = ["AK", "FL", "TX", "WA", "SD", "WY", "NV", "TN", "NH"]
OTHER_STATES = ["NY", "AZ", "PA", "VT", "CA", "IL", "OH", "GA", "NC", "MI"]
CHILD_RELS = ["son", "daughter"]
OTHER_RELS = ["parent", "niece", "nephew", "grandchild"]

# 2024 rule boundaries to oversample around
EDGE_INCOMES = [
    11_600, 16_550, 23_200, 47_150, 63_100, 94_300, 100_000, 100_525,
    125_000, 168_600, 191_950, 200_000, 200_525, 243_725, 250_000, 383_900,
    400_000, 10_330, 17_250, 22_720, 29_640, 25_000, 32_000, 34_000, 44_000,
]


def _money(rng: random.Random, lo: float, hi: float) -> int:
    """Log-uniform-ish integer dollar amount."""
    import math

    lo = max(lo, 1)
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def _dob(rng: random.Random, age: int) -> str:
    # age as of Jan 1, 2025 (engine convention)
    year = 2024 - age
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _ein(rng: random.Random) -> str:
    return f"{rng.randint(10, 99)}{rng.randint(1000000, 9999999)}"


def _w2(rng: random.Random, who: str, wages: int, employer: str, quirks: dict) -> dict:
    ss_wages = min(wages, 168_600)
    ss_tax = round(0.062 * ss_wages, 2)
    if quirks.get("over_withhold_ss"):
        ss_tax = round(ss_tax * rng.uniform(1.01, 1.15), 2)
    medicare_wages = wages
    medicare_tax = round(0.0145 * medicare_wages, 2)
    if medicare_wages > 200_000:
        medicare_tax = round(medicare_tax + 0.009 * (medicare_wages - 200_000), 2)
    w2 = {
        "who_applies_to": who,
        "ein": quirks.get("ein") or _ein(rng),
        "employer_name": rng.choice(EMPLOYERS),
        "employer_street_address": f"{rng.randint(1, 999)} Commerce St",
        "employer_city": "Springfield",
        "employer_state": quirks.get("state", "TX"),
        "employer_zip": f"{rng.randint(10000, 99999)}",
        "wages": wages,
        "withholding": int(wages * rng.uniform(0.0, 0.28)),
        "social_security_wages": ss_wages,
        "social_security_tax": ss_tax,
        "medicare_wages_and_tips": medicare_wages,
        "medicare_tax_withheld": medicare_tax,
        "social_security_tips": 0,
        "allocated_tips": 0,
        "dependent_care_benefits": 0,
        "nonqualified_plan": 0,
        "statutory_employee": False,
        "retirement_plan": False,
        "third_party_sick_pay": False,
    }
    if rng.random() < 0.15:
        codes = []
        for code, lo, hi in [
            ("D", 500, 23_000), ("DD", 1_000, 20_000), ("A", 10, 500),
            ("B", 10, 500), ("M", 10, 300), ("N", 10, 300), ("L", 50, 2_000),
        ]:
            if rng.random() < 0.3:
                codes.append({"box_12_code": code, "box_12_amount": _money(rng, lo, hi)})
        if codes:
            w2["employers_use_grp"] = codes[:3]
    # W-2 boxes 15-20 (state/local) — present in ~14/51 benchmark cases; the
    # engine ignores them for the federal return, so the model must learn to.
    st = quirks.get("state", "TX")
    if st not in ("AK", "FL", "TX", "WA", "SD", "WY", "NV", "TN", "NH") and rng.random() < 0.7:
        w2["w2_state_local_tax_grp"] = [{
            "state": st,
            "employer_state_id": f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
            "state_wages": wages,
            "state_income_tax": int(wages * rng.uniform(0.02, 0.06)),
            "local_wages": wages if rng.random() < 0.2 else 0,
            "local_income_tax": int(wages * 0.01) if rng.random() < 0.2 else 0,
            "locality": "CITY" if rng.random() < 0.2 else "",
        }]
    return w2


def _dependent(rng: random.Random, force_ctc_age: bool = False) -> dict:
    if force_ctc_age:
        age = rng.randint(0, 16)
    else:
        # cover CTC (<17), EITC (<19 / <24 student), and edge ages heavily
        age = rng.choice([rng.randint(0, 16)] * 5 + [17, 18, 19, 23, 24, rng.randint(19, 23)])
    rel = rng.choice(CHILD_RELS * 6 + OTHER_RELS)
    if rel in OTHER_RELS and rel == "parent":
        age = rng.randint(55, 90)
    student = age >= 19 and rng.random() < 0.7
    return {
        "dependent_first_name": rng.choice(FIRST_NAMES),
        "dependent_last_name": rng.choice(LAST_NAMES),
        "dependent_ssn": f"9004{rng.randint(10000, 99999)}",
        "dependent_date_of_birth": _dob(rng, age),
        "dependent_relationship": rel,
        "dependent_months_lived": rng.choice([12] * 8 + [rng.randint(1, 6), rng.randint(7, 11)]),
        "dependent_supported_by_tp": rng.random() > 0.05,
        "dependent_student_for_5_plus_months": student,
        "dependent_disabled": rng.random() < 0.03,
        "dependent_married": False,
        "dependent_us_citizen_national_or_resident_alien": rng.random() > 0.03,
        "dependent_relative_gross_income_eq_or_above_threshold": rel in OTHER_RELS
        and rng.random() < 0.2,
        "dependent_ssn_itin_atin": rng.random() > 0.08,
        "dependent_ssn_employment_valid": rng.random() > 0.05,
    }


def _schedule_c(rng: random.Random, who: str, scale: float) -> dict:
    gross = _money(rng, 500, 150_000 * scale) if rng.random() < 0.8 else 0
    sc: dict = {
        "who_applies_to": who,
        "business_name": f"{rng.choice(LAST_NAMES)} {rng.choice(['Consulting', 'Services', 'Design', 'Repair', 'Media'])}",
        "business_code": str(rng.choice([541990, 812990, 484110, 722511, 454110])),
        "business_ein": _ein(rng),
        "gross_receipts_cash": gross,
        "material_participate": True,
        "schc_at_risk": True,
        "qualified_business": True,
        "method_accounting": "cash",
        "use_calculated_qbi": True,
    }
    if rng.random() < 0.4:
        sc["irs1099_nec"] = [
            {
                "nec_payer_name": rng.choice(EMPLOYERS),
                "nec_comp": _money(rng, 500, 60_000 * scale),
                "nec_fed_wh": _money(rng, 0, 2_000) if rng.random() < 0.2 else 0,
                "nec_resale": False,
            }
            for _ in range(rng.randint(1, 2))
        ]
    if rng.random() < 0.2:
        sc["irs1099_k"] = [
            {
                "k_payer_name": "PayFast",
                "k_payer_ein": _ein(rng),
                "k_comp": _money(rng, 1_000, 50_000 * scale),
                "k_fed_wh": 0,
            }
        ]
    if rng.random() < 0.2:
        sc["irs1099_misc"] = [
            {
                "payer_name": rng.choice(EMPLOYERS),
                "payer_tin": _ein(rng),
                "rents": _money(rng, 500, 20_000) if rng.random() < 0.5 else 0,
                "other_income": _money(rng, 500, 15_000) if rng.random() < 0.5 else 0,
                "misc_fed_wh": 0,
                "royalties": 0,
                "fish_boat_proceeds": _money(rng, 100, 5_000) if rng.random() < 0.1 else 0,
                "medandhlth_care_pmts": _money(rng, 100, 5_000) if rng.random() < 0.1 else 0,
            }
        ]
    # expenses — sometimes enough to force a loss
    loss_bias = rng.random() < 0.2
    budget = (gross + 20_000) * (rng.uniform(0.8, 2.0) if loss_bias else rng.uniform(0.05, 0.7))
    # full Schedule C Part II expense line set (matches benchmark inputs, which
    # include mortgage_interest/other_rent/pension_psp/etc. in ~14/51 cases)
    for field in [
        "advertising", "commissions_fees", "contract_labor", "insurance",
        "legal_professional", "office_expense", "repairs_maintenance",
        "supplies", "tax_licenses", "travel", "utilities", "wages_expense",
        "mortgage_interest", "other_interest", "other_rent", "machinery_equip_rent",
        "pension_psp", "employee_benefit", "depletion",
    ]:
        if rng.random() < 0.28 and budget > 100:
            amt = _money(rng, 50, max(60, budget * 0.5))
            sc[field] = amt
            budget -= amt
    if rng.random() < 0.15:
        sc["other_income"] = _money(rng, 100, 8_000)
    # descriptive fields the benchmark carries (engine ignores them)
    sc["business_act"] = rng.choice(["consulting", "retail", "services", "freelance"])
    if rng.random() < 0.3:
        sc["former_employer"] = True
    if rng.random() < 0.25:
        sc["meal_entertainment"] = _money(rng, 50, 4_000)
    if rng.random() < 0.2:
        sc["other_expense_detail"] = [
            {
                "other_expense_detail_desc": rng.choice(["software", "phone", "misc", "fees"]),
                "other_expense_detail_amt": _money(rng, 50, 3_000),
            }
            for _ in range(rng.randint(1, 2))
        ]
    if rng.random() < 0.3:
        biz_miles = rng.randint(200, 20_000)
        sc["vehicle_info_group"] = [
            {
                "vehicle_service_date": _dob(rng, rng.randint(1, 10)),
                "vehicle_standard_mileage": True,
                "vehicle_business_miles": biz_miles,
                "vehicle_commuting_miles": rng.choice([0, 0, rng.randint(50, 5_000)]),
                "vehicle_other_miles": rng.choice([0, 0, rng.randint(50, 5_000)]),
                "car_truck_expense_parking": rng.choice([0, _money(rng, 10, 500)]),
                "car_truck_expense_tolls": rng.choice([0, _money(rng, 10, 500)]),
                "car_truck_expense_interest": rng.choice([0, _money(rng, 10, 800)]),
                "car_truck_expense_tax": rng.choice([0, _money(rng, 10, 500)]),
            }
        ]
    if rng.random() < 0.2:
        sc["business_home_area"] = rng.randint(5, 500)
        sc["total_home_area"] = sc["business_home_area"] + rng.randint(500, 3_000)
    if rng.random() < 0.15:
        sc["se_health_insurance"] = _money(rng, 1_000, 15_000)
    if rng.random() < 0.15:
        sc["specified_service"] = True
    if rng.random() < 0.1:
        sc["use_calculated_qbi"] = False
        sc["user_adjusted_qbi"] = int(gross * rng.uniform(-0.3, 0.8))
    if rng.random() < 0.1:
        sc["returns_allowances"] = _money(rng, 100, max(200, gross))
    if rng.random() < 0.1:
        sc["wages_paid"] = _money(rng, 1_000, 80_000)
        sc["ubia_qualified_property"] = _money(rng, 1_000, 200_000)
    return sc


def _form_8949(rng: random.Random) -> tuple[dict, dict]:
    rows = []
    for _ in range(rng.randint(1, 5)):
        basis = _money(rng, 0, 60_000) if rng.random() < 0.9 else 0
        proceeds = int(basis * rng.uniform(0.3, 2.5)) if basis else _money(rng, 500, 40_000)
        long_term = rng.random() < 0.5
        hold_days = rng.randint(400, 2_000) if long_term else rng.randint(10, 360)
        sold_m, sold_d = rng.randint(1, 12), rng.randint(1, 28)
        import datetime as dt

        sold = dt.date(2024, sold_m, sold_d)
        acquired = sold - dt.timedelta(days=hold_days)
        # occasionally the label disagrees with the dates (as in the benchmark);
        # dates are authoritative
        label = long_term if rng.random() > 0.05 else not long_term
        rows.append(
            {
                "desc_of_prop": rng.choice(["AAPL", "VTI", "TSLA", "BTC", "SPY", "Widget Co"]),
                "acquired_dt": acquired.isoformat(),
                "sold_disposed_dt": sold.isoformat(),
                "proceeds": proceeds,
                "cost_basis": basis,
                "wash_sale_loss": _money(rng, 50, 3_000)
                if (proceeds < basis and rng.random() < 0.15)
                else 0,
                "accrued_market_discount": 0,
                "fed_inc_tax_wh": 0,
                "input_sale_term": "long_term" if label else "short_term",
                "input_irs8949_code": "basis_reported",
                "reporting_method_input": "individual_transaction",
                "payer_ein": _ein(rng),
            }
        )
    f8949 = {"irs1099_b_grp": [{"payer_name": "BrokerCo", "who_applies_to": "taxpayer", "irs1099_b": rows}]}
    schedd = {}
    if rng.random() < 0.35:
        schedd = {
            "st_capital_loss_carryover_input": rng.choice([0, _money(rng, 500, 30_000)]),
            "lt_capital_loss_carryover_input": rng.choice([0, _money(rng, 500, 30_000)]),
        }
    return f8949, schedd


def generate_case(seed: int) -> dict:
    """Generate one flattened scenario (engine-ready)."""
    rng = random.Random(seed)

    status = rng.choices(
        ["single", "married_jointly", "head_of_household", "married_separately"],
        weights=[0.44, 0.35, 0.16, 0.05],
    )[0]
    joint = status == "married_jointly"

    # income profile: low / mid / high / edge-targeted
    profile = rng.choices(["low", "mid", "high", "edge"], weights=[0.35, 0.35, 0.18, 0.12])[0]
    if profile == "low":
        income_target = _money(rng, 3_000, 45_000)
    elif profile == "mid":
        income_target = _money(rng, 45_000, 130_000)
    elif profile == "high":
        income_target = _money(rng, 130_000, 550_000)
    else:
        income_target = int(rng.choice(EDGE_INCOMES) * rng.uniform(0.92, 1.08))

    placeholder_style = rng.random() < 0.3
    # "senior" archetype: elevate 65+ and blindness so 3-4 age/blind-box returns
    # appear (benchmark has them at ~2%/case; base generator produced 0).
    senior = rng.random() < 0.22
    if senior:
        tp_age = rng.randint(65, 88)
    else:
        tp_age = rng.choices(
            [rng.randint(19, 24), rng.randint(25, 64), rng.choice([64, 65, 66]), rng.randint(67, 85)],
            weights=[0.1, 0.6, 0.15, 0.15],
        )[0]
    tp_blind_p = 0.30 if senior else 0.04
    f1040: dict = {
        "filing_status": status,
        "tp_first_name": rng.choice(
            ["Test", "WTwo", "Single", "Filer", "Sample", "Alpha"]
        ) if placeholder_style else rng.choice(FIRST_NAMES),
        "tp_last_name": rng.choice(
            ["Case", "Complete", "Taxpayer", "Return", "Beta", "Demo"]
        ) if placeholder_style else rng.choice(LAST_NAMES),
        "tp_date_of_birth": _dob(rng, tp_age),
        "tp_blind": rng.random() < tp_blind_p,
        "tp_dependent": False,
        "tp_student": rng.random() < 0.05,
        "virtual_currency": rng.random() < 0.1,
        "nonresident_alien": False,
        "qualifying_child_of_another": False,
        "eic_not_allowed": rng.random() < 0.02,
        "credit_denied_reduced": False,
        "main_home_not_us": False,
        "nontaxable_combat_election": False,
        "has_medicaid_waiver_payment": False,
        "medicaid_waiver_payment": 0,
        "charitable_contribution": 0,
        "tp_elects_to_claim_dependent_credit": True,
        "applied_from_prior_year": 0,
        "paid_estimated_tax_pmts": False,
        "estimated_tax_payment_1": 0,
        "estimated_tax_payment_2": 0,
        "estimated_tax_payment_3": 0,
        "estimated_tax_payment_4": 0,
    }
    sp_age = None
    if joint:
        # for senior couples keep both spouses old so 3-4 box returns appear
        sp_age = rng.randint(65, 88) if senior else max(19, tp_age + rng.randint(-8, 8))
        f1040.update(
            {
                "sp_first_name": rng.choice(FIRST_NAMES),
                "sp_last_name": f1040["tp_last_name"],
                "sp_date_of_birth": _dob(rng, sp_age),
                "sp_blind": rng.random() < (0.30 if senior else 0.04),
                "sp_dependent": rng.random() < 0.02,
                "sp_student": False,
            }
        )
    # taxpayer claimable as dependent (young filers, no dependents of their own)
    if tp_age < 24 and not joint and rng.random() < 0.3:
        f1040["tp_dependent"] = True

    # dependents
    n_deps = 0
    if not f1040["tp_dependent"]:
        if status == "head_of_household":
            n_deps = rng.randint(1, 4)
        elif rng.random() < (0.5 if joint else 0.3):
            n_deps = rng.choices([1, 2, 3, 4, 5, 6], weights=[40, 30, 15, 8, 4, 3])[0]
    if n_deps:
        f1040["dependent_detail"] = [
            _dependent(rng, force_ctc_age=(i == 0 and rng.random() < 0.5)) for i in range(n_deps)
        ]

    # Benchmark inputs carry boilerplate sections regardless of relevance;
    # matching that surface exactly prevents off-distribution rambling.
    data: dict = {
        "irs1040": f1040,
        "residency_status": "us_citizen",
        "has_ssn": True,
        "earned_in_another_state": False,
        "worked_and_lived_in_different_states": False,
        "worked_in_multiple_states": False,
        "nonmonetary_charity": False,
        "other_expenses": False,
        "irs8995a_schedulec": {"prior_yr_qbi_loss_carryforward": 0},
        "irs2441": {
            "tp_earned_income_adjustment": 0,
            "sp_earned_income_adjustment": 0,
            "carryover": 0,
            "forfeited": 0,
        },
    }
    if rng.random() < 0.5:
        data["irs_payment"] = {
            "account_type": rng.choice(["checking", "savings"]),
            "bank_account_number": str(rng.randint(10**8, 10**10)),
            "rtn": str(rng.randint(10**8, 10**9)),
            "direct_debit_indicator": False,
            "iat_account_payment": False,
            "payment_date": "2025-04-15",
            "day_phone": f"90{rng.randint(10000000, 99999999)}",
        }
    f1040.update({
        "tp_middle_initial": rng.choice(["", "A", "J", "M", ""]),
        "tp_occupation": rng.choice(["Engineer", "Teacher", "Manager", "Analyst", "Driver", "Nurse"]),
        "tp_presidential_election_campaign": rng.random() < 0.2,
        "refund_method": "direct_deposit" if rng.random() < 0.7 else "check",
        "bank_account_type": rng.choice(["checking", "savings"]),
        "routing_number": str(rng.randint(10**8, 10**9)),
        "account_number": str(rng.randint(10**8, 10**11)),
        "iat_account_refund": False,
        "qualifying_child_of_another": f1040.get("qualifying_child_of_another", False),
        "hoh_planning_to_claim_child_or_dependent_credit": status == "head_of_household",
    })
    header = {
        "tp_ssn": f"900{rng.randint(100000, 999999)}",
        "address": rng.choice(["asd", "123 Street", "44 Test Way"]) if placeholder_style
        else f"{rng.randint(1, 9999)} {rng.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Juniper'])} St",
        "city": rng.choice(["ABC", "Juneau", "City"]) if placeholder_style
        else rng.choice(["Anchorage", "Miami", "Austin", "Seattle", "Cheyenne", "Nashville", "Reno"]),
        "state": rng.choice(NO_TAX_STATES * 9 + OTHER_STATES),
        "zip_code": f"{rng.randint(10000, 99999)}",
        "tp_signature_date": "2025-02-15",
        "tp_prior_year_agi": _money(rng, 0, 90_000) if rng.random() < 0.6 else 0,
        # computation-irrelevant fields the benchmark always includes; the model
        # must learn to ignore them (they appear in 20-100% of benchmark cases).
        "tp_signature_pin": f"{rng.randint(10000, 99999)}",
        "tp_received_ippin": rng.random() < 0.15,
        "prior_year_filed_jointly": joint and rng.random() < 0.8,
    }
    if joint:
        header.update({
            "sp_ssn": f"900{rng.randint(100000, 999999)}",
            "sp_signature_pin": f"{rng.randint(10000, 99999)}",
            "sp_signature_date": "2025-02-15",
            "sp_prior_year_agi": _money(rng, 0, 90_000) if rng.random() < 0.6 else 0,
        })

    # ---- income sources ----
    features = []
    has_sched_c = rng.random() < 0.28
    se_share = rng.uniform(0.3, 1.0) if has_sched_c else 0.0
    wage_budget = income_target * (1 - se_share * 0.7)

    w2s = []
    # Seniors are the benchmark's main no-W-2 population (retirement/SS income);
    # bias them toward zero W-2s to match benchmark's 11.8% no-W-2 returns.
    if senior:
        n_w2_tp = rng.choices([0, 1, 2], weights=[55, 35, 10])[0]
    else:
        n_w2_tp = rng.choices([0, 1, 2, 3], weights=[8, 70, 17, 5])[0]
    # only force a W-2 when the return would otherwise have no income at all
    if n_w2_tp == 0 and not has_sched_c and not joint and not senior:
        n_w2_tp = 1
    quirks = {"state": header["state"]}
    same_ein = rng.random() < 0.04
    if same_ein:
        quirks["ein"] = _ein(rng)
    for _ in range(n_w2_tp):
        w = max(1_000, int(wage_budget / max(1, n_w2_tp) * rng.uniform(0.6, 1.4)))
        w2s.append(_w2(rng, "taxpayer", w, rng.choice(EMPLOYERS), dict(quirks)))
    if joint and rng.random() < 0.8:
        for _ in range(rng.choices([1, 2], weights=[80, 20])[0]):
            w = max(1_000, int(wage_budget * rng.uniform(0.2, 0.8)))
            w2s.append(_w2(rng, "spouse", w, rng.choice(EMPLOYERS), {"state": header["state"]}))
    if w2s:
        data["w2"] = w2s
        features.append("w2")
        if n_w2_tp >= 2:
            features.append("multi_w2")
        # dependent care benefits only make sense with young dependents
        if n_deps and any(True for d in f1040.get("dependent_detail", [])) and rng.random() < 0.12:
            w2s[0]["dependent_care_benefits"] = rng.choice([2_500, 5_000, _money(rng, 500, 5_000)])
            features.append("dep_care_benefits")

    sched1: dict = {
        "student_interest": 0,
        "paid_student_loan_interest": False,
        "qualified_educator": False,
        "tp_educator_exp_amount": 0,
        "sp_educator_exp_amount": 0,
    }

    if has_sched_c:
        n_biz = rng.choices([1, 2, 3], weights=[45, 20, 35])[0]
        scale = max(0.2, income_target / 100_000)
        businesses = []
        for i in range(n_biz):
            who = "spouse" if (joint and rng.random() < 0.4) else "taxpayer"
            businesses.append(_schedule_c(rng, who, scale))
        data["irs1040_schedulec"] = businesses
        features.append("sched_c")
        if rng.random() < 0.2:
            data["irs8995a_schedulec"] = {
                "prior_yr_qbi_loss_carryforward": _money(rng, 500, 30_000)
            }
            features.append("qbi_carryforward")

    if rng.random() < 0.35:
        ints = []
        for _ in range(rng.randint(1, 3)):
            ints.append(
                {
                    "interest_1099int_payer": rng.choice(["First Bank", "CreditOne CU", "Treasury Direct", "MegaBank"]),
                    "interest_1099int_interest": _money(rng, 10, 15_000),
                    "interest_1099int_us_saving_bond_treasury_obligations": _money(rng, 50, 3_000) if rng.random() < 0.15 else 0,
                    "interest_1099int_tax_exempt": _money(rng, 100, 8_000) if rng.random() < 0.15 else 0,
                    "interest_1099int_early_withdrawal": _money(rng, 20, 800) if rng.random() < 0.12 else 0,
                    "interest_1099int_fed_inc_tax_withheld": _money(rng, 20, 1_500) if rng.random() < 0.15 else 0,
                    "interest_1099int_foreign_taxes_paid": _money(rng, 10, 250) if rng.random() < 0.08 else 0,
                    "interest_1099int_who_applies_to": "taxpayer",
                    "interest_1099int_market_discount": _money(rng, 20, 1_000) if rng.random() < 0.1 else 0,
                    "interest_1099int_bond_premium_treasury_obligations": _money(rng, 20, 800) if rng.random() < 0.1 else 0,
                    "interest_1099int_specified_private_act_bond": _money(rng, 20, 500) if rng.random() < 0.08 else 0,
                }
            )
        sb = data.setdefault("irs1040_scheduleb", {})
        sb["irs1099_int"] = ints
        sb["foreign_accounts_input"] = rng.random() < 0.05
        sb["foreign_trust_input"] = rng.random() < 0.02
        features.append("interest")
    if rng.random() < 0.25:
        divs = []
        for _ in range(rng.randint(1, 2)):
            ordinary = _money(rng, 50, 25_000)
            divs.append(
                {
                    "dividend_1099div_payer": rng.choice(["Vanguard", "Fidelity", "Schwab"]),
                    "dividend_1099div_ordinary_div": ordinary,
                    "dividend_1099div_qualified_div": int(ordinary * rng.uniform(0.3, 1.0)),
                    "dividend_1099div_capital_gain_distr": _money(rng, 100, 10_000) if rng.random() < 0.35 else 0,
                    "dividend_1099div_federal_inc_tax_withheld": _money(rng, 20, 1_500) if rng.random() < 0.1 else 0,
                    "dividend_1099div_foreign_tax_paid": _money(rng, 10, 250) if rng.random() < 0.1 else 0,
                }
            )
        data.setdefault("irs1040_scheduleb", {})["irs1099_div"] = divs
        features.append("dividends")

    if rng.random() < (0.75 if senior else 0.13):
        data["ssa_1099"] = {
            "ssa_1099_grp": [
                {
                    "ssa_who_applies_to": "taxpayer",
                    "ssa_net_benefits": _money(rng, 3_000, 45_000),
                    "ssa_federal_withholding": _money(rng, 0, 3_000) if rng.random() < 0.3 else 0,
                    "ssa_lump_sum_benefits": 0,
                    "ssa_medicare_premiums": 0,
                }
            ]
        }
        features.append("social_security")
    if rng.random() < (0.55 if senior else 0.1):
        data["irs1099_r"] = [
            {
                "who_applies_to": "taxpayer",
                "payer_name_business_name_input": "Pension Trust Co",
                "gross_distribution_amt_input": (g := _money(rng, 2_000, 80_000)),
                "taxable_amt_input": int(g * rng.uniform(0.5, 1.0)),
                "federal_income_tax_withheld_amt_input": _money(rng, 0, 8_000) if rng.random() < 0.5 else 0,
                "f1099_r_distribution_cd_input": "7",
            }
        ]
        features.append("retirement_1099r")
    if rng.random() < 0.15:
        comp = _money(rng, 800, 25_000)
        sched1["irs1099_g"] = [
            {
                "g_payer_name": "State Workforce Agency",
                "g_who_applies_to": "taxpayer",
                "g_payer_tin": _ein(rng),
                "g_unemployment_comp": comp,
                "g_amount_repaid_current_year": _money(rng, 100, max(200, comp // 3)) if rng.random() < 0.15 else 0,
                "g_amount_repaid_prior_years": 0,
                "g_fed_wh": int(comp * 0.1) if rng.random() < 0.3 else 0,
            }
        ]
        features.append("unemployment")
    if rng.random() < 0.12:
        f8949, schedd = _form_8949(rng)
        data["irs8949"] = f8949
        if schedd:
            data["irs1040_schedule_d"] = schedd
            features.append("loss_carryover")
        features.append("capital_gains")

    # adjustments
    if rng.random() < 0.1:
        sched1["qualified_educator"] = True
        sched1["tp_educator_exp_amount"] = rng.choice([150, 250, 300, 301, 450])
        if joint and rng.random() < 0.5:
            sched1["sp_educator_exp_amount"] = rng.choice([150, 300, 400])
        features.append("educator")
    if rng.random() < 0.12:
        sched1["paid_student_loan_interest"] = True
        sched1["student_interest"] = _money(rng, 100, 4_500)
        features.append("student_loan")
    if header["state"] == "AK" and rng.random() < 0.3:
        sched1["alaska_permanent_dividend_tp_input"] = rng.choice([1_312, 1_702, 3_284])
        features.append("ak_pfd")
    data["irs1040_schedule1"] = sched1

    # credits forms
    young_deps = [d for d in f1040.get("dependent_detail", []) if 2024 - int(d["dependent_date_of_birth"][:4]) < 13]
    if young_deps and rng.random() < 0.3:
        data["irs2441"] = {
            "tp_earned_income_adjustment": 0,
            "sp_earned_income_adjustment": 0,
            "carryover": rng.choice([0, 0, 0, _money(rng, 200, 3_000)]),
            "forfeited": 0,
            "nondependent_indicator": False,
            "qualifying_person_grp": [
                {"person": f"dependent_{i}", "qualifying_expenses_paid": _money(rng, 500, 12_000)}
                for i, _ in enumerate(young_deps[: rng.randint(1, min(2, len(young_deps)))])
            ],
            "care_provider_grp": [
                {
                    "provider_name": "Little Sprouts Care",
                    "provider_ssn": f"900{rng.randint(100000, 999999)}",
                    "tax_exempt": False,
                    "paid_to_provider": _money(rng, 500, 12_000),
                }
            ],
        }
        features.append("cdcc")
    student_candidates = [
        d for d in f1040.get("dependent_detail", [])
        if d["dependent_student_for_5_plus_months"]
    ]
    if (student_candidates or (18 <= tp_age <= 30)) and rng.random() < 0.12:
        is_tp_student = not student_candidates or rng.random() < 0.4
        data["irs8863"] = {
            "total_support_8863": _money(rng, 10_000, 60_000),
            "either_parent_alive": rng.random() < 0.8,
            "student_educational_instn_grp": [
                {
                    "student": "taxpayer_info" if is_tp_student else "dependent_0",
                    "academic_period_eligible_student": rng.random() < 0.8,
                    "post_secondary_education": rng.random() < 0.15,
                    "drug_felony_conviction": False,
                    "prior_year_credit_claimed": rng.random() < 0.3,
                    "educational_institution_group": [
                        {
                            "institution_name": "State University",
                            "ein": _ein(rng),
                            "current_year_1098t_received": True,
                            "qualified_expenses": _money(rng, 800, 30_000),
                        }
                    ],
                }
            ],
        }
        features.append("education_credit")
    if rng.random() < 0.06:
        slcsp = _money(rng, 2_400, 14_000)
        data["irs8962"] = {
            "received_1095a": True,
            "full_year_coverage_1095a": True,
            "annual_premium": int(slcsp * rng.uniform(0.7, 1.3)),
            "annual_premium_slcsp": slcsp,
            "annual_advanced_ptc": int(slcsp * rng.uniform(0.0, 1.0)),
            "dependents_magi": 0,
            "married_in_tax_year": False,
            "share_policy_alt_calc": False,
            "dependent_required_to_file": False,
        }
        features.append("ptc_8962")
    else:
        data["irs8962"] = {
            "received_1095a": False, "dependents_magi": 0, "annual_premium": 0,
            "annual_premium_slcsp": 0, "annual_advanced_ptc": 0,
        }

    # payments
    if rng.random() < 0.12:
        f1040["paid_estimated_tax_pmts"] = True
        for q in (1, 2, 3, 4):
            f1040[f"estimated_tax_payment_{q}"] = rng.choice([0, _money(rng, 200, 15_000)])
        features.append("estimated_payments")
    if rng.random() < 0.05:
        f1040["applied_from_prior_year"] = _money(rng, 100, 5_000)
    data["irs1040_schedule3"] = {
        "requested_extension": rng.random() < 0.05,
        "extension_payment": 0,
    }
    if data["irs1040_schedule3"]["requested_extension"]:
        data["irs1040_schedule3"]["extension_payment"] = _money(rng, 200, 10_000)
        features.append("extension")

    if f1040["tp_dependent"]:
        features.append("dependent_filer")
    if n_deps:
        features.append("dependents")

    # difficulty tier (for curriculum ordering)
    tier = 0
    if n_deps or "multi_w2" in features:
        tier = 1
    if set(features) & {"interest", "dividends", "unemployment", "social_security", "retirement_1099r"}:
        tier = max(tier, 2)
    if "sched_c" in features:
        tier = max(tier, 3)
    if set(features) & {"capital_gains", "cdcc", "education_credit", "ptc_8962", "dep_care_benefits"}:
        tier = max(tier, 4)

    return {
        "name": f"synth-{seed}",
        "header": header,
        "data": data,
        "meta": {"seed": seed, "tier": tier, "features": sorted(set(features)), "profile": profile},
    }
