"""Exact IRS TY2024 arithmetic the microsimulation engine doesn't reproduce.

PolicyEngine computes tax from rate formulas; Form 1040 line 16 for taxable
income under $100,000 uses the IRS Tax Table (rate schedule applied to the
midpoint of a $25/$50 bracket, rounded). The Qualified Dividends and Capital
Gain Tax Worksheet also routes its ordinary component through the tax table.
"""

from __future__ import annotations

import math

# 2024 ordinary rate schedules: (top_of_bracket, rate)
RATE_SCHEDULES: dict[str, list[tuple[float, float]]] = {
    "single": [
        (11_600, 0.10), (47_150, 0.12), (100_525, 0.22), (191_950, 0.24),
        (243_725, 0.32), (609_350, 0.35), (math.inf, 0.37),
    ],
    "married_jointly": [
        (23_200, 0.10), (94_300, 0.12), (201_050, 0.22), (383_900, 0.24),
        (487_450, 0.32), (731_200, 0.35), (math.inf, 0.37),
    ],
    "married_separately": [
        (11_600, 0.10), (47_150, 0.12), (100_525, 0.22), (191_950, 0.24),
        (243_725, 0.32), (365_600, 0.35), (math.inf, 0.37),
    ],
    "head_of_household": [
        (16_550, 0.10), (63_100, 0.12), (100_500, 0.22), (191_950, 0.24),
        (243_700, 0.32), (609_350, 0.35), (math.inf, 0.37),
    ],
}

# 2024 capital-gains breakpoints: (top of 0% bracket, top of 15% bracket)
CG_BRACKETS: dict[str, tuple[float, float]] = {
    "single": (47_025, 518_900),
    "married_jointly": (94_050, 583_750),
    "married_separately": (47_025, 291_850),
    "head_of_household": (63_000, 551_350),
}


def rnd(x: float) -> int:
    """IRS whole-dollar rounding: 50 cents rounds up (with float-noise guard)."""
    return int(math.floor(x + 0.5 + 1e-9)) if x >= 0 else -int(math.floor(-x + 0.5 + 1e-9))


def schedule_tax(taxable: float, status: str) -> float:
    """Tax from the rate schedule formula (unrounded)."""
    tax = 0.0
    lower = 0.0
    for top, rate in RATE_SCHEDULES[status]:
        if taxable <= lower:
            break
        tax += (min(taxable, top) - lower) * rate
        lower = top
    return tax


def tax_table(taxable: float, status: str) -> int:
    """Form 1040 line 16 tax via IRS Tax Table (income < $100k) or schedules."""
    taxable = max(0.0, taxable)
    if taxable >= 100_000:
        return rnd(schedule_tax(taxable, status))
    # Tax-table row midpoints: [0,5)->0 tax, [5,15), [15,25), then $25-wide rows
    # up to $3,000, then $50-wide rows up to $100,000.
    if taxable < 5:
        return 0
    if taxable < 15:
        mid = 10.0
    elif taxable < 25:
        mid = 20.0
    elif taxable < 3_000:
        lower = math.floor(taxable / 25) * 25
        mid = lower + 12.5
    else:
        lower = math.floor(taxable / 50) * 50
        mid = lower + 25.0
    return rnd(schedule_tax(mid, status))


def qdcgt_worksheet(
    taxable_income: float,
    qualified_dividends: float,
    net_capital_gain: float,
    status: str,
) -> int:
    """Qualified Dividends and Capital Gain Tax Worksheet (Form 1040 line 16).

    net_capital_gain: the long-term gain amount taxed at preferential rates
    (min of Schedule D lines 15 and 16, floored at 0; or capital gain
    distributions when no Schedule D is required).
    """
    l1 = max(0.0, taxable_income)
    l3 = max(0.0, qualified_dividends) + max(0.0, net_capital_gain)
    l3 = min(l3, l1)
    if l3 <= 0:
        return tax_table(l1, status)
    l5 = l1 - l3  # ordinary income
    t0, t15 = CG_BRACKETS[status]
    l6 = t0
    l7 = min(l1, l6)
    l8 = min(l5, l7)
    l9 = l7 - l8  # taxed at 0%
    l10 = min(l1, l3)
    l12 = l10 - l9  # remaining preferential income
    l13 = t15
    l14 = min(l1, l13)
    l15 = l5 + l9
    l16 = max(0.0, l14 - l15)
    l17 = min(l12, l16)  # taxed at 15%
    l18 = l17 * 0.15
    l19 = l9 + l17
    l20 = l10 - l19  # taxed at 20%
    l21 = l20 * 0.20
    l22 = tax_table(l5, status)
    l23 = l18 + l21 + l22
    l24 = tax_table(l1, status)
    return min(rnd(l23), l24)


# 2024 Social Security wage-base maximum employee tax (6.2% of $168,600)
MAX_EMPLOYEE_SS_TAX_2024 = 10_453.20
# 2024 standard mileage rate (business)
STANDARD_MILEAGE_RATE_2024 = 0.67
# Simplified home-office deduction: $5/sq ft, 300 sq ft cap
HOME_OFFICE_RATE, HOME_OFFICE_SQFT_CAP = 5.0, 300
