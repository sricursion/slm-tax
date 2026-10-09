# Reverse-engineered conventions of the TaxCalcBench reference engine

Discovered empirically while driving oracle agreement from 45% to 98% strict.
These are the deltas between "what the IRS forms say" naively implemented and
what Column Tax's engine (the benchmark's ground truth) actually computes.
Each was validated against expected `output.xml` values, often including the
intermediate form XML (Schedule C/D/SE, Form 8995).

## Rounding

- Whole-dollar rounding is half-up per form line (`rnd`), with the tax table
  and EITC table using the IRS midpoint-of-bracket method.
- **Schedule SE** rounds the 12.4% social security and 2.9% Medicare
  components *separately*, then the half-SE-tax ALD is `rnd(tax/2)`.
- **Schedule C** net profit is rounded per schedule; within the car/truck
  line, the standard-mileage product is rounded before adding prorated items,
  then the whole line is rounded.
- **Form 8995** rounds the 20% QBI component *per business* (confirmed via
  `QBIComponentAmt` XML values).

## Field semantics in the input JSON

- `dependent_ssn_itin_atin` is a **boolean** (truthy = has SSN), not a string.
- `qualified_educator` / `paid_student_loan_interest` booleans do **not** gate
  the deductions — the engine uses the entered amounts directly (capped $300
  per educator, $2,500 student loan with MAGI phase-out).
- `dependent_supported_by_tp: false` on a child-relationship dependent means
  the person is not a dependent at all (support test failed).
- `dependent_us_citizen_national_or_resident_alien: false` drops the
  dependent (citizenship test).
- `user_adjusted_qbi` (with `use_calculated_qbi: false`) replaces the
  business's computed QBI and is taken as-is (no ALD allocation to it).
- 1099-MISC `rents`, `royalties`, `fish_boat_proceeds`, `medandhlth_care_pmts`,
  `crop_ins_proceeds` are all treated as Schedule C gross receipts (no
  Schedule E in this dataset).
- Form 8949 term (short/long) is determined by the **acquired/sold dates**
  (held more than one year = long-term), not the `input_sale_term` label —
  at least one case has a mislabeled row.
- `tp_dependent: true` (taxpayer claimable as a dependent) blocks CTC/ODC,
  CDCC, education credits, EITC, and the student-loan ALD.

## Computation rules

- **Excess social security credit** (Sch 3 line 11): per person, cap each
  *employer's* (EIN's) withholding at the yearly max first; excess = capped
  total − max. Same-EIN W-2s are one employer; a single employer
  over-withholding yields no credit.
- **Form 8959 Part IV** (Medicare withholding reconciliation → line 25c):
  only when *total* Medicare wages (both spouses) exceed $200,000; amount =
  `rnd(total box 6) − rnd(1.45% × total box 5)`, floored at 0.
- **Vehicle expenses** (Sch C line 9): standard mileage × $0.67 + parking +
  tolls in full + (auto loan interest + personal property tax) × business-use
  percentage (business ÷ total miles).
- **Simplified home office** ($5/sqft, 300 sqft cap) cannot create or
  increase a loss (capped at profit-before-home-office, per business).
- **Form 8962**: %FPL floored to a whole percent; applicable figure linearly
  interpolated to 4 decimals; repayment caps by %FPL band.
- **EITC**: IRS table method — credit formula evaluated at the midpoint of
  the $50 bracket, for earned income and (if past phase-out start) AGI, take
  the smaller. 2024 maxima: 632 / 4,213 / 6,960 / 7,830.
- Taxable dependent-care benefits (Form 2441 Part III excess) are wage income
  (line 1e) and count as earned income for EITC/ACTC.
- QBI is reduced by SE-tax ALD + SE health insurance ALD allocated across
  *calculated* (non-overridden) businesses by profit share.

## Benchmark data issues found

- `single-senior-blind-over-65`: known broken (missing 1095-A inputs,
  [issue #66](https://github.com/column-tax/tax-calc-bench/issues/66)).
  Excluded from achievable-agreement reporting.
- `single-w2-schedulec-1099b-capital-loss-carryover`: one 8949 row is labeled
  `long_term` but is short-term by its dates; the expected Schedule D XML
  confirms date-based classification.
- `mfj-multiple-w2-schedule-c-qbi-income`: residual $1 disagreement in QBID
  ALD allocation across four businesses (their `AdjQualifiedBusinessIncomeAmt`
  split doesn't match any proportional allocation we tried); within lenient
  tolerance.
