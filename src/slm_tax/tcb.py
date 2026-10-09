"""Load and flatten TaxCalcBench TY24 test cases.

TaxCalcBench inputs wrap every leaf in {"label": ..., "value": ...}. This module
strips the wrappers into plain nested dicts/lists and exposes typed accessors for
the sections the oracle needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TY24_TEST_DATA = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "tax-calc-bench"
    / "tax_calc_bench"
    / "ty24"
    / "test_data"
)


def _unwrap(node: Any) -> Any:
    """Recursively replace {"label": ..., "value": v} wrappers with v."""
    if isinstance(node, dict):
        if "value" in node and set(node.keys()) <= {"label", "value"}:
            return node["value"]
        return {k: _unwrap(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_unwrap(item) for item in node]
    return node


def load_case(case_dir: Path) -> dict:
    """Load one test case; returns {"name", "header", "data", "expected_xml"}."""
    raw = json.loads((case_dir / "input.json").read_text(encoding="utf-8"))
    inp = _unwrap(raw["input"])
    return {
        "name": case_dir.name,
        "header": inp.get("return_header", {}),
        "data": inp.get("return_data", {}),
        "expected_xml": (case_dir / "output.xml").read_text(encoding="utf-8"),
    }


def load_all_cases(test_data_dir: Path = TY24_TEST_DATA) -> list[dict]:
    return [
        load_case(d)
        for d in sorted(test_data_dir.iterdir())
        if d.is_dir() and (d / "input.json").exists()
    ]


def num(x: Any) -> float:
    """Coerce a possibly-missing/None/str numeric field to float."""
    if x is None or x is False:
        return 0.0
    if x is True:
        return 1.0
    if isinstance(x, str):
        x = x.replace(",", "").replace("$", "").strip()
        if x == "":
            return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
