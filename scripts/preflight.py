"""One command to verify the whole pipeline before spending GPU.

Runs every gate the runbook mandates and reports a single pass/fail. Cheap
(~1-2 min CPU). If anything here fails, do NOT launch a training run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(desc: str, argv: list[str]) -> tuple[str, bool, str]:
    p = subprocess.run(
        [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True
    )
    return desc, p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1] if (p.stdout + p.stderr).strip() else ""


def inline_roundtrip(n: int = 2000) -> tuple[str, bool, str]:
    from slm_tax.engine import Return
    from slm_tax.generator import generate_case
    from slm_tax.render import END_OF_RETURN, render_target
    from slm_tax.score import LABEL_TO_TAG, score_output

    ok = 0
    for s in range(200000, 200000 + n):
        case = generate_case(s)
        r = Return(case)
        lines = r.compute()
        t = render_target(case, r)
        if (
            t.rstrip().endswith(END_OF_RETURN)
            and all(t.count(lbl) == 1 for lbl in LABEL_TO_TAG)
            and score_output(t, lines)["strict"]
        ):
            ok += 1
    return "round-trip (2000 fresh)", ok == n, f"{ok}/{n} strict-parse"


def main() -> None:
    checks = [
        run("oracle vs benchmark (>=98%)", ["scripts/validate_oracle.py"]),
        run("format ceiling (49/50, 50/50 lenient)", ["scripts/check_format_ceiling.py"]),
        run("trace self-consistency (0 bad)", ["scripts/check_trace_consistency.py", "3000"]),
        inline_roundtrip(),
    ]
    print("\n===== PRE-FLIGHT =====")
    all_ok = True
    for desc, ok, tail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc:42s} {tail}")
        all_ok &= ok
    print("=" * 22)
    print("ALL GREEN - safe to launch" if all_ok else "BLOCKED — fix before GPU")
    print("(also run `python scripts/coverage.py` and eyeball for UNDER-COVERED)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
