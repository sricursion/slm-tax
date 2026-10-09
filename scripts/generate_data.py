"""Generate an oracle-labeled synthetic tax-return dataset.

Each JSONL row: {id, seed, tier, features, profile, input_json, target, lines}.
The prompt is not stored per-row (constant template in slm_tax.render); apply
render.PROMPT_TEMPLATE.format(input_data=row["input_json"]) at training time.

Also runs three integrity checks:
1. round-trip: wrapped input JSON -> tcb unwrap -> engine gives identical lines
2. contamination: no generated case matches a TaxCalcBench case signature
3. crash-free rate across the full run
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_tax.engine import Return, compute_lines  # noqa: E402
from slm_tax.generator import generate_case  # noqa: E402
from slm_tax.render import build_input_json, render_target  # noqa: E402
from slm_tax.tcb import TY24_TEST_DATA, _unwrap, load_all_cases  # noqa: E402


def _canonical_input_hash(header: dict, data: dict) -> str:
    """SHA-256 of the canonical-JSON full input (sorted keys, no whitespace).

    Exact-input contamination guard: unlike the feature signature below, this
    is insensitive to which lines end up scored and catches a generated case
    that reproduces a benchmark input byte-for-byte at the semantic level.
    """
    canon = json.dumps(
        {"header": header, "data": data}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def benchmark_signatures() -> tuple[set, set]:
    """Return (feature signatures, canonical full-input hashes) of all 51 cases."""
    sigs: set = set()
    hashes: set = set()
    if not TY24_TEST_DATA.exists():
        return sigs, hashes
    for case in load_all_cases():
        hashes.add(_canonical_input_hash(case["header"], case["data"]))
        try:
            lines = compute_lines(case)
            sigs.add(
                (
                    case["data"].get("irs1040", {}).get("filing_status"),
                    lines["WagesAmt"],
                    lines["AdjustedGrossIncomeAmt"],
                )
            )
        except Exception:
            continue
    return sigs, hashes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/train.jsonl")
    ap.add_argument("--roundtrip-every", type=int, default=50)
    ap.add_argument("--compact-form", action="store_true", help="ablation: trace + 19 scored lines only")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bench_sigs, bench_hashes = benchmark_signatures()
    tiers = collections.Counter()
    features = collections.Counter()
    nonzero = collections.Counter()
    errors = 0
    roundtrip_fail = 0
    contaminated = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fh:
        written = 0
        seed = args.seed
        while written < args.n:
            case = generate_case(seed)
            seed += 1
            try:
                ret = Return(case)
                lines = ret.compute()
            except Exception:
                errors += 1
                if errors <= 5:
                    import traceback

                    traceback.print_exc(limit=3)
                continue

            input_json = build_input_json(case)
            target = render_target(case, ret, compact_form=args.compact_form)

            # integrity: round-trip through the benchmark's unwrap path
            if written % args.roundtrip_every == 0:
                rt = _unwrap(json.loads(input_json)["input"])
                rt_case = {"name": case["name"], "header": rt["return_header"], "data": rt["return_data"]}
                if compute_lines(rt_case) != lines:
                    roundtrip_fail += 1

            sig = (
                case["data"]["irs1040"]["filing_status"],
                lines["WagesAmt"],
                lines["AdjustedGrossIncomeAmt"],
            )
            if sig in bench_sigs:
                contaminated += 1
                continue
            if _canonical_input_hash(case["header"], case["data"]) in bench_hashes:
                contaminated += 1
                continue

            row = {
                "id": case["name"],
                "seed": case["meta"]["seed"],
                "tier": case["meta"]["tier"],
                "features": case["meta"]["features"],
                "profile": case["meta"]["profile"],
                "input_json": input_json,
                "target": target,
                "lines": lines,
            }
            fh.write(json.dumps(row) + "\n")
            written += 1
            tiers[case["meta"]["tier"]] += 1
            for f in case["meta"]["features"]:
                features[f] += 1
            for tag, val in lines.items():
                if val:
                    nonzero[tag] += 1

    dt = time.time() - t0
    n = args.n
    print(f"wrote {n} examples to {out_path} in {dt:.1f}s ({n / dt:.0f}/s)")
    print(f"engine errors: {errors} | round-trip failures: {roundtrip_fail} | contaminated skipped: {contaminated}")
    print("\ntier distribution:")
    for t in sorted(tiers):
        print(f"  tier {t}: {tiers[t]} ({tiers[t] / n:.1%})")
    print("\nfeature coverage:")
    for f, c in features.most_common():
        print(f"  {f:20s} {c:6d} ({c / n:.1%})")
    print("\nnonzero rate per scored line:")
    for tag, c in nonzero.most_common():
        print(f"  {tag:30s} {c / n:.1%}")

    stats = {
        "n": n, "seconds": dt, "errors": errors, "roundtrip_failures": roundtrip_fail,
        "contaminated_skipped": contaminated, "tiers": dict(tiers),
        "features": dict(features), "nonzero": dict(nonzero),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nstats: {stats_path}")


if __name__ == "__main__":
    main()
