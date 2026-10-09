"""Paper figures (E5.3). Emits PDF + PNG into paper/figures/.

Figures that need only existing data render immediately; the complexity-wall
figure (a) reads numbers from paper/figures/complexity_wall.json so it can be
re-rendered when the E0.3/E1 synthetic numbers land.

Usage: python scripts/figures.py [--only a|b|c|d]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tax-calc-bench"))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    # IEEE camera-ready: Times-family text to match the IEEEtran body font, and
    # TrueType (Type 42) embedding. Matplotlib defaults to pdf.fonttype 3, and
    # Type 3 fonts are routinely flagged by IEEE PDF eXpress.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote paper/figures/{name}.pdf/.png")


# --- (a) complexity wall -----------------------------------------------------

def fig_complexity_wall() -> None:
    src = OUT / "complexity_wall.json"
    if not src.exists():
        print("skip (a): paper/figures/complexity_wall.json not yet written")
        return
    d = json.loads(src.read_text())
    # d: {"benchmark": {model: {"simple": [k, n], "complex": [k, n]}},
    #     "synthetic": {model: {...}}}
    panels = [k for k in ("benchmark", "synthetic") if k in d]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.8), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        models = list(d[panel])
        x = range(len(models))
        w = 0.38
        for off, seg, color in ((-w / 2, "simple", "#4878a8"), (w / 2, "complex", "#c44e52")):
            vals, los, his = [], [], []
            for m in models:
                k, n = d[panel][m][seg]
                p = k / n if n else 0.0
                # Wilson 95% interval
                z = 1.96
                den = 1 + z * z / n
                ctr = (p + z * z / (2 * n)) / den
                rad = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
                vals.append(100 * p)
                los.append(100 * max(0.0, ctr - rad))
                his.append(100 * min(1.0, ctr + rad))
            err = [[v - lo for v, lo in zip(vals, los)], [hi - v for v, hi in zip(vals, his)]]
            ax.bar([i + off for i in x], vals, w, yerr=err, capsize=2,
                   label=seg, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models)
        ax.set_title(f"{panel} (n per bar varies)")
        ax.set_ylabel("strict %")
    axes[0].legend(frameon=False)
    save(fig, "complexity_wall")


# --- (b) ablation ladder -----------------------------------------------------

LADDER = [
    ("direct\nanswers", 5.9, "1.7B"),
    ("+traces", 2.0, "1.7B"),
    ("+term/compact\n/coverage", 2.0, "1.7B"),
    ("+bracket\nderivation", 11.8, "1.7B"),
    ("+SchC\ncomponents", 11.8, "1.7B"),
    ("4B\n(same data)", 31.4, "4B"),
    ("+SE/EITC\nfixes", 29.4, "4B"),
    ("+maj@4\nvoting", 27.5, "4B"),
]


def fig_ablation() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    xs = range(len(LADDER))
    ys = [r[1] for r in LADDER]
    colors = ["#4878a8" if r[2] == "1.7B" else "#c44e52" for r in LADDER]
    ax.plot(xs, ys, color="#888", lw=1, zorder=1)
    ax.scatter(xs, ys, c=colors, s=42, zorder=2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in LADDER], fontsize=7)
    ax.set_ylabel("benchmark strict %")
    ax.set_ylim(0, 38)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#4878a8", label="Qwen3-1.7B"),
        Line2D([], [], marker="o", ls="", color="#c44e52", label="Qwen3-4B"),
    ], frameon=False, loc="upper left")
    save(fig, "ablation_ladder")


# --- (c) first-divergence histogram ------------------------------------------

RUN_DIRS = [
    ("pilot-5", "artifacts/ckpt500-gen-texts"),
    ("pilot-6", "artifacts/pilot6-ckpt500-gen-texts"),
    ("pilot-7", "artifacts/pilot7-gen-texts"),
    ("run-8 (4B)", "artifacts/headline-4b-gen-texts"),
    ("run-9 (4B)", "artifacts/v8-gen-texts"),
]

CHAIN = ["1a", "9", "10", "11", "12", "15", "16", "19", "24", "25d", "26",
         "27", "28", "29", "32", "33", "34", "35a", "37"]


def first_divergence(gen_dir: Path) -> Counter:
    from lxml import etree

    from slm_tax.score import parse_line
    from slm_tax.tcb import load_all_cases
    from tax_calc_bench.tax_return_evaluator import LINES_TO_XPATH_VALUES

    tag_order = {line.split(":")[0].replace("Line ", ""): i
                 for i, line in enumerate(LINES_TO_XPATH_VALUES)}
    counts: Counter = Counter()
    for case in load_all_cases():
        f = gen_dir / f"{case['name']}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        tree = etree.fromstring(case["expected_xml"].encode())
        first = None
        for line, xp in LINES_TO_XPATH_VALUES.items():
            nodes = tree.xpath(xp)
            want = float(nodes[0].text) if nodes and nodes[0].text else 0.0
            if abs(parse_line(text, line) - want) >= 0.5:
                tag = line.split(":")[0].replace("Line ", "")
                if first is None or tag_order[tag] < tag_order[first]:
                    first = tag
        if first:
            counts[first] += 1
    return counts


def fig_first_divergence() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    width = 0.8 / len(RUN_DIRS)
    for i, (label, d) in enumerate(RUN_DIRS):
        p = ROOT / d
        if not p.exists():
            continue
        counts = first_divergence(p)
        xs = [j + i * width for j in range(len(CHAIN))]
        ax.bar(xs, [counts.get(t, 0) for t in CHAIN], width, label=label)
    ax.set_xticks([j + 0.4 for j in range(len(CHAIN))])
    ax.set_xticklabels(CHAIN, fontsize=7)
    ax.set_xlabel("first divergent line")
    ax.set_ylabel("cases")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "first_divergence")


# --- (d) by-line vs strict scatter -------------------------------------------

BASELINES = [
    ("kimi-k2p6", 31.4, 69.1), ("deepseek-v4-pro", 29.4, 80.3),
    ("glm-5p1", 23.5, 72.5), ("kimi-k2p7-code", 23.5, 71.0),
    ("glm-5p2", 17.6, 67.3), ("gpt-oss-120b", 0.0, 62.3),
    ("all-zeros", 0.0, 44.7), ("ours 4B", 31.4, 72.9), ("ours 1.7B", 11.8, 66.6),
    ("GPT-5.4 Pro (cited)", 62.75, 89.99), ("oracle", 96.1, 98.9),
]


def fig_scatter() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, strict, by_line in BASELINES:
        if name.startswith("ours"):
            c, m = "#c44e52", "D"
        elif name in ("oracle", "GPT-5.4 Pro (cited)"):
            c, m = "#777777", "s"
        elif name == "all-zeros":
            c, m = "#000000", "x"
        else:
            c, m = "#4878a8", "o"
        ax.scatter(by_line, strict, c=c, marker=m, s=34)
        ax.annotate(name, (by_line, strict), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5)
    ax.set_xlabel("by-line %")
    ax.set_ylabel("strict %")
    ax.set_xlim(40, 102)
    ax.set_ylim(-4, 102)
    save(fig, "byline_vs_strict")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["a", "b", "c", "d"], default=None)
    args = ap.parse_args()
    fns = {"a": fig_complexity_wall, "b": fig_ablation,
           "c": fig_first_divergence, "d": fig_scatter}
    for key, fn in fns.items():
        if args.only in (None, key):
            fn()
