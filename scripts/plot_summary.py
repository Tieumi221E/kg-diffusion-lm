#!/usr/bin/env python3
"""
Plot best QA EM (%) per split across different training-set scales.
X-axis: scale ascending (kgd_1000 -> kgd_200000).  Colour encodes split;
line style encodes decoding mode (random = dashed, low_confidence = solid).
By default scans runs/sft/kgd_* and reads metrics-step*.json files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

COLOR_MAP = {
    "all": "#3b8dbc",
    "people_reverse": "#f39c12",
    "people_forward": "#f6c344",
    "job": "#e74c3c",
    "two_hop": "#8a0fa7",
    "inversion": "#ff88a0",
    "symmetry": "#33c7c6",
    "incontext_inversion_with_ft": "#3cb7d6",
    "incontext_inversion_without_ft": "#8fd5ad",
    "incontext_symmetry_with_ft": "#f4a582",
    "incontext_symmetry_without_ft": "#1f988b",
}
LINE_STYLES = {"low_confidence": "-", "random": "--"}
MARKERS = {"low_confidence": "o", "random": "x"}

SPLIT_ORDER = [
    "all",
    "people_reverse",
    "people_forward",
    "job",
    "two_hop",
    "inversion",
    "symmetry",
    "incontext_inversion_with_ft",
    "incontext_inversion_without_ft",
    "incontext_symmetry_with_ft",
    "incontext_symmetry_without_ft",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot best QA EM per split across scales.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="List of runs dirs (e.g., runs/sft/kgd_1000). Default: glob runs/sft/kgd_*",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/sft/qa_best_by_scale.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--no-annotate", action="store_true", help="Disable value labels.")
    parser.add_argument(
        "--only-people",
        action="store_true",
        help="If set, plot only people_reverse and people_forward.",
    )
    return parser.parse_args()


def safe_em(entry: Dict[str, object]) -> float:
    if entry is None:
        return math.nan
    if entry.get("em") is not None:
        return float(entry["em"]) * 100.0
    correct = float(entry.get("correct", 0.0))
    count = float(entry.get("count", 0.0))
    return correct / count * 100.0 if count > 0 else math.nan


def load_best_from_dir(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Returns: split -> mode -> best_em."""
    files = sorted(run_dir.glob("metrics-step*.json"))
    best: Dict[str, Dict[str, float]] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data:
            ds = entry.get("dataset", "")
            if not isinstance(ds, str) or not ds.startswith("qa-"):
                continue
            parts = ds.split("-")
            if len(parts) < 3:
                continue
            split = "-".join(parts[1:-1])
            mode = parts[-1]
            em_val = safe_em(entry)
            cur = best.setdefault(split, {})
            prev = cur.get(mode, -math.inf)
            if math.isnan(em_val):
                continue
            if em_val > prev:
                cur[mode] = em_val
    return best


def main() -> None:
    args = parse_args()
    roots = args.roots
    if roots is None:
        roots = [str(p) for p in sorted(Path("runs/sft").glob("kgd_*")) if p.is_dir()]
    def scale_key(p: str) -> int:
        m = re.search(r"kgd_(\d+)", p)
        return int(m.group(1)) if m else 0

    roots = sorted(roots, key=scale_key)
    scales = [scale_key(r) for r in roots]

    series: Dict[str, Dict[str, List[float]]] = {}
    for run in roots:
        best = load_best_from_dir(Path(run))
        for split, modes in best.items():
            series.setdefault(split, {"random": [], "low_confidence": []})
        for split in list(series.keys()):
            modes = series[split]
            for mode in ["random", "low_confidence"]:
                em_val = best.get(split, {}).get(mode, math.nan)
                modes.setdefault(mode, []).append(em_val)

    splits_to_plot = ["people_reverse", "people_forward"] if args.only_people else SPLIT_ORDER

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    modes_to_plot = [("random", "Random"), ("low_confidence", "Low confidence")]

    for ax, (mode_key, mode_label) in zip(axes, modes_to_plot):
        for idx, split in enumerate(splits_to_plot):
            if split not in series:
                continue
            vals = series[split].get(mode_key, [])
            if not vals:
                continue
            color = COLOR_MAP.get(split, plt.cm.tab20.colors[idx % 20])
            ax.plot(
                scales,
                vals,
                color=color,
                linestyle=LINE_STYLES.get(mode_key, "-"),
                marker=MARKERS.get(mode_key, "o"),
                linewidth=2,
                markersize=6,
                label=split.replace("_", " "),
            )
            if not args.no_annotate:
                for x, y in zip(scales, vals):
                    if math.isnan(y):
                        continue
                    ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 6), textcoords="offset points", fontsize=8, color=color, ha="center")
        ax.set_title(f"{mode_label} (best EM)")
        ax.set_xlabel("Scale")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("EM (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, framealpha=0.9, borderaxespad=0.5, title="QA splits")
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"saved plot to {output}")


if __name__ == "__main__":
    main()
