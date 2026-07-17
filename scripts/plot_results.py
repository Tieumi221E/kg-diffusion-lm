#!/usr/bin/env python3
"""
Unified plotting tool for KGdLLM experiments.
Supports plotting from:
1. Training logs (train.log) - Loss and Token Accuracy.
2. Step metrics (metrics-step*.json) - QA Accuracy trends.
3. Bulk evaluation (eval_all_checkpoints.json) - Comprehensive sweep analysis.
"""

import argparse
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def parse_log(path: Path) -> Tuple[List[int], List[float]]:
    steps_loss, losses = [], []
    if not path.exists():
        return steps_loss, losses
    pat_loss = re.compile(r"Step\s+(\d+).+loss=([0-9.]+)")
    with path.open() as f:
        for line in f:
            m = pat_loss.search(line)
            if m:
                steps_loss.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps_loss, losses


def load_json_metrics(metrics_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    files = sorted(metrics_dir.glob("metrics-step*.json"))
    results = {}  # split -> {steps: [], values: []}
    for p in files:
        step = int(re.search(r"step(\d+)", p.name).group(1))
        data = json.loads(p.read_text())
        for entry in data:
            ds = entry.get("dataset", "")
            if ds.startswith("qa-") and ds.endswith("-low_confidence"):
                split = ds.replace("qa-", "").replace("-low_confidence", "")
                if split not in results:
                    results[split] = {"steps": [], "values": []}
                results[split]["steps"].append(step)
                results[split]["values"].append(entry.get("em", 0) * 100)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, help="Path to train.log")
    parser.add_argument(
        "--metrics-dir", type=Path, help="Directory with metrics-step*.json"
    )
    parser.add_argument(
        "--eval-json", type=Path, help="Path to eval_all_checkpoints.json"
    )
    parser.add_argument("--output", type=Path, default=Path("results.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Plot Loss
    if args.log:
        steps, losses = parse_log(args.log)
        if steps:
            axes[0].plot(steps, losses, label="Train Loss")
            axes[0].set_title("Training Loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

    # 2. Plot QA Accuracy
    if args.metrics_dir:
        qa_results = load_json_metrics(args.metrics_dir)
        for split, data in qa_results.items():
            axes[1].plot(data["steps"], data["values"], label=split)
        axes[1].set_title("QA Accuracy (EM %)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
