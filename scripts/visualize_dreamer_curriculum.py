#!/usr/bin/env python3
"""Build self-contained curriculum progress plots from Dreamer JSONL logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/dreamerv3"
DEST = SOURCE / "visualization"


def stage_key(path: Path):
    name = path.name
    prefixes = ("stage0", "stage1", "stage2", "stage3", "stage4", "stage5")
    rank = next((i for i, prefix in enumerate(prefixes) if name.startswith(prefix)), 99)
    return rank, name


def load_episodes(folder: Path):
    result = []
    path = folder / "metrics.jsonl"
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "episode/score" in row:
            result.append({
                "stage": folder.name, "step": int(row["step"]),
                "score": float(row["episode/score"]),
                "length": int(row["episode/length"]),
                "success": int(float(row["episode/score"]) > 0.0),
            })
    return result


def rolling(values, window=20):
    values = np.asarray(values, dtype=float)
    return np.array([
        values[max(0, index - window + 1):index + 1].mean()
        for index in range(len(values))
    ])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    stages = sorted(
        [path for path in SOURCE.iterdir() if path.is_dir() and path.name.startswith("stage")],
        key=stage_key,
    )
    episodes = {stage.name: load_episodes(stage) for stage in stages}
    episodes = {key: value for key, value in episodes.items() if value}

    with (DEST / "curriculum_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("stage", "step", "score", "length", "success"))
        writer.writeheader()
        for rows in episodes.values():
            writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for stage, rows in episodes.items():
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, rolling([row["success"] for row in rows]), label=stage)
        axes[1].plot(steps, rolling([row["score"] for row in rows]), label=stage)
        axes[2].plot(steps, rolling([row["length"] for row in rows]), label=stage)
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1, label="90% gate")
    axes[0].set_ylabel("Rolling success rate")
    axes[0].set_ylim(-0.02, 1.04)
    axes[1].set_ylabel("Rolling episode score")
    axes[2].set_ylabel("Rolling episode length")
    axes[2].set_xlabel("Environment steps within stage")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", fontsize=7)
    fig.suptitle("DreamerV3 UAV Curriculum Training Progress")
    fig.savefig(DEST / "curriculum_progress.png", dpi=170)
    plt.close(fig)

    evaluations = []
    for path in sorted(SOURCE.glob("eval_*/evaluation_summary.json")):
        data = json.loads(path.read_text())
        evaluations.append({"name": path.parent.name.removeprefix("eval_"), **data})
    if evaluations:
        names = [row["name"] for row in evaluations]
        successes = [row.get("success_rate", 0.0) for row in evaluations]
        collisions = [row.get("collisions", 0) / max(row.get("episodes", 1), 1)
                      for row in evaluations]
        timeouts = [row.get("timeouts", 0) / max(row.get("episodes", 1), 1)
                    for row in evaluations]
        y = np.arange(len(names))
        fig, axis = plt.subplots(figsize=(12, max(4, 0.45 * len(names) + 2)),
                                 constrained_layout=True)
        axis.barh(y, successes, label="success")
        axis.barh(y, collisions, left=successes, label="collision")
        axis.barh(y, timeouts, left=np.asarray(successes) + collisions, label="timeout")
        axis.axvline(0.90, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(y, names, fontsize=7)
        axis.set_xlim(0, 1.05)
        axis.set_xlabel("Episode fraction")
        axis.set_title("Independent Deterministic Evaluation")
        axis.legend()
        axis.grid(axis="x", alpha=0.25)
        fig.savefig(DEST / "evaluation_overview.png", dpi=170)
        plt.close(fig)

    summary = {
        "stages_with_episodes": len(episodes),
        "total_training_episodes": sum(map(len, episodes.values())),
        "evaluations": evaluations,
        "artifacts": ["curriculum_progress.png", "evaluation_overview.png",
                      "curriculum_episodes.csv"],
    }
    (DEST / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
