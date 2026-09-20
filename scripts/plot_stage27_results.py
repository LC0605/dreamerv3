from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["随机", "单横穿", "双横穿", "迎面", "垂直"]
BASELINE = [
    "outputs/ablations/stage25_threat_absolute_random/metrics.json",
    "outputs/ablations/stage25_threat_absolute_forced/metrics.json",
    "outputs/evaluation/stage25_best_double_crossing/metrics.json",
    "outputs/evaluation/stage25_best_head_on/metrics.json",
    "outputs/evaluation/stage25_best_vertical/metrics.json",
]
SHIELDED = [
    "outputs/evaluation/stage27_shield_random_100/metrics.json",
    "outputs/evaluation/stage27_shield_single_crossing_100/metrics.json",
    "outputs/evaluation/stage27_shield_double_crossing_100/metrics.json",
    "outputs/evaluation/stage27_shield_head_on_100/metrics.json",
    "outputs/evaluation/stage27_shield_vertical_100/metrics.json",
]


def load(paths: list[str]) -> list[dict]:
    return [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    baseline = load(BASELINE)
    shielded = load(SHIELDED)
    x = np.arange(len(SCENARIOS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    for axis, key, title in (
        (axes[0], "success_rate", "成功率（越高越好）"),
        (axes[1], "collision_rate", "碰撞率（越低越好）"),
    ):
        old = 100 * np.array([row[key] for row in baseline])
        new = 100 * np.array([row[key] for row in shielded])
        axis.bar(x - width / 2, old, width, label="威胁排序 PPO", color="#7096c8")
        axis.bar(x + width / 2, new, width, label="PPO + 动力学安全投影", color="#38a36b")
        axis.set_title(title)
        axis.set_xticks(x, SCENARIOS)
        axis.set_ylim(0, 105 if key == "success_rate" else 40)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel("百分比 / %")
        for index, value in enumerate(old):
            axis.text(index - width / 2, value + 1, f"{value:.0f}", ha="center", fontsize=9)
        for index, value in enumerate(new):
            axis.text(index + width / 2, value + 1, f"{value:.0f}", ha="center", fontsize=9)
    axes[0].legend(loc="lower left")
    output = ROOT / "outputs/report/stage27_safety_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("12 动态障碍独立评测（每类 100 回合）", fontsize=14)
    fig.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
