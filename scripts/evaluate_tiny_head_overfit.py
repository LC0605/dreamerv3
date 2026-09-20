#!/usr/bin/env python3
"""Evaluate the fixed selected rows used by the tiny head overfit probe."""

import argparse
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent
from embodied.run.bc_distill import _episodes


def config(path):
    loader = yaml.YAML(typ="safe")
    cfgs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    saved = loader.load(path.read_text())
    return elements.Config(cfgs["defaults"]).update(cfgs["quadrotor"]).update(saved)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    agent = make_agent(config(a.config))
    elements.checkpoint.load(a.checkpoint, {"agent": partial(
        agent.load, regex="^(?!(bcopt|opt)/).*")})
    episodes = _episodes(a.dataset / "train", 301)
    groups = {key: [] for key in ("ordinary", "pre_collision_5", "collision_terminal")}
    for episode in episodes:
        prev = np.zeros((1, 301, 4), np.float32)
        prev[:, 1:] = episode["action"][None, :-1]
        selected = np.flatnonzero(episode["loss_mask"])
        data = {
            **{key: episode[key][None] for key in (
                "vector", "reward", "is_first", "is_last", "is_terminal",
                "teacher", "action_policy_action", "loss_mask")},
            "diagnostic_prev_action": prev,
            "diagnostic_state_index": np.asarray([int(selected[-1])], np.int32),
            "diagnostic_candidate_action": np.zeros((1, 6, 4), np.float32),
        }
        out = agent.diagnose(data)
        pred_r = np.asarray(out["reward_pred"])[0]
        pred_c = np.asarray(out["continue_pred"])[0]
        collision_episode = bool((episode["reward"] < -50).any())
        for row in selected:
            cls = ("collision_terminal" if episode["reward"][row] < -50 else
                   "pre_collision_5" if collision_episode else "ordinary")
            groups[cls].append({
                "target_reward": float(episode["reward"][row]),
                "predicted_reward": float(pred_r[row]),
                "target_continue": float(not episode["is_terminal"][row]),
                "predicted_continue": float(pred_c[row]),
            })
    report = {}
    for cls, rows in groups.items():
        report[cls] = {
            "count": len(rows),
            "target_reward_mean": float(np.mean([x["target_reward"] for x in rows])),
            "predicted_reward_mean": float(np.mean([x["predicted_reward"] for x in rows])),
            "reward_mae": float(np.mean([abs(x["target_reward"] - x["predicted_reward"])
                                         for x in rows])),
            "predicted_continue_mean": float(np.mean([x["predicted_continue"] for x in rows])),
            "rows": rows,
        }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {q: v[q] for q in v if q != "rows"}
                      for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
