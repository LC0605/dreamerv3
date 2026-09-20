#!/usr/bin/env python3
"""Evaluate reward/continue heads on tagged risk-training rows."""

import argparse
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent


def config(path):
    y = yaml.YAML(typ="safe")
    cfgs = y.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    return elements.Config(cfgs["defaults"]).update(
        cfgs["quadrotor"]).update(y.load(path.read_text()))


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
    groups = {k: [] for k in ("ordinary", "pre_collision", "collision_terminal")}
    for path in sorted((a.dataset / "episodes").glob("*.npz")):
        with np.load(path) as source:
            d = {k: source[k] for k in source.files}
        n = len(d["reward"]); t = 301
        vector = np.zeros((1, t, 68), np.float32); vector[0, :n] = d["vector"]
        reward = np.zeros((1, t), np.float32); reward[0, :n] = d["reward"]
        first = np.ones((1, t), bool); first[0, :n] = d["is_first"]
        last = np.ones((1, t), bool); last[0, :n] = d["is_last"]
        terminal = np.zeros((1, t), bool); terminal[0, :n] = d["is_terminal"]
        prev = np.zeros((1, t, 4), np.float32); prev[0, 1:n] = d["action"][:n - 1]
        data = {
            "vector": vector, "reward": reward, "is_first": first,
            "is_last": last, "is_terminal": terminal,
            "teacher": np.zeros((1, t), bool),
            "action_policy_action": np.zeros((1, t, 4), np.float32),
            "loss_mask": np.arange(t)[None] < n,
            "diagnostic_prev_action": prev,
            "diagnostic_state_index": np.asarray([n - 1], np.int32),
            "diagnostic_candidate_action": np.zeros((1, 6, 4), np.float32),
        }
        out = agent.diagnose(data)
        pr, pc = np.asarray(out["reward_pred"])[0], np.asarray(out["continue_pred"])[0]
        indices = {
            "collision_terminal": np.flatnonzero(d["collision_terminal"]),
            "pre_collision": np.flatnonzero(d["pre_collision"]),
        }
        ordinary = np.flatnonzero((~d["is_first"]) & (~d["is_last"])
                                  & (~d["pre_collision"])
                                  & (~d["collision_terminal"])
                                  & (d["minimum_clearance"] >= .7))
        indices["ordinary"] = ordinary[::max(1, len(ordinary) // 20)][:20]
        for name, rows in indices.items():
            for row in rows:
                groups[name].append((float(d["reward"][row]), float(pr[row]),
                                     float(pc[row])))
    report = {}
    for name, rows in groups.items():
        report[name] = {
            "count": len(rows),
            "target_reward_mean": float(np.mean([x[0] for x in rows])),
            "predicted_reward_mean": float(np.mean([x[1] for x in rows])),
            "reward_mae": float(np.mean([abs(x[0] - x[1]) for x in rows])),
            "predicted_continue_mean": float(np.mean([x[2] for x in rows])),
        }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
