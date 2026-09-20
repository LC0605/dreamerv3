#!/usr/bin/env python3
"""Compare two Dreamer policies on identical recorded observation histories."""

import argparse
import csv
import gc
import json
from functools import partial
from pathlib import Path

import elements
import jax
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent


def load_config(path):
    loader = yaml.YAML(typ="safe")
    configs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    return elements.Config(configs["defaults"]).update(configs["quadrotor"]).update(
        loader.load(Path(path).read_text()))


def observation(row):
    return {
        "vector": np.asarray([[float(row[f"vector_{i:02d}"]) for i in range(68)]], np.float32),
        "reward": np.asarray([0.0], np.float32),
        "is_first": np.asarray([bool(int(row["is_first"]))]),
        "is_last": np.asarray([bool(int(row["is_last"]))]),
        "is_terminal": np.asarray([bool(float(row["collision"]) or float(row["success"]))]),
        "teacher": np.asarray([False]),
        "action_policy_action": np.zeros((1, 4), np.float32),
    }


def predictions(config, checkpoint, rows):
    agent = make_agent(load_config(config))
    elements.checkpoint.load(checkpoint, {"agent": partial(
        agent.load, regex="^(enc|dyn|dec|rew|con|pol|val|slowval|retnorm|valnorm|advnorm)/")})
    result = []
    carry = agent.init_policy(1); current_episode = None
    for row in rows:
        episode = (row["source"], int(row["episode"]))
        if episode != current_episode:
            carry = agent.init_policy(1); current_episode = episode
        applied = np.asarray([[float(row[key]) for key in (
            "action_forward", "action_lateral", "action_vertical", "action_yaw_rate")]], np.float32)
        carry = (*carry[:3], {"action": applied})
        with jax.transfer_guard("allow"):
            carry, action, _ = agent.policy(carry, observation(row), mode="eval")
        result.append(np.asarray(action["action"][0], np.float32))
    del agent; gc.collect()
    return np.stack(result)


def route_actions(actions, rows):
    result = []
    for action, row in zip(actions, rows):
        start = np.asarray([float(row[f"scene_start_{x}"]) for x in "xy"])
        goal = np.asarray([float(row[f"scene_goal_{x}"]) for x in "xy"])
        forward = goal - start; forward /= max(np.linalg.norm(forward), 1e-6)
        lateral = np.asarray([-forward[1], forward[0]])
        result.append((float(action[:2] @ forward), float(action[:2] @ lateral)))
    return np.asarray(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--parent-config", type=Path, required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--child-config", type=Path, required=True)
    p.add_argument("--child-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for path in a.input:
        with path.open() as stream:
            loaded = list(csv.DictReader(stream))
        for row in loaded:
            row["source"] = path.parent.name
        rows.extend(loaded)
    rows = [r for r in rows if not bool(int(r["is_first"]))]
    parent = predictions(a.parent_config, a.parent_checkpoint, rows)
    child = predictions(a.child_config, a.child_checkpoint, rows)
    parent_route = route_actions(parent, rows); child_route = route_actions(child, rows)
    near = np.asarray([float(r["minimum_clearance"]) < 0.70 for r in rows])

    def stats(mask):
        delta = child[mask] - parent[mask]
        route_delta = child_route[mask] - parent_route[mask]
        return {
            "states": int(mask.sum()),
            "action_mae_all_dims": float(np.abs(delta).mean()),
            "action_rmse_all_dims": float(np.sqrt(np.square(delta).mean())),
            "forward_delta": float(route_delta[:, 0].mean()),
            "lateral_delta": float(route_delta[:, 1].mean()),
            "absolute_lateral_delta": float(
                np.abs(child_route[mask, 1]).mean() - np.abs(parent_route[mask, 1]).mean()),
        }
    report = {"definition": "same recorded histories; deterministic action drift child minus parent",
              "all": stats(np.ones(len(rows), bool)), "near_obstacle_clearance_lt_0p70": stats(near)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
