#!/usr/bin/env python3
"""Replay one real history through multiple Dreamer posteriors and compare actions."""

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


MODELS = {
    "BC-only": (
        "outputs/dreamerv3/stageS1fA3_temporal3_actor_mean_bc/config.yaml",
        "outputs/dreamerv3/stageS1fA3_temporal3_actor_mean_bc/ckpt/20260825T163314F148682",
    ),
    "Mixed": (
        "outputs/dreamerv3/stageS1fB2_temporal3_teacher_takeover_midbc/config.yaml",
        "outputs/dreamerv3/stageS1fB2_temporal3_teacher_takeover_midbc/ckpt/20260825T222659F216085",
    ),
    "Pure-Dreamer": (
        "outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/config.yaml",
        "outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/20260825T223426F313624",
    ),
}


def load_config(path):
    loader = yaml.YAML(typ="safe")
    configs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    saved = loader.load(Path(path).read_text())
    return elements.Config(configs["defaults"]).update(
        configs["quadrotor"]).update(saved)


def observation(row):
    return {
        "vector": np.asarray([[float(row[f"vector_{i:02d}"]) for i in range(68)]], np.float32),
        "reward": np.asarray([0.0], np.float32),
        "is_first": np.asarray([bool(int(row["is_first"]))]),
        "is_last": np.asarray([bool(int(row["is_last"]))]),
        "is_terminal": np.asarray([
            bool(float(row["collision"]) or float(row["success"]))]),
        "teacher": np.asarray([False]),
        "action_policy_action": np.zeros((1, 4), np.float32),
    }


def phase_labels(rows):
    outcomes = {}
    for row in rows:
        episode = int(row["episode"])
        if float(row["collision"]) > 0.5:
            outcomes[episode] = "collision"
        elif float(row["success"]) > 0.5:
            outcomes[episode] = "success"
        elif float(row["timeout"]) > 0.5:
            outcomes[episode] = "timeout"
    episode_rows = {}
    for index, row in enumerate(rows):
        episode_rows.setdefault(int(row["episode"]), []).append(index)
    pre_collision = set()
    for episode, indices in episode_rows.items():
        if outcomes.get(episode) == "collision":
            pre_collision.update(indices[-6:])
    labels = []
    for index, row in enumerate(rows):
        pos = np.asarray([float(row[f"position_{axis}"]) for axis in "xyz"])
        start = np.asarray([float(row[f"scene_start_{axis}"]) for axis in "xyz"])
        goal = np.asarray([float(row[f"scene_goal_{axis}"]) for axis in "xyz"])
        obstacle = np.asarray([float(row[f"scene_obstacle_{axis}"]) for axis in "xyz"])
        direction = goal[:2] - start[:2]
        direction /= max(np.linalg.norm(direction), 1e-6)
        passed = np.dot(pos[:2] - obstacle[:2], direction) > 0.35
        clearance = float(row["minimum_clearance"])
        if index in pre_collision:
            label = "pre_collision"
        elif passed and clearance > 0.35:
            label = "post_obstacle_recovery"
        elif clearance < 0.25:
            label = "most_dangerous"
        elif clearance < 0.70:
            label = "near_obstacle"
        else:
            label = "far_obstacle"
        labels.append(label)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as stream:
        rows = list(csv.DictReader(stream))
    # Exclude the partial episode started after the fixed completed-episode gate.
    rows = [row for row in rows if int(row["episode"]) <= 60]
    labels = phase_labels(rows)
    predictions = {name: [] for name in MODELS}
    for name, (config_path, checkpoint) in MODELS.items():
        agent = make_agent(load_config(config_path))
        elements.checkpoint.load(
            checkpoint, {"agent": partial(agent.load, regex="^(?!bcopt/).*")})
        carry = agent.init_policy(1)
        current_episode = None
        for row in rows:
            episode = int(row["episode"])
            if episode != current_episode:
                carry = agent.init_policy(1)
                current_episode = episode
            applied = np.asarray([[float(row[key]) for key in (
                "action_forward", "action_lateral", "action_vertical", "action_yaw_rate")]],
                np.float32)
            carry = (*carry[:3], {"action": applied})
            # This diagnostic deliberately replaces the policy-generated
            # previous action with the real action that produced the current
            # observation. Permit that explicit host-to-device replay copy.
            with jax.transfer_guard("allow"):
                carry, action, _ = agent.policy(carry, observation(row), mode="eval")
            predictions[name].append(np.asarray(action["action"][0], np.float32))
        del agent
        gc.collect()
    names = ("vx", "vy", "vz", "yaw_rate")
    output_rows = []
    for index, row in enumerate(rows):
        out = {"episode": int(row["episode"]), "episode_step": int(row["episode_step"]),
               "phase": labels[index], "clearance": float(row["minimum_clearance"])}
        for model in MODELS:
            action = predictions[model][index]
            safe = model.lower().replace("-", "_")
            for dim, value in zip(names, action):
                out[f"{safe}_{dim}"] = float(value)
            out[f"{safe}_abs_vy_over_abs_vx"] = float(
                abs(action[1]) / (abs(action[0]) + 1e-6))
        output_rows.append(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader(); writer.writerows(output_rows)
    summary = {}
    for phase in sorted(set(labels)):
        indices = [i for i, label in enumerate(labels) if label == phase]
        summary[phase] = {"states": len(indices)}
        for model in MODELS:
            actions = np.stack([predictions[model][i] for i in indices])
            summary[phase][model] = {
                "vx": float(actions[:, 0].mean()), "vy": float(actions[:, 1].mean()),
                "vz": float(actions[:, 2].mean()), "yaw_rate": float(actions[:, 3].mean()),
                "abs_vx": float(np.abs(actions[:, 0]).mean()),
                "abs_vy": float(np.abs(actions[:, 1]).mean()),
                "abs_vy_over_abs_vx": float(
                    (np.abs(actions[:, 1]) / (np.abs(actions[:, 0]) + 1e-6)).mean()),
            }
    summary["risky_shortcut_pattern"] = None
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
