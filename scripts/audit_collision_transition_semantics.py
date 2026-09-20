#!/usr/bin/env python3
"""Collect one untouched Pure-Dreamer collision episode for timing audit."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent, make_env


def load_config(path):
    loader = yaml.YAML(typ="safe")
    configs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    saved = loader.load(Path(path).read_text())
    return elements.Config(configs["defaults"]).update(
        configs["quadrotor"]).update(saved)


def vec(value):
    return np.asarray(value, np.float32).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=51000)
    parser.add_argument("--max-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    base = load_config(args.config)
    agent = make_agent(base)
    elements.checkpoint.load(
        args.checkpoint, {"agent": partial(agent.load, regex="^(?!bcopt/).*")})
    selected = None
    attempted = []
    for episode in range(args.max_episodes):
        seed = args.seed_start + episode
        cfg = base.update({
            "env.quadrotor.seed": seed,
            "env.quadrotor.obstacle_lateral_offset": 0.3,
            "env.quadrotor.obstacle_randomization_level": 0,
            "env.quadrotor.dynamic_obstacles": False,
            "env.quadrotor.safety_projection": False,
        })
        env = make_env(cfg, 0)
        obs = env.step({"reset": True, "action": np.zeros(4, np.float32)})
        carry = agent.init_policy(1)
        transitions = []
        step = 0
        while not bool(obs["is_last"]):
            policy_obs = {k: np.asarray(v)[None] for k, v in obs.items()
                          if not k.startswith("log/")}
            carry, action, _ = agent.policy(carry, policy_obs, mode="eval")
            chosen = np.asarray(action["action"][0], np.float32)
            before = obs
            after = env.step({"reset": False, "action": chosen})
            transitions.append({
                "transition_step": step,
                "obs_t": vec(before["vector"]),
                "action_t": vec(chosen),
                "reward_t": float(after["reward"]),
                "next_obs": vec(after["vector"]),
                "is_first": bool(after["is_first"]),
                "is_last": bool(after["is_last"]),
                "is_terminal": bool(after["is_terminal"]),
                "timeout": bool(after.get("log/timeout", False)),
                "collision": bool(after.get("log/collision", False)),
                "clearance": float(after.get("log/minimum_clearance", np.nan)),
            })
            obs = after
            step += 1
        outcome = ("collision" if bool(obs.get("log/collision", False)) else
                   "timeout" if bool(obs.get("log/timeout", False)) else
                   "success" if bool(obs.get("log/success", False)) else "terminal")
        attempted.append({"seed": seed, "steps": step, "outcome": outcome})
        env.close()
        if outcome == "collision":
            selected = {"seed": seed, "outcome": outcome,
                        "steps": step, "transitions": transitions}
            break
    if selected is None:
        raise RuntimeError(f"no collision in {len(attempted)} episodes: {attempted}")

    payload = {
        "checkpoint": str(args.checkpoint),
        "offset_m": 0.3,
        "policy_mode": "eval_deterministic",
        "bc_scale": 0.0,
        "ppo": 0,
        "safety_projection": False,
        "holdout_seed": 35700,
        "holdout_included": False,
        "attempted": attempted,
        **selected,
    }
    (args.output / "collision_episode.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    fields = ["transition_step", "action_t", "reward_t", "is_first", "is_last",
              "is_terminal", "timeout", "collision", "clearance"]
    with (args.output / "last10.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in transitions[-10:]:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps({**{k: payload[k] for k in (
        "seed", "outcome", "steps", "attempted")},
        "last10": transitions[-10:]}, indent=2))


if __name__ == "__main__":
    main()
