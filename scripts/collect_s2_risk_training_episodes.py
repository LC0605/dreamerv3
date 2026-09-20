#!/usr/bin/env python3
"""Collect disjoint complete +0.3 episodes until collision coverage is adequate."""

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=52000)
    parser.add_argument("--min-episodes", type=int, default=20)
    parser.add_argument("--min-collisions", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--pre-collision-steps", type=int, default=20)
    args = parser.parse_args()
    if args.seed == 35700:
        raise ValueError("seed 35700 is the immutable audit hold-out")
    if args.output.exists() and any(args.output.rglob("episode_*.npz")):
        raise FileExistsError(args.output)
    raw = args.output / "episodes"
    raw.mkdir(parents=True)

    base = load_config(args.config)
    agent = make_agent(base)
    elements.checkpoint.load(
        args.checkpoint, {"agent": partial(agent.load, regex="^(?!bcopt/).*")})
    cfg = base.update({
        "env.quadrotor.seed": args.seed,
        "env.quadrotor.obstacle_lateral_offset": 0.3,
        "env.quadrotor.obstacle_randomization_level": 0,
        "env.quadrotor.dynamic_obstacles": False,
        "env.quadrotor.safety_projection": False,
    })
    env = make_env(cfg, 0)
    counts = dict(success=0, collision=0, timeout=0, other_terminal=0)
    manifest_eps = []
    total_steps = 0
    for episode_id in range(args.max_episodes):
        obs = env.step({"reset": True, "action": np.zeros(4, np.float32)})
        carry = agent.init_policy(1)
        rows = []
        while True:
            policy_obs = {k: np.asarray(v)[None] for k, v in obs.items()
                          if not k.startswith("log/")}
            carry, action, _ = agent.policy(carry, policy_obs, mode="eval")
            chosen = np.asarray(action["action"][0], np.float32)
            if bool(obs["is_last"]):
                chosen = np.zeros(4, np.float32)
            rows.append({
                "vector": np.asarray(obs["vector"], np.float32),
                "reward": np.asarray(obs["reward"], np.float32),
                "is_first": np.asarray(obs["is_first"], bool),
                "is_last": np.asarray(obs["is_last"], bool),
                "is_terminal": np.asarray(obs["is_terminal"], bool),
                "teacher": np.asarray(False, bool),
                "action_policy_action": np.zeros(4, np.float32),
                "action": chosen,
                "episode_id": np.asarray(episode_id, np.int32),
                "episode_step": np.asarray(len(rows), np.int32),
                "collision_terminal": np.asarray(
                    bool(obs.get("log/collision", False) and obs["is_terminal"]), bool),
                "pre_collision": np.asarray(False, bool),
                "timeout": np.asarray(bool(obs.get("log/timeout", False)), bool),
                "minimum_clearance": np.asarray(
                    obs.get("log/minimum_clearance", np.nan), np.float32),
            })
            if bool(obs["is_last"]):
                break
            obs = env.step({"reset": False, "action": chosen})
        collision = bool(rows[-1]["collision_terminal"])
        timeout = bool(rows[-1]["timeout"])
        success = bool(obs.get("log/success", False))
        if collision:
            outcome = "collision"
            start = max(0, len(rows) - 1 - args.pre_collision_steps)
            for index in range(start, len(rows) - 1):
                rows[index]["pre_collision"] = np.asarray(True, bool)
        elif timeout:
            outcome = "timeout"
        elif success:
            outcome = "success"
        else:
            outcome = "other_terminal"
        counts[outcome] += 1
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        path = raw / f"episode_{episode_id:04d}_{outcome}.npz"
        np.savez_compressed(path, **arrays)
        env_steps = len(rows) - 1
        total_steps += env_steps
        manifest_eps.append({
            "episode_id": episode_id, "seed_pool": args.seed,
            "rng_episode_index": episode_id, "outcome": outcome,
            "rows": len(rows), "environment_steps": env_steps,
            "pre_collision_rows": int(arrays["pre_collision"].sum()),
            "collision_terminal_rows": int(arrays["collision_terminal"].sum()),
            "file": str(path),
        })
        print(f"episode={episode_id} outcome={outcome} steps={env_steps} "
              f"collisions={counts['collision']} total={episode_id + 1}", flush=True)
        enough = (
            episode_id + 1 >= args.min_episodes
            and counts["collision"] >= args.min_collisions
            and counts["success"] >= 1
            and counts["timeout"] >= 1)
        if enough:
            break
    env.close()
    if not enough:
        raise RuntimeError(f"coverage gate failed after {args.max_episodes}: {counts}")
    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "offset_m": 0.3,
        "seed_pool": {"risk_training": args.seed, "heldout_audit": 35700},
        "seed_overlap": False,
        "deterministic_policy": True,
        "learning": False,
        "bc_scale": 0.0,
        "ppo": 0,
        "safety_projection": False,
        "complete_episodes_only": True,
        "pre_collision_steps": args.pre_collision_steps,
        "total_environment_steps": total_steps,
        "counts": counts,
        "episodes": manifest_eps,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"total_environment_steps": total_steps, "counts": counts}))


if __name__ == "__main__":
    main()
