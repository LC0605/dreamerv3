#!/usr/bin/env python3
"""Collect fixed-policy +0.8/+0.6/+0.3 episodes for Stage S2-WM."""

import argparse
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent, make_env


def config(path):
    loader = yaml.YAML(typ="safe")
    cfgs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    return elements.Config(cfgs["defaults"]).update(cfgs["quadrotor"]).update(
        loader.load(path.read_text()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=64000)
    p.add_argument("--max-episodes", type=int, default=250)
    p.add_argument("--min-collisions", type=int, default=40)
    p.add_argument("--min-successes", type=int, default=80)
    p.add_argument("--randomization-level", type=int, default=0)
    p.add_argument("--random-offset-low", type=float, default=-1.5)
    p.add_argument("--random-offset-high", type=float, default=1.5)
    a = p.parse_args()
    if a.seed == 35700: raise ValueError("permanent hold-out seed is forbidden")
    if a.random_offset_low > a.random_offset_high:
        raise ValueError("random offset lower bound exceeds upper bound")
    a.output.mkdir(parents=True, exist_ok=False); episodes = a.output / "episodes"
    episodes.mkdir()
    base = config(a.config); agent = make_agent(base)
    elements.checkpoint.load(a.checkpoint, {"agent": partial(
        agent.load, regex="^(?!(bcopt|opt)/).*")})
    offsets = (0.8, 0.6, 0.3)
    envs = {}
    for index, offset in enumerate(offsets):
        cfg = base.update({"env.quadrotor.seed": a.seed + index * 1000,
                           "env.quadrotor.obstacle_lateral_offset": offset,
                           "env.quadrotor.obstacle_randomization_level": a.randomization_level,
                           "env.quadrotor.random_offset_low": a.random_offset_low,
                           "env.quadrotor.random_offset_high": a.random_offset_high,
                           "env.quadrotor.dynamic_obstacles": False,
                           "env.quadrotor.safety_projection": False})
        envs[offset] = make_env(cfg, index)
    schedule = (0.8, 0.6, 0.3, 0.3, 0.3)
    counts = {"success": 0, "collision": 0, "out_of_bound": 0, "timeout": 0}
    by_offset = {str(x): {k: 0 for k in counts} for x in offsets}
    manifest = []
    try:
      for episode_id in range(a.max_episodes):
        offset = schedule[episode_id % len(schedule)]; env = envs[offset]
        obs = env.step({"reset": True, "action": np.zeros(4, np.float32)})
        carry = agent.init_policy(1); rows = []
        while True:
            policy_obs = {k: np.asarray(v)[None] for k, v in obs.items()
                          if not k.startswith("log/")}
            carry, action, _ = agent.policy(carry, policy_obs, mode="eval")
            chosen = np.asarray(action["action"][0], np.float32)
            if bool(obs["is_last"]): chosen = np.zeros(4, np.float32)
            success = bool(obs.get("log/success", False) and obs["is_terminal"])
            collision = bool(obs.get("log/collision", False) and obs["is_terminal"])
            oob = bool(obs.get("log/out_of_bounds", False) and obs["is_terminal"])
            outcome = 1 if success else 2 if collision else 3 if oob else 0
            terminal_component = 100.0 if outcome == 1 else -100.0 if outcome in (2, 3) else 0.0
            reward = float(obs["reward"])
            rows.append({
                "vector": np.asarray(obs["vector"], np.float32),
                "reward": np.asarray(reward, np.float32),
                "reward_dense": np.asarray(reward - terminal_component, np.float32),
                "terminal_outcome": np.asarray(outcome, np.int32),
                "is_first": np.asarray(obs["is_first"], bool),
                "is_last": np.asarray(obs["is_last"], bool),
                "is_terminal": np.asarray(obs["is_terminal"], bool),
                "teacher": np.asarray(False, bool),
                "action_policy_action": np.zeros(4, np.float32),
                "action": chosen, "episode_id": np.asarray(episode_id, np.int32),
                "episode_step": np.asarray(len(rows), np.int32),
                "curriculum_new": np.asarray(a.randomization_level > 0, bool),
                "offset": np.asarray(offset, np.float32),
                "obstacle_lateral_relative": np.asarray(
                    obs.get("log/obstacle_lateral_relative", np.nan), np.float32),
                "minimum_clearance": np.asarray(obs.get("log/minimum_clearance", np.nan), np.float32),
                "timeout": np.asarray(bool(obs.get("log/timeout", False)), bool)})
            if bool(obs["is_last"]): break
            obs = env.step({"reset": False, "action": chosen})
        final = int(rows[-1]["terminal_outcome"])
        lateral_trace = np.asarray(
            [r["obstacle_lateral_relative"] for r in rows], np.float32)
        realized = lateral_trace[np.isfinite(lateral_trace) & (np.abs(lateral_trace) > 1e-6)]
        initial_lateral = float(realized[0]) if len(realized) else float("nan")
        kind = ("success" if final == 1 else "collision" if final == 2 else
                "out_of_bound" if final == 3 else "timeout")
        counts[kind] += 1; by_offset[str(offset)][kind] += 1
        data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        residual = data["reward"] - (data["reward_dense"] + np.asarray(
            [0.0, 100.0, -100.0, -100.0], np.float32)[data["terminal_outcome"]])
        if np.max(np.abs(residual)) > 1e-5: raise RuntimeError("reward decomposition drift")
        path = episodes / f"episode_{episode_id:04d}_offset_{offset:.1f}_{kind}.npz"
        np.savez_compressed(path, **data)
        manifest.append({"episode_id": episode_id, "offset": offset,
                         "initial_obstacle_lateral_relative": initial_lateral,
                         "outcome": kind,
                         "rows": len(rows), "environment_steps": len(rows) - 1,
                         "file": str(path)})
        print(f"episode={episode_id + 1} offset={offset:+.1f} outcome={kind} "
              f"success={counts['success']} collision={counts['collision']}", flush=True)
        if counts["collision"] >= a.min_collisions and counts["success"] >= a.min_successes:
            break
    finally:
      for env in envs.values(): env.close()
    report = {"source_checkpoint": str(a.checkpoint), "policy_learning": False,
              "obstacle_randomization_level": a.randomization_level,
              "requested_random_offset_range": [a.random_offset_low, a.random_offset_high],
              "observed_initial_lateral_range": [
                  min(x["initial_obstacle_lateral_relative"] for x in manifest),
                  max(x["initial_obstacle_lateral_relative"] for x in manifest)],
              "seed_pools": {"collection_base": a.seed, "permanent_holdout": 35700},
              "seed_overlap": False, "requested_ratio": {"0.8": .2, "0.6": .2, "0.3": .6},
              "episodes": len(manifest), "counts": counts, "by_offset": by_offset,
              "targets": {"collision": a.min_collisions, "success": a.min_successes,
                          "max_episodes": a.max_episodes},
              "target_reached": counts["collision"] >= a.min_collisions and counts["success"] >= a.min_successes,
              "total_environment_steps": sum(x["environment_steps"] for x in manifest),
              "reward_decomposition_max_abs_residual": 0.0, "episode_manifest": manifest}
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("episodes", "counts", "by_offset",
                                             "target_reached", "total_environment_steps")}, indent=2))


if __name__ == "__main__": main()
