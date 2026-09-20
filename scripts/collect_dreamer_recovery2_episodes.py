#!/usr/bin/env python3
"""Collect the S3-A A1-Recovery-2 curriculum with a frozen Dreamer policy."""

import argparse
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent, make_env


SCENES = (
    ("low", 0.30, 0.45, 1, 45),
    ("mid", 0.45, 0.60, 1, 20),
    ("high", 0.60, 0.80, 1, 15),
    ("fixed0p3", 0.30, 0.30, 0, 10),
)


def load_config(path):
  loader = yaml.YAML(typ="safe")
  configs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
  saved = loader.load(Path(path).read_text())
  return elements.Config(configs["defaults"]).update(configs["quadrotor"]).update(saved)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=93000)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  (args.output / "episodes").mkdir(parents=True)

  base = load_config(args.config)
  agent = make_agent(base)
  elements.checkpoint.load(
      args.checkpoint, {"agent": partial(agent.load, regex="^(?!bcopt/).*")})
  envs = []
  for index, (_, low, high, randomization, _) in enumerate(SCENES):
    cfg = base.update({
        "env.quadrotor.seed": args.seed + 1000 * index,
        "env.quadrotor.obstacle_lateral_offset": low,
        "env.quadrotor.obstacle_randomization_level": randomization,
        "env.quadrotor.random_offset_low": low,
        "env.quadrotor.random_offset_high": high,
        "env.quadrotor.dynamic_obstacles": False,
        "env.quadrotor.safety_projection": False,
    })
    envs.append(make_env(cfg, 0))

  manifest = []
  episode_id = 0
  for scene_index, (name, low, high, randomization, count) in enumerate(SCENES):
    env = envs[scene_index]
    for _ in range(count):
      obs = env.step({"reset": True, "action": np.zeros(4, np.float32)})
      carry = agent.init_policy(1)
      rows = []
      while True:
        policy_obs = {key: np.asarray(value)[None] for key, value in obs.items()
                      if not key.startswith("log/")}
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
            "offset": np.asarray((low + high) / 2, np.float32),
        })
        if bool(obs["is_last"]):
          break
        obs = env.step({"reset": False, "action": chosen})
      arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
      filename = f"{name}_episode_{episode_id:04d}.npz"
      np.savez_compressed(args.output / "episodes" / filename, **arrays)
      manifest.append({"file": filename, "scene": name, "randomization": randomization,
                       "offset_low": low, "offset_high": high, "steps": len(rows) - 1})
      print(f"episode={episode_id} scene={name} steps={len(rows) - 1}", flush=True)
      episode_id += 1
  for env in envs:
    env.close()
  result = {"source_checkpoint": str(args.checkpoint), "seed_base": args.seed,
            "policy": "dreamer_eval", "ppo": 0, "bc_scale": 0.0,
            "safety_projection": False, "episodes": manifest}
  (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
  main()
