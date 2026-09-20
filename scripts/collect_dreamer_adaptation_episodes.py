#!/usr/bin/env python3
"""Collect complete Pure-Dreamer episodes under a strict interaction budget."""

import argparse
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent, make_env


OFFSETS = (0.8, 0.6, 0.3)
TARGETS = np.asarray((0.30, 0.30, 0.40), np.float64)


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
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--soft-budget", type=int, default=600,
                        help="Start no new episode after this; max_steps keeps total <= soft+300.")
    parser.add_argument("--hard-budget", type=int, default=900)
    args = parser.parse_args()
    if args.soft_budget + 300 > args.hard_budget:
        raise ValueError("soft budget plus max episode length exceeds hard budget")
    if args.output.exists() and any(args.output.rglob("episode_*.npz")):
        raise FileExistsError(args.output)
    (args.output / "train").mkdir(parents=True)
    (args.output / "validation").mkdir()

    base = load_config(args.config)
    agent = make_agent(base)
    elements.checkpoint.load(
        args.checkpoint,
        {"agent": partial(agent.load, regex="^(?!bcopt/).*")})
    envs = []
    for index, offset in enumerate(OFFSETS):
        cfg = base.update({
            "env.quadrotor.seed": args.seed + 1000 * index,
            "env.quadrotor.obstacle_lateral_offset": offset,
            "env.quadrotor.obstacle_randomization_level": 0,
            "env.quadrotor.dynamic_obstacles": False,
            "env.quadrotor.safety_projection": False,
        })
        envs.append(make_env(cfg, 0))

    steps_by_offset = np.zeros(len(OFFSETS), np.int64)
    episodes = []
    total_steps = 0
    episode_id = 0
    while total_steps < args.soft_budget:
        # Select the currently most underrepresented target share.
        normalized = steps_by_offset / TARGETS
        offset_index = int(np.argmin(normalized))
        offset = OFFSETS[offset_index]
        env = envs[offset_index]
        obs = env.step({"reset": True, "action": np.zeros(4, np.float32)})
        carry = agent.init_policy(1)
        rows = []
        episode_steps = 0
        while True:
            policy_obs = {
                key: np.asarray(value)[None]
                for key, value in obs.items() if not key.startswith("log/")}
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
            })
            if bool(obs["is_last"]):
                break
            obs = env.step({"reset": False, "action": chosen})
            episode_steps += 1
        if total_steps + episode_steps > args.hard_budget:
            raise RuntimeError("completed episode exceeded the hard interaction budget")
        total_steps += episode_steps
        steps_by_offset[offset_index] += episode_steps
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        episodes.append((episode_id, offset, episode_steps, arrays))
        print(f"episode={episode_id} offset={offset} steps={episode_steps} total={total_steps}")
        episode_id += 1

    # Reserve exactly one complete episode for local validation. The fixed
    # 60-episode audit set remains the external hold-out and is never copied.
    validation_id = episodes[-1][0]
    manifest_episodes = []
    for eid, offset, count, arrays in episodes:
        split = "validation" if eid == validation_id else "train"
        np.savez_compressed(args.output / split / f"episode_{eid:04d}.npz", **arrays)
        manifest_episodes.append({"episode_id": eid, "offset": offset,
                                  "environment_steps": count, "split": split})
    for env in envs:
        env.close()
    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "seed_base": args.seed,
        "audit_holdout_seed": 35700,
        "audit_holdout_included": False,
        "complete_episodes_only": True,
        "total_environment_steps": int(total_steps),
        "hard_budget": args.hard_budget,
        "steps_by_offset": {str(offset): int(value)
                            for offset, value in zip(OFFSETS, steps_by_offset)},
        "fractions_by_steps": {str(offset): float(value / total_steps)
                               for offset, value in zip(OFFSETS, steps_by_offset)},
        "episodes": manifest_episodes,
        "bc_scale": 0.0,
        "ppo": 0,
        "safety_projection": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
