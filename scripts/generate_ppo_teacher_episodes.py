#!/usr/bin/env python3
"""Generate episode-preserving PPO demonstrations for Dreamer actor distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from generate_ppo_teacher_replay import env_kwargs
from uav_navigation.train_ppo import make_env


ACTION_SCALE = np.asarray([0.5, 0.5, 0.3, 0.5], np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--validation-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--randomization-level", type=int, default=0)
    parser.add_argument("--obstacle-offset", type=float, default=0.8)
    parser.add_argument("--execution-noise-std", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 < args.validation_episodes < args.episodes:
        raise ValueError("validation episodes must be between zero and total episodes")
    if args.output.exists():
        (args.output / "train").mkdir(exist_ok=True)
        (args.output / "validation").mkdir(exist_ok=True)
    else:
        args.output.mkdir(parents=True)
        (args.output / "train").mkdir()
        (args.output / "validation").mkdir()

    teacher_env = env_kwargs(1)
    teacher_env.update(
        randomization_level=args.randomization_level,
        obstacle_offset=args.obstacle_offset,
        randomize_obstacle=False)
    raw = DummyVecEnv([lambda: Monitor(make_env(0, **teacher_env)())])
    normalization = VecNormalize.load(args.model_dir / "vec_normalize.pkl", raw)
    normalization.training = False
    normalization.norm_reward = False
    teacher = PPO.load(args.model_dir / "ppo_static_obstacle.zip", device="cpu")
    student_env = env_kwargs(3)
    student_env.update(
        randomization_level=args.randomization_level,
        obstacle_offset=args.obstacle_offset,
        randomize_obstacle=False)
    environment = make_env(args.seed, **student_env)()
    rng = np.random.default_rng(args.seed)

    def policy(observation: np.ndarray) -> np.ndarray:
        current = np.asarray(observation[:36], np.float32)[None]
        normalized = normalization.normalize_obs(current)
        action, _ = teacher.predict(normalized, deterministic=True)
        return np.asarray(action[0], np.float32)

    outcomes = dict(success=0, collision=0, out_of_bounds=0, timeout=0)
    lengths: list[int] = []
    existing = sorted(args.output.glob("*" + "/episode_*.npz"))
    for path in existing:
        with np.load(path) as data:
            lengths.append(len(data["reward"]))
            terminal_reward = float(data["reward"][-1])
            if not bool(data["is_terminal"][-1]):
                outcomes["timeout"] += 1
            elif terminal_reward > 50:
                outcomes["success"] += 1
            else:
                outcomes["collision"] += 1
    start_episode = len(existing)
    if start_episode:
        print(f"resuming at episode={start_episode}", flush=True)
    for episode_id in range(start_episode, args.episodes):
        observation, _ = environment.reset(seed=args.seed + episode_id)
        teacher_action = policy(observation)
        executed_action = np.clip(
            teacher_action + rng.normal(0, args.execution_noise_std, 4),
            -1, 1).astype(np.float32)
        rows = []

        def append(reward, first, last, terminal, action, target_action) -> None:
            step = len(rows)
            rows.append(dict(
                vector=np.asarray(observation, np.float32),
                reward=np.asarray(reward, np.float32),
                is_first=np.asarray(first, bool),
                is_last=np.asarray(last, bool),
                is_terminal=np.asarray(terminal, bool),
                teacher=np.asarray(not last, bool),
                action=np.asarray(action, np.float32),
                action_policy=np.asarray(target_action, np.float32),
                action_policy_action=np.asarray(target_action, np.float32),
                action_physical=np.asarray(action, np.float32) * ACTION_SCALE,
                episode_id=np.asarray(episode_id, np.int32),
                episode_step=np.asarray(step, np.int32),
            ))

        append(0.0, True, False, False, executed_action, teacher_action)
        outcome = "timeout"
        for _ in range(300):
            observation, reward, terminated, truncated, info = environment.step(executed_action)
            done = bool(terminated or truncated)
            next_teacher = np.zeros(4, np.float32) if done else policy(observation)
            next_executed = np.zeros(4, np.float32) if done else np.clip(
                next_teacher + rng.normal(0, args.execution_noise_std, 4),
                -1, 1).astype(np.float32)
            append(reward, False, done, bool(terminated), next_executed, next_teacher)
            teacher_action, executed_action = next_teacher, next_executed
            if done:
                if info.get("success"):
                    outcome = "success"
                elif info.get("collision"):
                    outcome = "collision"
                elif info.get("out_of_bounds"):
                    outcome = "out_of_bounds"
                break
        outcomes[outcome] += 1
        lengths.append(len(rows))
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        split = "validation" if episode_id >= args.episodes - args.validation_episodes else "train"
        np.savez_compressed(
            args.output / split / f"episode_{episode_id:04d}.npz", **arrays)
        if (episode_id + 1) % 25 == 0:
            print(f"episodes={episode_id + 1} last={outcome}", flush=True)

    environment.close()
    normalization.close()
    summary = dict(
        episodes=args.episodes,
        train_episodes=args.episodes - args.validation_episodes,
        validation_episodes=args.validation_episodes,
        transitions=int(sum(lengths)),
        mean_episode_rows=float(np.mean(lengths)),
        **outcomes,
        observation_dimensions=68,
        ppo_observation_dimensions=36,
        temporal_history=3,
        temporal_layout="state_and_L_t_then_L_t_minus_2_then_L_t_minus_1",
        action_policy_bounds=[-1.0, 1.0],
        action_physical_scale=ACTION_SCALE.tolist(),
        action_semantics=["vx_world", "vy_world", "vz_world", "yaw_rate"],
        safety_projection=False,
        obstacle_randomization_level=args.randomization_level,
        obstacle_offset=args.obstacle_offset,
        execution_noise_std=args.execution_noise_std,
        boundary_guard_distance=0.0,
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
