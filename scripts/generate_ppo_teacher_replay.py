#!/usr/bin/env python3
"""Generate Dreamer-compatible three-frame replay from a LiDAR-only PPO teacher."""

from __future__ import annotations

import argparse
import json
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uav_navigation.train_ppo import make_env


BASE62 = string.digits + string.ascii_letters


def base62(value: bytes) -> str:
    number = int.from_bytes(value, "big")
    chars: list[str] = []
    while number:
        chars.append(BASE62[number % 62])
        number //= 62
    return "".join(reversed(chars)).rjust(22, "0")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SF%f")


def env_kwargs(history: int) -> dict:
    return dict(
        obstacle_count=1,
        obstacle_offset=0.8,
        randomize_obstacle=False,
        randomization_level=0,
        safety_distance=0.5,
        safety_weight=2.0,
        dynamic_obstacles=False,
        include_lidar_delta=False,
        include_obstacle_state=False,
        arena_size=(16.0, 16.0, 5.0),
        random_start_goal=True,
        goal_distance_range=(2.0, 4.0),
        boundary_margin=3.0,
        horizontal_speed_limit=0.5,
        vertical_speed_limit=0.3,
        yaw_rate_limit=0.5,
        obstacle_half_xy_range=(0.45, 0.45),
        obstacle_height_range=(2.5, 2.5),
        obstacle_min_spacing=1.0,
        endpoint_clearance=0.5,
        max_steps=300,
        boundary_guard_distance=0.0,
        command_acceleration_limits=(0.8, 0.8, 0.5, 0.8),
        obstacle_layout="corridor",
        temporal_history=history,
        safety_projection=False,
    )


def save_chunk(output: Path, rows: list[dict[str, np.ndarray]], predecessor: bytes) -> bytes:
    chunk_id = uuid.uuid4().bytes
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    stepids = [np.frombuffer(chunk_id + index.to_bytes(4, "big"), np.uint8) for index in range(len(rows))]
    arrays["stepid"] = np.stack(stepids)
    filename = f"{timestamp()}-{base62(chunk_id)}-{base62(predecessor)}-{len(rows)}.npz"
    np.savez_compressed(output / filename, **arrays)
    return chunk_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    normalization_raw = DummyVecEnv(
        [lambda: Monitor(make_env(0, **env_kwargs(1))())]
    )
    normalization = VecNormalize.load(args.model_dir / "vec_normalize.pkl", normalization_raw)
    normalization.training = False
    normalization.norm_reward = False
    teacher = PPO.load(args.model_dir / "ppo_static_obstacle.zip", device="cpu")
    environment = make_env(args.seed, **env_kwargs(3))()

    rows: list[dict[str, np.ndarray]] = []
    predecessor = bytes(16)
    successes = collisions = out_of_bounds = timeouts = transitions = 0

    def teacher_action(observation: np.ndarray) -> np.ndarray:
        current = np.asarray(observation[:36], np.float32)[None]
        normalized = normalization.normalize_obs(current)
        action, _ = teacher.predict(normalized, deterministic=True)
        return np.asarray(action[0], np.float32)

    def append(observation, reward, first, last, terminal, action) -> None:
        nonlocal predecessor, transitions
        rows.append(dict(
            vector=np.asarray(observation, np.float32),
            reward=np.asarray(reward, np.float32),
            is_first=np.asarray(first, bool),
            is_last=np.asarray(last, bool),
            is_terminal=np.asarray(terminal, bool),
            teacher=np.asarray(not last, bool),
            action=np.asarray(action, np.float32),
        ))
        transitions += 1
        if len(rows) == args.chunk_size:
            predecessor = save_chunk(args.output, rows, predecessor)
            rows.clear()

    for episode in range(args.episodes):
        reset = environment.reset(seed=args.seed + episode)
        observation = reset[0] if isinstance(reset, tuple) else reset
        action = teacher_action(observation)
        append(observation, 0.0, True, False, False, action)
        outcome = "timeout"
        for _ in range(300):
            result = environment.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
                terminal = bool(terminated)
            else:
                observation, reward, done, info = result
                terminal = bool(info.get("is_terminal", done))
            next_action = np.zeros(4, np.float32) if done else teacher_action(observation)
            append(observation, reward, False, done, terminal, next_action)
            action = next_action
            if done:
                if info.get("success"):
                    successes += 1; outcome = "success"
                elif info.get("collision"):
                    collisions += 1; outcome = "collision"
                elif info.get("out_of_bounds"):
                    out_of_bounds += 1; outcome = "out_of_bounds"
                else:
                    timeouts += 1
                break
        else:
            timeouts += 1
        if (episode + 1) % 25 == 0:
            print(f"episodes={episode + 1} transitions={transitions} last={outcome}", flush=True)

    if rows:
        save_chunk(args.output, rows, predecessor)
    environment.close()
    normalization.close()
    summary = dict(
        episodes=args.episodes,
        transitions=transitions,
        successes=successes,
        collisions=collisions,
        out_of_bounds=out_of_bounds,
        timeouts=timeouts,
        success_rate=successes / args.episodes,
        collision_rate=collisions / args.episodes,
        safety_projection=False,
        temporal_history=3,
        teacher_observation_dimensions=36,
        replay_observation_dimensions=68,
    )
    (args.output / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
