#!/usr/bin/env python3
"""Evaluate matching PPO/VecNormalize checkpoints on one deterministic scenario set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uav_navigation.train_ppo import evaluate, make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--obstacles", type=int, required=True)
    parser.add_argument("--obstacle-layout", choices=["corridor", "arena"], default="corridor")
    parser.add_argument("--temporal-history", type=int, default=1)
    parser.add_argument("--obstacle-offset", type=float, default=0.0)
    parser.add_argument("--randomization-level", type=int, default=0)
    parser.add_argument("--offset-low", type=float, default=-1.5)
    parser.add_argument("--offset-high", type=float, default=1.5)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--motion-amplitude", type=float, default=0.75)
    parser.add_argument("--speed-low", type=float, default=0.2)
    parser.add_argument("--speed-high", type=float, default=0.5)
    parser.add_argument("--lidar-delta", action="store_true")
    parser.add_argument("--obstacle-state", action="store_true")
    parser.add_argument("--arena-x", type=float, default=10.0)
    parser.add_argument("--arena-y", type=float, default=10.0)
    parser.add_argument("--arena-z", type=float, default=3.0)
    parser.add_argument("--random-start-goal", action="store_true")
    parser.add_argument("--goal-distance-low", type=float, default=2.0)
    parser.add_argument("--goal-distance-high", type=float, default=5.85)
    parser.add_argument("--boundary-margin", type=float, default=1.0)
    parser.add_argument("--horizontal-speed", type=float, default=2.0)
    parser.add_argument("--vertical-speed", type=float, default=1.0)
    parser.add_argument("--yaw-rate", type=float, default=1.0)
    parser.add_argument("--obstacle-half-xy-low", type=float, default=0.3)
    parser.add_argument("--obstacle-half-xy-high", type=float, default=0.55)
    parser.add_argument("--obstacle-height-low", type=float, default=1.2)
    parser.add_argument("--obstacle-height-high", type=float, default=3.0)
    parser.add_argument("--obstacle-min-spacing", type=float, default=0.0)
    parser.add_argument("--endpoint-clearance", type=float, default=0.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--boundary-guard", type=float, default=0.0)
    parser.add_argument("--takeoff", action="store_true")
    parser.add_argument("--takeoff-speed", type=float, default=0.35)
    parser.add_argument("--takeoff-settle-time", type=float, default=2.0)
    parser.add_argument("--horizontal-acceleration", type=float)
    parser.add_argument("--vertical-acceleration", type=float)
    parser.add_argument("--yaw-acceleration", type=float)
    args = parser.parse_args()

    results = []
    for model_path in sorted(args.checkpoint_dir.glob("model_*.zip"), key=lambda path: int(path.stem.split("_")[1])):
        steps = int(model_path.stem.split("_")[1])
        normalize_path = args.checkpoint_dir / f"vec_normalize_{steps}.pkl"
        if not normalize_path.exists():
            continue
        raw_env = DummyVecEnv(
            [lambda: Monitor(make_env(
                0, args.obstacles, args.obstacle_offset, False,
                args.randomization_level, (args.offset_low, args.offset_high),
                dynamic_obstacles=args.dynamic_obstacles,
                obstacle_motion_amplitude=args.motion_amplitude,
                obstacle_speed_range=(args.speed_low, args.speed_high),
                include_lidar_delta=args.lidar_delta,
                include_obstacle_state=args.obstacle_state,
                arena_size=(args.arena_x, args.arena_y, args.arena_z),
                random_start_goal=args.random_start_goal,
                goal_distance_range=(args.goal_distance_low, args.goal_distance_high),
                boundary_margin=args.boundary_margin,
                horizontal_speed_limit=args.horizontal_speed,
                vertical_speed_limit=args.vertical_speed,
                yaw_rate_limit=args.yaw_rate,
                obstacle_half_xy_range=(args.obstacle_half_xy_low, args.obstacle_half_xy_high),
                obstacle_height_range=(args.obstacle_height_low, args.obstacle_height_high),
                obstacle_min_spacing=args.obstacle_min_spacing,
                endpoint_clearance=args.endpoint_clearance,
                smoothness_weight=args.smoothness_weight,
                max_steps=args.max_steps,
                boundary_guard_distance=args.boundary_guard,
                obstacle_layout=args.obstacle_layout,
                temporal_history=args.temporal_history,
                takeoff_enabled=args.takeoff,
                takeoff_speed=args.takeoff_speed,
                takeoff_settle_time=args.takeoff_settle_time,
                command_acceleration_limits=(
                    args.horizontal_acceleration,
                    args.horizontal_acceleration,
                    args.vertical_acceleration,
                    args.yaw_acceleration,
                )
                if all(
                    value is not None
                    for value in (
                        args.horizontal_acceleration,
                        args.vertical_acceleration,
                        args.yaw_acceleration,
                    )
                )
                else None,
            )())]
        )
        env = VecNormalize.load(normalize_path, raw_env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(model_path, env=env, device="cpu")
        metrics = evaluate(model, env, args.episodes, max_steps=args.max_steps)
        metrics["timesteps"] = steps
        results.append(metrics)
        env.close()
        print(json.dumps(metrics), flush=True)

    output = args.checkpoint_dir / "evaluation_summary.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
