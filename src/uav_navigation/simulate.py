import argparse
import time

import numpy as np

from .env import UAVNavigationEnv


def run(
    seed: int = 0,
    render_mode: str = "direct",
    max_steps: int = 300,
    obstacle_count: int = 1,
    hold_seconds: float = 0.0,
) -> dict:
    env = UAVNavigationEnv(
        render_mode=render_mode, max_steps=max_steps, seed=seed, obstacle_count=obstacle_count
    )
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    info = {}
    waypoint_index = 0
    try:
        for step in range(max_steps):
            waypoint = env.navigation_waypoints[waypoint_index]
            delta = waypoint - observation[:3]
            if np.linalg.norm(delta) < 0.35 and waypoint_index < len(env.navigation_waypoints) - 1:
                waypoint_index += 1
                delta = env.navigation_waypoints[waypoint_index] - observation[:3]
            velocity = np.clip(1.5 * delta, env.action_space.low[:3], env.action_space.high[:3])
            action = np.append(velocity, 0.0).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        if render_mode == "human" and hold_seconds > 0:
            time.sleep(hold_seconds)
    finally:
        env.close()
    return {"steps": step + 1, "total_reward": total_reward, **info}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--no-obstacle", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()
    result = run(
        args.seed,
        "human" if args.gui else "direct",
        args.max_steps,
        obstacle_count=0 if args.no_obstacle else 1,
        hold_seconds=args.hold_seconds if args.gui else 0.0,
    )
    print(
        f"success={result['success']} steps={result['steps']} "
        f"distance={result['distance']:.3f} collision={result['collision']} "
        f"clearance={result['minimum_clearance']:.3f} reward={result['total_reward']:.2f}"
    )


if __name__ == "__main__":
    main()
