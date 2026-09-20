from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from .env import UAVNavigationEnv


def record(output: Path, seed: int = 0, max_steps: int = 300) -> dict:
    env = UAVNavigationEnv(render_mode="direct", max_steps=max_steps, seed=seed)
    observation, _ = env.reset(seed=seed)
    frames: list[Image.Image] = []
    total_reward = 0.0
    info = {"success": False, "distance": float("inf")}
    waypoint_index = 0
    previous = observation[:3].copy()
    try:
        for step in range(max_steps):
            waypoint = env.navigation_waypoints[waypoint_index]
            delta = waypoint - observation[:3]
            if np.linalg.norm(delta) < 0.35 and waypoint_index < len(env.navigation_waypoints) - 1:
                waypoint_index += 1
                delta = env.navigation_waypoints[waypoint_index] - observation[:3]
            velocity = np.clip(1.5 * delta, env.action_space.low[:3], env.action_space.high[:3])
            observation, reward, terminated, truncated, info = env.step(
                np.append(velocity, 0.0).astype(np.float32)
            )
            total_reward += reward
            assert env.client_id is not None
            p.addUserDebugLine(
                previous,
                observation[:3],
                [1.0, 0.55, 0.05],
                lineWidth=4,
                lifeTime=0,
                physicsClientId=env.client_id,
            )
            previous = observation[:3].copy()
            frame = Image.fromarray(env.camera_frame())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((12, 12, 365, 103), fill=(0, 0, 0, 175))
            draw.text((24, 22), f"UAV navigation | seed {seed}", fill="white", font=ImageFont.load_default())
            draw.text(
                (24, 43),
                f"step {step + 1:03d}  distance {info['distance']:.2f} m",
                fill="white",
                font=ImageFont.load_default(),
            )
            draw.text(
                (24, 85),
                f"clearance {info['minimum_clearance']:.2f} m  collision {info['collision']}",
                fill="white",
                font=ImageFont.load_default(),
            )
            draw.text(
                (24, 64),
                f"reward {total_reward:.1f}",
                fill="white",
                font=ImageFont.load_default(),
            )
            frames.append(frame)
            if terminated or truncated:
                break
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames.extend([frames[-1]] * 8)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=130, loop=0, optimize=True)
    return {"steps": step + 1, "reward": total_reward, **info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a PyBullet UAV navigation episode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/navigation_seed0.gif"))
    args = parser.parse_args()
    result = record(args.output, args.seed, args.max_steps)
    print(
        f"saved={args.output.resolve()} success={result['success']} "
        f"steps={result['steps']} distance={result['distance']:.3f}"
    )


if __name__ == "__main__":
    main()
