from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .train_ppo import make_env


def _temporal_dashboard(
    camera: Image.Image,
    base_env,
    features: np.ndarray,
    telemetry: dict[str, list[float]],
    info: dict,
    action: np.ndarray,
) -> Image.Image:
    """Compose simulation, temporal perception and policy telemetry panels."""
    width, height = 1120, 480
    canvas = Image.new("RGB", (width, height), (15, 20, 28))
    canvas.paste(camera.resize((640, 480)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((655, 10), "8-frame LiDAR history  (red = near)", fill="white", font=font)
    history = list(base_env.perception_history)
    if history:
        history = [history[0]] * (8 - len(history)) + history[-8:]
        for row, sample in enumerate(history[-8:]):
            for ray, value in enumerate(np.asarray(sample[:16], dtype=float)):
                value = float(np.clip(value, 0.0, 1.0))
                color = (int(255 * (1.0 - value)), int(190 * value), int(220 * value))
                x0, y0 = 655 + ray * 14, 32 + row * 15
                draw.rectangle((x0, y0, x0 + 12, y0 + 12), fill=color)

    draw.text((900, 10), "Current polar scan", fill="white", font=font)
    center = np.array([1005.0, 92.0])
    radius = 65.0
    draw.ellipse((*tuple(center - radius), *tuple(center + radius)), outline=(90, 105, 125))
    if history:
        lidar = np.asarray(history[-1][:16], dtype=float)
        points = []
        for index, value in enumerate(lidar):
            angle = 2.0 * np.pi * index / len(lidar) - 0.5 * np.pi
            points.append(tuple(center + radius * value * np.array([np.cos(angle), np.sin(angle)])))
        if len(points) > 2:
            draw.polygon(points, outline=(70, 230, 255), fill=(30, 95, 130))
    draw.ellipse((center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3), fill="white")

    draw.text((655, 168), "Fused latent feature  64 static | 32 LiDAR | 32 motion", fill="white", font=font)
    feature_values = np.tanh(np.asarray(features, dtype=float).reshape(-1))
    for index, value in enumerate(feature_values[:128]):
        row, col = divmod(index, 32)
        magnitude = int(220 * abs(value))
        color = (40 + magnitude, 55, 80) if value >= 0 else (40, 70, 80 + magnitude)
        x0, y0 = 655 + col * 14, 190 + row * 15
        draw.rectangle((x0, y0, x0 + 12, y0 + 12), fill=color)

    draw.text((655, 264), "Flight telemetry (last 80 decisions)", fill="white", font=font)
    chart = (655, 286, 1105, 420)
    draw.rectangle(chart, outline=(75, 90, 110))
    colors = {
        "distance": (80, 220, 255),
        "speed": (255, 205, 70),
        "clearance": (110, 235, 125),
        "shield": (255, 105, 180),
    }
    scales = {"distance": 12.0, "speed": 1.2, "clearance": 4.0, "shield": 0.6}
    for name, color in colors.items():
        values = telemetry[name][-80:]
        if len(values) > 1:
            points = []
            for index, value in enumerate(values):
                x = chart[0] + (chart[2] - chart[0]) * index / 79.0
                y = chart[3] - (chart[3] - chart[1]) * np.clip(value / scales[name], 0.0, 1.0)
                points.append((x, y))
            draw.line(points, fill=color, width=2)
    draw.text((665, 426), "distance", fill=colors["distance"], font=font)
    draw.text((735, 426), "speed", fill=colors["speed"], font=font)
    draw.text((785, 426), "clearance", fill=colors["clearance"], font=font)
    draw.text((855, 426), "shield", fill=colors["shield"], font=font)
    draw.text(
        (655, 448),
        f"d={info.get('distance', 0):.2f}m  v={info.get('speed', 0):.2f}m/s  "
        f"clear={info.get('minimum_clearance', 0):.2f}m  "
        f"shield={info.get('shield_correction', 0):.2f}m/s  |a|={np.linalg.norm(action):.2f}",
        fill="white",
        font=font,
    )
    return canvas


def record(
    model_dir: Path,
    output: Path,
    seed: int,
    obstacle_count: int,
    dynamic_obstacles: bool = False,
    motion_amplitude: float = 0.75,
    speed_range: tuple[float, float] = (0.2, 0.5),
    include_lidar_delta: bool = False,
    include_obstacle_state: bool = False,
    randomization_level: int = 0,
    offset_range: tuple[float, float] = (-1.5, 1.5),
    arena_size: tuple[float, float, float] = (10.0, 10.0, 3.0),
    random_start_goal: bool = False,
    goal_distance_range: tuple[float, float] = (2.0, 5.85),
    boundary_margin: float = 1.0,
    horizontal_speed_limit: float = 2.0,
    vertical_speed_limit: float = 1.0,
    yaw_rate_limit: float = 1.0,
    obstacle_half_xy_range: tuple[float, float] = (0.3, 0.55),
    obstacle_height_range: tuple[float, float] = (1.2, 3.0),
    obstacle_min_spacing: float = 0.0,
    endpoint_clearance: float = 0.0,
    smoothness_weight: float = 0.02,
    max_steps: int = 200,
    boundary_guard_distance: float = 0.0,
    takeoff_enabled: bool = False,
    takeoff_speed: float = 0.35,
    takeoff_settle_time: float = 2.0,
    command_acceleration_limits: tuple[float, float, float, float] | None = None,
    landing_pad_enabled: bool = False,
    landing_approach_height: float = 1.0,
    arrival_slow_radius: float = 1.5,
    arrival_speed_tolerance: float = 0.15,
    arrival_hold_time: float = 1.0,
    arrival_velocity_weight: float = 1.5,
    obstacle_layout: str = "corridor",
    temporal_history: int = 1,
    temporal_dashboard: bool = False,
    forced_conflict_count: int = 0,
    forced_conflict_probability: float = 1.0,
    forced_conflict_mode: str = "crossing",
    conflict_nominal_speed: float = 0.7,
    conflict_time_jitter: float = 1.0,
    obstacle_state_mode: str = "nearest_absolute",
    safety_projection: bool = False,
    safety_projection_horizon: float = 2.5,
    safety_projection_margin: float = 0.55,
    safety_projection_gain: float = 0.45,
    autoland_enabled: bool = False,
    landing_descent_speed: float = 0.12,
    landing_flare_height: float = 0.30,
    landing_flare_speed: float = 0.05,
    landing_xy_tolerance: float = 0.20,
    touchdown_speed_tolerance: float = 0.18,
    touchdown_tilt_tolerance_degrees: float = 15.0,
    disarm_time: float = 0.40,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    raw_env = DummyVecEnv([make_env(
        seed,
        obstacle_count,
        dynamic_obstacles=dynamic_obstacles,
        obstacle_motion_amplitude=motion_amplitude,
        obstacle_speed_range=speed_range,
        include_lidar_delta=include_lidar_delta,
        include_obstacle_state=include_obstacle_state,
        randomization_level=randomization_level,
        random_offset_range=offset_range,
        arena_size=arena_size,
        random_start_goal=random_start_goal,
        goal_distance_range=goal_distance_range,
        boundary_margin=boundary_margin,
        horizontal_speed_limit=horizontal_speed_limit,
        vertical_speed_limit=vertical_speed_limit,
        yaw_rate_limit=yaw_rate_limit,
        obstacle_half_xy_range=obstacle_half_xy_range,
        obstacle_height_range=obstacle_height_range,
        obstacle_min_spacing=obstacle_min_spacing,
        endpoint_clearance=endpoint_clearance,
        smoothness_weight=smoothness_weight,
        max_steps=max_steps,
        boundary_guard_distance=boundary_guard_distance,
        takeoff_enabled=takeoff_enabled,
        takeoff_speed=takeoff_speed,
        takeoff_settle_time=takeoff_settle_time,
        command_acceleration_limits=command_acceleration_limits,
        capture_takeoff_frames=takeoff_enabled,
        landing_pad_enabled=landing_pad_enabled,
        landing_approach_height=landing_approach_height,
        arrival_slow_radius=arrival_slow_radius,
        arrival_speed_tolerance=arrival_speed_tolerance,
        arrival_hold_time=arrival_hold_time,
        arrival_velocity_weight=arrival_velocity_weight,
        obstacle_layout=obstacle_layout,
        temporal_history=temporal_history,
        forced_conflict_count=forced_conflict_count,
        forced_conflict_probability=forced_conflict_probability,
        forced_conflict_mode=forced_conflict_mode,
        conflict_nominal_speed=conflict_nominal_speed,
        conflict_time_jitter=conflict_time_jitter,
        obstacle_state_mode=obstacle_state_mode,
        safety_projection=safety_projection,
        safety_projection_horizon=safety_projection_horizon,
        safety_projection_margin=safety_projection_margin,
        safety_projection_gain=safety_projection_gain,
        autoland_enabled=autoland_enabled,
        landing_descent_speed=landing_descent_speed,
        landing_flare_height=landing_flare_height,
        landing_flare_speed=landing_flare_speed,
        landing_xy_tolerance=landing_xy_tolerance,
        touchdown_speed_tolerance=touchdown_speed_tolerance,
        touchdown_tilt_tolerance_degrees=touchdown_tilt_tolerance_degrees,
        disarm_time=disarm_time,
    )])
    env = VecNormalize.load(model_dir / "vec_normalize.pkl", raw_env)
    env.training = False
    env.norm_reward = False
    model_name = "ppo_no_obstacle" if obstacle_count == 0 else "ppo_static_obstacle"
    model = PPO.load(model_dir / model_name, env=env, device="cpu")
    env.seed(seed)
    observation = env.reset()
    base_env = raw_env.envs[0].unwrapped
    frames: list[Image.Image] = []
    telemetry = {"distance": [], "speed": [], "clearance": [], "shield": []}
    for index, takeoff_frame in enumerate(base_env.takeoff_frames):
        frame = Image.fromarray(takeoff_frame)
        if temporal_dashboard:
            takeoff_canvas = Image.new("RGB", (1120, 480), (15, 20, 28))
            takeoff_canvas.paste(frame.resize((640, 480)), (0, 0))
            takeoff_draw = ImageDraw.Draw(takeoff_canvas)
            takeoff_draw.text(
                (675, 210),
                "Temporal perception starts after takeoff / hover check",
                fill=(170, 195, 220),
                font=ImageFont.load_default(),
            )
            frame = takeoff_canvas
        draw = ImageDraw.Draw(frame)
        draw.rectangle((12, 10, 310, 64), fill=(0, 0, 0, 170))
        draw.text((24, 20), "phase TAKEOFF / HOVER", fill="white", font=ImageFont.load_default())
        draw.text((24, 43), f"preflight {index * base_env.control_dt:04.1f} s", fill="white", font=ImageFont.load_default())
        frames.append(frame)
    info = {}
    try:
        for step in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, infos = env.step(action)
            info = infos[0]
            frame = Image.fromarray(base_env.camera_frame())
            telemetry["distance"].append(float(info["distance"]))
            telemetry["speed"].append(float(info["speed"]))
            telemetry["clearance"].append(float(info["minimum_clearance"]))
            telemetry["shield"].append(float(info.get("shield_correction", 0.0)))
            if temporal_dashboard:
                import torch

                with torch.no_grad():
                    observation_tensor, _ = model.policy.obs_to_tensor(observation)
                    feature_tensor = model.policy.extract_features(observation_tensor)
                frame = _temporal_dashboard(
                    frame,
                    base_env,
                    feature_tensor.detach().cpu().numpy()[0],
                    telemetry,
                    info,
                    np.asarray(action)[0],
                )
            draw = ImageDraw.Draw(frame)
            draw.rectangle((12, 12, 415, 126), fill=(0, 0, 0))
            font = ImageFont.load_default()
            draw.text(
                (24, 22), f"PPO rollout | {info.get('flight_phase', 'navigation')}",
                fill="white", font=font
            )
            draw.text((24, 43), f"seed {seed}  step {step + 1:03d}", fill="white", font=font)
            landing_phase = info.get("flight_phase") in {
                "landing_descent", "landing_realign", "touchdown_disarm", "landed_disarmed"
            }
            if landing_phase:
                distance_label = (
                    f"pad error {info.get('landing_horizontal_error', 0):.2f} m  "
                    f"altitude {info.get('altitude_above_pad', 0):.2f} m"
                )
            else:
                distance_label = f"approach distance {info['distance']:.2f} m"
            draw.text((24, 64), distance_label, fill="white", font=font)
            draw.text(
                (24, 85),
                f"success {info['success']}  collision {info['collision']}",
                fill="white",
                font=font,
            )
            draw.text(
                (24, 106),
                f"contact {info.get('pad_contact', False)}  "
                f"motor mean {np.mean(info.get('motor_rpm', [0])):.0f} rpm",
                fill="white",
                font=font,
            )
            frames.append(frame)
            if done[0]:
                break
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames.extend([frames[-1]] * 8)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=130, loop=0, optimize=True)
    return {"steps": step + 1, **info}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/ppo_no_obstacle"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_no_obstacle/rollout.gif"))
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--obstacles", type=int, default=0)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--motion-amplitude", type=float, default=0.75)
    parser.add_argument("--speed-low", type=float, default=0.2)
    parser.add_argument("--speed-high", type=float, default=0.5)
    parser.add_argument("--lidar-delta", action="store_true")
    parser.add_argument("--obstacle-state", action="store_true")
    parser.add_argument("--randomization-level", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--offset-low", type=float, default=-1.5)
    parser.add_argument("--offset-high", type=float, default=1.5)
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
    parser.add_argument("--landing-pad", action="store_true")
    parser.add_argument("--landing-approach-height", type=float, default=1.0)
    parser.add_argument("--arrival-slow-radius", type=float, default=1.5)
    parser.add_argument("--arrival-speed-tolerance", type=float, default=0.15)
    parser.add_argument("--arrival-hold-time", type=float, default=1.0)
    parser.add_argument("--arrival-velocity-weight", type=float, default=1.5)
    parser.add_argument("--obstacle-layout", choices=["corridor", "arena"], default="corridor")
    parser.add_argument("--temporal-history", type=int, default=1)
    parser.add_argument("--temporal-dashboard", action="store_true")
    parser.add_argument("--forced-conflicts", type=int, default=0)
    parser.add_argument("--forced-conflict-probability", type=float, default=1.0)
    parser.add_argument(
        "--forced-conflict-mode", choices=["crossing", "head_on", "vertical"], default="crossing"
    )
    parser.add_argument("--conflict-nominal-speed", type=float, default=0.7)
    parser.add_argument("--conflict-time-jitter", type=float, default=1.0)
    parser.add_argument(
        "--obstacle-state-mode",
        choices=["nearest_absolute", "nearest_relative", "threat_absolute", "threat_relative"],
        default="nearest_absolute",
    )
    parser.add_argument("--safety-projection", action="store_true")
    parser.add_argument("--safety-projection-horizon", type=float, default=2.5)
    parser.add_argument("--safety-projection-margin", type=float, default=0.55)
    parser.add_argument("--safety-projection-gain", type=float, default=0.45)
    parser.add_argument("--autoland", action="store_true")
    parser.add_argument("--landing-descent-speed", type=float, default=0.12)
    parser.add_argument("--landing-flare-height", type=float, default=0.30)
    parser.add_argument("--landing-flare-speed", type=float, default=0.05)
    parser.add_argument("--landing-xy-tolerance", type=float, default=0.20)
    parser.add_argument("--touchdown-speed-tolerance", type=float, default=0.18)
    parser.add_argument("--touchdown-tilt-tolerance", type=float, default=15.0)
    parser.add_argument("--disarm-time", type=float, default=0.40)
    args = parser.parse_args()
    result = record(
        args.model_dir,
        args.output,
        args.seed,
        args.obstacles,
        args.dynamic_obstacles,
        args.motion_amplitude,
        (args.speed_low, args.speed_high),
        args.lidar_delta,
        args.obstacle_state,
        args.randomization_level,
        (args.offset_low, args.offset_high),
        (args.arena_x, args.arena_y, args.arena_z),
        args.random_start_goal,
        (args.goal_distance_low, args.goal_distance_high),
        args.boundary_margin,
        args.horizontal_speed,
        args.vertical_speed,
        args.yaw_rate,
        (args.obstacle_half_xy_low, args.obstacle_half_xy_high),
        (args.obstacle_height_low, args.obstacle_height_high),
        args.obstacle_min_spacing,
        args.endpoint_clearance,
        args.smoothness_weight,
        args.max_steps,
        args.boundary_guard,
        args.takeoff,
        args.takeoff_speed,
        args.takeoff_settle_time,
        (
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
        args.landing_pad,
        args.landing_approach_height,
        args.arrival_slow_radius,
        args.arrival_speed_tolerance,
        args.arrival_hold_time,
        args.arrival_velocity_weight,
        args.obstacle_layout,
        args.temporal_history,
        args.temporal_dashboard,
        args.forced_conflicts,
        args.forced_conflict_probability,
        args.forced_conflict_mode,
        args.conflict_nominal_speed,
        args.conflict_time_jitter,
        args.obstacle_state_mode,
        args.safety_projection,
        args.safety_projection_horizon,
        args.safety_projection_margin,
        args.safety_projection_gain,
        args.autoland,
        args.landing_descent_speed,
        args.landing_flare_height,
        args.landing_flare_speed,
        args.landing_xy_tolerance,
        args.touchdown_speed_tolerance,
        args.touchdown_tilt_tolerance,
        args.disarm_time,
    )
    print(result)


if __name__ == "__main__":
    main()
