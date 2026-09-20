from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import gymnasium as gym

from .env import UAVNavigationEnv


def obstacle_observation_indices(temporal_history: int) -> np.ndarray:
    """Indices occupied by the three obstacle-state slots in all frames."""
    indices = list(range(36, 54))
    for frame in range(temporal_history - 1):
        start = 54 + frame * 34 + 16
        indices.extend(range(start, start + 18))
    return np.asarray(indices, dtype=np.int64)


def reset_new_obstacle_statistics(vec_env, temporal_history: int) -> None:
    """Give newly activated obstacle channels a neutral normalization prior."""
    indices = obstacle_observation_indices(temporal_history)
    vec_env.obs_rms.mean[indices] = 0.0
    vec_env.obs_rms.var[indices] = 1.0


def configure_trainable_scope(model, scope: str) -> None:
    """Optionally adapt only the temporal obstacle encoder after a semantic switch."""
    if scope == "all":
        for parameter in model.policy.parameters():
            parameter.requires_grad = True
        return
    if scope != "obstacle_encoder":
        raise ValueError("trainable scope must be 'all' or 'obstacle_encoder'")
    for parameter in model.policy.parameters():
        parameter.requires_grad = False
    feature_extractor = model.policy.features_extractor
    if not hasattr(feature_extractor, "obstacle_gru"):
        raise ValueError("obstacle_encoder scope requires the temporal feature extractor")
    for parameter in feature_extractor.obstacle_gru.parameters():
        parameter.requires_grad = True


def make_env(
    seed: int,
    obstacle_count: int = 1,
    obstacle_offset: float = 0.0,
    randomize_obstacle: bool = False,
    randomization_level: int = 0,
    random_offset_range: tuple[float, float] = (-1.5, 1.5),
    safety_distance: float = 0.5,
    safety_weight: float = 2.0,
    failure_penalty: float = 100.0,
    dynamic_obstacles: bool = False,
    obstacle_motion_amplitude: float = 0.75,
    obstacle_speed_range: tuple[float, float] = (0.2, 0.5),
    include_lidar_delta: bool = False,
    include_obstacle_state: bool = False,
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
    capture_takeoff_frames: bool = False,
    landing_pad_enabled: bool = False,
    landing_approach_height: float = 1.0,
    arrival_slow_radius: float = 1.5,
    arrival_speed_tolerance: float = 0.15,
    arrival_hold_time: float = 1.0,
    arrival_velocity_weight: float = 1.5,
    obstacle_layout: str = "corridor",
    temporal_history: int = 1,
    forced_conflict_count: int = 0,
    forced_conflict_probability: float = 1.0,
    conflict_nominal_speed: float = 0.7,
    conflict_time_jitter: float = 1.0,
    obstacle_state_mode: str = "nearest_absolute",
    forced_conflict_mode: str = "crossing",
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
):
    def factory():
        return gym.wrappers.RescaleAction(
            UAVNavigationEnv(
                render_mode="direct",
                max_steps=max_steps,
                seed=seed,
                obstacle_count=obstacle_count,
                obstacle_lateral_offset=obstacle_offset,
                randomize_obstacle=randomize_obstacle,
                obstacle_randomization_level=randomization_level,
                random_offset_range=random_offset_range,
                safety_distance=safety_distance,
                safety_weight=safety_weight,
                failure_penalty=failure_penalty,
                dynamic_obstacles=dynamic_obstacles,
                obstacle_motion_amplitude=obstacle_motion_amplitude,
                obstacle_speed_range=obstacle_speed_range,
                include_lidar_delta=include_lidar_delta,
                include_obstacle_state=include_obstacle_state,
                arena_size=arena_size,
                random_start_goal=random_start_goal,
                goal_distance_range=goal_distance_range,
                boundary_margin=boundary_margin,
                start_height_range=(1.0, 2.0) if random_start_goal else (1.0, 1.0),
                goal_height_range=(1.0, 3.0) if random_start_goal else (0.5, 2.5),
                horizontal_speed_limit=horizontal_speed_limit,
                vertical_speed_limit=vertical_speed_limit,
                yaw_rate_limit=yaw_rate_limit,
                obstacle_half_xy_range=obstacle_half_xy_range,
                obstacle_height_range=obstacle_height_range,
                obstacle_min_spacing=obstacle_min_spacing,
                endpoint_clearance=endpoint_clearance,
                smoothness_weight=smoothness_weight,
                boundary_guard_distance=boundary_guard_distance,
                takeoff_enabled=takeoff_enabled,
                takeoff_speed=takeoff_speed,
                takeoff_settle_time=takeoff_settle_time,
                command_acceleration_limits=command_acceleration_limits,
                capture_takeoff_frames=capture_takeoff_frames,
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
                conflict_nominal_speed=conflict_nominal_speed,
                conflict_time_jitter=conflict_time_jitter,
                obstacle_state_mode=obstacle_state_mode,
                forced_conflict_mode=forced_conflict_mode,
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
            ),
            min_action=-1.0,
            max_action=1.0,
        )

    return factory


def evaluate(
    model, vec_env, episodes: int, seed_offset: int = 10_000, max_steps: int = 200
) -> dict:
    successes = 0
    collisions = 0
    out_of_bounds = 0
    lengths: list[int] = []
    clearances: list[float] = []
    final_distances: list[float] = []
    speeds: list[float] = []
    tilts: list[float] = []
    action_changes: list[float] = []
    shield_corrections: list[float] = []
    shield_active_steps = 0
    total_steps = 0
    for episode in range(episodes):
        vec_env.seed(seed_offset + episode)
        observation = vec_env.reset()
        minimum_clearance = float("inf")
        previous_applied_action = None
        for step in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, infos = vec_env.step(action)
            info = infos[0]
            total_steps += 1
            minimum_clearance = min(minimum_clearance, float(info["minimum_clearance"]))
            speeds.append(float(info["speed"]))
            tilts.append(float(info["tilt"]))
            applied_action = np.asarray(info["applied_action"], dtype=np.float32)
            if previous_applied_action is not None:
                action_changes.append(float(np.linalg.norm(applied_action - previous_applied_action)))
            previous_applied_action = applied_action
            shield_correction = float(info.get("shield_correction", 0.0))
            shield_corrections.append(shield_correction)
            shield_active_steps += int(info.get("shield_constraints", 0) > 0)
            if done[0]:
                successes += int(info["success"])
                collisions += int(info["collision"])
                out_of_bounds += int(info["out_of_bounds"])
                lengths.append(step + 1)
                clearances.append(minimum_clearance)
                final_distances.append(float(info["distance"]))
                break
    timeouts = episodes - successes - collisions - out_of_bounds
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "out_of_bounds_rate": out_of_bounds / episodes,
        "timeout_rate": timeouts / episodes,
        "mean_length": float(np.mean(lengths)) if lengths else float(max_steps),
        "mean_minimum_clearance": float(np.mean(clearances)) if clearances else 0.0,
        "p10_minimum_clearance": float(np.percentile(clearances, 10)) if clearances else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "p95_speed": float(np.percentile(speeds, 95)) if speeds else 0.0,
        "mean_tilt_degrees": float(np.degrees(np.mean(tilts))) if tilts else 0.0,
        "p95_tilt_degrees": (
            float(np.degrees(np.percentile(tilts, 95))) if tilts else 0.0
        ),
        "mean_action_change": float(np.mean(action_changes)) if action_changes else 0.0,
        "shield_activation_rate": shield_active_steps / max(total_steps, 1),
        "mean_shield_correction": (
            float(np.mean(shield_corrections)) if shield_corrections else 0.0
        ),
        "p95_shield_correction": (
            float(np.percentile(shield_corrections, 95)) if shield_corrections else 0.0
        ),
    }


def train(
    timesteps: int,
    output: Path,
    seed: int,
    evaluation_episodes: int,
    resume: Path | None = None,
    obstacle_count: int = 1,
    obstacle_offset: float = 0.0,
    resume_model: str | None = None,
    randomize_obstacle: bool = False,
    randomization_level: int = 0,
    random_offset_range: tuple[float, float] = (-1.5, 1.5),
    safety_distance: float = 0.5,
    safety_weight: float = 2.0,
    failure_penalty: float = 100.0,
    dynamic_obstacles: bool = False,
    obstacle_motion_amplitude: float = 0.75,
    obstacle_speed_range: tuple[float, float] = (0.2, 0.5),
    include_lidar_delta: bool = False,
    include_obstacle_state: bool = False,
    transfer_observation: bool = False,
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
    learning_rate: float | None = None,
    temporal_conv: bool = False,
    forced_conflict_count: int = 0,
    forced_conflict_probability: float = 1.0,
    conflict_nominal_speed: float = 0.7,
    conflict_time_jitter: float = 1.0,
    obstacle_state_mode: str = "nearest_absolute",
    trainable_scope: str = "all",
    forced_conflict_mode: str = "crossing",
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
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as error:
        raise SystemExit("Stable-Baselines3 is required; run this command in the drone_rl environment.") from error

    output.mkdir(parents=True, exist_ok=True)
    if temporal_conv and temporal_history < 2:
        raise ValueError("temporal convolution requires --temporal-history of at least 2")
    if temporal_conv and transfer_observation:
        raise ValueError("temporal convolution uses a new architecture and cannot transfer MLP weights")
    model_name = "ppo_no_obstacle" if obstacle_count == 0 else "ppo_static_obstacle"
    raw_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    seed,
                    obstacle_count,
                    obstacle_offset,
                    randomize_obstacle,
                    randomization_level,
                    random_offset_range,
                    safety_distance,
                    safety_weight,
                    failure_penalty,
                    dynamic_obstacles,
                    obstacle_motion_amplitude,
                    obstacle_speed_range,
                    include_lidar_delta,
                    include_obstacle_state,
                    arena_size,
                    random_start_goal,
                    goal_distance_range,
                    boundary_margin,
                    horizontal_speed_limit,
                    vertical_speed_limit,
                    yaw_rate_limit,
                    obstacle_half_xy_range,
                    obstacle_height_range,
                    obstacle_min_spacing,
                    endpoint_clearance,
                    smoothness_weight,
                    max_steps,
                    boundary_guard_distance,
                    takeoff_enabled,
                    takeoff_speed,
                    takeoff_settle_time,
                    command_acceleration_limits,
                    False,
                    landing_pad_enabled,
                    landing_approach_height,
                    arrival_slow_radius,
                    arrival_speed_tolerance,
                    arrival_hold_time,
                    arrival_velocity_weight,
                    obstacle_layout,
                    temporal_history,
                    forced_conflict_count,
                    forced_conflict_probability,
                    conflict_nominal_speed,
                    conflict_time_jitter,
                    obstacle_state_mode,
                    forced_conflict_mode,
                    safety_projection,
                    safety_projection_horizon,
                    safety_projection_margin,
                    safety_projection_gain,
                    autoland_enabled,
                    landing_descent_speed,
                    landing_flare_height,
                    landing_flare_speed,
                    landing_xy_tolerance,
                    touchdown_speed_tolerance,
                    touchdown_tilt_tolerance_degrees,
                    disarm_time,
                )()
            )
        ]
    )
    if resume and transfer_observation:
        import pickle
        import torch

        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        old_model = PPO.load(resume / (resume_model or model_name), device="cpu")
        model = PPO(
            "MlpPolicy", env, learning_rate=learning_rate or 1e-4, n_steps=1024, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.001,
            policy_kwargs={"net_arch": [128, 128]}, seed=seed, verbose=1, device="cpu",
        )
        old_state = old_model.policy.state_dict()
        new_state = model.policy.state_dict()
        for key, old_value in old_state.items():
            if key not in new_state:
                continue
            if new_state[key].shape == old_value.shape:
                new_state[key] = old_value
            elif key in {
                "mlp_extractor.policy_net.0.weight",
                "mlp_extractor.value_net.0.weight",
            } and new_state[key].shape[1] > old_value.shape[1]:
                expanded = torch.zeros_like(new_state[key])
                expanded[:, : old_value.shape[1]] = old_value
                new_state[key] = expanded
        model.policy.load_state_dict(new_state)
        with (resume / "vec_normalize.pkl").open("rb") as handle:
            old_normalizer = pickle.load(handle)
        old_size = old_normalizer.obs_rms.mean.shape[0]
        env.obs_rms.mean[:old_size] = old_normalizer.obs_rms.mean
        env.obs_rms.var[:old_size] = old_normalizer.obs_rms.var
        env.obs_rms.count = old_normalizer.obs_rms.count
        reset_num_timesteps = True
    elif resume:
        env = VecNormalize.load(resume / "vec_normalize.pkl", raw_env)
        previous_metrics_path = resume / "metrics.json"
        if include_obstacle_state and obstacle_count > 0 and previous_metrics_path.exists():
            previous_metrics = json.loads(previous_metrics_path.read_text(encoding="utf-8"))
            previous_state_mode = previous_metrics.get(
                "obstacle_state_mode", "nearest_absolute"
            )
            if (
                int(previous_metrics.get("obstacle_count", obstacle_count)) == 0
                or previous_state_mode != obstacle_state_mode
            ):
                reset_new_obstacle_statistics(env, temporal_history)
        env.training = True
        env.norm_reward = True
        model = PPO.load(resume / (resume_model or model_name), env=env, device="cpu")
        model.verbose = 1
        model.ent_coef = 0.001
        resume_learning_rate = learning_rate or 1e-4
        model.learning_rate = resume_learning_rate
        model.lr_schedule = lambda _: resume_learning_rate
        reset_num_timesteps = False
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        policy_kwargs = {"net_arch": [128, 128]}
        if temporal_conv:
            from .temporal_features import TemporalObstacleFeatureExtractor

            policy_kwargs.update(
                features_extractor_class=TemporalObstacleFeatureExtractor,
                features_extractor_kwargs={"history_length": temporal_history},
            )
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate or 3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=1,
            device="cpu",
        )
        reset_num_timesteps = True
    from stable_baselines3.common.callbacks import BaseCallback

    configure_trainable_scope(model, trainable_scope)

    class TrainingMonitorCallback(BaseCallback):
        def __init__(self, save_frequency: int = 10_000):
            super().__init__()
            self.save_frequency = save_frequency
            self.next_save = 0
            self.successes = deque(maxlen=100)
            self.collisions = deque(maxlen=100)
            self.boundaries = deque(maxlen=100)

        def _on_training_start(self) -> None:
            self.next_save = ((self.model.num_timesteps // self.save_frequency) + 1) * self.save_frequency

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if done:
                    self.successes.append(float(info.get("success", False)))
                    self.collisions.append(float(info.get("collision", False)))
                    self.boundaries.append(float(info.get("out_of_bounds", False)))
            if self.successes:
                self.logger.record("navigation/success_rate_100", np.mean(self.successes))
                self.logger.record("navigation/collision_rate_100", np.mean(self.collisions))
                self.logger.record("navigation/out_of_bounds_rate_100", np.mean(self.boundaries))
            if self.model.num_timesteps >= self.next_save:
                checkpoint = output / "checkpoints"
                checkpoint.mkdir(parents=True, exist_ok=True)
                self.model.save(checkpoint / f"model_{self.model.num_timesteps}")
                self.model.get_env().save(checkpoint / f"vec_normalize_{self.model.num_timesteps}.pkl")
                self.next_save += self.save_frequency
            return True

    model.tensorboard_log = str(output / "tensorboard")
    model.learn(
        total_timesteps=timesteps,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
        callback=TrainingMonitorCallback(),
        tb_log_name="PPO",
    )
    model.save(output / model_name)
    env.save(output / "vec_normalize.pkl")
    env.training = False
    env.norm_reward = False
    metrics = evaluate(model, env, evaluation_episodes, max_steps=max_steps)
    metrics.update(
        {
            "timesteps": model.num_timesteps,
            "seed": seed,
            "obstacle_count": obstacle_count,
            "obstacle_lateral_offset": obstacle_offset,
            "randomize_obstacle": randomize_obstacle,
            "obstacle_randomization_level": randomization_level,
            "random_offset_range": list(random_offset_range),
            "safety_distance": safety_distance,
            "safety_weight": safety_weight,
            "failure_penalty": failure_penalty,
            "dynamic_obstacles": dynamic_obstacles,
            "obstacle_motion_amplitude": obstacle_motion_amplitude,
            "obstacle_speed_range": list(obstacle_speed_range),
            "include_lidar_delta": include_lidar_delta,
            "include_obstacle_state": include_obstacle_state,
            "transfer_observation": transfer_observation,
            "arena_size": list(arena_size),
            "random_start_goal": random_start_goal,
            "goal_distance_range": list(goal_distance_range),
            "boundary_margin": boundary_margin,
            "horizontal_speed_limit": horizontal_speed_limit,
            "vertical_speed_limit": vertical_speed_limit,
            "yaw_rate_limit": yaw_rate_limit,
            "obstacle_half_xy_range": list(obstacle_half_xy_range),
            "obstacle_height_range": list(obstacle_height_range),
            "obstacle_min_spacing": obstacle_min_spacing,
            "endpoint_clearance": endpoint_clearance,
            "smoothness_weight": smoothness_weight,
            "max_steps": max_steps,
            "boundary_guard_distance": boundary_guard_distance,
            "takeoff_enabled": takeoff_enabled,
            "takeoff_speed": takeoff_speed,
            "takeoff_settle_time": takeoff_settle_time,
            "command_acceleration_limits": (
                list(command_acceleration_limits)
                if command_acceleration_limits is not None else None
            ),
            "landing_pad_enabled": landing_pad_enabled,
            "landing_approach_height": landing_approach_height,
            "arrival_slow_radius": arrival_slow_radius,
            "arrival_speed_tolerance": arrival_speed_tolerance,
            "arrival_hold_time": arrival_hold_time,
            "arrival_velocity_weight": arrival_velocity_weight,
            "obstacle_layout": obstacle_layout,
            "temporal_history": temporal_history,
            "learning_rate": learning_rate,
            "temporal_conv": temporal_conv,
            "forced_conflict_count": forced_conflict_count,
            "forced_conflict_probability": forced_conflict_probability,
            "conflict_nominal_speed": conflict_nominal_speed,
            "conflict_time_jitter": conflict_time_jitter,
            "obstacle_state_mode": obstacle_state_mode,
            "trainable_scope": trainable_scope,
            "forced_conflict_mode": forced_conflict_mode,
            "safety_projection": safety_projection,
            "safety_projection_horizon": safety_projection_horizon,
            "safety_projection_margin": safety_projection_margin,
            "safety_projection_gain": safety_projection_gain,
            "autoland_enabled": autoland_enabled,
            "landing_descent_speed": landing_descent_speed,
            "landing_flare_height": landing_flare_height,
            "landing_flare_speed": landing_flare_speed,
            "landing_xy_tolerance": landing_xy_tolerance,
            "touchdown_speed_tolerance": touchdown_speed_tolerance,
            "touchdown_tilt_tolerance_degrees": touchdown_tilt_tolerance_degrees,
            "disarm_time": disarm_time,
        }
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    env.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO baseline on static-obstacle navigation.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_static_obstacle"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--obstacle-offset", type=float, default=0.0)
    parser.add_argument("--resume-model")
    parser.add_argument("--random-obstacle", action="store_true")
    parser.add_argument("--randomization-level", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--offset-low", type=float, default=-1.5)
    parser.add_argument("--offset-high", type=float, default=1.5)
    parser.add_argument("--safety-distance", type=float, default=0.5)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    parser.add_argument("--failure-penalty", type=float, default=100.0)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--motion-amplitude", type=float, default=0.75)
    parser.add_argument("--speed-low", type=float, default=0.2)
    parser.add_argument("--speed-high", type=float, default=0.5)
    parser.add_argument("--lidar-delta", action="store_true")
    parser.add_argument("--obstacle-state", action="store_true")
    parser.add_argument("--transfer-observation", action="store_true")
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
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temporal-conv", action="store_true")
    parser.add_argument("--forced-conflicts", type=int, default=0)
    parser.add_argument("--forced-conflict-probability", type=float, default=1.0)
    parser.add_argument("--conflict-nominal-speed", type=float, default=0.7)
    parser.add_argument("--conflict-time-jitter", type=float, default=1.0)
    parser.add_argument(
        "--obstacle-state-mode",
        choices=[
            "nearest_absolute",
            "nearest_relative",
            "threat_absolute",
            "threat_relative",
        ],
        default="nearest_absolute",
    )
    parser.add_argument(
        "--trainable-scope", choices=["all", "obstacle_encoder"], default="all"
    )
    parser.add_argument(
        "--forced-conflict-mode",
        choices=["crossing", "head_on", "vertical"],
        default="crossing",
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
    metrics = train(
        args.timesteps,
        args.output,
        args.seed,
        args.eval_episodes,
        args.resume,
        args.obstacles,
        args.obstacle_offset,
        args.resume_model,
        args.random_obstacle,
        args.randomization_level,
        (args.offset_low, args.offset_high),
        args.safety_distance,
        args.safety_weight,
        args.failure_penalty,
        args.dynamic_obstacles,
        args.motion_amplitude,
        (args.speed_low, args.speed_high),
        args.lidar_delta,
        args.obstacle_state,
        args.transfer_observation,
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
        args.learning_rate,
        args.temporal_conv,
        args.forced_conflicts,
        args.forced_conflict_probability,
        args.conflict_nominal_speed,
        args.conflict_time_jitter,
        args.obstacle_state_mode,
        args.trainable_scope,
        args.forced_conflict_mode,
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
