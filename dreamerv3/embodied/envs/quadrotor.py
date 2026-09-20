"""DreamerV3 adapter for the project's physics-based Gymnasium environment."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import elements
import embodied
import numpy as np


class Quadrotor(embodied.Env):
  """Expose UAVNavigationEnv through the Embodied environment contract.

  Dreamer chooses normalized velocity/yaw-rate commands. UAVNavigationEnv then
  executes them through its velocity-attitude cascade, motor mixer, motor lag,
  and 240 Hz PyBullet rigid-body simulation. This adapter never overwrites the
  vehicle pose or velocity.
  """

  def __init__(
      self,
      task='navigation',
      project_root='..',
      seed=0,
      max_steps=450,
      obstacle_count=0,
      obstacle_spawn_probability=1.0,
      obstacle_lateral_offset=0.0,
      obstacle_randomization_level=0,
      random_offset_low=-1.5,
      random_offset_high=1.5,
      dynamic_obstacles=False,
      obstacle_motion_amplitude=0.60,
      obstacle_speed_low=0.08,
      obstacle_speed_high=0.15,
      temporal_history=1,
      include_obstacle_state=False,
      obstacle_state_mode='threat_relative',
      random_start_goal=True,
      arena_x=16.0,
      arena_y=16.0,
      arena_z=5.0,
      goal_distance_low=2.0,
      goal_distance_high=5.85,
      horizontal_speed_limit=0.60,
      vertical_speed_limit=0.30,
      yaw_rate_limit=0.50,
      obstacle_half_xy_low=0.25,
      obstacle_half_xy_high=0.45,
      obstacle_height_low=1.0,
      obstacle_height_high=2.5,
      obstacle_min_spacing=1.0,
      endpoint_clearance=0.5,
      obstacle_layout='corridor',
      safety_distance=0.50,
      safety_weight=2.0,
      failure_penalty=100.0,
      boundary_margin=1.0,
      boundary_guard_distance=0.0,
      vertical_boundary_guard_distance=0.2,
      smoothness_weight=0.02,
      acceleration_xy=0.8,
      acceleration_z=0.5,
      acceleration_yaw=0.8,
      forced_conflict_count=0,
      forced_conflict_probability=1.0,
      forced_conflict_mode='crossing',
      conflict_nominal_speed=0.5,
      conflict_time_jitter=1.0,
      conflict_speed_low=0.3,
      conflict_speed_high=0.7,
      safety_projection=False,
      safety_projection_horizon=2.5,
      safety_projection_margin=0.55,
      safety_projection_gain=0.45,
      ttc_reward_weight=2.0,
      boundary_reward_weight=0.5,
      lidar_noise_std=0.0,
      observation_noise_std=0.0,
      observation_dropout_probability=0.0,
      observation_delay_steps=0,
      action_delay_steps=0,
      mass_scale=1.0,
      inertia_scale=1.0,
      thrust_scale=1.0,
      wind_acceleration_x=0.0,
      wind_acceleration_y=0.0,
      wind_acceleration_z=0.0,
      landing_pad_enabled=False,
      autoland_enabled=False,
  ):
    if task != 'navigation':
      raise ValueError(f'Unknown quadrotor task: {task!r}')
    root = (Path(__file__).resolve().parents[2] / project_root).resolve()
    source = root / 'src'
    if str(source) not in sys.path:
      sys.path.insert(0, str(source))
    from uav_navigation.env import UAVNavigationEnv

    self._env = UAVNavigationEnv(
        render_mode='direct',
        seed=int(seed),
        max_steps=int(max_steps),
        obstacle_count=int(obstacle_count),
        obstacle_spawn_probability=float(obstacle_spawn_probability),
        obstacle_lateral_offset=float(obstacle_lateral_offset),
        obstacle_randomization_level=int(obstacle_randomization_level),
        random_offset_range=(float(random_offset_low), float(random_offset_high)),
        dynamic_obstacles=bool(dynamic_obstacles),
        obstacle_motion_amplitude=float(obstacle_motion_amplitude),
        obstacle_speed_range=(float(obstacle_speed_low), float(obstacle_speed_high)),
        temporal_history=int(temporal_history),
        include_obstacle_state=bool(include_obstacle_state),
        obstacle_state_mode=str(obstacle_state_mode),
        random_start_goal=bool(random_start_goal),
        arena_size=(float(arena_x), float(arena_y), float(arena_z)),
        goal_distance_range=(float(goal_distance_low), float(goal_distance_high)),
        horizontal_speed_limit=float(horizontal_speed_limit),
        vertical_speed_limit=float(vertical_speed_limit),
        yaw_rate_limit=float(yaw_rate_limit),
        obstacle_half_xy_range=(float(obstacle_half_xy_low), float(obstacle_half_xy_high)),
        obstacle_height_range=(float(obstacle_height_low), float(obstacle_height_high)),
        obstacle_min_spacing=float(obstacle_min_spacing),
        endpoint_clearance=float(endpoint_clearance),
        obstacle_layout=str(obstacle_layout),
        safety_distance=float(safety_distance),
        safety_weight=float(safety_weight),
        failure_penalty=float(failure_penalty),
        boundary_margin=float(boundary_margin),
        boundary_guard_distance=float(boundary_guard_distance),
        vertical_boundary_guard_distance=float(vertical_boundary_guard_distance),
        smoothness_weight=float(smoothness_weight),
        command_acceleration_limits=(
            float(acceleration_xy), float(acceleration_xy),
            float(acceleration_z), float(acceleration_yaw)),
        forced_conflict_count=int(forced_conflict_count),
        forced_conflict_probability=float(forced_conflict_probability),
        forced_conflict_mode=str(forced_conflict_mode),
        conflict_nominal_speed=float(conflict_nominal_speed),
        conflict_time_jitter=float(conflict_time_jitter),
        conflict_speed_range=(float(conflict_speed_low), float(conflict_speed_high)),
        safety_projection=bool(safety_projection),
        safety_projection_horizon=float(safety_projection_horizon),
        safety_projection_margin=float(safety_projection_margin),
        safety_projection_gain=float(safety_projection_gain),
        ttc_reward_weight=float(ttc_reward_weight),
        boundary_reward_weight=float(boundary_reward_weight),
        mass_scale=float(mass_scale),
        inertia_scale=float(inertia_scale),
        thrust_scale=float(thrust_scale),
        wind_acceleration=(float(wind_acceleration_x), float(wind_acceleration_y),
                           float(wind_acceleration_z)),
        landing_pad_enabled=bool(landing_pad_enabled),
        autoland_enabled=bool(autoland_enabled),
    )
    if not 0.0 <= float(observation_dropout_probability) <= 1.0:
      raise ValueError('observation dropout probability must be in [0, 1]')
    if int(observation_delay_steps) < 0 or int(action_delay_steps) < 0:
      raise ValueError('observation and action delays must be non-negative')
    if float(lidar_noise_std) < 0.0 or float(observation_noise_std) < 0.0:
      raise ValueError('observation noise standard deviations must be non-negative')
    self._rng = np.random.default_rng(int(seed) + 7919)
    self._lidar_noise_std = float(lidar_noise_std)
    self._observation_noise_std = float(observation_noise_std)
    self._observation_dropout_probability = float(observation_dropout_probability)
    self._observation_delay_steps = int(observation_delay_steps)
    self._action_delay_steps = int(action_delay_steps)
    self._observation_queue = deque(maxlen=self._observation_delay_steps + 1)
    self._action_queue = deque(maxlen=self._action_delay_steps + 1)
    self._last_vector = None
    self._episode_obstacle_present = False
    self._done = True
    self._scene_start = np.zeros(3, np.float32)
    self._scene_goal = np.zeros(3, np.float32)
    self._scene_obstacle = np.zeros(3, np.float32)
    self._scene_obstacle_half = np.zeros(3, np.float32)

  @property
  def obs_space(self):
    space = self._env.observation_space
    return {
        'vector': elements.Space(np.float32, space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        # Online transitions are not demonstrations. Offline replay chunks can
        # set this field to select behavior-cloning supervision explicitly.
        'teacher': elements.Space(bool),
        'action_policy_action': elements.Space(np.float32, (4,), -1.0, 1.0),
        'log/distance': elements.Space(np.float32),
        'log/success': elements.Space(np.float32, (), 0.0, 1.0),
        'log/collision': elements.Space(np.float32, (), 0.0, 1.0),
        'log/out_of_bounds': elements.Space(np.float32, (), 0.0, 1.0),
        'log/timeout': elements.Space(np.float32, (), 0.0, 1.0),
        'log/obstacle_present': elements.Space(np.float32, (), 0.0, 1.0),
        'log/obstacle_episode': elements.Space(np.float32, (), 0.0, 1.0),
        'log/success_with_obstacle': elements.Space(np.float32, (), 0.0, 1.0),
        'log/collision_with_obstacle': elements.Space(np.float32, (), 0.0, 1.0),
        'log/timeout_with_obstacle': elements.Space(np.float32, (), 0.0, 1.0),
        'log/speed': elements.Space(np.float32),
        'log/tilt': elements.Space(np.float32),
        'log/minimum_clearance': elements.Space(np.float32),
        'log/boundary_clearance': elements.Space(np.float32),
        'log/path_length': elements.Space(np.float32),
        'log/action_delta': elements.Space(np.float32),
        'log/action_magnitude': elements.Space(np.float32),
        'log/action_forward': elements.Space(np.float32),
        'log/action_lateral': elements.Space(np.float32),
        'log/action_vertical': elements.Space(np.float32),
        'log/action_yaw_rate': elements.Space(np.float32),
        'log/action_route_lateral': elements.Space(np.float32),
        'log/obstacle_lateral_relative': elements.Space(np.float32),
        'log/position_x': elements.Space(np.float32),
        'log/position_y': elements.Space(np.float32),
        'log/position_z': elements.Space(np.float32),
        'log/scene_start_x': elements.Space(np.float32),
        'log/scene_start_y': elements.Space(np.float32),
        'log/scene_start_z': elements.Space(np.float32),
        'log/scene_goal_x': elements.Space(np.float32),
        'log/scene_goal_y': elements.Space(np.float32),
        'log/scene_goal_z': elements.Space(np.float32),
        'log/scene_obstacle_x': elements.Space(np.float32),
        'log/scene_obstacle_y': elements.Space(np.float32),
        'log/scene_obstacle_z': elements.Space(np.float32),
        'log/scene_obstacle_half_x': elements.Space(np.float32),
        'log/scene_obstacle_half_y': elements.Space(np.float32),
        'log/scene_obstacle_half_z': elements.Space(np.float32),
        'log/path_length_ratio': elements.Space(np.float32),
        'log/altitude': elements.Space(np.float32),
    }

  @property
  def act_space(self):
    space = self._env.action_space
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, space.shape, space.low, space.high),
    }

  def step(self, action):
    if bool(action['reset']) or self._done:
      observation, reset_info = self._env.reset()
      self._scene_start = np.asarray(reset_info['start'], np.float32)
      self._scene_goal = np.asarray(reset_info['goal'], np.float32)
      obstacles = reset_info.get('obstacles', ())
      halves = reset_info.get('obstacle_half_extents', ())
      self._scene_obstacle = np.asarray(
          obstacles[0] if obstacles else np.zeros(3), np.float32)
      self._scene_obstacle_half = np.asarray(
          halves[0] if halves else np.zeros(3), np.float32)
      self._episode_obstacle_present = bool(reset_info.get('obstacles', ()))
      self._observation_queue.clear()
      self._action_queue.clear()
      self._last_vector = None
      zero_action = np.zeros(self._env.action_space.shape, dtype=np.float32)
      for _ in range(self._action_delay_steps):
        self._action_queue.append(zero_action.copy())
      processed = self._process_observation(observation)
      for _ in range(self._observation_delay_steps):
        self._observation_queue.append(processed.copy())
      self._done = False
      return self._observation(processed, 0.0, {}, is_first=True)
    self._action_queue.append(np.asarray(action['action'], dtype=np.float32))
    executed_action = self._action_queue[0]
    observation, reward, terminated, truncated, info = self._env.step(
        executed_action)
    observation = self._process_observation(observation)
    self._done = bool(terminated or truncated)
    return self._observation(
        observation,
        reward,
        info,
        is_last=self._done,
        is_terminal=bool(terminated),
    )

  def _observation(
      self, vector, reward, info, is_first=False, is_last=False, is_terminal=False):
    obstacle_present = self._episode_obstacle_present
    success = bool(info.get('success', False))
    collision = bool(info.get('collision', False))
    timeout = bool(is_last and not is_terminal)
    applied_action = np.asarray(
        info.get('applied_action', np.zeros(4, np.float32)), np.float32)
    position = np.asarray(info.get('position', np.zeros(3, np.float32)), np.float32)
    return {
        'vector': np.asarray(vector, dtype=np.float32),
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'teacher': False,
        'action_policy_action': np.zeros(4, np.float32),
        'log/distance': np.float32(info.get('distance', 0.0)),
        'log/success': np.float32(success),
        'log/collision': np.float32(collision),
        'log/out_of_bounds': np.float32(info.get('out_of_bounds', False)),
        'log/timeout': np.float32(timeout),
        'log/obstacle_present': np.float32(obstacle_present),
        'log/obstacle_episode': np.float32(is_last and obstacle_present),
        'log/success_with_obstacle': np.float32(success and obstacle_present),
        'log/collision_with_obstacle': np.float32(collision and obstacle_present),
        'log/timeout_with_obstacle': np.float32(timeout and obstacle_present),
        'log/speed': np.float32(info.get('speed', 0.0)),
        'log/tilt': np.float32(info.get('tilt', 0.0)),
        'log/minimum_clearance': np.float32(info.get('minimum_clearance', 10.0)),
        'log/boundary_clearance': np.float32(info.get('boundary_clearance', 10.0)),
        'log/path_length': np.float32(info.get('path_length', 0.0)),
        'log/action_delta': np.float32(info.get('action_delta', 0.0)),
        'log/action_magnitude': np.float32(
            np.linalg.norm(applied_action)),
        'log/action_forward': np.float32(applied_action[0]),
        'log/action_lateral': np.float32(applied_action[1]),
        'log/action_vertical': np.float32(applied_action[2]),
        'log/action_yaw_rate': np.float32(applied_action[3]),
        'log/action_route_lateral': np.float32(
            info.get('route_lateral_action', 0.0)),
        'log/obstacle_lateral_relative': np.float32(
            info.get('obstacle_lateral_relative', 0.0)),
        'log/position_x': np.float32(position[0]),
        'log/position_y': np.float32(position[1]),
        'log/position_z': np.float32(position[2]),
        'log/scene_start_x': np.float32(self._scene_start[0]),
        'log/scene_start_y': np.float32(self._scene_start[1]),
        'log/scene_start_z': np.float32(self._scene_start[2]),
        'log/scene_goal_x': np.float32(self._scene_goal[0]),
        'log/scene_goal_y': np.float32(self._scene_goal[1]),
        'log/scene_goal_z': np.float32(self._scene_goal[2]),
        'log/scene_obstacle_x': np.float32(self._scene_obstacle[0]),
        'log/scene_obstacle_y': np.float32(self._scene_obstacle[1]),
        'log/scene_obstacle_z': np.float32(self._scene_obstacle[2]),
        'log/scene_obstacle_half_x': np.float32(self._scene_obstacle_half[0]),
        'log/scene_obstacle_half_y': np.float32(self._scene_obstacle_half[1]),
        'log/scene_obstacle_half_z': np.float32(self._scene_obstacle_half[2]),
        'log/path_length_ratio': np.float32(info.get('path_length_ratio', 0.0)),
        'log/altitude': np.float32(info.get('altitude', 0.0)),
    }

  def _process_observation(self, vector):
    vector = np.asarray(vector, dtype=np.float32).copy()
    if self._observation_noise_std:
      # State noise is deliberately small and dimensionless; LiDAR has its own
      # independently controllable noise level below.
      vector += self._rng.normal(0.0, self._observation_noise_std, vector.shape)
    if self._lidar_noise_std:
      # The current-frame 16-ray LiDAR always starts at offset 20.
      vector[20:36] = np.clip(
          vector[20:36] + self._rng.normal(0.0, self._lidar_noise_std, 16), 0.0, 1.0)
    if (self._last_vector is not None and self._observation_dropout_probability
        and self._rng.random() < self._observation_dropout_probability):
      vector = self._last_vector.copy()
    self._last_vector = vector.copy()
    self._observation_queue.append(vector)
    return self._observation_queue[0].copy()

  def render(self):
    return self._env.camera_frame()

  def close(self):
    self._env.close()
