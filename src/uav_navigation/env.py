from __future__ import annotations

from pathlib import Path
import time
from collections import deque

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from .dynamics import CF2XParameters, VelocityAttitudeController


class UAVNavigationEnv(gym.Env):
    """Quadrotor navigation with configurable arena and moving obstacles."""

    metadata = {"render_modes": ["human", "direct"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str = "direct",
        max_steps: int = 300,
        seed: int | None = None,
        drone_urdf: str | Path | None = None,
        obstacle_count: int = 1,
        obstacle_spawn_probability: float = 1.0,
        obstacle_lateral_offset: float = 0.0,
        randomize_obstacle: bool = False,
        obstacle_randomization_level: int = 0,
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
        start_height_range: tuple[float, float] = (1.0, 1.0),
        goal_height_range: tuple[float, float] = (0.5, 2.5),
        horizontal_speed_limit: float = 2.0,
        vertical_speed_limit: float = 1.0,
        yaw_rate_limit: float = 1.0,
        obstacle_half_xy_range: tuple[float, float] = (0.3, 0.55),
        obstacle_height_range: tuple[float, float] = (1.2, 3.0),
        obstacle_min_spacing: float = 0.0,
        endpoint_clearance: float = 0.0,
        smoothness_weight: float = 0.02,
        boundary_guard_distance: float = 0.0,
        vertical_boundary_guard_distance: float = 0.2,
        takeoff_enabled: bool = False,
        takeoff_speed: float = 0.35,
        takeoff_settle_time: float = 2.0,
        command_acceleration_limits: tuple[float, float, float, float] | None = None,
        capture_takeoff_frames: bool = False,
        landing_pad_enabled: bool = False,
        landing_approach_height: float = 1.0,
        autoland_enabled: bool = False,
        landing_descent_speed: float = 0.12,
        landing_flare_height: float = 0.30,
        landing_flare_speed: float = 0.05,
        landing_xy_tolerance: float = 0.20,
        touchdown_speed_tolerance: float = 0.18,
        touchdown_tilt_tolerance_degrees: float = 15.0,
        disarm_time: float = 0.40,
        arrival_slow_radius: float = 1.5,
        arrival_speed_tolerance: float = 0.15,
        arrival_hold_time: float = 1.0,
        arrival_velocity_weight: float = 1.5,
        terminal_position_control_radius: float = 0.8,
        obstacle_layout: str = "corridor",
        temporal_history: int = 1,
        forced_conflict_count: int = 0,
        forced_conflict_probability: float = 1.0,
        conflict_nominal_speed: float = 0.7,
        conflict_time_jitter: float = 1.0,
        conflict_speed_range: tuple[float, float] = (0.3, 0.7),
        obstacle_state_mode: str = "nearest_absolute",
        forced_conflict_mode: str = "crossing",
        safety_projection: bool = False,
        safety_projection_horizon: float = 2.5,
        safety_projection_margin: float = 0.55,
        safety_projection_gain: float = 0.45,
        ttc_reward_weight: float = 2.0,
        boundary_reward_weight: float = 0.5,
        mass_scale: float = 1.0,
        inertia_scale: float = 1.0,
        thrust_scale: float = 1.0,
        wind_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if obstacle_count < 0:
            raise ValueError("obstacle_count must be non-negative")
        if not 0.0 <= obstacle_spawn_probability <= 1.0:
            raise ValueError("obstacle spawn probability must be between zero and one")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.obstacle_spawn_probability = float(obstacle_spawn_probability)
        self.obstacle_lateral_offset = obstacle_lateral_offset
        self.randomize_obstacle = randomize_obstacle
        self.obstacle_randomization_level = max(
            obstacle_randomization_level, 3 if randomize_obstacle else 0
        )
        self.random_offset_range = random_offset_range
        self.safety_distance = safety_distance
        self.safety_weight = safety_weight
        self.failure_penalty = failure_penalty
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacle_motion_amplitude = obstacle_motion_amplitude
        self.obstacle_speed_range = obstacle_speed_range
        self.include_lidar_delta = include_lidar_delta
        self.include_obstacle_state = include_obstacle_state
        self.arena_size = np.asarray(arena_size, dtype=np.float32)
        self.random_start_goal = random_start_goal
        self.goal_distance_range = goal_distance_range
        self.boundary_margin = boundary_margin
        self.start_height_range = start_height_range
        self.goal_height_range = goal_height_range
        self.obstacle_half_xy_range = obstacle_half_xy_range
        self.obstacle_height_range = obstacle_height_range
        self.obstacle_min_spacing = obstacle_min_spacing
        self.endpoint_clearance = endpoint_clearance
        self.smoothness_weight = smoothness_weight
        self.boundary_guard_distance = boundary_guard_distance
        self.vertical_boundary_guard_distance = vertical_boundary_guard_distance
        self.takeoff_enabled = takeoff_enabled
        self.takeoff_speed = takeoff_speed
        self.takeoff_settle_time = takeoff_settle_time
        self.command_acceleration_limits = np.asarray(
            command_acceleration_limits
            if command_acceleration_limits is not None
            else (np.inf, np.inf, np.inf, np.inf),
            dtype=np.float32,
        )
        self.capture_takeoff_frames = capture_takeoff_frames
        self.landing_pad_enabled = landing_pad_enabled
        self.landing_approach_height = landing_approach_height
        if autoland_enabled and not landing_pad_enabled:
            raise ValueError("autoland requires the landing pad")
        if landing_descent_speed <= 0.0 or landing_flare_speed <= 0.0:
            raise ValueError("landing descent speeds must be positive")
        if not 0.0 < landing_flare_height < landing_approach_height:
            raise ValueError("landing flare height must be below the approach height")
        if landing_xy_tolerance <= 0.0 or touchdown_speed_tolerance <= 0.0:
            raise ValueError("invalid landing or touchdown tolerance")
        if touchdown_tilt_tolerance_degrees <= 0.0 or disarm_time < 0.0:
            raise ValueError("invalid touchdown tilt tolerance or disarm time")
        self.autoland_enabled = autoland_enabled
        self.landing_descent_speed = landing_descent_speed
        self.landing_flare_height = landing_flare_height
        self.landing_flare_speed = landing_flare_speed
        self.landing_xy_tolerance = landing_xy_tolerance
        self.touchdown_speed_tolerance = touchdown_speed_tolerance
        self.touchdown_tilt_tolerance = np.deg2rad(touchdown_tilt_tolerance_degrees)
        self.disarm_time = disarm_time
        self.arrival_slow_radius = arrival_slow_radius
        self.arrival_speed_tolerance = arrival_speed_tolerance
        self.arrival_hold_time = arrival_hold_time
        self.arrival_velocity_weight = arrival_velocity_weight
        self.terminal_position_control_radius = terminal_position_control_radius
        if obstacle_layout not in {"corridor", "arena"}:
            raise ValueError("obstacle layout must be 'corridor' or 'arena'")
        self.obstacle_layout = obstacle_layout
        if not 0 <= forced_conflict_count <= obstacle_count:
            raise ValueError("forced conflict count must be between zero and obstacle count")
        if forced_conflict_count and (obstacle_layout != "arena" or not dynamic_obstacles):
            raise ValueError("forced conflicts require dynamic obstacles in the arena layout")
        self.forced_conflict_count = forced_conflict_count
        if not 0.0 <= forced_conflict_probability <= 1.0:
            raise ValueError("forced conflict probability must be between zero and one")
        if conflict_nominal_speed <= 0.0 or conflict_time_jitter < 0.0:
            raise ValueError("invalid conflict timing parameters")
        if not 0.0 < conflict_speed_range[0] <= conflict_speed_range[1]:
            raise ValueError("invalid conflict speed range")
        self.forced_conflict_probability = forced_conflict_probability
        self.conflict_nominal_speed = conflict_nominal_speed
        self.conflict_time_jitter = conflict_time_jitter
        self.conflict_speed_range = conflict_speed_range
        self.active_forced_conflict_count = 0
        if obstacle_state_mode not in {
            "nearest_absolute",
            "nearest_relative",
            "threat_absolute",
            "threat_relative",
        }:
            raise ValueError("invalid obstacle state mode")
        self.obstacle_state_mode = obstacle_state_mode
        if forced_conflict_mode not in {"crossing", "head_on", "vertical", "mixed"}:
            raise ValueError("invalid forced conflict mode")
        self.forced_conflict_mode = forced_conflict_mode
        if safety_projection_horizon <= 0.0 or safety_projection_margin <= 0.0:
            raise ValueError("invalid safety projection horizon or margin")
        if safety_projection_gain < 0.0:
            raise ValueError("safety projection gain must be non-negative")
        self.safety_projection = safety_projection
        self.safety_projection_horizon = safety_projection_horizon
        self.safety_projection_margin = safety_projection_margin
        self.safety_projection_gain = safety_projection_gain
        if ttc_reward_weight < 0.0 or boundary_reward_weight < 0.0:
            raise ValueError("reward ablation weights must be non-negative")
        self.ttc_reward_weight = ttc_reward_weight
        self.boundary_reward_weight = boundary_reward_weight
        if mass_scale <= 0.0 or inertia_scale <= 0.0 or thrust_scale <= 0.0:
            raise ValueError("dynamics scales must be positive")
        self.mass_scale = float(mass_scale)
        self.inertia_scale = float(inertia_scale)
        self.thrust_scale = float(thrust_scale)
        self.wind_acceleration = np.asarray(wind_acceleration, dtype=np.float32)
        if self.wind_acceleration.shape != (3,) or not np.all(
            np.isfinite(self.wind_acceleration)
        ):
            raise ValueError("wind acceleration must contain three finite values")
        if temporal_history < 1:
            raise ValueError("temporal history must be at least one frame")
        self.temporal_history = temporal_history
        if np.any(self.arena_size <= 0):
            raise ValueError("arena dimensions must be positive")
        if not 0 < goal_distance_range[0] <= goal_distance_range[1]:
            raise ValueError("invalid goal distance range")
        if self.command_acceleration_limits.shape != (4,) or np.any(
            self.command_acceleration_limits <= 0.0
        ):
            raise ValueError("command acceleration limits must contain four positive values")
        if takeoff_speed <= 0.0 or takeoff_settle_time < 0.0:
            raise ValueError("invalid takeoff speed or settle time")
        if landing_approach_height <= 0.0 or arrival_slow_radius <= 0.0:
            raise ValueError("invalid landing approach height or slow radius")
        if arrival_speed_tolerance <= 0.0 or arrival_hold_time < 0.0:
            raise ValueError("invalid arrival speed tolerance or hold time")
        if terminal_position_control_radius <= 0.0:
            raise ValueError("terminal position control radius must be positive")
        default_urdf = Path("/home/user/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf")
        self.drone_urdf = Path(drone_urdf) if drone_urdf else default_urdf
        self.control_dt = 0.1
        self.physics_hz = 240
        self.physics_dt = 1.0 / self.physics_hz
        self.physics_steps_per_action = 24
        self.bounds_low = np.array(
            [-0.5 * self.arena_size[0], -0.5 * self.arena_size[1], 0.1], dtype=np.float32
        )
        self.bounds_high = np.array(
            [0.5 * self.arena_size[0], 0.5 * self.arena_size[1], self.arena_size[2]],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array(
                [-horizontal_speed_limit, -horizontal_speed_limit, -vertical_speed_limit, -yaw_rate_limit],
                dtype=np.float32,
            ),
            high=np.array(
                [horizontal_speed_limit, horizontal_speed_limit, vertical_speed_limit, yaw_rate_limit],
                dtype=np.float32,
            ),
        )
        self.lidar_rays = 16
        self.lidar_range = 4.0
        # Keep temporal obstacle states dimensionless. This prevents a
        # curriculum transition from injecting metre-scale values into
        # channels that were all zero in the obstacle-free stage.
        self.obstacle_position_scale = np.asarray(arena_size, dtype=np.float32)
        self.obstacle_velocity_scale = np.array(
            [horizontal_speed_limit, horizontal_speed_limit, vertical_speed_limit],
            dtype=np.float32,
        )
        base_observation_size = (
            36 + (self.lidar_rays if include_lidar_delta else 0)
            + (18 if include_obstacle_state else 0)
        )
        temporal_perception_size = self.lidar_rays + (18 if include_obstacle_state else 0)
        observation_size = base_observation_size + (temporal_history - 1) * temporal_perception_size
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )
        self.np_random = np.random.default_rng(seed)
        self.client_id: int | None = None
        self.drone_id: int | None = None
        self.goal_id: int | None = None
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacle_ids: list[int] = []
        self.obstacle_centers: list[np.ndarray] = []
        self.obstacle_half_extents: list[np.ndarray] = []
        self.obstacle_anchors: list[np.ndarray] = []
        self.obstacle_motion_axes: list[np.ndarray] = []
        self.obstacle_motion_rates: list[float] = []
        self.obstacle_motion_phases: list[float] = []
        self.navigation_waypoints: list[np.ndarray] = []
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.parameters = CF2XParameters()
        self.controller = VelocityAttitudeController(self.parameters)
        self.motor_rpm = np.full(4, self.parameters.hover_rpm)
        self.steps = 0
        self.previous_distance = 0.0
        self.initial_distance = 0.0
        self.sim_time = 0.0
        self.previous_lidar: np.ndarray | None = None
        self.perception_history: deque[np.ndarray] = deque(maxlen=temporal_history)
        self.flight_phase = "navigation"
        self.takeoff_duration = 0.0
        self.takeoff_frames: list[np.ndarray] = []
        self.arrival_hold_steps = 0
        self.disarm_duration = 0.0
        self.touchdown = False
        self.path_length = 0.0
        self.minimum_episode_clearance = 10.0

    def _sample_start_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.random_start_goal:
            start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            goal = self.np_random.uniform([-4.0, -4.0, 0.5], [4.0, 4.0, 2.5]).astype(np.float32)
            while np.linalg.norm(goal - start) < self.goal_distance_range[0]:
                goal = self.np_random.uniform([-4.0, -4.0, 0.5], [4.0, 4.0, 2.5]).astype(np.float32)
            return start, goal

        xy_low = self.bounds_low[:2] + self.boundary_margin
        xy_high = self.bounds_high[:2] - self.boundary_margin
        if np.any(xy_low >= xy_high):
            raise ValueError("boundary margin leaves no valid start/goal area")
        for _ in range(2_000):
            start = np.array(
                [
                    *self.np_random.uniform(xy_low, xy_high),
                    self.np_random.uniform(*self.start_height_range),
                ],
                dtype=np.float32,
            )
            horizontal_distance = float(self.np_random.uniform(*self.goal_distance_range))
            angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            goal = np.array(
                [
                    start[0] + horizontal_distance * np.cos(angle),
                    start[1] + horizontal_distance * np.sin(angle),
                    self.np_random.uniform(*self.goal_height_range),
                ],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(goal - start))
            if (
                np.all(goal[:2] >= xy_low)
                and np.all(goal[:2] <= xy_high)
                and self.goal_distance_range[0] <= distance <= self.goal_distance_range[1]
            ):
                return start, goal
        raise RuntimeError("could not sample valid start and goal after 2000 attempts")

    def _connect(self) -> None:
        if self.client_id is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._connect()
        assert self.client_id is not None
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -self.parameters.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        start, self.goal = self._sample_start_and_goal()
        self.episode_start = start.copy()
        route_direction_xy = self.goal[:2] - start[:2]
        route_direction_xy /= max(float(np.linalg.norm(route_direction_xy)), 1e-6)
        self.route_perpendicular_xy = np.array(
            [-route_direction_xy[1], route_direction_xy[0]], dtype=np.float32)
        if self.landing_pad_enabled:
            self.goal[2] = self.landing_approach_height
        if not self.drone_urdf.is_file():
            raise FileNotFoundError(f"Quadrotor URDF not found: {self.drone_urdf}")
        spawn = start.copy()
        if self.takeoff_enabled:
            spawn[2] = max(0.08, float(self.bounds_low[2]))
        self.drone_id = p.loadURDF(
            str(self.drone_urdf),
            basePosition=spawn,
            useFixedBase=False,
            globalScaling=1.0,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        nominal_inertia = np.array([1.4e-5, 1.4e-5, 2.17e-5], dtype=np.float64)
        p.changeDynamics(
            self.drone_id,
            -1,
            mass=self.parameters.mass * self.mass_scale,
            localInertiaDiagonal=nominal_inertia * self.inertia_scale,
            linearDamping=0,
            angularDamping=0,
            physicsClientId=self.client_id,
        )
        if self.landing_pad_enabled:
            pad_half_extents = [0.6, 0.6, 0.025]
            goal_collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=pad_half_extents, physicsClientId=self.client_id
            )
            goal_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=pad_half_extents,
                rgbaColor=[0.08, 0.65, 0.18, 1.0],
                physicsClientId=self.client_id,
            )
            self.goal_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=goal_collision,
                baseVisualShapeIndex=goal_visual,
                basePosition=[float(self.goal[0]), float(self.goal[1]), 0.025],
                physicsClientId=self.client_id,
            )
            z_mark = 0.055
            x, y = map(float, self.goal[:2])
            for start_mark, end_mark in (
                ((x - 0.25, y - 0.3, z_mark), (x - 0.25, y + 0.3, z_mark)),
                ((x + 0.25, y - 0.3, z_mark), (x + 0.25, y + 0.3, z_mark)),
                ((x - 0.25, y, z_mark), (x + 0.25, y, z_mark)),
            ):
                p.addUserDebugLine(
                    start_mark, end_mark, [1.0, 1.0, 1.0], lineWidth=5,
                    physicsClientId=self.client_id,
                )
        else:
            goal_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.15,
                rgbaColor=[0.1, 0.85, 0.2, 0.75],
                physicsClientId=self.client_id,
            )
            self.goal_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=goal_visual,
                basePosition=self.goal,
                physicsClientId=self.client_id,
            )
        self.obstacle_ids.clear()
        self.obstacle_centers.clear()
        self.obstacle_half_extents.clear()
        self.obstacle_anchors.clear()
        self.obstacle_motion_axes.clear()
        self.obstacle_motion_rates.clear()
        self.obstacle_motion_phases.clear()
        self.navigation_waypoints.clear()
        # Do not advance the RNG for the legacy all-random case. This keeps
        # fixed-seed benchmark scenes bit-for-bit comparable across versions.
        self.active_forced_conflict_count = 0
        # Preserve the established fixed-seed benchmark exactly when the
        # probability is one; only progressive curricula consume this draw.
        spawn_obstacles = bool(
            self.obstacle_count
            and (
                self.obstacle_spawn_probability >= 1.0
                or self.np_random.random() < self.obstacle_spawn_probability
            )
        )
        if self.forced_conflict_count and spawn_obstacles:
            self.active_forced_conflict_count = (
                self.forced_conflict_count
                if self.np_random.random() < self.forced_conflict_probability
                else 0
            )
        if spawn_obstacles:
            direction_xy = self.goal[:2] - start[:2]
            direction_xy /= max(float(np.linalg.norm(direction_xy)), 1e-6)
            perpendicular = np.array([-direction_xy[1], direction_xy[0]])
            nominal_fractions = np.linspace(
                0.15 if self.obstacle_count > 3 else 0.3,
                0.85 if self.obstacle_count > 3 else 0.7,
                self.obstacle_count,
            )
            arena_cells: list[tuple[np.ndarray, np.ndarray]] = []
            if self.obstacle_layout == "arena":
                grid_size = int(np.ceil(np.sqrt(self.obstacle_count)))
                x_edges = np.linspace(self.bounds_low[0] + 0.7, self.bounds_high[0] - 0.7, grid_size + 1)
                y_edges = np.linspace(self.bounds_low[1] + 0.7, self.bounds_high[1] - 0.7, grid_size + 1)
                cells = [
                    (
                        np.array([x_edges[ix], y_edges[iy]], dtype=np.float32),
                        np.array([x_edges[ix + 1], y_edges[iy + 1]], dtype=np.float32),
                    )
                    for ix in range(grid_size)
                    for iy in range(grid_size)
                ]
                selected = self.np_random.permutation(len(cells))[: self.obstacle_count]
                arena_cells = [cells[int(index)] for index in selected]
            for obstacle_index, nominal_fraction in enumerate(nominal_fractions):
                forced_conflict = obstacle_index < self.active_forced_conflict_count
                conflict_mode = self.forced_conflict_mode
                if forced_conflict and conflict_mode == "mixed":
                    conflict_mode = str(
                        self.np_random.choice(("crossing", "head_on", "vertical"))
                    )
                speed_range = (
                    self.conflict_speed_range if forced_conflict
                    else self.obstacle_speed_range
                )
                candidate_speed = float(self.np_random.uniform(*speed_range))
                motion_rate = candidate_speed / max(float(self.obstacle_motion_amplitude), 1e-6)
                for placement_attempt in range(300):
                    if self.obstacle_randomization_level >= 1:
                        lateral_offset = float(self.np_random.uniform(*self.random_offset_range))
                    else:
                        # A fixed offset can point outside the arena when the
                        # sampled route runs near a boundary. Alternate the
                        # geometrically equivalent side of the route instead
                        # of retrying the same impossible center 300 times.
                        lateral_offset = (
                            self.obstacle_lateral_offset
                            if placement_attempt % 2 == 0
                            else -self.obstacle_lateral_offset
                        )
                    if self.obstacle_randomization_level >= 2:
                        lower_fraction = 0.1 if self.obstacle_count > 3 else 0.2
                        upper_fraction = 0.9 if self.obstacle_count > 3 else 0.8
                        path_fraction = float(
                            np.clip(
                                nominal_fraction + self.np_random.uniform(-0.08, 0.08),
                                lower_fraction,
                                upper_fraction,
                            )
                        )
                    else:
                        path_fraction = float(nominal_fraction)
                        # For a fixed corridor scene, retries otherwise repeat
                        # the same invalid longitudinal position.  This occurs
                        # for short routes when the nominal 0.3 fraction plus a
                        # small lateral offset misses endpoint clearance by a
                        # few millimetres.  Keep the requested lateral offset
                        # exact and move only as far along the route as needed
                        # to make the static, fixed-size obstacle valid.
                        if not self.dynamic_obstacles:
                            route_xy_length = float(
                                np.linalg.norm(self.goal[:2] - start[:2])
                            )
                            # Small numerical margin avoids float32 rounding
                            # putting the center microscopically inside the
                            # strict endpoint exclusion radius.
                            endpoint_radius = self.endpoint_clearance + 0.45 + 1e-4
                            longitudinal_clearance = float(
                                np.sqrt(max(
                                    endpoint_radius ** 2 - lateral_offset ** 2,
                                    0.0,
                                ))
                            )
                            minimum_fraction = (
                                longitudinal_clearance
                                / max(route_xy_length, 1e-6)
                            )
                            if minimum_fraction <= 0.5:
                                path_fraction = float(np.clip(
                                    path_fraction,
                                    minimum_fraction,
                                    1.0 - minimum_fraction,
                                ))
                    if self.obstacle_randomization_level >= 3 or self.obstacle_layout == "arena":
                        half_xy = float(self.np_random.uniform(*self.obstacle_half_xy_range))
                        half_z = 0.5 * float(self.np_random.uniform(*self.obstacle_height_range))
                    else:
                        half_xy = 0.45
                        half_z = 1.25
                    if self.obstacle_layout == "arena":
                        if forced_conflict:
                            conflict_fraction = (obstacle_index + 1) / (
                                self.active_forced_conflict_count + 1
                            )
                            center = start + conflict_fraction * (self.goal - start)
                            center[:2] += float(self.np_random.uniform(-0.15, 0.15)) * perpendicular
                            if conflict_mode == "head_on":
                                motion_axis = np.array(
                                    [-direction_xy[0], -direction_xy[1], 0.0], dtype=np.float32
                                )
                            elif conflict_mode == "vertical":
                                half_z = half_xy
                                motion_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                            else:
                                motion_axis = np.array(
                                    [perpendicular[0], perpendicular[1], 0.0], dtype=np.float32
                                )
                        else:
                            # A selected cell can be almost entirely covered by
                            # an endpoint exclusion zone. Rotate through cells
                            # instead of silently reducing the obstacle count.
                            cell_low, cell_high = arena_cells[
                                (obstacle_index + placement_attempt) % len(arena_cells)
                            ]
                            center = np.array(
                                [*self.np_random.uniform(cell_low, cell_high), 0.0],
                                dtype=np.float32,
                            )
                            motion_angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                            motion_axis = np.array(
                                [np.cos(motion_angle), np.sin(motion_angle), 0.0], dtype=np.float32
                            )
                    else:
                        center = start + path_fraction * (self.goal - start)
                        center[:2] += lateral_offset * perpendicular
                        motion_axis = np.array(
                            [perpendicular[0], perpendicular[1], 0.0], dtype=np.float32
                        )
                    center[2] = half_z
                    if forced_conflict and conflict_mode == "vertical":
                        route_height = float(
                            start[2] + conflict_fraction * (self.goal[2] - start[2])
                        )
                        center[2] = np.clip(
                            route_height,
                            self.bounds_low[2] + half_z + self.obstacle_motion_amplitude,
                            self.bounds_high[2] - half_z - self.obstacle_motion_amplitude,
                        )
                    if forced_conflict:
                        distance_to_crossing = float(
                            np.linalg.norm(center[:2] - start[:2])
                        )
                        expected_arrival = (
                            distance_to_crossing / self.conflict_nominal_speed
                            + float(
                                self.np_random.uniform(
                                    -self.conflict_time_jitter, self.conflict_time_jitter
                                )
                            )
                        )
                        expected_arrival = max(expected_arrival, self.control_dt)
                        phase = float((-motion_rate * expected_arrival) % (2.0 * np.pi))
                    else:
                        phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                    initial_center = center.copy()
                    if self.dynamic_obstacles:
                        initial_center += (
                            self.obstacle_motion_amplitude * np.sin(phase) * motion_axis
                        )
                    required_edge = half_xy + (
                        self.obstacle_motion_amplitude if self.dynamic_obstacles else 0.0
                    )
                    inside_arena = bool(
                        np.all(center[:2] >= self.bounds_low[:2] + required_edge)
                        and np.all(center[:2] <= self.bounds_high[:2] - required_edge)
                        and (
                            not motion_axis[2]
                            or (
                                center[2]
                                >= self.bounds_low[2] + half_z + self.obstacle_motion_amplitude
                                and center[2]
                                <= self.bounds_high[2] - half_z - self.obstacle_motion_amplitude
                            )
                        )
                    )
                    endpoint_reference = initial_center if forced_conflict else center
                    endpoint_edge = half_xy if forced_conflict else required_edge
                    clear_endpoints = self.endpoint_clearance <= 0.0 or bool(
                        np.linalg.norm(endpoint_reference[:2] - start[:2])
                        >= self.endpoint_clearance + endpoint_edge
                        and np.linalg.norm(endpoint_reference[:2] - self.goal[:2])
                        >= self.endpoint_clearance + endpoint_edge
                    )
                    separated = all(
                        np.linalg.norm(initial_center[:2] - prior[:2]) >= self.obstacle_min_spacing
                        for prior in self.obstacle_centers
                    )
                    if inside_arena and clear_endpoints and separated:
                        break
                else:
                    raise RuntimeError(
                        f"could not place obstacle {obstacle_index + 1}/{self.obstacle_count} "
                        "without overlap after 300 attempts"
                    )
                half_extents = np.array([half_xy, half_xy, half_z])
                collision = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
                )
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=[0.85, 0.12 + 0.08 * (obstacle_index % 3), 0.08, 1.0],
                    physicsClientId=self.client_id,
                )
                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=initial_center,
                    physicsClientId=self.client_id,
                )
                self.obstacle_ids.append(obstacle_id)
                self.obstacle_centers.append(initial_center.astype(np.float32))
                self.obstacle_half_extents.append(half_extents.astype(np.float32))
                self.obstacle_anchors.append(center.astype(np.float32).copy())
                self.obstacle_motion_axes.append(
                    motion_axis
                )
                self.obstacle_motion_rates.append(motion_rate)
                self.obstacle_motion_phases.append(phase)
                if self.obstacle_layout == "corridor":
                    detour = center.copy()
                    detour[:2] += (1.15 if obstacle_index % 2 == 0 else -1.15) * perpendicular
                    detour[2] = float(
                        np.clip(0.5 * (start[2] + self.goal[2]), 0.5, self.bounds_high[2] - 0.5)
                    )
                    self.navigation_waypoints.append(detour.astype(np.float32))
        self.navigation_waypoints.append(self.goal.copy())
        x_low, y_low = map(float, self.bounds_low[:2])
        x_high, y_high = map(float, self.bounds_high[:2])
        corners = [
            (x_low, y_low, 0.02),
            (x_high, y_low, 0.02),
            (x_high, y_high, 0.02),
            (x_low, y_high, 0.02),
        ]
        for index, corner in enumerate(corners):
            p.addUserDebugLine(
                corner,
                corners[(index + 1) % len(corners)],
                [0.9, 0.2, 0.2],
                lineWidth=2,
                physicsClientId=self.client_id,
            )
        self.steps = 0
        self.sim_time = 0.0
        self.previous_lidar = None
        self.perception_history.clear()
        self.previous_action[:] = 0.0
        self.motor_rpm[:] = self.parameters.hover_rpm
        self.controller.reset()
        self.flight_phase = "takeoff" if self.takeoff_enabled else "navigation"
        self.takeoff_duration = 0.0
        self.takeoff_frames.clear()
        self.arrival_hold_steps = 0
        self.disarm_duration = 0.0
        self.touchdown = False
        self.path_length = 0.0
        self.minimum_episode_clearance = 10.0
        if self.takeoff_enabled:
            self._run_takeoff_and_settle(float(start[2]))
            self.flight_phase = "navigation"
            self.previous_action[:] = 0.0
            self.previous_lidar = None
        self.previous_distance = float(np.linalg.norm(self.goal - start))
        self.initial_distance = self.previous_distance
        return self._observation(), {
            "start": start.copy(),
            "spawn": spawn.copy(),
            "goal": self.goal.copy(),
            "obstacles": [center.copy() for center in self.obstacle_centers],
            "obstacle_half_extents": [
                half.copy() for half in self.obstacle_half_extents],
            "obstacle_layout": self.obstacle_layout,
            "forced_conflict_count": self.forced_conflict_count,
            "active_forced_conflict_count": self.active_forced_conflict_count,
            "forced_conflict_mode": self.forced_conflict_mode,
            "flight_phase": self.flight_phase,
            "takeoff_duration": self.takeoff_duration,
            "autoland_enabled": self.autoland_enabled,
        }

    def _nearest_obstacle(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if not self.obstacle_centers:
            return np.full(3, 10.0), np.zeros(3), 10.0
        clearances = []
        for center, half_extents in zip(self.obstacle_centers, self.obstacle_half_extents):
            outside = np.maximum(np.abs(position - center) - half_extents, 0.0)
            clearances.append(float(np.linalg.norm(outside)))
        index = int(np.argmin(clearances))
        return (
            self.obstacle_centers[index] - position,
            self.obstacle_half_extents[index],
            clearances[index],
        )

    def _predicted_collision_risk(
        self, position: np.ndarray, velocity: np.ndarray, horizon: float = 2.0
    ) -> tuple[float, float]:
        """Return earliest time-to-closest-approach and predicted AABB clearance."""
        best_ttc = float("inf")
        best_clearance = float("inf")
        for index, (center, half_extents) in enumerate(
            zip(self.obstacle_centers, self.obstacle_half_extents)
        ):
            obstacle_velocity = np.zeros(3, dtype=np.float32)
            if self.dynamic_obstacles:
                obstacle_velocity = (
                    self.obstacle_motion_amplitude
                    * self.obstacle_motion_rates[index]
                    * np.cos(
                        self.obstacle_motion_rates[index] * self.sim_time
                        + self.obstacle_motion_phases[index]
                    )
                    * self.obstacle_motion_axes[index]
                ).astype(np.float32)
            relative_position = center - position
            relative_velocity = obstacle_velocity - velocity
            relative_speed_squared = float(relative_velocity @ relative_velocity)
            if relative_speed_squared < 1e-6:
                continue
            ttc = -float(relative_position @ relative_velocity) / relative_speed_squared
            if not 0.0 < ttc <= horizon:
                continue
            future_relative = relative_position + ttc * relative_velocity
            outside = np.maximum(np.abs(future_relative) - half_extents, 0.0)
            clearance = float(np.linalg.norm(outside))
            if clearance < best_clearance:
                best_ttc = ttc
                best_clearance = clearance
        return best_ttc, best_clearance

    def _project_safe_velocity(
        self, position: np.ndarray, desired_velocity: np.ndarray
    ) -> tuple[np.ndarray, float, int]:
        """Project a world-frame velocity away from imminent moving AABB conflicts.

        The projection changes only the high-level velocity request. The normal
        acceleration limiter and rigid-body flight controller remain in the
        loop, so the shield cannot teleport or instantaneously redirect the UAV.
        """
        projected = desired_velocity.astype(np.float32, copy=True)
        correction = np.zeros(3, dtype=np.float32)
        active_constraints = 0
        for index, (center, half_extents) in enumerate(
            zip(self.obstacle_centers, self.obstacle_half_extents)
        ):
            obstacle_velocity = np.zeros(3, dtype=np.float32)
            if self.dynamic_obstacles:
                obstacle_velocity = (
                    self.obstacle_motion_amplitude
                    * self.obstacle_motion_rates[index]
                    * np.cos(
                        self.obstacle_motion_rates[index] * self.sim_time
                        + self.obstacle_motion_phases[index]
                    )
                    * self.obstacle_motion_axes[index]
                ).astype(np.float32)
            relative_position = center - position
            relative_velocity = obstacle_velocity - projected
            speed_squared = float(relative_velocity @ relative_velocity)
            if speed_squared < 1e-6:
                continue
            ttc = -float(relative_position @ relative_velocity) / speed_squared
            if not 0.0 < ttc <= self.safety_projection_horizon:
                continue
            future_relative = relative_position + ttc * relative_velocity
            outside = np.maximum(np.abs(future_relative) - half_extents, 0.0)
            predicted_clearance = float(np.linalg.norm(outside))
            if predicted_clearance >= self.safety_projection_margin:
                continue
            # Prefer a geometrically meaningful side of the obstacle. At an
            # exact predicted intersection, choose a horizontal side-step
            # perpendicular to the closing motion rather than an arbitrary axis.
            avoidance = -np.sign(future_relative) * outside
            if float(np.linalg.norm(avoidance)) < 1e-5:
                closing_xy = relative_velocity[:2]
                avoidance = np.array(
                    [-closing_xy[1], closing_xy[0], 0.0], dtype=np.float32
                )
                if float(np.dot(avoidance, -relative_position)) < 0.0:
                    avoidance *= -1.0
            norm = float(np.linalg.norm(avoidance))
            if norm < 1e-6:
                continue
            urgency = (
                (1.0 - predicted_clearance / self.safety_projection_margin)
                * (1.0 - ttc / self.safety_projection_horizon)
            )
            correction += self.safety_projection_gain * urgency * avoidance / norm
            active_constraints += 1
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > self.safety_projection_gain:
            correction *= self.safety_projection_gain / correction_norm
            correction_norm = self.safety_projection_gain
        projected += correction
        return projected, correction_norm, active_constraints

    def _observation(self) -> np.ndarray:
        assert self.client_id is not None and self.drone_id is not None
        position, orientation = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        velocity, angular_velocity = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        body_velocity = rotation.T @ np.asarray(velocity)
        body_rates = rotation.T @ np.asarray(angular_velocity)
        lidar = self._lidar(np.asarray(position), rotation)
        components = [
                position,
                orientation,
                body_velocity,
                body_rates,
                self.goal - np.asarray(position),
                self.previous_action,
                lidar,
            ]
        if self.include_lidar_delta:
            lidar_delta = (
                np.zeros_like(lidar)
                if self.previous_lidar is None
                else (lidar - self.previous_lidar) / self.control_dt
            )
            components.append(lidar_delta)
        obstacle_state_flat = np.empty(0, dtype=np.float32)
        if self.include_obstacle_state:
            obstacle_state = np.zeros((3, 6), dtype=np.float32)
            entries = []
            drone_velocity = np.asarray(velocity, dtype=np.float32)
            for index, center in enumerate(self.obstacle_centers):
                relative_position = center - np.asarray(position)
                obstacle_velocity = np.zeros(3, dtype=np.float32)
                if self.dynamic_obstacles:
                    obstacle_velocity = (
                        self.obstacle_motion_amplitude
                        * self.obstacle_motion_rates[index]
                        * np.cos(
                            self.obstacle_motion_rates[index] * self.sim_time
                            + self.obstacle_motion_phases[index]
                        )
                        * self.obstacle_motion_axes[index]
                    ).astype(np.float32)
                relative_velocity = obstacle_velocity - drone_velocity
                encoded_velocity = (
                    relative_velocity
                    if self.obstacle_state_mode.endswith("relative")
                    else obstacle_velocity
                )
                if self.obstacle_state_mode.startswith("threat"):
                    speed_squared = float(relative_velocity @ relative_velocity)
                    ttc = (
                        float(
                            np.clip(
                                -float(relative_position @ relative_velocity) / speed_squared,
                                0.0,
                                3.0,
                            )
                        )
                        if speed_squared > 1e-6 else 0.0
                    )
                    future_relative = relative_position + ttc * relative_velocity
                    future_clearance = float(
                        np.linalg.norm(
                            np.maximum(
                                np.abs(future_relative) - self.obstacle_half_extents[index], 0.0
                            )
                        )
                    )
                    closing = float(relative_position @ relative_velocity) < 0.0
                    rank = future_clearance + 0.15 * ttc + (0.0 if closing else 4.0)
                else:
                    encoded_velocity = obstacle_velocity
                    rank = float(np.linalg.norm(relative_position))
                entries.append((rank, relative_position, encoded_velocity))
            ranked_entries = sorted(entries, key=lambda item: item[0])
            for slot, (_, relative_position, encoded_velocity) in enumerate(ranked_entries[:3]):
                obstacle_state[slot, :3] = (
                    rotation.T @ relative_position
                ) / self.obstacle_position_scale
                obstacle_state[slot, 3:] = (
                    rotation.T @ encoded_velocity
                ) / self.obstacle_velocity_scale
            obstacle_state_flat = obstacle_state.reshape(-1)
            components.append(obstacle_state_flat)
        perception = np.concatenate((lidar, obstacle_state_flat)).astype(np.float32)
        self.perception_history.append(perception)
        if self.temporal_history > 1:
            frames = list(self.perception_history)
            padding = [frames[0]] * (self.temporal_history - len(frames))
            ordered_history = padding + frames
            components.extend(ordered_history[:-1])
        self.previous_lidar = lidar.copy()
        obs = np.concatenate(components)
        return obs.astype(np.float32)

    def _lidar(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Normalized horizontal body-frame ranges; 1 means no hit within range."""
        assert self.client_id is not None
        angles = np.linspace(0.0, 2.0 * np.pi, self.lidar_rays, endpoint=False)
        body_directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        world_directions = body_directions @ rotation.T
        ray_from = position + 0.12 * world_directions
        ray_to = position + self.lidar_range * world_directions
        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.client_id)
        readings = np.ones(self.lidar_rays, dtype=np.float32)
        for index, result in enumerate(results):
            object_id, hit_fraction = result[0], result[2]
            if object_id >= 0 and object_id != self.drone_id:
                readings[index] = np.clip(hit_fraction, 0.0, 1.0)
        return readings

    def _update_dynamic_obstacles(self) -> None:
        if not self.dynamic_obstacles:
            return
        assert self.client_id is not None
        for index, obstacle_id in enumerate(self.obstacle_ids):
            displacement = self.obstacle_motion_amplitude * np.sin(
                self.obstacle_motion_rates[index] * self.sim_time
                + self.obstacle_motion_phases[index]
            )
            center = self.obstacle_anchors[index] + displacement * self.obstacle_motion_axes[index]
            self.obstacle_centers[index] = center.astype(np.float32)
            p.resetBasePositionAndOrientation(
                obstacle_id, center, [0.0, 0.0, 0.0, 1.0], physicsClientId=self.client_id
            )

    def _simulate_command(self, action: np.ndarray, update_obstacles: bool = True) -> None:
        """Advance one policy interval through the motor and rigid-body dynamics."""
        assert self.client_id is not None and self.drone_id is not None
        for _ in range(self.physics_steps_per_action):
            if update_obstacles:
                self.sim_time += self.physics_dt
                self._update_dynamic_obstacles()
            _, orientation = p.getBasePositionAndOrientation(
                self.drone_id, physicsClientId=self.client_id
            )
            velocity, angular_velocity = p.getBaseVelocity(
                self.drone_id, physicsClientId=self.client_id
            )
            rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            body_rates = rotation.T @ np.asarray(angular_velocity)
            rpm_command = self.controller.compute(
                action[:3], action[3], rotation, np.asarray(velocity), body_rates, self.physics_dt
            )
            alpha = 1.0 - np.exp(-self.physics_dt / self.parameters.motor_time_constant)
            self.motor_rpm += alpha * (rpm_command - self.motor_rpm)
            forces = self.thrust_scale * self.parameters.kf * np.square(self.motor_rpm)
            for motor_link, force in enumerate(forces):
                p.applyExternalForce(
                    self.drone_id,
                    motor_link,
                    [0, 0, float(force)],
                    [0, 0, 0],
                    p.LINK_FRAME,
                    physicsClientId=self.client_id,
                )
            yaw_torque = self.thrust_scale * self.parameters.km * float(
                -self.motor_rpm[0] ** 2 + self.motor_rpm[1] ** 2
                - self.motor_rpm[2] ** 2 + self.motor_rpm[3] ** 2
            )
            p.applyExternalTorque(
                self.drone_id,
                -1,
                [0, 0, yaw_torque],
                p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
            if np.any(self.wind_acceleration):
                p.applyExternalForce(
                    self.drone_id,
                    -1,
                    (self.parameters.mass * self.mass_scale * self.wind_acceleration).tolist(),
                    [0.0, 0.0, 0.0],
                    p.WORLD_FRAME,
                    physicsClientId=self.client_id,
                )
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(self.physics_dt)

    def _run_takeoff_and_settle(self, target_height: float) -> None:
        """Use the flight controller to climb and establish a stable hover before RL control."""
        assert self.client_id is not None and self.drone_id is not None
        stable_steps = 0
        required_stable_steps = int(np.ceil(self.takeoff_settle_time / self.control_dt))
        max_preflight_steps = int(np.ceil(12.0 / self.control_dt))
        for preflight_step in range(max_preflight_steps):
            position, _ = p.getBasePositionAndOrientation(
                self.drone_id, physicsClientId=self.client_id
            )
            velocity, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
            height_error = target_height - float(position[2])
            vertical_command = float(np.clip(1.2 * height_error, -0.2, self.takeoff_speed))
            command = np.array([0.0, 0.0, vertical_command, 0.0], dtype=np.float32)
            self.flight_phase = "hover_settle" if abs(height_error) <= 0.1 else "takeoff"
            self._simulate_command(command, update_obstacles=False)
            if self.capture_takeoff_frames:
                self.takeoff_frames.append(self.camera_frame())
            stable = abs(height_error) <= 0.1 and np.linalg.norm(velocity) <= 0.15
            stable_steps = stable_steps + 1 if stable else 0
            self.takeoff_duration = (preflight_step + 1) * self.control_dt
            if stable_steps >= required_stable_steps:
                return
        raise RuntimeError("quadrotor failed to establish a stable hover during takeoff")

    def _run_motor_disarm(self) -> None:
        """Ramp motor thrust to zero after a validated pad touchdown."""
        assert self.client_id is not None and self.drone_id is not None
        total_steps = max(1, int(np.ceil(self.disarm_time / self.physics_dt)))
        initial_rpm = self.motor_rpm.copy()
        for disarm_step in range(total_steps):
            fraction = max(0.0, 1.0 - (disarm_step + 1) / total_steps)
            self.motor_rpm[:] = initial_rpm * fraction
            forces = self.thrust_scale * self.parameters.kf * np.square(self.motor_rpm)
            for motor_link, force in enumerate(forces):
                p.applyExternalForce(
                    self.drone_id,
                    motor_link,
                    [0, 0, float(force)],
                    [0, 0, 0],
                    p.LINK_FRAME,
                    physicsClientId=self.client_id,
                )
            yaw_torque = self.thrust_scale * self.parameters.km * float(
                -self.motor_rpm[0] ** 2 + self.motor_rpm[1] ** 2
                - self.motor_rpm[2] ** 2 + self.motor_rpm[3] ** 2
            )
            p.applyExternalTorque(
                self.drone_id,
                -1,
                [0, 0, yaw_torque],
                p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
            self.sim_time += self.physics_dt
            self._update_dynamic_obstacles()
            p.stepSimulation(physicsClientId=self.client_id)
        self.motor_rpm[:] = 0.0
        self.previous_action[:] = 0.0
        self.disarm_duration = total_steps * self.physics_dt

    def step(self, action):
        assert self.client_id is not None and self.drone_id is not None
        requested_action = np.clip(
            np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high
        )
        position_before, _ = p.getBasePositionAndOrientation(
            self.drone_id, physicsClientId=self.client_id
        )
        distance_before = float(np.linalg.norm(self.goal - np.asarray(position_before)))
        landing_state_active = self.autoland_enabled and self.flight_phase in {
            "landing_descent", "landing_realign"
        }
        if landing_state_active:
            horizontal_error = self.goal[:2] - np.asarray(position_before[:2])
            requested_action[:2] = np.clip(0.8 * horizontal_error, -0.20, 0.20)
            altitude_above_pad = float(position_before[2] - 0.05)
            if np.linalg.norm(horizontal_error) > self.landing_xy_tolerance:
                requested_action[2] = 0.0
                self.flight_phase = "landing_realign"
            else:
                descent_speed = (
                    self.landing_flare_speed
                    if altitude_above_pad <= self.landing_flare_height
                    else self.landing_descent_speed
                )
                requested_action[2] = -descent_speed
                self.flight_phase = "landing_descent"
            requested_action[3] = 0.0
        elif self.landing_pad_enabled and distance_before < self.terminal_position_control_radius:
            position_error = self.goal - np.asarray(position_before)
            requested_action[:2] = np.clip(position_error[:2], -0.3, 0.3)
            requested_action[2] = np.clip(position_error[2], -0.2, 0.2)
            requested_action[3] = 0.0
            self.flight_phase = "terminal_hover"
        else:
            self.flight_phase = "navigation"
        if (
            self.landing_pad_enabled
            and not landing_state_active
            and distance_before < self.arrival_slow_radius
        ):
            speed_scale = float(np.clip(distance_before / self.arrival_slow_radius, 0.15, 1.0))
            horizontal_norm = float(np.linalg.norm(requested_action[:2]))
            horizontal_cap = float(self.action_space.high[0]) * speed_scale
            if horizontal_norm > horizontal_cap:
                requested_action[:2] *= horizontal_cap / max(horizontal_norm, 1e-6)
            requested_action[2] = np.clip(
                requested_action[2],
                -float(self.action_space.high[2]) * speed_scale,
                float(self.action_space.high[2]) * speed_scale,
            )
        shield_correction = 0.0
        shield_constraints = 0
        if self.safety_projection:
            safe_velocity, shield_correction, shield_constraints = self._project_safe_velocity(
                np.asarray(position_before, dtype=np.float32), requested_action[:3]
            )
            requested_action[:3] = np.clip(
                safe_velocity, self.action_space.low[:3], self.action_space.high[:3]
            )
        max_delta = self.command_acceleration_limits * self.control_dt
        action = self.previous_action + np.clip(
            requested_action - self.previous_action, -max_delta, max_delta
        )
        if self.boundary_guard_distance > 0.0:
            position, _ = p.getBasePositionAndOrientation(
                self.drone_id, physicsClientId=self.client_id
            )
            # Arena termination is three-dimensional, so the execution-time
            # boundary guard must constrain altitude as well as X/Y. Leaving Z
            # unguarded allowed obstacle-avoidance corrections to trade a safe
            # lateral path for an avoidable ceiling/floor violation.
            for axis in (0, 1, 2):
                guard_distance = (
                    self.vertical_boundary_guard_distance
                    if axis == 2 else self.boundary_guard_distance
                )
                if (
                    position[axis] <= self.bounds_low[axis] + guard_distance
                    and action[axis] < 0.0
                ) or (
                    position[axis] >= self.bounds_high[axis] - guard_distance
                    and action[axis] > 0.0
                ):
                    action[axis] = 0.0
        prior_action = self.previous_action.copy()
        self._simulate_command(action)
        self.previous_action = action.copy()
        self.steps += 1
        obs = self._observation()
        distance = float(np.linalg.norm(obs[13:16]))
        speed = float(np.linalg.norm(obs[7:10]))
        world_velocity, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        orientation = obs[3:7]
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        pad_contact = bool(
            self.goal_id is not None
            and p.getContactPoints(self.drone_id, self.goal_id, physicsClientId=self.client_id)
        )
        landing_horizontal_error = float(np.linalg.norm(obs[13:15]))
        inside_arrival_zone = distance < 0.3
        stable_arrival = inside_arrival_zone and speed < self.arrival_speed_tolerance
        reached = False
        if self.autoland_enabled:
            if self.flight_phase == "terminal_hover":
                self.arrival_hold_steps = self.arrival_hold_steps + 1 if stable_arrival else 0
                required_hold_steps = max(1, int(np.ceil(self.arrival_hold_time / self.control_dt)))
                if self.arrival_hold_steps >= required_hold_steps:
                    self.flight_phase = "landing_descent"
                    self.arrival_hold_steps = 0
            if self.flight_phase in {"landing_descent", "landing_realign"}:
                touchdown_stable = (
                    pad_contact
                    and landing_horizontal_error <= self.landing_xy_tolerance
                    and abs(float(world_velocity[2])) <= self.touchdown_speed_tolerance
                    and float(np.linalg.norm(world_velocity[:2])) <= self.arrival_speed_tolerance
                    and tilt <= self.touchdown_tilt_tolerance
                )
                if touchdown_stable:
                    self.touchdown = True
                    self.flight_phase = "touchdown_disarm"
                    self._run_motor_disarm()
                    obs = self._observation()
                    world_velocity, _ = p.getBaseVelocity(
                        self.drone_id, physicsClientId=self.client_id
                    )
                    speed = float(np.linalg.norm(world_velocity))
                    reached = True
                    self.flight_phase = "landed_disarmed"
        elif self.landing_pad_enabled:
            self.arrival_hold_steps = self.arrival_hold_steps + 1 if stable_arrival else 0
            required_hold_steps = max(1, int(np.ceil(self.arrival_hold_time / self.control_dt)))
            reached = self.arrival_hold_steps >= required_hold_steps
        else:
            reached = inside_arrival_zone
        effective_bounds_low = self.bounds_low.copy()
        if self.autoland_enabled and self.flight_phase in {
            "landing_descent", "landing_realign", "touchdown_disarm", "landed_disarmed"
        }:
            effective_bounds_low[2] = -0.05
        out_of_bounds = bool(
            np.any(obs[:3] < effective_bounds_low) or np.any(obs[:3] > self.bounds_high)
        )
        out_of_bounds_axes = np.logical_or(
            obs[:3] < effective_bounds_low, obs[:3] > self.bounds_high
        )
        collided = any(
            p.getContactPoints(self.drone_id, obstacle_id, physicsClientId=self.client_id)
            for obstacle_id in self.obstacle_ids
        )
        _, _, minimum_clearance = self._nearest_obstacle(obs[:3])
        predicted_ttc, predicted_clearance = self._predicted_collision_risk(
            obs[:3], np.asarray(world_velocity, dtype=np.float32)
        )
        boundary_clearance = float(
            min(
                obs[0] - self.bounds_low[0],
                self.bounds_high[0] - obs[0],
                obs[1] - self.bounds_low[1],
                self.bounds_high[1] - obs[1],
            )
        )
        reward_terms = {
            "progress": (
                0.0
                if self.autoland_enabled and self.flight_phase in {
                    "landing_descent", "landing_realign", "landed_disarmed"
                }
                else 5.0 * (self.previous_distance - distance)
            ),
            "goal": 100.0 if reached else 0.0,
            "failure": -self.failure_penalty if (collided or out_of_bounds) else 0.0,
            "safety": -self.safety_weight * max(0.0, self.safety_distance - minimum_clearance),
            "smoothness": -self.smoothness_weight * float(np.square(action - prior_action).sum()),
            "arrival_velocity": (
                -self.arrival_velocity_weight
                * max(0.0, 1.0 - distance / self.arrival_slow_radius)
                * speed
                if self.landing_pad_enabled else 0.0
            ),
            "arrival_stability": (
                (1.0 if inside_arrival_zone else 0.0)
                + (2.0 * (1.0 - speed / self.arrival_speed_tolerance) if stable_arrival else 0.0)
                if self.landing_pad_enabled else 0.0
            ),
            "boundary": (
                -self.boundary_reward_weight * max(0.0, 1.5 - boundary_clearance)
                if self.obstacle_layout == "arena" else 0.0
            ),
            "ttc": (
                -self.ttc_reward_weight
                * (1.0 - predicted_ttc / 2.0)
                * max(0.0, self.safety_distance - predicted_clearance)
                / max(self.safety_distance, 1e-6)
                if predicted_ttc <= 2.0 else 0.0
            ),
            "time": -0.1,
        }
        reward = float(sum(reward_terms.values()))
        self.path_length += float(np.linalg.norm(obs[:3] - np.asarray(position_before)))
        self.minimum_episode_clearance = min(
            self.minimum_episode_clearance, minimum_clearance
        )
        action_delta = float(np.linalg.norm(action - prior_action))
        nearest_obstacle = (
            min(
                self.obstacle_centers,
                key=lambda center: float(np.linalg.norm(center[:2] - obs[:2])),
            )
            if self.obstacle_centers else None
        )
        obstacle_lateral_relative = (
            float(np.dot(
                nearest_obstacle[:2] - obs[:2], self.route_perpendicular_xy))
            if nearest_obstacle is not None else 0.0
        )
        route_lateral_action = float(
            np.dot(action[:2], self.route_perpendicular_xy))
        self.previous_distance = distance
        terminated = reached or out_of_bounds or collided
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, {
            "distance": distance,
            "success": reached,
            "out_of_bounds": out_of_bounds,
            "out_of_bounds_axes": out_of_bounds_axes.copy(),
            "position": obs[:3].copy(),
            "collision": collided,
            "minimum_clearance": minimum_clearance,
            "minimum_episode_clearance": self.minimum_episode_clearance,
            "boundary_clearance": boundary_clearance,
            "path_length": self.path_length,
            "path_length_ratio": self.path_length / max(self.initial_distance, 1e-6),
            "altitude": float(obs[2]),
            "action_delta": action_delta,
            "reward_terms": reward_terms,
            "motor_rpm": self.motor_rpm.copy(),
            "obstacle_positions": [center.copy() for center in self.obstacle_centers],
            "requested_action": requested_action.copy(),
            "applied_action": action.copy(),
            "obstacle_lateral_relative": obstacle_lateral_relative,
            "route_lateral_action": route_lateral_action,
            "flight_phase": self.flight_phase,
            "takeoff_duration": self.takeoff_duration,
            "arrival_hold_steps": self.arrival_hold_steps,
            "speed": speed,
            "tilt": tilt,
            "inside_arrival_zone": inside_arrival_zone,
            "pad_contact": pad_contact,
            "touchdown": self.touchdown,
            "landing_horizontal_error": landing_horizontal_error,
            "altitude_above_pad": float(obs[2] - 0.05),
            "disarm_duration": self.disarm_duration,
            "boundary_clearance": boundary_clearance,
            "predicted_ttc": predicted_ttc,
            "predicted_clearance": predicted_clearance,
            "shield_correction": shield_correction,
            "shield_constraints": shield_constraints,
        }

    def camera_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Return an RGB overview frame for recording in DIRECT or GUI mode."""
        assert self.client_id is not None and self.drone_id is not None
        position, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        position = np.asarray(position)
        target = 0.5 * (position + self.goal)
        separation = float(np.linalg.norm(position - self.goal))
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=max(2.2, 1.35 * separation),
            yaw=45,
            pitch=-38,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(55, width / height, 0.1, 30)
        image = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        return np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)[..., :3]

    def close(self):
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
            self.drone_id = None
            self.goal_id = None
