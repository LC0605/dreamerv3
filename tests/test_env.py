import numpy as np
import pybullet as p
import torch

from uav_navigation import UAVNavigationEnv
from uav_navigation.simulate import run
from uav_navigation.temporal_features import TemporalObstacleFeatureExtractor
from uav_navigation.train_ppo import obstacle_observation_indices


def test_temporal_obstacle_indices_cover_each_frame_once():
    indices = obstacle_observation_indices(8)
    assert indices.shape == (8 * 18,)
    assert len(np.unique(indices)) == len(indices)
    assert indices.min() == 36
    assert indices.max() == 291


def test_gymnasium_contract():
    env = UAVNavigationEnv(seed=1, max_steps=3)
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert observation.shape == (36,)
    assert info["goal"].shape == (3,)
    result = env.step(np.zeros(4, dtype=np.float32))
    assert len(result) == 5
    env.close()


def test_rule_controller_reaches_goal():
    result = run(seed=2)
    assert result["success"]
    assert result["distance"] < 0.3


def test_hover_uses_motor_forces_without_state_teleportation():
    env = UAVNavigationEnv(seed=3, max_steps=100, obstacle_count=0)
    start, _ = env.reset(seed=3)
    for _ in range(50):
        observation, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert not terminated and not truncated
    assert abs(float(observation[2] - start[2])) < 0.15
    assert np.all(info["motor_rpm"] > 0)
    env.close()


def test_rule_controller_avoids_static_obstacle():
    result = run(seed=4, obstacle_count=1)
    assert result["success"]
    assert not result["collision"]
    assert result["minimum_clearance"] > 0.0


def test_multiple_obstacles_are_created():
    env = UAVNavigationEnv(seed=7, max_steps=10, obstacle_count=3, obstacle_randomization_level=3)
    try:
        _, info = env.reset()
        assert len(info["obstacles"]) == 3
        assert len(env.obstacle_ids) == 3
        assert len(env.obstacle_half_extents) == 3
    finally:
        env.close()


def test_obstacle_spawn_probability_supports_presence_curriculum():
    absent = UAVNavigationEnv(seed=70, obstacle_count=1,
                              obstacle_spawn_probability=0.0)
    present = UAVNavigationEnv(seed=70, obstacle_count=1,
                               obstacle_spawn_probability=1.0)
    mixed = UAVNavigationEnv(seed=70, obstacle_count=1,
                             obstacle_spawn_probability=0.5)
    try:
        absent.reset(seed=70)
        present.reset(seed=70)
        assert len(absent.obstacle_ids) == 0
        assert len(present.obstacle_ids) == 1
        observed_counts = []
        for seed in range(70, 110):
            mixed.reset(seed=seed)
            observed_counts.append(len(mixed.obstacle_ids))
        assert 0 in observed_counts and 1 in observed_counts
    finally:
        absent.close()
        present.close()
        mixed.close()


def test_fixed_corridor_offset_flips_inward_near_arena_boundary():
    env = UAVNavigationEnv(
        seed=0,
        max_steps=2,
        obstacle_count=1,
        obstacle_layout="corridor",
        obstacle_randomization_level=0,
        obstacle_lateral_offset=1.5,
        arena_size=(16.0, 16.0, 5.0),
        goal_distance_range=(4.0, 8.0),
        boundary_margin=1.0,
        endpoint_clearance=0.5,
    )
    try:
        # Exercise routes along every arena edge. Before the inward-side
        # fallback, some of these deterministic resets raised after retrying
        # the same invalid center 300 times.
        for seed in range(100):
            env.reset(seed=seed)
            assert len(env.obstacle_centers) == 1
            center = env.obstacle_centers[0]
            half_xy = env.obstacle_half_extents[0][0]
            assert np.all(center[:2] >= env.bounds_low[:2] + half_xy)
            assert np.all(center[:2] <= env.bounds_high[:2] - half_xy)
    finally:
        env.close()


def test_dynamic_obstacle_moves_and_collision_geometry_follows():
    env = UAVNavigationEnv(
        seed=11,
        max_steps=10,
        obstacle_count=1,
        dynamic_obstacles=True,
        obstacle_motion_amplitude=0.5,
        obstacle_speed_range=(0.4, 0.4),
    )
    try:
        env.reset(seed=11)
        initial = env.obstacle_centers[0].copy()
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            assert not terminated and not truncated
        current = env.obstacle_centers[0]
        assert env.client_id is not None
        body_position, _ = p.getBasePositionAndOrientation(
            env.obstacle_ids[0], physicsClientId=env.client_id
        )
        assert np.linalg.norm(current - initial) > 1e-3
        assert np.allclose(current, body_position, atol=1e-6)
    finally:
        env.close()


def test_lidar_delta_adds_temporal_observation_without_changing_base_features():
    env = UAVNavigationEnv(
        seed=13,
        max_steps=5,
        obstacle_count=1,
        dynamic_obstacles=True,
        include_lidar_delta=True,
    )
    try:
        observation, _ = env.reset(seed=13)
        assert observation.shape == (52,)
        assert np.allclose(observation[36:], 0.0)
        next_observation, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        assert env.observation_space.contains(next_observation)
        assert np.all(np.isfinite(next_observation[36:]))
    finally:
        env.close()


def test_obstacle_state_contains_relative_position_and_dynamic_velocity():
    env = UAVNavigationEnv(
        seed=17,
        obstacle_count=2,
        dynamic_obstacles=True,
        include_obstacle_state=True,
        obstacle_speed_range=(0.3, 0.3),
    )
    try:
        observation, _ = env.reset(seed=17)
        assert observation.shape == (54,)
        state = observation[36:].reshape(3, 6)
        assert np.any(np.abs(state[:2, :3]) > 0.0)
        assert np.any(np.abs(state[:2, 3:]) > 0.0)
        assert np.allclose(state[2], 0.0)
    finally:
        env.close()


def test_threat_obstacle_state_encodes_relative_velocity():
    env = UAVNavigationEnv(
        seed=19,
        obstacle_count=1,
        dynamic_obstacles=False,
        include_obstacle_state=True,
        obstacle_state_mode="threat_relative",
    )
    try:
        env.reset(seed=19)
        assert env.client_id is not None and env.drone_id is not None
        p.resetBaseVelocity(
            env.drone_id,
            linearVelocity=[0.5, 0.0, 0.0],
            physicsClientId=env.client_id,
        )
        observation = env._observation()
        encoded_body_velocity = observation[39:42]
        assert np.isclose(np.linalg.norm(encoded_body_velocity), 0.25, atol=1e-3)
    finally:
        env.close()


def test_threat_ranking_handles_exactly_tied_scores():
    env = UAVNavigationEnv(
        seed=21,
        obstacle_count=2,
        include_obstacle_state=True,
        obstacle_state_mode="threat_absolute",
    )
    try:
        env.reset(seed=21)
        env.obstacle_centers[1] = env.obstacle_centers[0].copy()
        env.obstacle_half_extents[1] = env.obstacle_half_extents[0].copy()
        observation = env._observation()
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_large_arena_slow_dense_random_scene_constraints():
    env = UAVNavigationEnv(
        seed=23,
        obstacle_count=6,
        obstacle_randomization_level=3,
        random_offset_range=(-1.5, 1.5),
        dynamic_obstacles=True,
        obstacle_motion_amplitude=0.3,
        obstacle_speed_range=(0.08, 0.25),
        arena_size=(16.0, 16.0, 5.0),
        random_start_goal=True,
        goal_distance_range=(6.0, 10.0),
        boundary_margin=1.5,
        start_height_range=(1.0, 2.0),
        goal_height_range=(1.0, 3.0),
        horizontal_speed_limit=1.0,
        vertical_speed_limit=0.4,
        yaw_rate_limit=0.5,
        obstacle_half_xy_range=(0.2, 0.4),
        obstacle_height_range=(0.8, 2.5),
        obstacle_min_spacing=1.2,
        endpoint_clearance=1.2,
    )
    try:
        _, info = env.reset(seed=23)
        start, goal = info["start"], info["goal"]
        assert 6.0 <= np.linalg.norm(goal - start) <= 10.0
        assert np.allclose(env.action_space.high, [1.0, 1.0, 0.4, 0.5])
        assert len(env.obstacle_centers) == 6
        for half_extents in env.obstacle_half_extents:
            assert 0.2 <= half_extents[0] <= 0.4
            assert 0.8 <= 2.0 * half_extents[2] <= 2.5
        for index, center in enumerate(env.obstacle_centers):
            for other in env.obstacle_centers[index + 1 :]:
                assert np.linalg.norm(center[:2] - other[:2]) >= 1.2
    finally:
        env.close()


def test_boundary_guard_blocks_outward_horizontal_command_without_teleporting():
    env = UAVNavigationEnv(
        seed=29,
        obstacle_count=0,
        arena_size=(16.0, 16.0, 5.0),
        boundary_guard_distance=1.0,
        horizontal_speed_limit=1.0,
    )
    try:
        env.reset(seed=29)
        assert env.client_id is not None and env.drone_id is not None
        p.resetBasePositionAndOrientation(
            env.drone_id, [7.2, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.client_id,
        )
        env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert env.previous_action[0] == 0.0
    finally:
        env.close()


def test_takeoff_uses_dynamics_then_hands_over_from_stable_hover():
    env = UAVNavigationEnv(
        seed=31,
        obstacle_count=0,
        takeoff_enabled=True,
        takeoff_speed=0.35,
        takeoff_settle_time=1.0,
    )
    try:
        observation, info = env.reset(seed=31)
        assert info["spawn"][2] <= 0.11
        assert abs(float(observation[2] - info["start"][2])) <= 0.1
        assert info["flight_phase"] == "navigation"
        assert 1.0 <= info["takeoff_duration"] <= 12.0
        assert env.client_id is not None and env.drone_id is not None
        velocity, _ = p.getBaseVelocity(env.drone_id, physicsClientId=env.client_id)
        assert np.linalg.norm(velocity) <= 0.15
    finally:
        env.close()


def test_navigation_command_is_limited_by_acceleration():
    env = UAVNavigationEnv(
        seed=37,
        obstacle_count=0,
        horizontal_speed_limit=1.0,
        vertical_speed_limit=0.4,
        yaw_rate_limit=0.5,
        command_acceleration_limits=(0.5, 0.5, 0.25, 0.3),
    )
    try:
        env.reset(seed=37)
        _, _, _, _, info = env.step(np.array([1.0, -1.0, 0.4, 0.5], dtype=np.float32))
        assert np.allclose(info["requested_action"], [1.0, -1.0, 0.4, 0.5])
        assert np.allclose(info["applied_action"], [0.05, -0.05, 0.025, 0.03])
        assert np.allclose(env.previous_action, info["applied_action"])
    finally:
        env.close()


def test_landing_pad_requires_low_speed_stable_hold():
    env = UAVNavigationEnv(
        seed=41,
        obstacle_count=0,
        landing_pad_enabled=True,
        landing_approach_height=1.0,
        arrival_speed_tolerance=0.15,
        arrival_hold_time=0.3,
    )
    try:
        env.reset(seed=41)
        assert env.client_id is not None and env.drone_id is not None
        p.resetBasePositionAndOrientation(
            env.drone_id, env.goal, [0.0, 0.0, 0.0, 1.0], physicsClientId=env.client_id
        )
        p.resetBaseVelocity(
            env.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=env.client_id
        )
        terminated = False
        info = {}
        for _ in range(3):
            _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert terminated
        assert info["success"]
        assert info["inside_arrival_zone"]
        assert info["speed"] < 0.15
    finally:
        env.close()


def test_autoland_descends_contacts_pad_and_disarms_motors():
    env = UAVNavigationEnv(
        seed=42,
        max_steps=250,
        obstacle_count=0,
        landing_pad_enabled=True,
        landing_approach_height=1.0,
        autoland_enabled=True,
        landing_descent_speed=0.12,
        landing_flare_height=0.30,
        landing_flare_speed=0.05,
        landing_xy_tolerance=0.20,
        arrival_speed_tolerance=0.15,
        arrival_hold_time=0.3,
        command_acceleration_limits=(1.5, 1.5, 0.6, 1.0),
    )
    try:
        env.reset(seed=42)
        assert env.client_id is not None and env.drone_id is not None
        p.resetBasePositionAndOrientation(
            env.drone_id, env.goal, [0.0, 0.0, 0.0, 1.0], physicsClientId=env.client_id
        )
        p.resetBaseVelocity(
            env.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=env.client_id
        )
        terminated = truncated = False
        info = {}
        for _ in range(220):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["success"] and info["touchdown"] and info["pad_contact"]
        assert info["flight_phase"] == "landed_disarmed"
        assert info["altitude_above_pad"] < 0.10
        assert np.allclose(info["motor_rpm"], 0.0)
        assert info["disarm_duration"] >= 0.39
    finally:
        env.close()


def test_arena_layout_spreads_dense_obstacles_across_flight_area():
    env = UAVNavigationEnv(
        seed=43,
        obstacle_count=12,
        obstacle_layout="arena",
        obstacle_randomization_level=3,
        dynamic_obstacles=True,
        obstacle_motion_amplitude=0.3,
        obstacle_speed_range=(0.08, 0.15),
        arena_size=(16.0, 16.0, 5.0),
        random_start_goal=True,
        goal_distance_range=(6.0, 10.0),
        boundary_margin=2.0,
        obstacle_half_xy_range=(0.2, 0.4),
        obstacle_height_range=(0.8, 2.5),
        obstacle_min_spacing=1.2,
        endpoint_clearance=1.2,
    )
    try:
        _, info = env.reset(seed=43)
        centers = np.asarray(info["obstacles"])
        assert info["obstacle_layout"] == "arena"
        assert centers.shape == (12, 3)
        assert np.ptp(centers[:, 0]) > 9.0
        assert np.ptp(centers[:, 1]) > 9.0
        for index, center in enumerate(centers):
            assert np.linalg.norm(center[:2] - info["start"][:2]) >= 1.2
            assert np.linalg.norm(center[:2] - info["goal"][:2]) >= 1.2
            for other in centers[index + 1 :]:
                assert np.linalg.norm(center[:2] - other[:2]) >= 1.2
    finally:
        env.close()


def test_forced_dynamic_conflicts_cross_the_nominal_route_near_arrival_time():
    env = UAVNavigationEnv(
        seed=61,
        obstacle_count=12,
        obstacle_layout="arena",
        forced_conflict_count=2,
        dynamic_obstacles=True,
        obstacle_motion_amplitude=0.6,
        obstacle_speed_range=(0.08, 0.15),
        arena_size=(16.0, 16.0, 5.0),
        random_start_goal=True,
        goal_distance_range=(6.0, 10.0),
        boundary_margin=2.0,
        obstacle_half_xy_range=(0.2, 0.35),
        obstacle_height_range=(1.2, 2.8),
        obstacle_min_spacing=1.5,
        endpoint_clearance=1.5,
        horizontal_speed_limit=1.0,
        conflict_nominal_speed=0.7,
        conflict_time_jitter=0.0,
    )
    try:
        _, info = env.reset(seed=61)
        start = info["start"]
        route = info["goal"][:2] - start[:2]
        route_unit = route / np.linalg.norm(route)
        for index in range(2):
            anchor = env.obstacle_anchors[index]
            offset = anchor[:2] - start[:2]
            lateral_distance = abs(route_unit[0] * offset[1] - route_unit[1] * offset[0])
            assert lateral_distance <= 0.16
            expected_arrival = np.linalg.norm(offset) / env.conflict_nominal_speed
            crossing_offset = env.obstacle_motion_amplitude * np.sin(
                env.obstacle_motion_rates[index] * expected_arrival
                + env.obstacle_motion_phases[index]
            )
            assert abs(crossing_offset) < 1e-5
    finally:
        env.close()


def test_head_on_and_vertical_conflict_motion_axes_are_physical():
    common = dict(
        seed=63,
        obstacle_count=4,
        obstacle_layout="arena",
        forced_conflict_count=1,
        dynamic_obstacles=True,
        obstacle_motion_amplitude=0.4,
        arena_size=(16.0, 16.0, 5.0),
        random_start_goal=True,
        goal_distance_range=(6.0, 8.0),
        boundary_margin=2.0,
        obstacle_min_spacing=1.0,
        endpoint_clearance=1.0,
    )
    head_on = UAVNavigationEnv(**common, forced_conflict_mode="head_on")
    vertical = UAVNavigationEnv(**common, forced_conflict_mode="vertical")
    try:
        _, info = head_on.reset(seed=63)
        route = info["goal"][:2] - info["start"][:2]
        route /= np.linalg.norm(route)
        assert np.dot(head_on.obstacle_motion_axes[0][:2], route) < -0.99

        vertical.reset(seed=63)
        assert np.allclose(vertical.obstacle_motion_axes[0], [0.0, 0.0, 1.0])
        assert np.isclose(
            vertical.obstacle_half_extents[0][2], vertical.obstacle_half_extents[0][0]
        )
        anchor = vertical.obstacle_anchors[0]
        half_z = vertical.obstacle_half_extents[0][2]
        assert anchor[2] - half_z - vertical.obstacle_motion_amplitude >= vertical.bounds_low[2]
        assert anchor[2] + half_z + vertical.obstacle_motion_amplitude <= vertical.bounds_high[2]
    finally:
        head_on.close()
        vertical.close()


def test_conflict_obstacles_use_separate_speed_range_and_mixed_modes():
    observed_axes = set()
    for seed in range(8):
        env = UAVNavigationEnv(
            seed=seed,
            obstacle_count=6,
            obstacle_layout="arena",
            forced_conflict_count=2,
            forced_conflict_mode="mixed",
            dynamic_obstacles=True,
            obstacle_motion_amplitude=0.6,
            obstacle_speed_range=(0.08, 0.15),
            conflict_speed_range=(0.3, 0.7),
            arena_size=(16.0, 16.0, 5.0),
            random_start_goal=True,
            goal_distance_range=(6.0, 8.0),
            boundary_margin=2.0,
            obstacle_min_spacing=1.0,
            endpoint_clearance=1.0,
        )
        try:
            env.reset(seed=seed)
            forced_speeds = np.asarray(env.obstacle_motion_rates[:2]) * 0.6
            slow_speeds = np.asarray(env.obstacle_motion_rates[2:]) * 0.6
            assert np.all((0.3 <= forced_speeds) & (forced_speeds <= 0.7))
            assert np.all((0.08 <= slow_speeds) & (slow_speeds <= 0.15))
            for axis in env.obstacle_motion_axes[:2]:
                observed_axes.add("vertical" if abs(axis[2]) > 0.9 else "horizontal")
        finally:
            env.close()
    assert observed_axes == {"horizontal", "vertical"}


def test_dynamics_randomization_changes_physical_body_and_remains_finite():
    env = UAVNavigationEnv(
        seed=71,
        obstacle_count=0,
        mass_scale=1.15,
        inertia_scale=1.20,
        thrust_scale=0.90,
        wind_acceleration=(0.15, -0.10, 0.0),
    )
    try:
        observation, _ = env.reset(seed=71)
        dynamics = p.getDynamicsInfo(env.drone_id, -1, physicsClientId=env.client_id)
        assert np.isclose(dynamics[0], env.parameters.mass * 1.15)
        assert np.allclose(
            dynamics[2], np.array([1.4e-5, 1.4e-5, 2.17e-5]) * 1.20
        )
        for _ in range(10):
            observation, _, terminated, truncated, _ = env.step(np.zeros(4))
            assert np.all(np.isfinite(observation))
            if terminated or truncated:
                break
    finally:
        env.close()


def test_safety_projection_only_changes_imminent_collision_velocity():
    env = UAVNavigationEnv(
        seed=65,
        obstacle_count=1,
        dynamic_obstacles=False,
        safety_projection=True,
        safety_projection_horizon=2.5,
        safety_projection_margin=0.55,
        safety_projection_gain=0.45,
    )
    try:
        env.reset(seed=65)
        env.obstacle_centers[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        env.obstacle_half_extents[0] = np.array([0.3, 0.3, 0.6], dtype=np.float32)
        desired = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        projected, correction, constraints = env._project_safe_velocity(
            np.array([-2.0, 0.0, 1.0], dtype=np.float32), desired
        )
        assert constraints == 1
        assert 0.0 < correction <= 0.45
        assert not np.allclose(projected, desired)

        unchanged, correction, constraints = env._project_safe_velocity(
            np.array([-2.0, 0.0, 1.0], dtype=np.float32), -desired
        )
        assert constraints == 0
        assert correction == 0.0
        assert np.allclose(unchanged, -desired)
    finally:
        env.close()


def test_forced_conflict_probability_mixes_hard_and_random_episodes():
    env = UAVNavigationEnv(
        seed=67,
        obstacle_count=4,
        obstacle_layout="arena",
        forced_conflict_count=1,
        forced_conflict_probability=0.5,
        dynamic_obstacles=True,
        arena_size=(16.0, 16.0, 5.0),
        random_start_goal=True,
        goal_distance_range=(6.0, 8.0),
        boundary_margin=2.0,
        obstacle_min_spacing=1.0,
        endpoint_clearance=1.0,
    )
    try:
        active_counts = [env.reset()[1]["active_forced_conflict_count"] for _ in range(40)]
        assert 0 in active_counts
        assert 1 in active_counts
    finally:
        env.close()


def test_arena_layout_penalizes_edge_hugging_shortcut():
    env = UAVNavigationEnv(
        seed=47,
        obstacle_count=0,
        obstacle_layout="arena",
        arena_size=(16.0, 16.0, 5.0),
    )
    try:
        env.reset(seed=47)
        assert env.client_id is not None and env.drone_id is not None
        p.resetBasePositionAndOrientation(
            env.drone_id, [7.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.client_id,
        )
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["boundary_clearance"] < 1.5
        assert info["reward_terms"]["boundary"] < 0.0
    finally:
        env.close()


def test_ttc_predicts_closing_motion_but_ignores_receding_motion():
    env = UAVNavigationEnv(seed=53, obstacle_count=1)
    try:
        env.reset(seed=53)
        center = env.obstacle_centers[0]
        approaching_position = center - np.array([2.0, 0.0, 0.0], dtype=np.float32)
        ttc, clearance = env._predicted_collision_risk(
            approaching_position, np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        assert 0.0 < ttc <= 2.0
        assert clearance < env.safety_distance
        receding_ttc, _ = env._predicted_collision_risk(
            approaching_position, np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        )
        assert np.isinf(receding_ttc)
    finally:
        env.close()


def test_eight_frame_lightweight_history_has_expected_shape_and_rolls():
    env = UAVNavigationEnv(
        seed=59,
        obstacle_count=2,
        dynamic_obstacles=True,
        include_obstacle_state=True,
        temporal_history=8,
    )
    try:
        observation, _ = env.reset(seed=59)
        assert observation.shape == (292,)
        assert env.observation_space.contains(observation)
        perception_size = env.lidar_rays + 18
        history = observation[54:].reshape(7, perception_size)
        current_perception = np.concatenate((observation[20:36], observation[36:54]))
        assert np.allclose(history, current_perception)
        next_observation, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        next_history = next_observation[54:].reshape(7, perception_size)
        assert np.allclose(next_history[-1], current_perception)
    finally:
        env.close()


def test_temporal_encoder_produces_compact_finite_features():
    env = UAVNavigationEnv(
        seed=61,
        obstacle_count=2,
        include_obstacle_state=True,
        temporal_history=8,
    )
    try:
        observation, _ = env.reset(seed=61)
        extractor = TemporalObstacleFeatureExtractor(env.observation_space, history_length=8)
        batch = torch.as_tensor(observation).unsqueeze(0)
        features = extractor(batch)
        assert features.shape == (1, 128)
        assert torch.all(torch.isfinite(features))
    finally:
        env.close()
