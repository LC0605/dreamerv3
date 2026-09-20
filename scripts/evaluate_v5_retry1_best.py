#!/usr/bin/env python3
"""Independently evaluate the strongest observed retry1 milestone."""

from pathlib import Path

import continue_dreamer_curriculum as curriculum


SNAPSHOT = Path(
    "/home/user/UAV_Dreamer/outputs/dreamerv3/earlystop_snapshots/"
    "stage1a_fixed_v5_recovery_presence_0p65_retry1/"
    "step_21490_20260823T100609F825158"
)
SCENE = {
    "obstacle_count": 1,
    "obstacle_layout": "corridor",
    "obstacle_randomization_level": 0,
    "dynamic_obstacles": False,
    "obstacle_lateral_offset": 1.5,
    "obstacle_spawn_probability": 0.65,
    "safety_distance": 0.9,
    "safety_weight": 0.3,
    "ttc_reward_weight": 0.5,
    "horizontal_speed_limit": 0.42,
    "safety_projection": False,
}


if __name__ == "__main__":
    curriculum.evaluate(
        "eval_stage1a_fixed_v5_recovery_presence_0p65_retry1_milestone_"
        "step_21490_20260823T100609F825158",
        SNAPSHOT,
        (0.8, 0.5, 0.8),
        SCENE,
    )
