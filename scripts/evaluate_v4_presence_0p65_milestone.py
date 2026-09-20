#!/usr/bin/env python3
"""Independently evaluate the first immutable v4 presence milestone."""

from pathlib import Path

import continue_dreamer_curriculum as curriculum


SNAPSHOT = Path(
    "/home/user/UAV_Dreamer/outputs/dreamerv3/earlystop_snapshots/"
    "stage1a_fixed_v4_presence_0p65/"
    "step_8470_20260822T232553F562186"
)
SCENE = {
    "obstacle_count": 1,
    "obstacle_layout": "corridor",
    "obstacle_randomization_level": 0,
    "dynamic_obstacles": False,
    "obstacle_lateral_offset": 1.5,
    "obstacle_spawn_probability": 0.65,
    "safety_distance": 0.8,
    "safety_weight": 0.5,
    "ttc_reward_weight": 0.75,
    "horizontal_speed_limit": 0.45,
    "safety_projection": False,
}


if __name__ == "__main__":
    curriculum.evaluate(
        "eval_stage1a_fixed_v4_presence_0p65_milestone_"
        "step_8470_20260822T232553F562186",
        SNAPSHOT,
        (0.8, 0.5, 0.8),
        SCENE,
    )
