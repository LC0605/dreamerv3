from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


class _Space:
    def __init__(self, dtype, shape=(), low=None, high=None):
        self.dtype = dtype
        self.shape = shape
        self.low = low
        self.high = high


def test_dreamerv3_adapter_uses_project_environment(monkeypatch):
    elements = types.ModuleType("elements")
    elements.Space = _Space
    embodied = types.ModuleType("embodied")
    embodied.Env = object
    monkeypatch.setitem(sys.modules, "elements", elements)
    monkeypatch.setitem(sys.modules, "embodied", embodied)

    path = Path(__file__).parents[1] / "dreamerv3/embodied/envs/quadrotor.py"
    spec = importlib.util.spec_from_file_location("uav_dreamerv3_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    env = module.Quadrotor(
        project_root="..",
        random_start_goal=False,
        max_steps=2,
        arena_x=16.0,
        arena_y=16.0,
        arena_z=5.0,
        horizontal_speed_limit=0.6,
        vertical_speed_limit=0.3,
    )
    try:
        first = env.step({"reset": True, "action": np.zeros(4, np.float32)})
        assert first["is_first"] and not first["is_last"]
        assert first["log/success"] == 0.0
        assert first["vector"].shape == env.obs_space["vector"].shape
        assert env.act_space["action"].shape == (4,)
        np.testing.assert_allclose(
            env.act_space["action"].high, [0.6, 0.6, 0.3, 0.5]
        )
        np.testing.assert_allclose(env._env.arena_size, [16.0, 16.0, 5.0])

        transition = env.step(
            {"reset": False, "action": np.zeros(4, dtype=np.float32)}
        )
        assert transition["vector"].dtype == np.float32
        assert isinstance(transition["reward"], np.float32)
        assert "log/distance" in transition
        assert "log/out_of_bounds" in transition
    finally:
        env.close()
