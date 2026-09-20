#!/usr/bin/env python3
"""Recover chronologically valid episodes from the legacy teacher NPZ arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_SCALE = np.asarray([0.5, 0.5, 0.3, 0.5], np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "train").mkdir()
    (args.output / "validation").mkdir()

    keys = ("vector", "reward", "is_first", "is_last", "is_terminal", "action")
    pieces = {key: [] for key in keys}
    for path in sorted(args.source.glob("*.npz")):
        with np.load(path) as source:
            for key in keys:
                pieces[key].append(source[key])
    data = {key: np.concatenate(value) for key, value in pieces.items()}
    starts = np.flatnonzero(data["is_first"])
    ends = np.flatnonzero(data["is_last"])
    if len(starts) != len(ends) or np.any(starts > ends):
        raise ValueError("invalid episode boundary counts")
    if starts[0] != 0 or ends[-1] != len(data["vector"]) - 1:
        raise ValueError("dataset does not start/end on episode boundaries")
    if len(starts) > 1 and not np.array_equal(starts[1:], ends[:-1] + 1):
        raise ValueError("gap or overlap between episodes")
    if not 0 < args.validation_episodes < len(starts):
        raise ValueError("invalid validation episode count")

    outcomes = dict(terminal=0, timeout=0)
    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        slc = slice(start, end + 1)
        length = end - start + 1
        action_policy = data["action"][slc].astype(np.float32)
        episode = {
            "vector": data["vector"][slc].astype(np.float32),
            "reward": data["reward"][slc].astype(np.float32),
            "is_first": data["is_first"][slc].astype(bool),
            "is_last": data["is_last"][slc].astype(bool),
            "is_terminal": data["is_terminal"][slc].astype(bool),
            "teacher": ~data["is_last"][slc].astype(bool),
            "action": action_policy,
            "action_policy": action_policy,
            "action_policy_action": action_policy,
            "action_physical": action_policy * ACTION_SCALE,
            "episode_id": np.full(length, episode_id, np.int32),
            "episode_step": np.arange(length, dtype=np.int32),
        }
        split = "validation" if episode_id >= len(starts) - args.validation_episodes else "train"
        np.savez_compressed(
            args.output / split / f"episode_{episode_id:04d}.npz", **episode)
        outcomes["terminal" if episode["is_terminal"][-1] else "timeout"] += 1

    summary = dict(
        episodes=len(starts),
        train_episodes=len(starts) - args.validation_episodes,
        validation_episodes=args.validation_episodes,
        transitions=len(data["vector"]),
        **outcomes,
        source=str(args.source),
        source_arrays_chronologically_resorted=True,
        legacy_chunk_links_ignored=True,
        observation_dimensions=68,
        ppo_observation_dimensions=36,
        temporal_history=3,
        temporal_layout="state_and_L_t_then_L_t_minus_2_then_L_t_minus_1",
        action_policy_bounds=[-1.0, 1.0],
        action_physical_scale=ACTION_SCALE.tolist(),
        action_semantics=["vx_world", "vy_world", "vz_world", "yaw_rate"],
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
