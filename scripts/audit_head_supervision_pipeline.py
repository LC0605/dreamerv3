#!/usr/bin/env python3
"""Quantify terminal supervision weight and verify RSSM/reward indexing."""

import argparse
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent
from embodied.run.bc_distill import _episodes, _stream


def load_config(path):
    loader = yaml.YAML(typ="safe")
    configs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    saved = loader.load(Path(path).read_text())
    return elements.Config(configs["defaults"]).update(
        configs["quadrotor"]).update(saved)


def diagnostic_fields(batch):
    b, t = batch["reward"].shape
    prev = np.zeros_like(batch["action"])
    prev[:, 1:] = batch["action"][:, :-1]
    indices = np.maximum(batch["loss_mask"].sum(1).astype(np.int32) - 1, 0)
    candidates = np.zeros((b, 6, 4), np.float32)
    return {
        **{key: batch[key] for key in (
            "vector", "reward", "is_first", "is_last", "is_terminal",
            "teacher", "action_policy_action", "loss_mask")},
        "diagnostic_prev_action": prev,
        "diagnostic_state_index": indices,
        "diagnostic_candidate_action": candidates,
    }


def scalar(outputs, key):
    return float(np.asarray(outputs[key]).reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collision-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = _episodes(args.dataset / "train", 301)
    stream = _stream(episodes, 10, seed=0, shuffle=True, stratified=True)
    fractions = []
    batches = []
    for _ in range(100):
        batch = next(stream)
        valid = batch["loss_mask"]
        collision = valid & batch["is_terminal"] & (batch["reward"] < -50)
        fractions.append(float(collision.sum() / valid.sum()))
        if not batches:
            batches.append(batch)

    agent = make_agent(load_config(args.config))
    elements.checkpoint.load(
        args.checkpoint,
        {"agent": partial(agent.load, regex="^(?!(bcopt|opt)/).*")})
    batch_out = agent.diagnose(diagnostic_fields(batches[0]))

    collision_data = json.loads(args.collision_json.read_text())
    trans = collision_data["transitions"]
    rows = len(trans) + 1
    vector = np.zeros((1, rows, 68), np.float32)
    reward = np.zeros((1, rows), np.float32)
    first = np.zeros((1, rows), bool); first[0, 0] = True
    last = np.zeros((1, rows), bool)
    terminal = np.zeros((1, rows), bool)
    current = np.zeros((1, rows, 4), np.float32)
    vector[0, 0] = np.asarray(trans[0]["obs_t"], np.float32)
    for index, transition in enumerate(trans):
        row = index + 1
        vector[0, row] = np.asarray(transition["next_obs"], np.float32)
        reward[0, row] = transition["reward_t"]
        last[0, row] = transition["is_last"]
        terminal[0, row] = transition["is_terminal"]
        current[0, index] = np.asarray(transition["action_t"], np.float32)
    prev = np.zeros_like(current)
    prev[:, 1:] = current[:, :-1]
    trace_data = {
        "vector": vector, "reward": reward, "is_first": first,
        "is_last": last, "is_terminal": terminal,
        "teacher": np.zeros((1, rows), bool),
        "action_policy_action": np.zeros((1, rows, 4), np.float32),
        "loss_mask": np.ones((1, rows), bool),
        "diagnostic_prev_action": prev,
        "diagnostic_state_index": np.asarray([rows - 1], np.int32),
        "diagnostic_candidate_action": np.zeros((1, 6, 4), np.float32),
    }
    trace_out = agent.diagnose(trace_data)
    deter = np.asarray(trace_out["rssm_deter"])[0]
    stoch = np.asarray(trace_out["rssm_stoch"])[0]
    reward_pred = np.asarray(trace_out["reward_pred"])[0]
    continue_pred = np.asarray(trace_out["continue_pred"])[0]
    trace = []
    for row in range(max(0, rows - 10), rows):
        trace.append({
            "feature_row": row,
            "transition_action_index_consumed": row - 1 if row else None,
            "previous_action": prev[0, row].tolist(),
            "target_reward": float(reward[0, row]),
            "target_terminal": bool(terminal[0, row]),
            "predicted_reward": float(reward_pred[row]),
            "predicted_continue": float(continue_pred[row]),
            "deter_norm": float(np.linalg.norm(deter[row])),
            "stoch_norm": float(np.linalg.norm(stoch[row])),
            "deter_delta": float(np.linalg.norm(deter[row] - deter[row - 1])) if row else 0.0,
            "stoch_delta": float(np.linalg.norm(stoch[row] - stoch[row - 1])) if row else 0.0,
        })

    # symexp_twohot uses raw reward values with bins spaced uniformly in
    # symlog coordinates; it does not symlog the target before interpolation.
    half = np.linspace(-20, 0, 128, dtype=np.float32)
    half = np.sign(half) * np.expm1(np.abs(half))
    bins = np.concatenate([half, -half[:-1][::-1]])
    target = float(trans[-1]["reward_t"])
    below = int(np.clip(np.searchsorted(bins, target, side="right") - 1, 0, 254))
    above = int(np.clip(np.searchsorted(bins, target, side="left"), 0, 254))
    if below == above:
        weights = [1.0, 0.0]
    else:
        total = abs(float(bins[below]) - target) + abs(float(bins[above]) - target)
        weights = [abs(float(bins[above]) - target) / total,
                   abs(float(bins[below]) - target) / total]

    result = {
        "batch_terminal_fraction_100": {
            "mean": float(np.mean(fractions)), "min": float(np.min(fractions)),
            "max": float(np.max(fractions)), "percent": float(100 * np.mean(fractions))},
        "representative_batch": {
            key: scalar(batch_out, key) for key in (
                "valid_count", "collision_terminal_count",
                "collision_terminal_fraction", "reward_terminal_loss_mean",
                "reward_other_loss_mean", "continue_terminal_loss_mean",
                "continue_other_loss_mean", "reward_terminal_grad_norm",
                "reward_other_grad_norm", "continue_terminal_grad_norm",
                "continue_other_grad_norm", "reward_terminal_weighted_grad_norm",
                "continue_terminal_weighted_grad_norm")},
        "reward_transform": {
            "name": "symexp_twohot", "raw_target": target,
            "target_is_symlogged_before_loss": False,
            "bracketing_bins_raw": [float(bins[below]), float(bins[above])],
            "twohot_weights": weights,
            "reported_prediction_space": "raw reward (probability-weighted raw bins)"},
        "continue_transform": {
            "name": "binary Bernoulli NLL", "target": "1-is_terminal",
            "contdisc": False},
        "collision_feature_trace": trace,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
