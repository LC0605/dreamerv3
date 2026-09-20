#!/usr/bin/env python3
"""Audit Pure-Dreamer calibration and route-lateral counterfactuals."""

import csv
import argparse
import json
import math
from functools import partial
from pathlib import Path

import elements
import numpy as np
from PIL import Image, ImageDraw
import ruamel.yaml as yaml

from dreamerv3.main import make_agent


ROOT = Path("outputs/dreamerv3")
INPUT = ROOT / "eval_stageS2_modeldiag60_pure_offset_pos0p3/trajectory_actions.csv"
SAME = ROOT / "diagnostics/same_state_policy_offset_pos0p3.csv"
OUT = ROOT / "diagnostics/pure_model_risk_offset_pos0p3.json"
CHECKPOINT = ROOT / "stageS1fB5_temporal3_online_pure_short/ckpt/20260825T223426F313624"
CONFIG = ROOT / "stageS1fB5_temporal3_online_pure_short/config.yaml"
CANDIDATE_LATERAL = np.asarray([-0.20, -0.35, -0.50, -0.65, -0.80, -1.00], np.float32)


def config(path=CONFIG):
    loader = yaml.YAML(typ="safe")
    defaults = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    saved = loader.load(path.read_text())
    return elements.Config(defaults["defaults"]).update(
        defaults["quadrotor"]).update(saved)


def mean(values, mask):
    values = np.asarray(values); mask = np.asarray(mask, bool)
    return float(values[mask].mean()) if mask.any() else None


def auc(target, score):
    target = np.asarray(target, bool); score = np.asarray(score)
    pos, neg = int(target.sum()), int((~target).sum())
    if not pos or not neg:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), np.float64); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[target].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(target, score):
    target = np.asarray(target, bool); score = np.asarray(score)
    if not target.any(): return None
    order = np.argsort(-score, kind="stable"); ranked = target[order]
    return float((np.cumsum(ranked) / np.arange(1, len(ranked) + 1))[ranked].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    with INPUT.open() as stream:
        raw = [row for row in csv.DictReader(stream) if int(row["episode"]) <= 60]
    with SAME.open() as stream:
        same = list(csv.DictReader(stream))
    assert len(raw) == len(same)
    episodes = [[row for row in raw if int(row["episode"]) == episode]
                for episode in range(1, 61)]
    T = max(len(rows) for rows in episodes)
    B = len(episodes)
    vectors = np.zeros((B, T, 68), np.float32)
    rewards = np.zeros((B, T), np.float32)
    first = np.ones((B, T), bool)
    last = np.ones((B, T), bool)
    terminal = np.zeros((B, T), bool)
    prev_action = np.zeros((B, T, 4), np.float32)
    current_action = np.zeros((B, T, 4), np.float32)
    valid = np.zeros((B, T), bool)
    clearance = np.full((B, T), 10.0, np.float32)
    outcome_collision = np.zeros(B, bool)
    selected = np.zeros(B, np.int32)
    candidates = np.zeros((B, len(CANDIDATE_LATERAL), 4), np.float32)
    flat_index = 0
    selected_policy_route_lateral = {name: [] for name in (
        "bc_only", "mixed", "pure_dreamer")}
    effective_candidate_lateral = []
    for batch, rows in enumerate(episodes):
        same_rows = same[flat_index:flat_index + len(rows)]
        flat_index += len(rows)
        for step, row in enumerate(rows):
            vectors[batch, step] = [float(row[f"vector_{i:02d}"]) for i in range(68)]
            rewards[batch, step] = float(row["reward"])
            first[batch, step] = bool(int(row["is_first"]))
            last[batch, step] = bool(int(row["is_last"]))
            terminal[batch, step] = bool(float(row["collision"]) or float(row["success"]))
            current_action[batch, step] = [float(row[key]) for key in (
                "action_forward", "action_lateral", "action_vertical", "action_yaw_rate")]
            valid[batch, step] = True
            clearance[batch, step] = float(row["minimum_clearance"])
        outcome_collision[batch] = any(float(row["collision"]) > .5 for row in rows)
        # CSV row t is (obs_t, action_t, reward_from_previous_transition).
        # RSSM posterior at obs_t must consume action_(t-1), not action_t.
        # The first posterior is reset, so its previous action is zero.
        if len(rows) > 1:
            prev_action[batch, 1:len(rows)] = current_action[batch, :len(rows) - 1]
        usable = np.arange(1, max(2, len(rows) - 1))
        selected[batch] = int(usable[np.argmin(clearance[batch, usable])])
        row = rows[selected[batch]]
        same_row = same_rows[selected[batch]]
        dx = float(row["scene_goal_x"]) - float(row["scene_start_x"])
        dy = float(row["scene_goal_y"]) - float(row["scene_start_y"])
        norm = max(math.hypot(dx, dy), 1e-6); dx /= norm; dy /= norm
        pure = np.asarray([float(same_row[f"pure_dreamer_{key}"])
                           for key in ("vx", "vy", "vz", "yaw_rate")], np.float32)
        route_forward = pure[0] * dx + pure[1] * dy
        for model in selected_policy_route_lateral:
            action = np.asarray([float(same_row[f"{model}_{key}"])
                                 for key in ("vx", "vy")])
            selected_policy_route_lateral[model].append(
                float(-action[0] * dy + action[1] * dx))
        for index, route_lateral in enumerate(CANDIDATE_LATERAL):
            candidates[batch, index] = np.asarray([
                route_forward * dx - route_lateral * dy,
                route_forward * dy + route_lateral * dx,
                pure[2], pure[3]], np.float32)
        candidates[batch] = np.clip(candidates[batch], -1, 1)
        effective_candidate_lateral.append(
            (-candidates[batch, :, 0] * dy
             + candidates[batch, :, 1] * dx).tolist())

    data = {
        "vector": vectors, "reward": rewards, "is_first": first,
        "is_last": last, "is_terminal": terminal,
        "teacher": np.zeros((B, T), bool),
        "action_policy_action": np.zeros((B, T, 4), np.float32),
        "diagnostic_prev_action": prev_action,
        "diagnostic_state_index": selected,
        "diagnostic_candidate_action": candidates,
    }
    agent = make_agent(config(args.config))
    elements.checkpoint.load(
        args.checkpoint, {"agent": partial(
            agent.load, regex="^(?!(bcopt|opt)/).*")})
    outputs = agent.diagnose(data)
    pred_reward = np.asarray(outputs["reward_pred"])
    pred_dense_reward = np.asarray(outputs.get("dense_reward_pred", pred_reward))
    pred_terminal_reward = np.asarray(outputs.get(
        "terminal_reward_pred", np.zeros_like(pred_reward)))
    pred_continue = np.asarray(outputs["continue_pred"])
    pred_success = np.asarray(outputs.get(
        "outcome_success_prob", np.zeros_like(pred_reward)))
    pred_collision = np.asarray(outputs.get(
        "outcome_collision_prob", np.zeros_like(pred_reward)))
    pred_vector = np.asarray(outputs["vector_pred"])
    actor_action_mean = np.asarray(outputs["actor_action_mean"])
    near = valid & (clearance < .70)
    ordinary = valid & (clearance >= .70) & ~terminal
    collision_steps = valid & outcome_collision[:, None]
    safe_steps = valid & ~outcome_collision[:, None]
    pre5 = np.zeros_like(valid)
    for batch, rows in enumerate(episodes):
        if outcome_collision[batch]:
            pre5[batch, max(0, len(rows) - 6):len(rows) - 1] = True
    true_continue = (~terminal).astype(np.float32) * (1 - 1 / 333)
    reward_abs_error = np.abs(pred_reward - rewards)
    terminal_component = np.where(terminal & outcome_collision[:, None], -100.0,
                                  np.where(terminal, 100.0, 0.0))
    dense_rewards = rewards - terminal_component
    dense_reward_abs_error = np.abs(pred_dense_reward - dense_rewards)
    continue_abs_error = np.abs(pred_continue - true_continue)
    vector_sq_error = np.square(pred_vector - vectors)
    lidar_abs_error = np.abs(pred_vector[..., 20:36] - vectors[..., 20:36]).mean(-1)

    report = {
        "episodes": B,
        "collision_episodes": int(outcome_collision.sum()),
        "selected_state_clearance_mean": float(
            clearance[np.arange(B), selected].mean()),
        "calibration": {},
        "counterfactual_route_lateral_candidates": CANDIDATE_LATERAL.tolist(),
        "counterfactual_effective_route_lateral_mean":
            np.asarray(effective_candidate_lateral).mean(0).tolist(),
        "selected_policy_route_lateral_mean": {
            name: float(np.mean(values))
            for name, values in selected_policy_route_lateral.items()},
        "counterfactual": {},
    }
    reference_action = np.zeros_like(actor_action_mean)
    flat_index = 0
    for batch, rows in enumerate(episodes):
        same_rows = same[flat_index:flat_index + len(rows)]
        flat_index += len(rows)
        for step, row in enumerate(same_rows):
            reference_action[batch, step] = [float(row[f"pure_dreamer_{key}"])
                                              for key in ("vx", "vy", "vz", "yaw_rate")]
    action_error = np.abs(actor_action_mean - reference_action)
    report["policy_interface_drift"] = {
        "overall_mae": float(action_error[valid].mean()),
        "per_dimension_mae_vx_vy_vz_yaw": action_error[valid].mean(0).tolist(),
        "near_obstacle_mae": float(action_error[near].mean()),
        "collision_before_5_mae": float(action_error[pre5].mean()),
    }
    terminal_rows = terminal & valid
    terminal_collision = np.broadcast_to(outcome_collision[:, None], terminal.shape)[terminal_rows]
    terminal_score = pred_collision[terminal_rows]
    terminal_pred = terminal_score >= .5
    tp = int((terminal_pred & terminal_collision).sum())
    fp = int((terminal_pred & ~terminal_collision).sum())
    fn = int((~terminal_pred & terminal_collision).sum())
    tn = int((~terminal_pred & ~terminal_collision).sum())
    report["conditional_outcome"] = {
        "collision_auroc": auc(terminal_collision, terminal_score),
        "collision_auprc": average_precision(terminal_collision, terminal_score),
        "collision_precision": tp / max(tp + fp, 1),
        "collision_recall": tp / max(tp + fn, 1),
        "success_auroc": auc(~terminal_collision, 1 - terminal_score),
        "success_auprc": average_precision(~terminal_collision, 1 - terminal_score),
        "success_precision": tn / max(tn + fn, 1),
        "success_recall": tn / max(tn + fp, 1),
        "confusion_success_collision": [[tn, fp], [fn, tp]],
    }
    groups = {"all": valid, "near_obstacle": near, "ordinary": ordinary,
              "safe_episode": safe_steps, "collision_episode": collision_steps,
              "collision_before_5": pre5,
              "success_terminal": terminal & ~outcome_collision[:, None],
              "collision_terminal": terminal & outcome_collision[:, None]}
    for name, mask in groups.items():
        report["calibration"][name] = {
            "states": int(mask.sum()),
            "real_reward": mean(rewards, mask),
            "predicted_reward": mean(pred_reward, mask),
            "reward_mae": mean(reward_abs_error, mask),
            "real_dense_reward": mean(dense_rewards, mask),
            "predicted_dense_reward": mean(pred_dense_reward, mask),
            "dense_reward_mae": mean(dense_reward_abs_error, mask),
            "predicted_terminal_reward": mean(pred_terminal_reward, mask),
            "predicted_terminal_probability": mean(1 - pred_continue, mask),
            "predicted_success_given_terminal": mean(pred_success, mask),
            "predicted_collision_given_terminal": mean(pred_collision, mask),
            "real_continue": mean(true_continue, mask),
            "predicted_continue": mean(pred_continue, mask),
            "continue_mae": mean(continue_abs_error, mask),
            "vector_rmse": math.sqrt(mean(vector_sq_error.mean(-1), mask)),
            "lidar_mae": mean(lidar_abs_error, mask),
            "dynamics_loss": mean(outputs["dyn_loss"], mask),
            "representation_loss": mean(outputs["rep_loss"], mask),
            "value": mean(outputs["value_pred"], mask),
        }
    for horizon in (1, 3, 5, 15):
        report["counterfactual"][f"horizon_{horizon}"] = {}
        for index, lateral in enumerate(CANDIDATE_LATERAL):
            report["counterfactual"][f"horizon_{horizon}"][str(float(lateral))] = {
                "predicted_return": float(np.asarray(outputs[f"h{horizon}_return"])[:, index].mean()),
                "lambda_return": float(np.asarray(outputs[f"h{horizon}_lambda_return"])[:, index].mean()),
                "value": float(np.asarray(outputs[f"h{horizon}_value"])[:, index].mean()),
                "predicted_continue": float(np.asarray(outputs[f"h{horizon}_continue"])[:, index].mean()),
                "predicted_risk": float(np.asarray(outputs[f"h{horizon}_risk"])[:, index].mean()),
                "predicted_collision_given_terminal": float(np.asarray(
                    outputs[f"h{horizon}_collision_prob"])[:, index].mean()),
                "predicted_lidar_min": float(np.asarray(outputs[f"h{horizon}_lidar_min"])[:, index].mean()),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    # Compact return/value plot without adding a plotting dependency.
    image = Image.new("RGB", (900, 500), "white"); draw = ImageDraw.Draw(image)
    draw.rectangle((70, 40, 850, 440), outline="black", width=2)
    colors = {1: (60, 150, 60), 3: (40, 100, 210), 5: (180, 90, 30), 15: (150, 40, 150)}
    all_values = [report["counterfactual"][f"horizon_{h}"][str(float(v))]["lambda_return"]
                  for h in colors for v in CANDIDATE_LATERAL]
    lo, hi = min(all_values), max(all_values); span = max(hi - lo, 1e-6)
    xs = [90 + i * 740 / (len(CANDIDATE_LATERAL) - 1)
          for i in range(len(CANDIDATE_LATERAL))]
    for horizon, color in colors.items():
        vals = [report["counterfactual"][f"horizon_{horizon}"][str(float(v))]["lambda_return"]
                for v in CANDIDATE_LATERAL]
        ys = [420 - (value - lo) / span * 350 for value in vals]
        draw.line(list(zip(xs, ys)), fill=color, width=3)
        draw.text((700, 55 + 20 * list(colors).index(horizon)),
                  f"H={horizon}", fill=color)
    for x, value in zip(xs, CANDIDATE_LATERAL):
        draw.text((x - 18, 450), f"{value:.2f}", fill="black")
    draw.text((310, 475), "route-lateral policy action", fill="black")
    draw.text((15, 15), "Counterfactual imagined lambda return", fill="black")
    image.save(args.output.with_suffix(".png"))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
