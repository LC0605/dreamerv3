#!/usr/bin/env python3
"""Five train/hold-out linear probes on frozen Original Pure [h,z]."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from reward_decomposition_r1_offline import arrays, auc, average_precision, decompose


def geometry(cache, risk_data, holdout_csv):
    train_goal = []
    for path in sorted((risk_data / "episodes").glob("*.npz")):
        with np.load(path) as src:
            train_goal.append(np.linalg.norm(src["vector"][:, 13:16], axis=-1))
    with holdout_csv.open() as stream: rows = list(csv.DictReader(stream))
    hold_goal = []
    for episode in range(1, 61):
        part = [r for r in rows if int(r["episode"]) == episode]
        hold_goal.append(np.asarray([float(r["distance"]) for r in part]))
    return np.concatenate(train_goal), np.concatenate(hold_goal)


def score_stats(score):
    return {"count": int(len(score)), "mean": float(score.mean()),
            "std": float(score.std()), "min": float(score.min()),
            "max": float(score.max()), "p10": float(np.percentile(score, 10)),
            "p50": float(np.percentile(score, 50)),
            "p90": float(np.percentile(score, 90))}


def fit_probe(xtr, ytr, xte, yte):
    mean = xtr.mean(0); std = xtr.std(0) + 1e-4
    a = np.clip((xtr - mean) / std, -10, 10)
    b = np.clip((xte - mean) / std, -10, 10)
    y = ytr.astype(float)
    weights = np.where(ytr, .5 / ytr.mean(), .5 / (~ytr).mean())

    def objective(w):
        z = a @ w[:-1] + w[-1]
        loss = np.average(np.logaddexp(0, z) - y * z, weights=weights)
        gradz = weights * (expit(z) - y) / weights.sum()
        grad = np.r_[a.T @ gradz, gradz.sum()]
        loss += 1e-4 * np.square(w[:-1]).sum()
        grad[:-1] += 2e-4 * w[:-1]
        return loss, grad

    fit = minimize(objective, np.zeros(a.shape[1] + 1), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500})
    score = b @ fit.x[:-1] + fit.x[-1]
    pred = score >= 0
    tp = int((yte & pred).sum()); fp = int((~yte & pred).sum())
    fn = int((yte & ~pred).sum()); tn = int((~yte & ~pred).sum())
    raw_pos = xtr[ytr].mean(0); raw_neg = xtr[~ytr].mean(0)
    std_pos = a[ytr].mean(0); std_neg = a[~ytr].mean(0)
    return {
        "train_positive": int(ytr.sum()), "train_negative": int((~ytr).sum()),
        "holdout_positive": int(yte.sum()), "holdout_negative": int((~yte).sum()),
        "auroc": auc(yte, score), "auprc": average_precision(yte, score),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "confusion_real_negative_positive": [[tn, fp], [fn, tp]],
        "score_distribution": {"positive": score_stats(score[yte]),
                               "negative": score_stats(score[~yte])},
        "centroid_distance": {
            "raw_euclidean": float(np.linalg.norm(raw_pos - raw_neg)),
            "train_standardized_euclidean": float(np.linalg.norm(std_pos - std_neg)),
            "train_standardized_per_sqrt_dim": float(
                np.linalg.norm(std_pos - std_neg) / np.sqrt(a.shape[1]))},
        "fit_success": bool(fit.success), "fit_iterations": int(fit.nit)}


def masks(data, goal_distance):
    _, outcome, _, _ = decompose(data)
    nonterminal = outcome == 0
    success = outcome == 1
    collision = outcome == 2
    pre = data["stratum"] == 1
    near_obstacle = data["near"].astype(bool)
    # Goal success condition is distance <0.3. Use <0.7 nonterminal rows as
    # hard negatives covering the final approach without including terminal.
    near_goal = nonterminal & (goal_distance < .7)
    collision_hard = nonterminal & (pre | near_obstacle)
    return {"nonterminal": nonterminal, "success": success, "collision": collision,
            "pre": pre, "near_obstacle": near_obstacle, "near_goal": near_goal,
            "collision_hard": collision_hard, "terminal": success | collision}


def select(feature, mask, positive):
    return feature[mask], positive[mask]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--risk-data", type=Path, required=True)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    train = arrays(a.cache / "risk_training_features.npz")
    hold = arrays(a.cache / "holdout_features.npz")
    train_goal, hold_goal = geometry(a.cache, a.risk_data, a.holdout_csv)
    assert len(train_goal) == len(train["feature"])
    assert len(hold_goal) == len(hold["feature"])
    tm = masks(train, train_goal); hm = masks(hold, hold_goal)
    definitions = {
        "collision_vs_all_nonterminal": (
            tm["collision"] | tm["nonterminal"], tm["collision"],
            hm["collision"] | hm["nonterminal"], hm["collision"]),
        "collision_vs_hard_negative": (
            tm["collision"] | tm["collision_hard"], tm["collision"],
            hm["collision"] | hm["collision_hard"], hm["collision"]),
        "success_vs_all_nonterminal": (
            tm["success"] | tm["nonterminal"], tm["success"],
            hm["success"] | hm["nonterminal"], hm["success"]),
        "success_vs_near_goal": (
            tm["success"] | tm["near_goal"], tm["success"],
            hm["success"] | hm["near_goal"], hm["success"]),
        "all_terminal_vs_all_nonterminal": (
            tm["terminal"] | tm["nonterminal"], tm["terminal"],
            hm["terminal"] | hm["nonterminal"], hm["terminal"]),
    }
    report = {"training_seed_pool": 52000, "holdout_seed_pool": 35700,
              "feature": "Original Pure concat(h,z), dim=640", "network_updates": 0,
              "near_goal_definition": "nonterminal goal distance <0.7m",
              "collision_hard_negative_definition": "nonterminal and (pre_collision or clearance<0.7m)",
              "probes": {}}
    for name, (trmask, trpositive, hmask, hpositive) in definitions.items():
        xtr, ytr = select(train["feature"], trmask, trpositive)
        xte, yte = select(hold["feature"], hmask, hpositive)
        report["probes"][name] = fit_probe(xtr, ytr, xte, yte)
        print(name, json.dumps(report["probes"][name]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
