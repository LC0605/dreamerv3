#!/usr/bin/env python3
"""Independent frozen-latent collision and success terminal hazards."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from reward_decomposition_r1_offline import arrays, auc, average_precision, decompose


def goal_distances(risk_data, holdout_csv):
    train = []
    for path in sorted((risk_data / "episodes").glob("*.npz")):
        with np.load(path) as src:
            train.append(np.linalg.norm(src["vector"][:, 13:16], axis=-1))
    with holdout_csv.open() as stream: rows = list(csv.DictReader(stream))
    hold = []
    for episode in range(1, 61):
        part = [r for r in rows if int(r["episode"]) == episode]
        hold.append(np.asarray([float(r["distance"]) for r in part]))
    return np.concatenate(train), np.concatenate(hold)


def fit_hazard(feature, positive, negative, l2=1e-4):
    selected = positive | negative; x = feature[selected]; y = positive[selected]
    mean, std = x.mean(0), x.std(0) + 1e-4
    z = np.clip((x - mean) / std, -10, 10)
    weight = np.where(y, .5 / y.mean(), .5 / (~y).mean())

    def objective(w):
        score = z @ w[:-1] + w[-1]
        value = np.average(np.logaddexp(0, score) - y * score, weights=weight)
        gradz = weight * (expit(score) - y) / weight.sum()
        grad = np.r_[z.T @ gradz, gradz.sum()]
        value += l2 * np.square(w[:-1]).sum(); grad[:-1] += 2 * l2 * w[:-1]
        return value, grad

    fit = minimize(objective, np.zeros(z.shape[1] + 1), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500})
    prior = float(y.mean())
    correction = np.log(prior / (1 - prior))  # sampler fitting prior was 0.5
    train_prob = expit(z @ fit.x[:-1] + fit.x[-1] + correction)
    # Threshold is selected from training only by maximum F1.
    candidates = np.unique(train_prob)
    best = (0.0, .5)
    for threshold in candidates:
        pred = train_prob >= threshold
        tp = (y & pred).sum(); fp = ((~y) & pred).sum(); fn = (y & ~pred).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]: best = (float(f1), float(threshold))
    return {"weight": fit.x[:-1], "bias": fit.x[-1], "mean": mean, "std": std,
            "prior": prior, "prior_logit_correction": correction,
            "threshold": best[1], "training_f1": best[0],
            "fit_success": bool(fit.success), "fit_iterations": int(fit.nit)}


def probability(model, feature):
    x = np.clip((feature - model["mean"]) / model["std"], -10, 10)
    return expit(x @ model["weight"] + model["bias"] + model["prior_logit_correction"])


def distribution(prob):
    return {"mean": float(prob.mean()), "std": float(prob.std()),
            "p50": float(np.percentile(prob, 50)), "p90": float(np.percentile(prob, 90)),
            "p99": float(np.percentile(prob, 99))}


def primary_metrics(prob, positive, negative, threshold):
    mask = positive | negative; y = positive[mask]; p = prob[mask]; pred = p >= threshold
    tp = int((y & pred).sum()); fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())
    return {"positive_count": int(y.sum()), "negative_count": int((~y).sum()),
            "auroc": auc(y, p), "auprc": average_precision(y, p),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "confusion_real_negative_positive": [[tn, fp], [fn, tp]],
            "positive_probability": distribution(p[y]),
            "negative_probability": distribution(p[~y])}


def masks(data, goal):
    _, outcome, _, _ = decompose(data)
    nonterminal = outcome == 0; pre = data["stratum"] == 1
    near_obstacle = nonterminal & data["near"].astype(bool)
    near_goal = nonterminal & (goal < .7)
    return {"nonterminal": nonterminal, "ordinary": nonterminal & (data["stratum"] == 0),
            "pre": pre, "near_obstacle": near_obstacle, "near_goal": near_goal,
            "collision": outcome == 2, "success": outcome == 1,
            "collision_hard": nonterminal & (pre | data["near"].astype(bool))}


def false_positive_groups(prob, groups, threshold):
    result = {}
    for name in ("ordinary", "pre", "near_obstacle", "near_goal"):
        mask = groups[name]
        result[name] = {"count": int(mask.sum()), "mean_probability": float(prob[mask].mean()),
                        "false_positive_rate": float((prob[mask] >= threshold).mean())}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--risk-data", type=Path, required=True)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    train = arrays(a.cache / "risk_training_features.npz")
    hold = arrays(a.cache / "holdout_features.npz")
    train_goal, hold_goal = goal_distances(a.risk_data, a.holdout_csv)
    tm, hm = masks(train, train_goal), masks(hold, hold_goal)
    definitions = {
        "collision_hazard": (tm["collision"], tm["collision_hard"],
                             hm["collision"], hm["collision_hard"]),
        "success_hazard": (tm["success"], tm["near_goal"],
                           hm["success"], hm["near_goal"])}
    report = {"phase": "cause_specific_hazards", "dreamer_network_updates": 0,
              "training_seed": 52000, "holdout_seed": 35700, "hazards": {}}
    saved = {}
    for name, (tpos, tneg, hpos, hneg) in definitions.items():
        model = fit_hazard(train["feature"], tpos, tneg)
        prob = probability(model, hold["feature"])
        report["hazards"][name] = {
            "training_positive": int(tpos.sum()), "training_negative": int(tneg.sum()),
            "natural_training_prior": model["prior"],
            "prior_logit_correction": model["prior_logit_correction"],
            "training_selected_threshold": model["threshold"],
            "training_f1": model["training_f1"],
            "holdout_primary": primary_metrics(prob, hpos, hneg, model["threshold"]),
            "holdout_false_positive_groups": false_positive_groups(prob, hm, model["threshold"]),
            "terminal_recall": float((prob[hpos] >= model["threshold"]).mean()),
            "terminal_mean_probability": float(prob[hpos].mean())}
        for key in ("weight", "bias", "mean", "std"):
            saved[f"{name}/{key}"] = np.asarray(model[key])
        saved[f"{name}/prior_logit_correction"] = np.asarray(model["prior_logit_correction"])
        saved[f"{name}/threshold"] = np.asarray(model["threshold"])
        print(name, json.dumps(report["hazards"][name]))
    np.savez_compressed(a.output / "cause_specific_hazards.npz", **saved)
    (a.output / "hazard_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
