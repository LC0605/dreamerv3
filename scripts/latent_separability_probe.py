#!/usr/bin/env python3
"""Linear and nearest-neighbor probes on frozen Dreamer representations."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist

from dreamerv3.main import make_agent


NAMES = ("ordinary", "pre_collision", "collision_terminal")


def cfg(path):
    loader = yaml.YAML(typ="safe")
    configs = loader.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    saved = loader.load(path.read_text())
    return elements.Config(configs["defaults"]).update(
        configs["quadrotor"]).update(saved)


def risk_episodes(root):
    episodes = []
    for path in sorted((root / "episodes").glob("*.npz")):
        with np.load(path) as source:
            d = {key: source[key] for key in source.files}
        n = len(d["reward"]); labels = np.full(n, -1, np.int32)
        labels[d["pre_collision"]] = 1
        labels[d["collision_terminal"]] = 2
        ordinary = np.flatnonzero(
            (~d["is_first"]) & (~d["is_last"])
            & (~d["pre_collision"]) & (d["minimum_clearance"] >= .7))
        if len(ordinary):
            take = ordinary[np.linspace(0, len(ordinary) - 1,
                                        min(20, len(ordinary))).astype(int)]
            labels[take] = 0
        episodes.append({**d, "labels": labels})
    return episodes


def holdout_episodes(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for eid in range(1, 61):
        part = [row for row in rows if int(row["episode"]) == eid]
        n = len(part); labels = np.full(n, -1, np.int32)
        collision = any(float(row["collision"]) > .5 for row in part)
        if collision:
            labels[n - 1] = 2
            labels[max(1, n - 11):n - 1] = 1
        ordinary = [i for i, row in enumerate(part)
                    if i > 0 and not int(row["is_last"])
                    and labels[i] < 0 and float(row["minimum_clearance"]) >= .7]
        if ordinary:
            take = np.asarray(ordinary)[np.linspace(
                0, len(ordinary) - 1, min(20, len(ordinary))).astype(int)]
            labels[take] = 0
        action = np.asarray([[float(row[key]) for key in (
            "action_forward", "action_lateral", "action_vertical", "action_yaw_rate")]
            for row in part], np.float32)
        result.append({
            "vector": np.asarray([[float(row[f"vector_{i:02d}"]) for i in range(68)]
                                  for row in part], np.float32),
            "reward": np.asarray([float(row["reward"]) for row in part], np.float32),
            "is_first": np.asarray([bool(int(row["is_first"])) for row in part]),
            "is_last": np.asarray([bool(int(row["is_last"])) for row in part]),
            "is_terminal": np.asarray([
                bool(float(row["collision"]) or float(row["success"])) for row in part]),
            "action": action, "labels": labels,
        })
    return result


def extract(agent, episodes, batch_size=10):
    collected = {key: [] for key in ("encoder", "deter", "stoch", "concat")}
    ys = []
    for start in range(0, len(episodes), batch_size):
        group = episodes[start:start + batch_size]
        b, t = len(group), max(len(x["reward"]) for x in group)
        vector = np.zeros((b, t, 68), np.float32)
        reward = np.zeros((b, t), np.float32)
        first = np.ones((b, t), bool); last = np.ones((b, t), bool)
        terminal = np.zeros((b, t), bool); prev = np.zeros((b, t, 4), np.float32)
        labels = np.full((b, t), -1, np.int32); valid = np.zeros((b, t), bool)
        for i, ep in enumerate(group):
            n = len(ep["reward"])
            vector[i, :n] = ep["vector"]; reward[i, :n] = ep["reward"]
            first[i, :n] = ep["is_first"]; last[i, :n] = ep["is_last"]
            terminal[i, :n] = ep["is_terminal"]; labels[i, :n] = ep["labels"]
            valid[i, :n] = True
            if n > 1:
                prev[i, 1:n] = ep["action"][:n - 1]
        data = {
            "vector": vector, "reward": reward, "is_first": first,
            "is_last": last, "is_terminal": terminal,
            "teacher": np.zeros((b, t), bool),
            "action_policy_action": np.zeros((b, t, 4), np.float32),
            "loss_mask": valid,
            "diagnostic_prev_action": prev,
            "diagnostic_state_index": np.zeros(b, np.int32),
            "diagnostic_candidate_action": np.zeros((b, 6, 4), np.float32),
        }
        out = agent.diagnose(data)
        mask = labels >= 0
        enc = np.asarray(out["encoder_token"]).reshape(b, t, -1)[mask]
        deter = np.asarray(out["rssm_deter"]).reshape(b, t, -1)[mask]
        stoch = np.asarray(out["rssm_stoch"]).reshape(b, t, -1)[mask]
        collected["encoder"].append(enc)
        collected["deter"].append(deter)
        collected["stoch"].append(stoch)
        collected["concat"].append(np.concatenate([deter, stoch], -1))
        ys.append(labels[mask])
    return {key: np.concatenate(value) for key, value in collected.items()}, np.concatenate(ys)


def auroc(y, score):
    y = np.asarray(y, bool); pos, neg = y.sum(), (~y).sum()
    if not pos or not neg: return None
    order = np.argsort(score); ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(y, score):
    y = np.asarray(y, bool); pos = y.sum()
    if not pos: return None
    order = np.argsort(-score); ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(y) + 1)
    return float((precision * ranked).sum() / pos)


def linear_probe(xtr, ytr, xte, yte):
    mean, std = xtr.mean(0), xtr.std(0) + 1e-4
    a = np.clip((xtr - mean) / std, -10, 10)
    b = np.clip((xte - mean) / std, -10, 10)
    scores = []
    for cls in range(3):
        y = (ytr == cls).astype(float)
        weight = np.where(y > 0, .5 / max(y.mean(), 1e-6),
                          .5 / max(1 - y.mean(), 1e-6))
        def objective(w):
            z = a @ w[:-1] + w[-1]
            loss = np.average(np.logaddexp(0, z) - y * z, weights=weight)
            gradz = weight * (expit(z) - y) / weight.sum()
            grad = np.r_[a.T @ gradz, gradz.sum()]
            loss += 1e-4 * np.square(w[:-1]).sum()
            grad[:-1] += 2e-4 * w[:-1]
            return loss, grad
        fit = minimize(objective, np.zeros(a.shape[1] + 1), jac=True,
                       method="L-BFGS-B", options={"maxiter": 250})
        scores.append(b @ fit.x[:-1] + fit.x[-1])
    scores = np.stack(scores, 1)
    metrics = {NAMES[c]: {"auroc": auroc(yte == c, scores[:, c]),
                          "auprc": auprc(yte == c, scores[:, c])}
               for c in range(3)}
    metrics["accuracy"] = float((scores.argmax(1) == yte).mean())

    # Standardized nearest-neighbor geometry; subsample ordinary train points
    # only for bounded diagnostic memory.
    keep = np.concatenate([np.flatnonzero(ytr == c)[:300] for c in range(3)])
    dist = cdist(b, a[keep], metric="euclidean")
    nearest = ytr[keep][dist.argmin(1)]
    metrics["knn1_accuracy"] = float((nearest == yte).mean())
    metrics["nearest_distance"] = {}
    for c, name in enumerate(NAMES):
        mask = yte == c
        same = dist[mask][:, ytr[keep] == c].min(1)
        other = dist[mask][:, ytr[keep] != c].min(1)
        metrics["nearest_distance"][name] = {
            "same_class_mean": float(same.mean()),
            "other_class_mean": float(other.mean()),
            "same_closer_fraction": float((same < other).mean())}
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--risk-data", type=Path, required=True)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    agent = make_agent(cfg(a.config))
    elements.checkpoint.load(a.checkpoint, {"agent": partial(
        agent.load, regex="^(?!(bcopt|opt)/).*")})
    xtr, ytr = extract(agent, risk_episodes(a.risk_data))
    xte, yte = extract(agent, holdout_episodes(a.holdout_csv))
    report = {
        "train_counts": {NAMES[c]: int((ytr == c).sum()) for c in range(3)},
        "holdout_counts": {NAMES[c]: int((yte == c).sum()) for c in range(3)},
        "representations": {key: linear_probe(xtr[key], ytr, xte[key], yte)
                            for key in xtr},
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
