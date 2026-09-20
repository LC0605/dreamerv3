#!/usr/bin/env python3
"""Read-only four-class reward-gradient diagnostic on frozen cached features."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from reward_head_cached_lr_sweep import REW_KEYS, ce, load_params, logits, predict


NAMES = ("success_terminal", "ordinary_nonterminal", "pre_collision",
         "collision_terminal")


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in tree.values()))


def cosine(a, b):
    return float(sum(jnp.vdot(a[k], b[k]) for k in a) /
                 (tree_norm(a) * tree_norm(b) + 1e-20))


def gradients(params, data, masks):
    result = {}
    for name, mask in masks.items():
        x = jnp.asarray(data["feature"][mask])
        y = jnp.asarray(data["twohot"][mask])
        result[name] = jax.grad(lambda p: ce(p, x, y).mean())(params)
    return result


def matrix(grads, keys=REW_KEYS):
    return {a: {b: cosine({k: grads[a][k] for k in keys},
                          {k: grads[b][k] for k in keys})
                for b in NAMES} for a in NAMES}


def reward_stats(values):
    return {"count": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()),
            "max": float(values.max()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90))}


def masks(data):
    terminal = data["is_terminal"].astype(bool)
    pre = data["stratum"] == 1
    collision = terminal & (data["reward"] < -50)
    success = terminal & (data["reward"] > 50)
    ordinary = ~terminal & ~pre
    result = dict(success_terminal=success, ordinary_nonterminal=ordinary,
                  pre_collision=pre, collision_terminal=collision)
    stacked = np.stack(list(result.values()))
    assert np.all(stacked.sum(0) <= 1), "Four classes must be mutually exclusive"
    return result


def time_buckets(data, params, collision_grad):
    distance = np.full(len(data["reward"]), -1, np.int32)
    for episode in np.unique(data["episode"]):
        idx = np.flatnonzero(data["episode"] == episode)
        terminal = idx[(data["is_terminal"][idx].astype(bool)) &
                       (data["reward"][idx] < -50)]
        if len(terminal):
            distance[idx] = data["timestep"][terminal[-1]] - data["timestep"][idx]
    definitions = {
        "t-1": lambda d: d == 1,
        "t-2_to_t-3": lambda d: (d >= 2) & (d <= 3),
        "t-4_to_t-5": lambda d: (d >= 4) & (d <= 5),
        "t-6_to_t-10": lambda d: (d >= 6) & (d <= 10),
        "earlier": lambda d: d > 10,
    }
    pred = predict(params, data["feature"])
    result = {}
    for name, fn in definitions.items():
        mask = fn(distance) & (data["stratum"] == 1)
        if not mask.any():
            result[name] = {"count": 0, "latent_probe_risk_score": None,
                            "probe_note": "No stored per-sample probe score."}
            continue
        grad = jax.grad(lambda p: ce(
            p, jnp.asarray(data["feature"][mask]),
            jnp.asarray(data["twohot"][mask])).mean())(params)
        result[name] = {
            **reward_stats(data["reward"][mask]),
            "predicted_reward_mean": float(pred[mask].mean()),
            "predicted_reward_std": float(pred[mask].std()),
            "cosine_with_collision_gradient": cosine(grad, collision_grad),
            "latent_probe_risk_score": None,
            "probe_note": "Existing probe artifact stores aggregate AUROC/AUPRC, not per-sample scores.",
        }
    return result


def diagnose(data, params):
    class_masks = masks(data)
    grads = gradients(params, data, class_masks)
    groups = {
        "shared_trunk_all": REW_KEYS[:3],
        "shared_mlp_linear": REW_KEYS[:2],
        "shared_trunk_norm": REW_KEYS[2:3],
        "twohot_logit_projection": REW_KEYS[3:],
        "twohot_logit_kernel": REW_KEYS[3:4],
        "twohot_logit_bias": REW_KEYS[4:5],
    }
    layerwise = {}
    for group, keys in groups.items():
        layerwise[group] = {
            "norms": {name: float(tree_norm({k: grads[name][k] for k in keys}))
                      for name in NAMES},
            "cosine": matrix(grads, keys),
        }
    return {
        "architecture": {
            "shared_mlp_layers": 1,
            "note": "Actual Reward Head has one shared 640->64 MLP layer plus RMS norm/SILU, then a 64->255 two-hot logits projection; no middle shared layers exist."},
        "class_overlap_count": int((np.stack(list(class_masks.values())).sum(0) > 1).sum()),
        "unclassified_count": int((np.stack(list(class_masks.values())).sum(0) == 0).sum()),
        "reward_distribution": {name: reward_stats(data["reward"][mask])
                                for name, mask in class_masks.items()},
        "gradient_norms": {name: float(tree_norm(grads[name])) for name in NAMES},
        "cosine_matrix": matrix(grads),
        "layerwise": layerwise,
        "pre_collision_time_buckets": time_buckets(
            data, params, grads["collision_terminal"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    params, _ = load_params(a.checkpoint / "agent.pkl")
    report = {"read_only": True, "optimizer_updates": 0, "splits": {}}
    for split, filename in (("training", "risk_training_features.npz"),
                            ("holdout", "holdout_features.npz")):
        with np.load(a.cache / filename) as src:
            data = {key: src[key] for key in src.files}
        report["splits"][split] = diagnose(data, params)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    for split, value in report["splits"].items():
        print(split, json.dumps({"norms": value["gradient_norms"],
                                 "cosines": value["cosine_matrix"]}))


if __name__ == "__main__":
    main()
