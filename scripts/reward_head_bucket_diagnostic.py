#!/usr/bin/env python3
"""Inspect symexp-twohot targets and output-logit gradient conflict."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from reward_head_cached_lr_sweep import load_params, logits, bins


def cosine(a, b):
    return float(jnp.vdot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-20))


def summarize(data, params):
    result = {}
    raw_bins = np.asarray(bins())
    names = ("ordinary", "pre_collision", "collision")
    mean_logit_grads = []
    for value, name in enumerate(names):
        idx = np.flatnonzero(data["stratum"] == value)
        x = jnp.asarray(data["feature"][idx])
        target = jnp.asarray(data["twohot"][idx])
        prob = jax.nn.softmax(logits(params, x), -1)
        grad = prob - target
        mean_grad = grad.mean(0)
        mean_logit_grads.append(mean_grad)
        below = data["twohot_below"][idx]
        above = data["twohot_above"][idx]
        active = np.unique(np.concatenate([below, above]))
        result[name] = {
            "count": int(len(idx)),
            "raw_reward_mean": float(data["reward"][idx].mean()),
            "raw_reward_min": float(data["reward"][idx].min()),
            "raw_reward_max": float(data["reward"][idx].max()),
            "target_bin_below_mean": float(below.mean()),
            "target_bin_above_mean": float(above.mean()),
            "active_target_bin_min": int(active.min()),
            "active_target_bin_max": int(active.max()),
            "active_target_raw_min": float(raw_bins[active].min()),
            "active_target_raw_max": float(raw_bins[active].max()),
            "mean_logit_gradient_norm": float(jnp.linalg.norm(mean_grad)),
            "top_abs_logit_gradient_bins": [
                {"bin": int(i), "raw_value": float(raw_bins[i]),
                 "gradient": float(mean_grad[i])}
                for i in np.argsort(np.abs(np.asarray(mean_grad)))[-10:][::-1]],
        }
    result["logit_gradient_cosine"] = {
        "ordinary_collision": cosine(mean_logit_grads[0], mean_logit_grads[2]),
        "pre_collision_collision": cosine(mean_logit_grads[1], mean_logit_grads[2]),
        "ordinary_pre_collision": cosine(mean_logit_grads[0], mean_logit_grads[1]),
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    params, _ = load_params(a.checkpoint / "agent.pkl")
    report = {}
    for split, filename in (("training", "risk_training_features.npz"),
                            ("holdout", "holdout_features.npz")):
        with np.load(a.cache / filename) as src:
            data = {k: src[k] for k in src.files}
        report[split] = summarize(data, params)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({split: value["logit_gradient_cosine"]
                      for split, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()
