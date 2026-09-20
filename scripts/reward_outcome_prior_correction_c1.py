#!/usr/bin/env python3
"""Phase C1: read-only natural-prior correction of frozen R1 logits."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from reward_decomposition_r1_offline import (
    TERMINAL_VALUES, arrays, auc, average_precision, decompose,
    outcome_logits, raw_dense_pred, stats)


def load_heads(path):
    with np.load(path) as src:
        dense = {k.split("::", 1)[1]: jnp.asarray(src[k])
                 for k in src.files if k.startswith("dense::")}
        outcome = {k.split("::", 1)[1]: jnp.asarray(src[k])
                   for k in src.files if k.startswith("outcome::")}
    return dense, outcome


def probabilities(params, feature, correction=None, chunk=2048):
    result = []
    for start in range(0, len(feature), chunk):
        value = outcome_logits(params, jnp.asarray(feature[start:start + chunk]))
        # OOB is unsupported in R1 and excluded from the effective softmax.
        value = value[:, :3]
        if correction is not None:
            value = value + jnp.asarray(correction)
        result.append(np.asarray(jax.nn.softmax(value, -1)))
    return np.concatenate(result)


def classification(outcome, probs):
    pred = probs.argmax(-1)
    confusion = np.zeros((3, 3), np.int32)
    np.add.at(confusion, (outcome, pred), 1)
    result = {"confusion_matrix_rows_real_cols_pred": confusion.tolist()}
    for cls, name in ((1, "success"), (2, "collision")):
        y = outcome == cls; positive = pred == cls
        tp = int((y & positive).sum()); fp = int((~y & positive).sum())
        fn = int((y & ~positive).sum())
        result[name] = {
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "auroc": auc(y, probs[:, cls]), "auprc": average_precision(y, probs[:, cls]),
            "mean_probability_positive": float(probs[y, cls].mean()),
            "mean_probability_negative": float(probs[~y, cls].mean())}
    ordinary = outcome == 0
    result["ordinary"] = {
        "mean_P_nonterminal": float(probs[ordinary, 0].mean()),
        "mean_P_success": float(probs[ordinary, 1].mean()),
        "mean_P_collision": float(probs[ordinary, 2].mean())}
    return result


def combined(data, dense_target, outcome, dense_pred, probs):
    terminal_values = TERMINAL_VALUES[:3]
    pred = dense_pred + probs @ terminal_values
    groups = {
        "ordinary_nonterminal": (outcome == 0) & (data["stratum"] == 0),
        "pre_collision": data["stratum"] == 1,
        "success_terminal": outcome == 1,
        "collision_terminal": outcome == 2}
    return {name: stats(data["reward"][mask], pred[mask])
            for name, mask in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--heads", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=500)
    a = p.parse_args()
    train = arrays(a.cache / "risk_training_features.npz")
    hold = arrays(a.cache / "holdout_features.npz")
    train_dense, train_outcome, _, _ = decompose(train)
    hold_dense, hold_outcome, _, _ = decompose(hold)
    dense_params, outcome_params = load_heads(a.heads)
    natural_counts = np.bincount(train_outcome, minlength=4)
    natural = natural_counts / natural_counts.sum()
    # Actual sampler emitted exactly 128 rows of each supported class/update.
    sampled_counts = np.asarray([128, 128, 128, 0], np.int64) * a.updates
    sampled = sampled_counts / sampled_counts.sum()
    correction = np.log(natural[:3] + 1e-12) - np.log(sampled[:3] + 1e-12)
    dense_pred = raw_dense_pred(dense_params, hold["feature"])
    raw = probabilities(outcome_params, hold["feature"])
    corrected = probabilities(outcome_params, hold["feature"], correction)
    report = {
        "phase": "C1", "training": False, "optimizer_steps": 0,
        "priors": {
            "class_order": ["nonterminal", "success", "collision", "out_of_bound"],
            "natural_counts": natural_counts.tolist(), "p_env": natural.tolist(),
            "sampled_counts_actual": sampled_counts.tolist(), "q_train": sampled.tolist(),
            "supported_correction_classes": ["nonterminal", "success", "collision"],
            "oob_handling": "excluded from correction softmax; zero R1 samples; no guessed prior",
            "log_prior_correction_supported": correction.tolist()},
        "raw": {"classification": classification(hold_outcome, raw),
                "combined": combined(hold, hold_dense, hold_outcome, dense_pred, raw)},
        "corrected": {
            "classification": classification(hold_outcome, corrected),
            "combined": combined(hold, hold_dense, hold_outcome, dense_pred, corrected)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
