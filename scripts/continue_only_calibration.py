#!/usr/bin/env python3
"""Frozen-latent Continue-only LR sweep with binary prior calibration."""

import argparse
import csv
import json
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from embodied.jax.opt import clip_by_agc, scale_by_momentum, scale_by_rms
from conditional_outcome_c2_offline import conditional_logits
from reward_decomposition_r1_offline import arrays, auc, average_precision, decompose, raw_dense_pred, stats


CON_KEYS = ("con/mlp/linear0/kernel", "con/mlp/linear0/bias",
            "con/mlp/norm0/scale", "con/head/logit/kernel", "con/head/logit/bias")


def load_continue(path):
    with path.open("rb") as stream: tree = pickle.load(stream)["params"]
    return {k: jnp.asarray(tree[k]) for k in CON_KEYS}


def load_r1_dense(path):
    with np.load(path) as src:
        return {k.split("::", 1)[1]: jnp.asarray(src[k])
                for k in src.files if k.startswith("dense::")}


def load_conditional(path):
    with np.load(path) as src: return {k: jnp.asarray(src[k]) for k in src.files}


def con_logits(params, feature):
    x = feature.astype(jnp.bfloat16)
    x = x @ params[CON_KEYS[0]].astype(x.dtype) + params[CON_KEYS[1]].astype(x.dtype)
    dtype = x.dtype; x = x.astype(jnp.float32)
    x *= jax.lax.rsqrt(jnp.mean(x ** 2, -1, keepdims=True) + 1e-4)
    x *= params[CON_KEYS[2]]
    x = jax.nn.silu(x.astype(dtype))
    x = x @ params[CON_KEYS[3]].astype(x.dtype) + params[CON_KEYS[4]].astype(x.dtype)
    return x.astype(jnp.float32).reshape(-1)


def loss(params, feature, terminal):
    # Train P(terminal) directly as sigmoid(-continue_logit); equivalent to
    # the existing Continue target 1-is_terminal.
    terminal_logit = -con_logits(params, feature)
    return optax.sigmoid_binary_cross_entropy(terminal_logit, terminal).mean()


def optimizer(lr):
    return optax.chain(clip_by_agc(.3), scale_by_rms(.999, 1e-20),
                       scale_by_momentum(.9), optax.scale_by_learning_rate(lr))


@partial(jax.jit, static_argnums=4)
def update(params, state, feature, terminal, opt):
    value, grad = jax.value_and_grad(loss)(params, feature, terminal)
    updates, state = opt.update(grad, state, params)
    return optax.apply_updates(params, updates), state, value


def probabilities(params, feature, correction=0.0, chunk=2048):
    values = []
    for start in range(0, len(feature), chunk):
        terminal_logit = -con_logits(params, jnp.asarray(feature[start:start + chunk]))
        values.append(np.asarray(jax.nn.sigmoid(terminal_logit + correction)))
    return np.concatenate(values)


def timeout_masks(train, hold, risk_data, holdout_csv):
    manifest = json.loads((risk_data / "manifest.json").read_text())
    timeout_eps = {int(x["episode_id"]) for x in manifest["episodes"]
                   if x["outcome"] == "timeout"}
    train_timeout = np.asarray([int(e) in timeout_eps for e in train["episode"]])
    with holdout_csv.open() as stream: rows = list(csv.DictReader(stream))
    hold_timeout_eps = {int(r["episode"]) for r in rows if float(r["timeout"]) > .5}
    hold_timeout = np.asarray([int(e) in hold_timeout_eps for e in hold["episode"]])
    return train_timeout, hold_timeout


def binary_metrics(target, prob):
    pred = prob >= .5; target = target.astype(bool)
    tp = int((target & pred).sum()); fp = int((~target & pred).sum())
    fn = int((target & ~pred).sum()); tn = int((~target & ~pred).sum())
    eps = 1e-7; clipped = np.clip(prob, eps, 1 - eps)
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(tp + fn, 1),
            "auroc": auc(target, prob), "auprc": average_precision(target, prob),
            "brier": float(np.mean((prob - target) ** 2)),
            "bce": float(np.mean(-(target * np.log(clipped) + (~target) * np.log(1 - clipped)))),
            "confusion_real_nonterminal_terminal": [[tn, fp], [fn, tp]]}


def evaluate(data, dense_target, outcome, timeout, dense_params, conditional_params,
             con_params, correction):
    terminal = outcome > 0
    raw = probabilities(con_params, data["feature"])
    corrected = probabilities(con_params, data["feature"], correction)
    conditional = np.concatenate([np.asarray(jax.nn.softmax(conditional_logits(
        conditional_params, jnp.asarray(data["feature"][s:s + 2048])), -1))
        for s in range(0, len(terminal), 2048)])
    dense = raw_dense_pred(dense_params, data["feature"])
    groups = {"ordinary": (outcome == 0) & (data["stratum"] == 0) & ~timeout,
              "pre_collision": data["stratum"] == 1,
              "success_terminal": outcome == 1,
              "collision_terminal": outcome == 2,
              "timeout": timeout}
    result = {"raw_metrics": binary_metrics(terminal, raw),
              "corrected_metrics": binary_metrics(terminal, corrected),
              "groups": {}}
    expected = corrected * (100 * conditional[:, 0] - 100 * conditional[:, 1])
    combined = dense + expected
    for name, mask in groups.items():
        result["groups"][name] = {
            "count": int(mask.sum()), "raw_P_terminal": float(raw[mask].mean()),
            "corrected_P_terminal": float(corrected[mask].mean()),
            "expected_terminal_reward": float(expected[mask].mean()),
            "combined": stats(data["reward"][mask], combined[mask])}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--risk-data", type=Path, required=True)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--r1-heads", type=Path, required=True)
    p.add_argument("--conditional", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--seed", type=int, default=93041)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    train = arrays(a.cache / "risk_training_features.npz")
    hold = arrays(a.cache / "holdout_features.npz")
    train_dense, train_outcome, _, _ = decompose(train)
    hold_dense, hold_outcome, _, _ = decompose(hold)
    train_timeout, hold_timeout = timeout_masks(train, hold, a.risk_data, a.holdout_csv)
    dense_params = load_r1_dense(a.r1_heads); conditional_params = load_conditional(a.conditional)
    initial = load_continue(a.checkpoint / "agent.pkl")
    natural_terminal = float((train_outcome > 0).mean()); sampled_terminal = .5
    correction = float(np.log(natural_terminal / (1 - natural_terminal)) -
                       np.log(sampled_terminal / (1 - sampled_terminal)))
    terminals = np.flatnonzero(train_outcome > 0)
    nonterminals = np.flatnonzero(train_outcome == 0)
    rng_plan = np.random.default_rng(a.seed)
    plan = [(rng_plan.choice(nonterminals, 128, replace=False),
             rng_plan.choice(terminals, 128, replace=True)) for _ in range(a.updates)]
    report = {"phase": "Continue-only", "updates": a.updates,
              "natural_terminal_count": int(len(terminals)),
              "natural_nonterminal_count": int(len(nonterminals)),
              "p_env_terminal": natural_terminal,
              "sampled_terminal_count": int(a.updates * 128),
              "sampled_nonterminal_count": int(a.updates * 128),
              "q_train_terminal": sampled_terminal,
              "terminal_logit_prior_correction": correction,
              "timeout_training_rows": int(train_timeout.sum()),
              "timeout_holdout_rows": int(hold_timeout.sum()), "runs": []}
    for lr in (1e-5, 3e-5, 1e-4):
        params = {k: v.copy() for k, v in initial.items()}
        opt = optimizer(lr); state = opt.init(params); trace = []
        for step, (ni, ti) in enumerate(plan, 1):
            idx = np.concatenate([ni, ti]); target = np.concatenate([
                np.zeros(128, np.float32), np.ones(128, np.float32)])
            params, state, value = update(params, state,
                jnp.asarray(train["feature"][idx]), jnp.asarray(target), opt)
            if step in (1, 25, 50, 100): trace.append({"update": step, "loss": float(value)})
        entry = {"lr": lr, "trace_training_only": trace,
                 "training": evaluate(train, train_dense, train_outcome, train_timeout,
                                      dense_params, conditional_params, params, correction),
                 "permanent_holdout": evaluate(hold, hold_dense, hold_outcome, hold_timeout,
                                               dense_params, conditional_params, params, correction)}
        report["runs"].append(entry)
        np.savez_compressed(a.output / f"continue_lr_{lr:.0e}.npz",
                            **{k: np.asarray(v) for k, v in params.items()})
        print(json.dumps({"lr": lr, "holdout": entry["permanent_holdout"]}))
    (a.output / "continue_sweep_100.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
