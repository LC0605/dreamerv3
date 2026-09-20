#!/usr/bin/env python3
"""Phase R1: frozen-latent dense reward + terminal outcome validation."""

import argparse
import json
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from embodied.jax.opt import clip_by_agc, scale_by_momentum, scale_by_rms
from reward_head_cached_lr_sweep import REW_KEYS, bins, load_params, logits


TERMINAL_VALUES = np.asarray([0.0, 100.0, -100.0, -100.0], np.float32)
OUTCOME_NAMES = ("nonterminal", "success", "collision", "out_of_bound")


def arrays(path):
    with np.load(path) as src:
        return {k: src[k] for k in src.files}


def decompose(data):
    terminal = data["is_terminal"].astype(bool)
    collision = data["stratum"] == 2
    success = terminal & ~collision
    # The immutable +0.3 datasets contain no OOB episodes. The four-class API
    # reserves class 3 because env.py applies the same failure_penalty to OOB.
    outcome = np.zeros(len(terminal), np.int32)
    outcome[success] = 1
    outcome[collision] = 2
    terminal_reward = TERMINAL_VALUES[outcome]
    dense = data["reward"].astype(np.float32) - terminal_reward
    residual = data["reward"] - (dense + terminal_reward)
    assert np.max(np.abs(residual)) <= 1e-6
    return dense, outcome, terminal_reward, residual


def twohot(target):
    raw_bins = np.asarray(bins())
    below = np.clip(np.searchsorted(raw_bins, target, side="right") - 1, 0, 254)
    above = np.clip(np.searchsorted(raw_bins, target, side="left"), 0, 254)
    equal = below == above
    db = np.where(equal, 1, np.abs(raw_bins[below] - target))
    da = np.where(equal, 1, np.abs(raw_bins[above] - target))
    total = db + da
    result = np.zeros((len(target), 255), np.float32)
    result[np.arange(len(target)), below] += da / total
    result[np.arange(len(target)), above] += db / total
    return result


def dense_loss(params, feature, target):
    return -(target * jax.nn.log_softmax(logits(params, feature), -1)).sum(-1).mean()


def outcome_logits(params, feature):
    x = feature.astype(jnp.bfloat16)
    x = x @ params["outcome/mlp/linear0/kernel"].astype(x.dtype)
    x = x + params["outcome/mlp/linear0/bias"].astype(x.dtype)
    dtype = x.dtype
    x = x.astype(jnp.float32)
    x = x * (jax.lax.rsqrt(jnp.mean(x ** 2, -1, keepdims=True) + 1e-4) *
             params["outcome/mlp/norm0/scale"])
    x = jax.nn.silu(x.astype(dtype))
    x = x @ params["outcome/head/logits/kernel"].astype(x.dtype)
    x = x + params["outcome/head/logits/bias"].astype(x.dtype)
    return x.astype(jnp.float32)


def outcome_loss(params, feature, target):
    return -jax.nn.log_softmax(outcome_logits(params, feature), -1)[
        jnp.arange(len(target)), target].mean()


def trunc_normal(key, shape, fanin):
    return jax.random.truncated_normal(key, -2, 2, shape) * (1.1368 / np.sqrt(fanin))


def init_outcome(seed=91027):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "outcome/mlp/linear0/kernel": trunc_normal(k1, (640, 32), 640),
        "outcome/mlp/linear0/bias": jnp.zeros(32, jnp.float32),
        "outcome/mlp/norm0/scale": jnp.ones(32, jnp.float32),
        "outcome/head/logits/kernel": trunc_normal(k2, (32, 4), 32),
        "outcome/head/logits/bias": jnp.zeros(4, jnp.float32),
    }


def optimizer(lr):
    return optax.chain(clip_by_agc(.3), scale_by_rms(.999, 1e-20),
                       scale_by_momentum(.9), optax.scale_by_learning_rate(lr))


@partial(jax.jit, static_argnums=4)
def dense_update(params, state, x, y, opt):
    value, grad = jax.value_and_grad(dense_loss)(params, x, y)
    updates, state = opt.update(grad, state, params)
    return optax.apply_updates(params, updates), state, value


@partial(jax.jit, static_argnums=4)
def outcome_update(params, state, x, y, opt):
    value, grad = jax.value_and_grad(outcome_loss)(params, x, y)
    updates, state = opt.update(grad, state, params)
    return optax.apply_updates(params, updates), state, value


def sample(rng, pool, count):
    return rng.choice(pool, count, replace=len(pool) < count)


def raw_dense_pred(params, feature, chunk=2048):
    raw_bins = bins(); result = []
    for start in range(0, len(feature), chunk):
        prob = jax.nn.softmax(logits(params, jnp.asarray(feature[start:start + chunk])), -1)
        middle = (prob[:, 127:128] * raw_bins[127:128]).sum(-1)
        paired = ((prob[:, :127] * raw_bins[:127])[:, ::-1] +
                  prob[:, 128:] * raw_bins[128:]).sum(-1)
        result.append(np.asarray(middle + paired))
    return np.concatenate(result)


def outcome_prob(params, feature, chunk=2048):
    result = []
    for start in range(0, len(feature), chunk):
        result.append(np.asarray(jax.nn.softmax(outcome_logits(
            params, jnp.asarray(feature[start:start + chunk])), -1)))
    return np.concatenate(result)


def auc(y, score):
    pos = score[y]; neg = score[~y]
    if not len(pos) or not len(neg): return None
    return float(((pos[:, None] > neg[None]).sum() +
                  .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def average_precision(y, score):
    if not y.sum(): return None
    order = np.argsort(-score, kind="stable"); ys = y[order]
    precision = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(precision[ys].mean())


def stats(target, pred):
    return {"count": int(len(target)), "real_mean": float(target.mean()),
            "predicted_mean": float(pred.mean()),
            "mae": float(np.abs(target - pred).mean())}


def evaluate(data, dense_target, outcome, dense_params, outcome_params):
    dense_pred = raw_dense_pred(dense_params, data["feature"])
    probs = outcome_prob(outcome_params, data["feature"])
    combined = dense_pred + probs @ TERMINAL_VALUES
    predicted_class = probs.argmax(-1)
    groups = {
        "ordinary_nonterminal": (outcome == 0) & (data["stratum"] == 0),
        "pre_collision": data["stratum"] == 1,
        "success_terminal": outcome == 1,
        "collision_terminal": outcome == 2,
    }
    dense_metrics = {name: stats(dense_target[mask], dense_pred[mask])
                     for name, mask in groups.items()}
    combined_metrics = {name: stats(data["reward"][mask], combined[mask])
                        for name, mask in groups.items()}
    # Distance buckets are available from cached episode/timestep metadata.
    distance = np.full(len(outcome), -1, np.int32)
    for episode in np.unique(data["episode"]):
        idx = np.flatnonzero(data["episode"] == episode)
        col = idx[outcome[idx] == 2]
        if len(col): distance[idx] = data["timestep"][col[-1]] - data["timestep"][idx]
    for name, mask in {
        "pre_t1": distance == 1, "pre_t2_3": (distance >= 2) & (distance <= 3),
        "pre_t4_5": (distance >= 4) & (distance <= 5),
        "pre_t6_10": (distance >= 6) & (distance <= 10)}.items():
        mask &= data["stratum"] == 1
        dense_metrics[name] = stats(dense_target[mask], dense_pred[mask])
    confusion = np.zeros((4, 4), np.int32)
    np.add.at(confusion, (outcome, predicted_class), 1)
    classification = {"confusion_matrix_rows_real_cols_pred": confusion.tolist()}
    for cls, name in ((1, "success"), (2, "collision"), (3, "out_of_bound")):
        y = outcome == cls; pred = predicted_class == cls
        tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
        classification[name] = {
            "count": int(y.sum()), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "fnr": fn / max(tp + fn, 1),
            "fpr": fp / max((~y).sum(), 1), "auroc": auc(y, probs[:, cls]),
            "auprc": average_precision(y, probs[:, cls]),
            "mean_probability_positive": float(probs[y, cls].mean()) if y.any() else None,
            "mean_probability_negative": float(probs[~y, cls].mean())}
    nonterminal = outcome == 0
    classification["ordinary_false_probability"] = {
        "mean_P_success": float(probs[nonterminal, 1].mean()),
        "mean_P_collision": float(probs[nonterminal, 2].mean()),
        "mean_P_oob": float(probs[nonterminal, 3].mean())}
    return {"dense": dense_metrics, "outcome": classification,
            "combined": combined_metrics,
            "prediction_ranges": {"dense_min": float(dense_pred.min()),
                                  "dense_max": float(dense_pred.max()),
                                  "combined_min": float(combined.min()),
                                  "combined_max": float(combined.max())}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--dense-lr", type=float, default=1e-5)
    p.add_argument("--outcome-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=91027)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    train = arrays(a.cache / "risk_training_features.npz")
    holdout = arrays(a.cache / "holdout_features.npz")
    train_dense, train_outcome, train_terminal, train_residual = decompose(train)
    hold_dense, hold_outcome, hold_terminal, hold_residual = decompose(holdout)
    dense_params, _ = load_params(a.checkpoint / "agent.pkl")
    outcome_params = init_outcome(a.seed)
    dense_opt = optimizer(a.dense_lr); outcome_opt = optimizer(a.outcome_lr)
    dense_state = dense_opt.init(dense_params); outcome_state = outcome_opt.init(outcome_params)
    rng = np.random.default_rng(a.seed)
    pools = {
        "ordinary": np.flatnonzero((train_outcome == 0) & (train["stratum"] == 0)),
        "pre": np.flatnonzero(train["stratum"] == 1),
        "success": np.flatnonzero(train_outcome == 1),
        "collision": np.flatnonzero(train_outcome == 2)}
    dense_twohot = twohot(train_dense)
    trace = []
    for step in range(1, a.updates + 1):
        didx = np.concatenate([sample(rng, pools["ordinary"], 205),
                               sample(rng, pools["pre"], 26),
                               sample(rng, pools["success"], 13),
                               sample(rng, pools["collision"], 12)])
        oidx = np.concatenate([sample(rng, pools["ordinary"], 128),
                               sample(rng, pools["success"], 128),
                               sample(rng, pools["collision"], 128)])
        dense_params, dense_state, dloss = dense_update(
            dense_params, dense_state, jnp.asarray(train["feature"][didx]),
            jnp.asarray(dense_twohot[didx]), dense_opt)
        outcome_params, outcome_state, oloss = outcome_update(
            outcome_params, outcome_state, jnp.asarray(train["feature"][oidx]),
            jnp.asarray(train_outcome[oidx]), outcome_opt)
        if step in (1, 100, 250, a.updates):
            trace.append({"update": step, "dense_loss": float(dloss),
                          "outcome_loss": float(oloss)})
    report = {
        "phase": "R1", "parent_checkpoint": str(a.checkpoint),
        "frozen": ["encoder", "rssm", "decoder", "continue", "value", "actor"],
        "trained": ["dense_reward_head", "terminal_outcome_head"],
        "environment_reward_source": {
            "file": "src/uav_navigation/env.py", "goal_component": 100.0,
            "failure_penalty_config": 100.0,
            "terminal_values": TERMINAL_VALUES.tolist(),
            "oob_supported_by_environment": True,
            "oob_samples_in_r1_data": int((train_outcome == 3).sum()),
            "timeout_outcome": "nonterminal"},
        "decomposition_audit": {
            "training_max_abs_residual": float(np.max(np.abs(train_residual))),
            "holdout_max_abs_residual": float(np.max(np.abs(hold_residual))),
            "training_terminal_dense_mean": {
                "success": float(train_dense[train_outcome == 1].mean()),
                "collision": float(train_dense[train_outcome == 2].mean())},
            "holdout_terminal_dense_mean": {
                "success": float(hold_dense[hold_outcome == 1].mean()),
                "collision": float(hold_dense[hold_outcome == 2].mean())}},
        "training_config": {"updates": a.updates, "dense_lr": a.dense_lr,
                            "outcome_lr": a.outcome_lr, "seed": a.seed,
                            "dense_batch_mix": "205 ordinary/26 pre/13 success/12 collision",
                            "outcome_batch_mix": "128 nonterminal/128 success/128 collision",
                            "oob_training": "unavailable in existing R1 datasets"},
        "trace_training_only": trace,
        "training": evaluate(train, train_dense, train_outcome, dense_params, outcome_params),
        "permanent_holdout": evaluate(
            holdout, hold_dense, hold_outcome, dense_params, outcome_params)}
    np.savez_compressed(a.output / "reward_decomposition_heads.npz",
                        **{**{f"dense::{k}": np.asarray(v) for k, v in dense_params.items()},
                           **{f"outcome::{k}": np.asarray(v) for k, v in outcome_params.items()}})
    (a.output / "parent_checkpoint.txt").write_text(str(a.checkpoint) + "\n")
    (a.output / "r1_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"audit": report["decomposition_audit"],
                      "holdout": report["permanent_holdout"]}, indent=2))


if __name__ == "__main__":
    main()
