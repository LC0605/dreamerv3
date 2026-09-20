#!/usr/bin/env python3
"""Phase C2: terminal-only conditional success/collision head validation."""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from embodied.jax.opt import clip_by_agc, scale_by_momentum, scale_by_rms
from reward_decomposition_r1_offline import (
    arrays, auc, average_precision, decompose, raw_dense_pred, stats,
    trunc_normal)


def load_dense_and_conditional(path, seed):
    with np.load(path) as src:
        dense = {k.split("::", 1)[1]: jnp.asarray(src[k])
                 for k in src.files if k.startswith("dense::")}
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    conditional = {
        "conditional/mlp/linear0/kernel": trunc_normal(k1, (640, 32), 640),
        "conditional/mlp/linear0/bias": jnp.zeros(32, jnp.float32),
        "conditional/mlp/norm0/scale": jnp.ones(32, jnp.float32),
        "conditional/head/logits/kernel": trunc_normal(k2, (32, 2), 32),
        "conditional/head/logits/bias": jnp.zeros(2, jnp.float32)}
    return dense, conditional


def conditional_logits(params, feature):
    x = feature.astype(jnp.bfloat16)
    x = x @ params["conditional/mlp/linear0/kernel"].astype(x.dtype)
    x += params["conditional/mlp/linear0/bias"].astype(x.dtype)
    dtype = x.dtype; x = x.astype(jnp.float32)
    x *= jax.lax.rsqrt(jnp.mean(x ** 2, -1, keepdims=True) + 1e-4)
    x *= params["conditional/mlp/norm0/scale"]
    x = jax.nn.silu(x.astype(dtype))
    x = x @ params["conditional/head/logits/kernel"].astype(x.dtype)
    x += params["conditional/head/logits/bias"].astype(x.dtype)
    return x.astype(jnp.float32)


def loss(params, feature, target):
    logp = jax.nn.log_softmax(conditional_logits(params, feature), -1)
    return -logp[jnp.arange(len(target)), target].mean()


def optimizer(lr):
    return optax.chain(clip_by_agc(.3), scale_by_rms(.999, 1e-20),
                       scale_by_momentum(.9), optax.scale_by_learning_rate(lr))


@partial(jax.jit, static_argnums=4)
def update(params, state, feature, target, opt):
    value, grad = jax.value_and_grad(loss)(params, feature, target)
    updates, state = opt.update(grad, state, params)
    return optax.apply_updates(params, updates), state, value


def conditional_prob(params, feature, chunk=2048):
    return np.concatenate([np.asarray(jax.nn.softmax(conditional_logits(
        params, jnp.asarray(feature[start:start + chunk])), -1))
        for start in range(0, len(feature), chunk)])


def load_continue(path):
    import pickle
    with path.open("rb") as stream: tree = pickle.load(stream)["params"]
    keys = ("con/mlp/linear0/kernel", "con/mlp/linear0/bias",
            "con/mlp/norm0/scale", "con/head/logit/kernel", "con/head/logit/bias")
    return {k: jnp.asarray(tree[k]) for k in keys}


def continue_prob(params, feature, chunk=2048):
    values = []
    for start in range(0, len(feature), chunk):
        x = jnp.asarray(feature[start:start + chunk]).astype(jnp.bfloat16)
        x = x @ params["con/mlp/linear0/kernel"].astype(x.dtype)
        x += params["con/mlp/linear0/bias"].astype(x.dtype)
        dtype = x.dtype; x = x.astype(jnp.float32)
        x *= jax.lax.rsqrt(jnp.mean(x ** 2, -1, keepdims=True) + 1e-4)
        x *= params["con/mlp/norm0/scale"]
        x = jax.nn.silu(x.astype(dtype))
        x = x @ params["con/head/logit/kernel"].astype(x.dtype)
        x += params["con/head/logit/bias"].astype(x.dtype)
        values.append(np.asarray(jax.nn.sigmoid(x.astype(jnp.float32))).reshape(-1))
    return np.concatenate(values)


def terminal_metrics(outcome, probs):
    mask = outcome > 0
    # Conditional labels: 0 success, 1 collision. OOB is unsupported.
    target = outcome[mask] - 1; p = probs[mask]; pred = p.argmax(-1)
    confusion = np.zeros((2, 2), np.int32); np.add.at(confusion, (target, pred), 1)
    result = {"count": int(mask.sum()),
              "confusion_rows_real_success_collision": confusion.tolist(),
              "mean_probability_correct": float(p[np.arange(len(p)), target].mean())}
    for cls, name in ((0, "success"), (1, "collision")):
        y = target == cls; positive = pred == cls
        tp = int((y & positive).sum()); fp = int((~y & positive).sum())
        fn = int((y & ~positive).sum())
        result[name] = {"precision": tp / max(tp + fp, 1),
                        "recall": tp / max(tp + fn, 1),
                        "auroc": auc(y, p[:, cls]),
                        "auprc": average_precision(y, p[:, cls]),
                        "mean_probability": float(p[y, cls].mean())}
    return result


def evaluate(data, dense_target, outcome, dense_params, conditional_params, con_params):
    dense = raw_dense_pred(dense_params, data["feature"])
    conditional = conditional_prob(conditional_params, data["feature"])
    p_continue = continue_prob(con_params, data["feature"])
    p_terminal = 1 - p_continue
    expected_terminal = p_terminal * (conditional[:, 0] * 100 - conditional[:, 1] * 100)
    combined = dense + expected_terminal
    groups = {"ordinary_nonterminal": (outcome == 0) & (data["stratum"] == 0),
              "pre_collision": data["stratum"] == 1,
              "success_terminal": outcome == 1,
              "collision_terminal": outcome == 2}
    decomposition = {}
    for name, mask in groups.items():
        decomposition[name] = {
            "count": int(mask.sum()), "mean_P_terminal": float(p_terminal[mask].mean()),
            "mean_P_success_given_terminal": float(conditional[mask, 0].mean()),
            "mean_P_collision_given_terminal": float(conditional[mask, 1].mean()),
            "mean_expected_terminal_reward": float(expected_terminal[mask].mean()),
            "dense_predicted_mean": float(dense[mask].mean()),
            "combined": stats(data["reward"][mask], combined[mask])}
    return {"conditional_terminal_only": terminal_metrics(outcome, conditional),
            "components_by_real_outcome": decomposition}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--r1-heads", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=92031)
    a = p.parse_args()
    train = arrays(a.cache / "risk_training_features.npz")
    hold = arrays(a.cache / "holdout_features.npz")
    train_dense, train_outcome, _, _ = decompose(train)
    hold_dense, hold_outcome, _, _ = decompose(hold)
    dense_params, params = load_dense_and_conditional(a.r1_heads, a.seed)
    con_params = load_continue(a.checkpoint / "agent.pkl")
    opt = optimizer(a.lr); state = opt.init(params); rng = np.random.default_rng(a.seed)
    success = np.flatnonzero(train_outcome == 1); collision = np.flatnonzero(train_outcome == 2)
    trace = []
    for step in range(1, a.updates + 1):
        si = rng.choice(success, 128, replace=len(success) < 128)
        ci = rng.choice(collision, 128, replace=len(collision) < 128)
        idx = np.concatenate([si, ci]); target = np.concatenate([
            np.zeros(128, np.int32), np.ones(128, np.int32)])
        params, state, value = update(params, state, jnp.asarray(train["feature"][idx]),
                                      jnp.asarray(target), opt)
        if step in (1, 100, 250, a.updates): trace.append({"update": step, "loss": float(value)})
    report = {"phase": "C2", "trained_module": "conditional_outcome_only",
              "frozen": ["encoder", "rssm", "decoder", "dense_reward", "continue",
                         "value", "actor"],
              "classes": ["success_given_terminal", "collision_given_terminal"],
              "oob": "reserved for future extension; excluded because zero samples",
              "updates": a.updates, "lr": a.lr, "trace_training_only": trace,
              "training": evaluate(train, train_dense, train_outcome, dense_params, params, con_params),
              "permanent_holdout": evaluate(hold, hold_dense, hold_outcome,
                                            dense_params, params, con_params)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(a.output.with_suffix(".npz"),
                        **{k: np.asarray(v) for k, v in params.items()})
    print(json.dumps(report["permanent_holdout"], indent=2))


if __name__ == "__main__":
    main()
