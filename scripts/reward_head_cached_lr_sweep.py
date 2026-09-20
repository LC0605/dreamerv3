#!/usr/bin/env python3
"""Offline frozen-feature reward-head LR and optimizer-state diagnostic."""

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


REW_KEYS = (
    "rew/mlp/linear0/kernel", "rew/mlp/linear0/bias",
    "rew/mlp/norm0/scale", "rew/head/logits/kernel",
    "rew/head/logits/bias")


def load_params(path):
    with path.open("rb") as stream:
        tree = pickle.load(stream)["params"]
    params = {key: jnp.asarray(tree[key], jnp.float32) for key in REW_KEYS}
    restored = {
        "rms_step": jnp.asarray(tree["opt/state/1/0"], jnp.int32),
        "mom_step": jnp.asarray(tree["opt/state/2/0"], jnp.int32),
        "nu": {key: jnp.asarray(tree[f"opt/state/1/1/{key}"], jnp.float32)
               for key in REW_KEYS},
        "mu": {key: jnp.asarray(tree[f"opt/state/2/1/{key}"], jnp.float32)
               for key in REW_KEYS},
    }
    return params, restored


def logits(params, feature):
    # Match nets.MLP/Linear mixed-precision semantics: linear math is bf16,
    # RMS normalization is f32 internally, then returns to bf16.
    x = feature.astype(jnp.bfloat16)
    x = x @ params[REW_KEYS[0]].astype(x.dtype) + params[REW_KEYS[1]].astype(x.dtype)
    dtype = x.dtype
    x = x.astype(jnp.float32)
    x = x * (jax.lax.rsqrt(jnp.mean(jnp.square(x), -1, keepdims=True) + 1e-4) *
             params[REW_KEYS[2]])
    x = x.astype(dtype)
    x = jax.nn.silu(x)
    x = x @ params[REW_KEYS[3]].astype(x.dtype) + params[REW_KEYS[4]].astype(x.dtype)
    return x.astype(jnp.float32)


def bins():
    half = np.linspace(-20, 0, 128, dtype=np.float32)
    half = np.sign(half) * np.expm1(np.abs(half))
    return jnp.asarray(np.concatenate([half, -half[:-1][::-1]]))


def ce(params, feature, target):
    return -(target * jax.nn.log_softmax(logits(params, feature), -1)).sum(-1)


def loss(params, batches, weights):
    return sum(w * ce(params, x, y).mean()
               for w, (x, y) in zip(weights, batches))


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in tree.values()))


def cosine(a, b):
    return float(sum(jnp.vdot(a[k], b[k]) for k in a) /
                 (tree_norm(a) * tree_norm(b) + 1e-20))


def grad_diagnostic(params, batches):
    grads = []
    for x, y in batches:
        grads.append(jax.grad(lambda p: ce(p, x, y).mean())(params))
    result = {
        "norm_ordinary": float(tree_norm(grads[0])),
        "norm_pre_collision": float(tree_norm(grads[1])),
        "norm_collision": float(tree_norm(grads[2])),
        "cos_ordinary_collision": cosine(grads[0], grads[2]),
        "cos_pre_collision_collision": cosine(grads[1], grads[2]),
        "cos_ordinary_pre_collision": cosine(grads[0], grads[1]),
    }
    for key in params:
        short = key.replace("rew/", "").replace("/", ".")
        result[f"layer/{short}/norm_ordinary"] = float(jnp.linalg.norm(grads[0][key]))
        result[f"layer/{short}/norm_pre_collision"] = float(jnp.linalg.norm(grads[1][key]))
        result[f"layer/{short}/norm_collision"] = float(jnp.linalg.norm(grads[2][key]))
        result[f"layer/{short}/cos_ordinary_collision"] = cosine(
            {key: grads[0][key]}, {key: grads[2][key]})
        result[f"layer/{short}/cos_pre_collision_collision"] = cosine(
            {key: grads[1][key]}, {key: grads[2][key]})
    groups = {
        "trunk": REW_KEYS[:3], "projection": REW_KEYS[3:],
        "trunk_linear": REW_KEYS[:2], "trunk_norm": REW_KEYS[2:3]}
    for name, keys in groups.items():
        ga = {k: grads[0][k] for k in keys}; gp = {k: grads[1][k] for k in keys}
        gc = {k: grads[2][k] for k in keys}
        result[f"group/{name}/norm_ordinary"] = float(tree_norm(ga))
        result[f"group/{name}/norm_pre_collision"] = float(tree_norm(gp))
        result[f"group/{name}/norm_collision"] = float(tree_norm(gc))
        result[f"group/{name}/cos_ordinary_collision"] = cosine(ga, gc)
        result[f"group/{name}/cos_pre_collision_collision"] = cosine(gp, gc)
    return result


def fresh_state(params):
    zeros = {key: jnp.zeros_like(value) for key, value in params.items()}
    return {"rms_step": jnp.array(0, jnp.int32),
            "mom_step": jnp.array(0, jnp.int32),
            "nu": zeros, "mu": {k: v.copy() for k, v in zeros.items()}}


@jax.jit
def update(params, state, batches, lr, weights):
    value, grads = jax.value_and_grad(loss)(params, batches, weights)
    clipped = {}
    for key in params:
        unorm = jnp.linalg.norm(grads[key].reshape(-1), 2)
        pnorm = jnp.linalg.norm(params[key].reshape(-1), 2)
        upper = 0.3 * jnp.maximum(1e-3, pnorm)
        clipped[key] = grads[key] / jnp.maximum(1.0, unorm / upper)
    rs = state["rms_step"] + 1
    nu = {k: .999 * state["nu"][k] + .001 * clipped[k] ** 2 for k in params}
    rms = {k: clipped[k] / (jnp.sqrt(nu[k] / (1 - .999 ** rs)) + 1e-20)
           for k in params}
    ms = state["mom_step"] + 1
    mu = {k: .9 * state["mu"][k] + .1 * rms[k] for k in params}
    momentum = {k: mu[k] / (1 - .9 ** ms) for k in params}
    updates = {k: -lr * momentum[k] for k in params}
    params = {k: params[k] + updates[k] for k in params}
    state = {"rms_step": rs, "mom_step": ms, "nu": nu, "mu": mu}
    return params, state, value, tree_norm(grads), tree_norm(updates)


def arrays(path):
    with np.load(path) as src:
        return {k: src[k] for k in src.files}


def predict(params, feature, chunk=2048):
    raw_bins = bins()
    values = []
    losses = []
    # target is supplied separately in evaluation; retain probabilities here.
    for start in range(0, len(feature), chunk):
        prob = jax.nn.softmax(logits(params, jnp.asarray(feature[start:start + chunk])), -1)
        # Match outs.TwoHot.pred()'s symmetric accumulation exactly.
        middle = (prob[:, 127:128] * raw_bins[127:128]).sum(-1)
        paired = ((prob[:, :127] * raw_bins[:127])[:, ::-1] +
                  prob[:, 128:] * raw_bins[128:]).sum(-1)
        values.append(np.asarray(middle + paired))
    return np.concatenate(values)


def evaluate(params, data):
    pred = predict(params, data["feature"])
    result = {"prediction_distribution": {}}
    names = {0: "ordinary", 1: "pre_collision", 2: "collision"}
    for value, name in names.items():
        mask = data["stratum"] == value
        result[f"{name}_predicted_reward"] = float(pred[mask].mean())
        result[f"{name}_reward_mae"] = float(np.abs(pred[mask] - data["reward"][mask]).mean())
        result["prediction_distribution"][name] = {
            "mean": float(pred[mask].mean()), "std": float(pred[mask].std()),
            "min": float(pred[mask].min()), "max": float(pred[mask].max())}
    near = data["near"].astype(bool)
    result["near_obstacle_reward_mae"] = float(
        np.abs(pred[near] - data["reward"][near]).mean())
    logits_all = []
    for start in range(0, len(pred), 2048):
        logits_all.append(np.asarray(logits(params, jnp.asarray(
            data["feature"][start:start + 2048]))))
    logits_all = np.concatenate(logits_all)
    result["twohot_loss"] = float((
        -data["twohot"] * jax.nn.log_softmax(jnp.asarray(logits_all), -1)).sum(-1).mean())
    return result


def batch_plan(data, updates, size, seed):
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(data["stratum"] == i) for i in range(3)]
    return [[rng.choice(pool, size=size, replace=len(pool) < size) for pool in pools]
            for _ in range(updates)]


def materialize(data, indices):
    return tuple((jnp.asarray(data["feature"][idx]),
                  jnp.asarray(data["twohot"][idx])) for idx in indices)


def run(initial, opt_state, data, plan, lr, checkpoints, weights):
    params = {k: v.copy() for k, v in initial.items()}
    state = opt_state
    start = {k: v.copy() for k, v in params.items()}
    report = {"gradient": {}, "optimization": {}, "metrics": {}}
    diagnostic = materialize(data, plan[0])
    report["gradient"]["0"] = grad_diagnostic(params, diagnostic)
    for step, indices in enumerate(plan, 1):
        batch = materialize(data, indices)
        params, state, value, gnorm, unorm = update(
            params, state, batch, lr, jnp.asarray(weights, jnp.float32))
        if step in checkpoints:
            report["gradient"][str(step)] = grad_diagnostic(params, diagnostic)
            delta = {k: params[k] - start[k] for k in params}
            report["optimization"][str(step)] = {
                "loss": float(value), "gradient_norm": float(gnorm),
                "last_update_norm": float(unorm),
                "parameter_delta_norm": float(tree_norm(delta))}
    return params, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-per-stratum", type=int, default=128)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--lrs", type=float, nargs="+", default=(1e-5, 3e-5, 1e-4))
    parser.add_argument("--optimizers", nargs="+", choices=("fresh", "restored"),
                        default=("fresh", "restored"))
    parser.add_argument("--weights", nargs="+", default=("0.3,0.2,0.5",),
                        help="One or more ordinary,pre,collision triples")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = arrays(args.cache / "risk_training_features.npz")
    holdout = arrays(args.cache / "holdout_features.npz")
    initial, restored = load_params(args.checkpoint / "agent.pkl")
    plan = batch_plan(train, args.updates, args.batch_per_stratum, args.seed)
    weight_sets = [tuple(float(x) for x in item.split(",")) for item in args.weights]
    assert all(len(x) == 3 and abs(sum(x) - 1) < 1e-6 for x in weight_sets)
    result = {"configuration": {
        "updates": args.updates, "batch_per_stratum": args.batch_per_stratum,
        "seed": args.seed, "weight_sets": weight_sets,
        "frozen_feature": True, "optimizer_chain": "AGC-RMS-Momentum-constant_lr"},
        "initial": {"training": evaluate(initial, train),
                    "holdout": evaluate(initial, holdout)}, "runs": []}
    checkpoints = tuple(x for x in (25, 50, 100, 250, 500) if x <= args.updates)
    for kind in args.optimizers:
        for lr in args.lrs:
          for weights in weight_sets:
            state = fresh_state(initial) if kind == "fresh" else {
                "rms_step": restored["rms_step"], "mom_step": restored["mom_step"],
                "nu": {k: v.copy() for k, v in restored["nu"].items()},
                "mu": {k: v.copy() for k, v in restored["mu"].items()}}
            params, trace = run(initial, state, train, plan, lr, checkpoints, weights)
            entry = {"optimizer": kind, "lr": lr, "updates": args.updates,
                     "weights": weights,
                     **trace, "training": evaluate(params, train),
                     "holdout": evaluate(params, holdout)}
            result["runs"].append(entry)
            print(json.dumps({"optimizer": kind, "lr": lr, "weights": weights,
                "train_collision_pred": entry["training"]["collision_predicted_reward"],
                "holdout_collision_pred": entry["holdout"]["collision_predicted_reward"],
                "ordinary_mae": entry["holdout"]["ordinary_reward_mae"]}))
    (args.output / f"sweep_{args.updates}.json").write_text(
        json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
