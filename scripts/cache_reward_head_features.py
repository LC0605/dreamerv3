#!/usr/bin/env python3
"""Cache exact [RSSM deter, flattened stoch] inputs for reward-only sweeps."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent


def config(path):
    y = yaml.YAML(typ="safe")
    cfgs = y.load(Path("dreamerv3/dreamerv3/configs.yaml").read_text())
    return elements.Config(cfgs["defaults"]).update(
        cfgs["quadrotor"]).update(y.load(path.read_text()))


def risk_episodes(root):
    result = []
    for path in sorted((root / "episodes").glob("*.npz")):
        with np.load(path) as src:
            d = {key: src[key] for key in src.files}
        n = len(d["reward"])
        strata = np.zeros(n, np.int32)
        strata[d["pre_collision"]] = 1
        strata[d["collision_terminal"]] = 2
        result.append({**d, "stratum": strata,
                       "episode": np.full(n, int(d["episode_id"][0]), np.int32),
                       "seed": np.full(n, 52000, np.int32),
                       "near": d["minimum_clearance"] < .7})
    return result


def holdout_episodes(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for eid in range(1, 61):
        part = [row for row in rows if int(row["episode"]) == eid]
        n = len(part); strata = np.zeros(n, np.int32)
        collision = any(float(row["collision"]) > .5 for row in part)
        if collision:
            strata[max(1, n - 11):n - 1] = 1
            strata[n - 1] = 2
        result.append({
            "vector": np.asarray([[float(row[f"vector_{i:02d}"]) for i in range(68)]
                                  for row in part], np.float32),
            "reward": np.asarray([float(row["reward"]) for row in part], np.float32),
            "is_first": np.asarray([bool(int(row["is_first"])) for row in part]),
            "is_last": np.asarray([bool(int(row["is_last"])) for row in part]),
            "is_terminal": np.asarray([
                bool(float(row["collision"]) or float(row["success"])) for row in part]),
            "action": np.asarray([[float(row[key]) for key in (
                "action_forward", "action_lateral", "action_vertical", "action_yaw_rate")]
                for row in part], np.float32),
            "stratum": strata,
            "episode": np.full(n, eid, np.int32),
            "seed": np.full(n, 35700, np.int32),
            "minimum_clearance": np.asarray(
                [float(row["minimum_clearance"]) for row in part], np.float32),
            "near": np.asarray([float(row["minimum_clearance"]) < .7 for row in part]),
        })
    return result


def cache(agent, episodes):
    output = {key: [] for key in (
        "feature", "reward", "stratum", "episode", "seed", "timestep",
        "is_terminal", "near", "official_reward_pred")}
    for start in range(0, len(episodes), 10):
        group = episodes[start:start + 10]
        b, t = len(group), max(len(ep["reward"]) for ep in group)
        vector = np.zeros((b, t, 68), np.float32)
        reward = np.zeros((b, t), np.float32)
        first = np.ones((b, t), bool); last = np.ones((b, t), bool)
        terminal = np.zeros((b, t), bool); prev = np.zeros((b, t, 4), np.float32)
        valid = np.zeros((b, t), bool)
        for i, ep in enumerate(group):
            n = len(ep["reward"]); valid[i, :n] = True
            vector[i, :n] = ep["vector"]; reward[i, :n] = ep["reward"]
            first[i, :n] = ep["is_first"]; last[i, :n] = ep["is_last"]
            terminal[i, :n] = ep["is_terminal"]
            if n > 1: prev[i, 1:n] = ep["action"][:n - 1]
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
        deter = np.asarray(out["rssm_deter"])
        stoch = np.asarray(out["rssm_stoch"])
        feature = np.concatenate([deter, stoch], -1)
        for i, ep in enumerate(group):
            n = len(ep["reward"])
            output["feature"].append(feature[i, :n])
            for key in ("reward", "stratum", "episode", "seed", "is_terminal", "near"):
                output[key].append(np.asarray(ep[key]))
            output["official_reward_pred"].append(
                np.asarray(out["reward_pred"])[i, :n])
            output["timestep"].append(np.arange(n, dtype=np.int32))
    output = {key: np.concatenate(value) for key, value in output.items()}
    # Exact compact representation of the 255-way two-hot target.
    half = np.linspace(-20, 0, 128, dtype=np.float32)
    half = np.sign(half) * np.expm1(np.abs(half))
    bins = np.concatenate([half, -half[:-1][::-1]])
    target = output["reward"]
    below = np.clip(np.searchsorted(bins, target, side="right") - 1, 0, 254)
    above = np.clip(np.searchsorted(bins, target, side="left"), 0, 254)
    equal = below == above
    db = np.where(equal, 1, np.abs(bins[below] - target))
    da = np.where(equal, 1, np.abs(bins[above] - target))
    total = db + da
    wb, wa = da / total, db / total
    twohot = np.zeros((len(target), 255), np.float32)
    twohot[np.arange(len(target)), below] += wb
    twohot[np.arange(len(target)), above] += wa
    output.update(twohot=twohot, twohot_below=below.astype(np.int16),
                  twohot_above=above.astype(np.int16),
                  twohot_weight_below=wb.astype(np.float32),
                  twohot_weight_above=wa.astype(np.float32))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--risk-data", type=Path, required=True)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    agent = make_agent(config(a.config))
    elements.checkpoint.load(a.checkpoint, {"agent": partial(
        agent.load, regex="^(?!(bcopt|opt)/).*")})
    train = cache(agent, risk_episodes(a.risk_data))
    holdout = cache(agent, holdout_episodes(a.holdout_csv))
    np.savez_compressed(a.output / "risk_training_features.npz", **train)
    np.savez_compressed(a.output / "holdout_features.npz", **holdout)
    manifest = {
        "checkpoint": str(a.checkpoint), "feature_definition": "concat(deter, flatten(stoch))",
        "feature_dim": int(train["feature"].shape[1]),
        "train_rows": int(len(train["reward"])), "holdout_rows": int(len(holdout["reward"])),
        "train_strata": {str(i): int((train["stratum"] == i).sum()) for i in range(3)},
        "holdout_strata": {str(i): int((holdout["stratum"] == i).sum()) for i in range(3)},
        "seeds": {"training": 52000, "holdout": 35700},
        "encoder_rssm_reused_during_sweep": False,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
