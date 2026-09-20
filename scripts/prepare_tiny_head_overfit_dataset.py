#!/usr/bin/env python3
"""Build fixed frozen-latent samples for an official-head overfit diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.rglob("*.npz")):
        raise FileExistsError(args.output)
    train = args.output / "train"; valid = args.output / "validation"
    train.mkdir(parents=True); valid.mkdir()
    collisions = sorted((args.source / "episodes").glob("*collision.npz"))[:10]
    successes = sorted((args.source / "episodes").glob("*success.npz"))[:10]
    if len(collisions) != 10 or len(successes) != 10:
        raise ValueError((len(collisions), len(successes)))
    records = []
    for kind, paths in (("collision", collisions), ("ordinary", successes)):
        for number, path in enumerate(paths):
            with np.load(path) as source:
                data = {key: source[key] for key in source.files}
            mask = np.zeros(len(data["reward"]), bool)
            selected = []
            if kind == "collision":
                terminal = int(np.flatnonzero(data["collision_terminal"])[0])
                pre = max(1, terminal - 5)
                mask[[pre, terminal]] = True
                selected = [{"class": "pre_collision_5", "row": pre},
                            {"class": "collision_terminal", "row": terminal}]
            else:
                candidates = np.flatnonzero(
                    (~data["is_first"]) & (~data["is_last"])
                    & (data["minimum_clearance"] >= 0.7))
                row = int(candidates[len(candidates) // 2] if len(candidates)
                          else max(1, len(mask) // 2))
                mask[row] = True
                selected = [{"class": "ordinary", "row": row}]
            data["loss_mask"] = mask
            name = f"episode_{kind}_{number:02d}.npz"
            for directory in (train, valid):
                np.savez_compressed(directory / name, **data)
            records.append({"source": str(path), "file": name,
                            "selected": selected})
    manifest = {
        "purpose": "diagnostic frozen-latent official-head overfit only",
        "samples": {"collision_terminal": 10, "pre_collision_5": 10,
                    "ordinary": 10},
        "sequences": 20, "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["samples"]))


if __name__ == "__main__":
    main()
