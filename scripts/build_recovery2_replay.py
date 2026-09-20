#!/usr/bin/env python3
"""Build an immutable 100-episode Recovery-2 replay without copying data."""

import argparse
import json
import os
from pathlib import Path


def link(source, destination):
  destination.symlink_to(source.resolve())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--new", type=Path, required=True)
  parser.add_argument("--old", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  train = args.output / "train"
  valid = args.output / "validation"
  train.mkdir(parents=True)
  valid.mkdir()

  selected = []
  for source in sorted((args.new / "episodes").glob("*.npz")):
    selected.append(("new", source))
  # Old rehearsal contributes 5 mid and 5 high episodes. New collection was
  # reduced by exactly those counts, preserving the requested total mixture.
  for prefix in ("mid", "high"):
    candidates = sorted((args.old / "train").glob(f"{prefix}_*.npz"))[:5]
    if len(candidates) != 5:
      raise RuntimeError(f"need five old {prefix} episodes, found {len(candidates)}")
    selected.extend(("old", path) for path in candidates)
  if len(selected) != 100:
    raise RuntimeError(f"expected 100 episodes, found {len(selected)}")
  for age, source in selected:
    link(source, train / f"{age}_{source.name}")
  # Validation is a read-only view and does not remove episodes from training.
  for index in (44, 64, 79, 89):
    age, source = selected[index]
    link(source, valid / f"{age}_{source.name}")
  composition = {"low": 45, "mid": 25, "high": 20, "fixed0p3": 10,
                 "new": 90, "old": 10}
  (args.output / "manifest.json").write_text(json.dumps({
      "composition_percent": composition,
      "old_replay": str(args.old), "new_replay": str(args.new),
      "sampling": "uniform_episode; batch_length=301",
  }, indent=2) + "\n")


if __name__ == "__main__":
  main()
