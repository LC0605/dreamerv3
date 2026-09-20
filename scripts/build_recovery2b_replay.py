#!/usr/bin/env python3
"""Build Recovery-2B replay: 60% new curriculum, 40% mastered replay."""

import argparse
import json
from pathlib import Path


NEW_COUNTS = {"low": 27, "mid": 15, "high": 12, "fixed0p3": 6}
OLD_COUNTS = {"low": 16, "mid": 12, "high": 8, "fixed_new": 4}


def take(directory, prefix, count):
  paths = sorted(directory.glob(f"{prefix}_*.npz"))[:count]
  if len(paths) != count:
    raise RuntimeError(f"need {count} {prefix} episodes in {directory}, found {len(paths)}")
  return paths


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
  for prefix, count in NEW_COUNTS.items():
    selected += [("new", prefix, path) for path in take(args.new / "episodes", prefix, count)]
  for prefix, count in OLD_COUNTS.items():
    selected += [("old", prefix, path) for path in take(args.old / "train", prefix, count)]
  if len(selected) != 100:
    raise RuntimeError(f"expected 100 episodes, found {len(selected)}")
  for age, scene, source in selected:
    (train / f"{age}_{scene}_{source.name}").symlink_to(source.resolve())
  # One non-exclusive validation view per curriculum band.
  for index in (26, 41, 53, 59):
    age, scene, source = selected[index]
    (valid / f"{age}_{scene}_{source.name}").symlink_to(source.resolve())
  manifest = {
      "parent_checkpoint": "outputs/dreamerv3/stageS3A_A1_Recovery_4000/ckpt/20260830T153553F674727",
      "new_percent": 60,
      "old_mastered_percent": 40,
      "new_internal_percent": {"low_0p30_0p45": 45, "mid_0p45_0p60": 25,
                               "high_0p60_0p80": 20, "fixed_0p30": 10},
      "new_counts": NEW_COUNTS, "old_counts": OLD_COUNTS,
      "new_source": str(args.new), "old_source": str(args.old),
      "sampling": "uniform episode; batch_length=301",
  }
  (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
