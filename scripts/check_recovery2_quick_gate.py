#!/usr/bin/env python3
"""Apply immutable Recovery-2 stop conditions to completed quick evaluations."""

import argparse
import csv
import json
import math
from pathlib import Path


def metrics(directory):
  rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
  row = next(row for row in reversed(rows) if "epstats/episode_count" in row)
  success = float(row["epstats/log/success/max"])
  collision = float(row["epstats/log/collision/max"])
  if not all(math.isfinite(value) for value in (success, collision)):
    raise RuntimeError(f"non-finite evaluation metric in {directory}")
  return success, collision


def danger(directory, collision):
  with (directory / "trajectory_actions.csv").open() as handle:
    rows = list(csv.DictReader(handle))
  near = [row for row in rows if float(row["minimum_clearance"]) < 0.70]
  forward = sum(float(row["action_forward"]) for row in near) / max(len(near), 1)
  lateral = sum(abs(float(row["action_route_lateral"])) for row in near) / max(len(near), 1)
  by_episode = {}
  for row in rows:
    by_episode.setdefault(row["episode"], []).append(float(row["minimum_clearance"]))
  clearance = sum(min(values) for values in by_episode.values()) / max(len(by_episode), 1)
  return forward > .0440 and lateral < .1648 and clearance < .352 and collision > .02


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", required=True)
  parser.add_argument("--history", type=Path, required=True)
  args = parser.parse_args()
  root = Path("outputs/dreamerv3")
  values = {}
  for scene in ("random", "fixed0p8", "fixed0p6", "fixed0p3", "low"):
    values[scene] = metrics(root / f"eval_{args.label}_{scene}")
  failures = []
  if values["fixed0p8"][0] < .83:
    failures.append("fixed +0.8 success fell more than 5pp below parent 88%")
  if values["fixed0p6"][0] < .84:
    failures.append("fixed +0.6 success fell more than 5pp below parent 89%")
  if danger(root / f"eval_{args.label}_random", values["random"][1]):
    failures.append("forward/lateral/clearance/collision danger pattern returned")
  history = json.loads(args.history.read_text()) if args.history.exists() else []
  current = {"label": args.label,
             "mean_success": sum(x[0] for x in values.values()) / len(values),
             "mean_collision": sum(x[1] for x in values.values()) / len(values),
             "scenes": values}
  history.append(current)
  args.history.parent.mkdir(parents=True, exist_ok=True)
  args.history.write_text(json.dumps(history, indent=2) + "\n")
  if len(history) >= 3:
    if history[-1]["mean_collision"] > history[-2]["mean_collision"] > history[-3]["mean_collision"]:
      failures.append("mean collision worsened at two consecutive checkpoints")
    if history[-1]["mean_success"] < history[-2]["mean_success"] < history[-3]["mean_success"]:
      failures.append("overall success degraded at two consecutive checkpoints")
  print(json.dumps(current, indent=2))
  if failures:
    raise SystemExit("STOP: " + "; ".join(failures))


if __name__ == "__main__":
  main()
