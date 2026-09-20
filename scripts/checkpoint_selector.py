#!/usr/bin/env python3
"""Summarize five-scene gates and retain hierarchical validation candidates."""

import argparse
import csv
import datetime
import json
from collections import defaultdict
from pathlib import Path


SCENES = ('random', 'fixed0p8', 'fixed0p6', 'fixed0p3', 'low')


def scene_metrics(directory):
  episodes = defaultdict(list)
  with (directory / 'trajectory_actions.csv').open() as handle:
    for row in csv.DictReader(handle):
      episodes[int(row['episode'])].append(row)
  complete = []
  for episode, rows in sorted(episodes.items()):
    terminal = [row for row in rows if float(row['is_last'])]
    if terminal:
      complete.append((episode, rows, terminal[-1]))
  if not complete:
    raise RuntimeError(f'no completed episodes in {directory}')
  count = len(complete)
  totals = {
      key: sum(float(terminal[key]) for _, _, terminal in complete)
      for key in ('success', 'collision', 'timeout')}
  clearances = [
      min(float(row['minimum_clearance']) for row in rows)
      for _, rows, _ in complete]
  navigation = [
      max(int(row['episode_step']) for row in rows) * 0.1
      for _, rows, _ in complete]
  return {
      'episodes': count,
      'successes': int(totals['success']),
      'collisions': int(totals['collision']),
      'timeouts': int(totals['timeout']),
      'success_rate': totals['success'] / count,
      'collision_rate': totals['collision'] / count,
      'timeout_rate': totals['timeout'] / count,
      'mean_minimum_clearance': sum(clearances) / count,
      'minimum_clearance': min(clearances),
      'mean_navigation_time': sum(navigation) / count,
      'evaluation_directory': str(directory),
  }


def collect(prefix, overrides):
  result = {}
  for scene in SCENES:
    directory = overrides.get(scene, Path(f'{prefix}_{scene}'))
    result[scene] = scene_metrics(directory)
  return result


def retention(candidate, parent, tolerance=0.05):
  checks = {}
  states = []
  for scene in ('fixed0p8', 'fixed0p6'):
    parent_rate = parent[scene]['success_rate']
    candidate_rate = candidate[scene]['success_rate']
    threshold = parent_rate - tolerance
    resolution = 1.0 / candidate[scene]['episodes']
    if candidate_rate >= threshold:
      state = 'PASS'
    elif candidate_rate >= threshold - resolution:
      state = 'REVIEW'
    else:
      state = 'FAIL'
    checks[scene] = {
        'status': state,
        'parent_success_rate': parent_rate,
        'allowed_drop': tolerance,
        'threshold': threshold,
        'candidate_success_rate': candidate_rate,
        'episode_resolution': resolution,
    }
    states.append(state)
  overall = 'FAIL' if 'FAIL' in states else ('REVIEW' if 'REVIEW' in states else 'PASS')
  return overall, checks


def rank_key(entry):
  """Layered lexicographic ordering; deliberately not a weighted score."""
  scenes = entry['metrics']['scenes']
  gate = {'PASS': 2, 'REVIEW': 1}[entry['screening']]
  primary = [scenes[name] for name in ('random', 'low')]
  return (
      gate,
      min(x['success_rate'] for x in primary),
      sum(x['success_rate'] for x in primary),
      -max(x['collision_rate'] for x in primary),
      -sum(x['collision_rate'] for x in primary),
      -max(x['timeout_rate'] for x in primary),
      -sum(x['timeout_rate'] for x in primary),
      scenes['fixed0p3']['success_rate'],
      -scenes['fixed0p3']['collision_rate'],
      -scenes['fixed0p3']['timeout_rate'],
      sum(x['mean_minimum_clearance'] for x in scenes.values()),
      -sum(x['mean_navigation_time'] for x in scenes.values()),
  )


def parse_scene_paths(values):
  result = {}
  for value in values:
    scene, path = value.split('=', 1)
    if scene not in SCENES:
      raise ValueError(f'unknown scene: {scene}')
    result[scene] = Path(path)
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--update', required=True, type=int)
  parser.add_argument('--candidate-prefix', required=True)
  parser.add_argument('--candidate-eval', action='append', default=[])
  parser.add_argument('--parent-prefix', required=True)
  parser.add_argument('--parent-eval', action='append', default=[])
  parser.add_argument('--parent-checkpoint', required=True)
  parser.add_argument('--history', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--top-k', type=int, choices=(2, 3), default=3)
  parser.add_argument('--retention-tolerance', type=float, default=0.05)
  args = parser.parse_args()

  candidate = collect(args.candidate_prefix, parse_scene_paths(args.candidate_eval))
  parent = collect(args.parent_prefix, parse_scene_paths(args.parent_eval))
  screening, checks = retention(candidate, parent, args.retention_tolerance)
  entry = {
      'checkpoint': args.checkpoint,
      'update': args.update,
      'parent_checkpoint': args.parent_checkpoint,
      'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'screening': screening,
      'retention': checks,
      'metrics': {'scenes': candidate},
      'parent_metrics': {'scenes': parent},
  }
  history = json.loads(args.history.read_text()) if args.history.exists() else []
  history = [old for old in history if old['checkpoint'] != args.checkpoint]
  history.append(entry)
  eligible = [old for old in history if old['screening'] in ('PASS', 'REVIEW')]
  eligible.sort(key=rank_key, reverse=True)
  selected = eligible[:args.top_k]
  table = [{
      'rank': index + 1,
      'checkpoint': item['checkpoint'],
      'update': item['update'],
      'screening': item['screening'],
  } for index, item in enumerate(selected)]
  result = {**entry, 'top_candidates': table, 'ranking_method': [
      'retention fixed +0.8/+0.6',
      'random and low-band success/collision/timeout',
      'fixed +0.3 success/collision/timeout',
      'clearance and navigation time tie-breakers',
  ]}
  args.history.parent.mkdir(parents=True, exist_ok=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.history.write_text(json.dumps(history, indent=2) + '\n')
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  (args.output.parent / 'top_candidates.json').write_text(
      json.dumps(table, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
