#!/usr/bin/env python3
"""Aggregate independent static-obstacle evaluation batches."""

import argparse
import json
from pathlib import Path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('output', type=Path)
  parser.add_argument('evaldirs', nargs='+', type=Path)
  parser.add_argument('--control-dt', type=float, default=0.1)
  args = parser.parse_args()

  totals = {key: 0.0 for key in (
      'episodes', 'successes', 'collisions', 'out_of_bounds', 'timeouts',
      'episode_steps', 'minimum_clearance', 'path_length', 'action_magnitude')}
  batches = []
  for directory in args.evaldirs:
    summary = json.loads((directory / 'evaluation_summary.json').read_text())
    count = float(summary['episodes'])
    # Recover lengths from per-episode rows. This also supports evaluations
    # produced before mean_episode_length was fixed in the shell wrapper.
    steps = 0.0
    for line in (directory / 'metrics.jsonl').read_text().splitlines():
      steps += float(json.loads(line).get('episode/length', 0.0))
    totals['episodes'] += count
    totals['episode_steps'] += steps
    for key in ('successes', 'collisions', 'out_of_bounds', 'timeouts'):
      totals[key] += float(summary[key])
    for key in ('minimum_clearance', 'path_length', 'action_magnitude'):
      totals[key] += count * float(summary[f'mean_{key}'])
    batches.append({'directory': str(directory), 'episodes': int(count)})

  episodes = totals['episodes']
  result = {
      'batches': batches,
      'episodes': int(totals['episodes']),
      'successes': int(totals['successes']),
      'collisions': int(totals['collisions']),
      'out_of_bounds': int(totals['out_of_bounds']),
      'timeouts': int(totals['timeouts']),
  }
  for singular, plural in (
      ('success', 'successes'), ('collision', 'collisions'),
      ('out_of_bounds', 'out_of_bounds'), ('timeout', 'timeouts')):
    result[f'{singular}_rate'] = totals[plural] / episodes
  result['mean_episode_steps'] = totals['episode_steps'] / episodes
  result['mean_navigation_time_seconds'] = (
      result['mean_episode_steps'] * args.control_dt)
  for key in ('minimum_clearance', 'path_length', 'action_magnitude'):
    result[f'mean_{key}'] = totals[key] / episodes
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
