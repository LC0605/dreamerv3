#!/usr/bin/env python3
"""Create a hard-linked teacher dataset containing only successful episodes."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('source', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--validation-fraction', type=float, default=0.2)
  args = parser.parse_args()
  files = sorted(args.source.glob('*/episode_*.npz'))
  successful = []
  rejected = {'collision': 0, 'timeout': 0}
  rows = 0
  for path in files:
    with np.load(path) as data:
      terminal = bool(data['is_terminal'][-1])
      terminal_reward = float(data['reward'][-1])
      if terminal and terminal_reward > 50:
        successful.append(path)
        rows += len(data['reward'])
      elif terminal:
        rejected['collision'] += 1
      else:
        rejected['timeout'] += 1
  if not successful:
    raise RuntimeError('no successful episodes found')
  args.output.mkdir(parents=True, exist_ok=False)
  (args.output / 'train').mkdir()
  (args.output / 'validation').mkdir()
  valid_count = max(1, round(len(successful) * args.validation_fraction))
  split_at = len(successful) - valid_count
  for index, source in enumerate(successful):
    split = 'train' if index < split_at else 'validation'
    os.link(source, args.output / split / source.name)
  summary = {
      'source': str(args.source),
      'success_only': True,
      'episodes': len(successful),
      'train_episodes': split_at,
      'validation_episodes': valid_count,
      'transitions': rows,
      'rejected': rejected,
  }
  (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
