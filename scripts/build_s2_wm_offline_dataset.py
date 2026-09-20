#!/usr/bin/env python3
"""Build the immutable mixed replay used by Stage S2-WM."""

import argparse
from pathlib import Path

import numpy as np


def converted(source):
  with np.load(source) as item:
    data = {key: item[key] for key in item.files}
  if 'terminal_outcome' not in data:
    outcome = np.zeros(len(data['reward']), np.int32)
    terminal = data['is_terminal'].astype(bool)
    collision = data.get('collision_terminal', np.zeros(len(outcome), bool)).astype(bool)
    outcome[terminal & ~collision] = 1
    outcome[collision] = 2
    data['terminal_outcome'] = outcome
    component = np.zeros(len(outcome), np.float32)
    component[outcome == 1] = 100.0
    component[outcome == 2] = -100.0
    data['reward_dense'] = data['reward'].astype(np.float32) - component
  residual = data['reward'] - (data['reward_dense'] + np.choose(
      data['terminal_outcome'], [0.0, 100.0, -100.0, -100.0]))
  if np.max(np.abs(residual)) > 1e-5:
    raise ValueError((source, float(np.max(np.abs(residual)))))
  return data


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--new', type=Path, required=True)
  parser.add_argument('--old', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  for split in ('train', 'validation'):
    (args.output / split).mkdir(parents=True, exist_ok=True)
  new = sorted((args.new / 'episodes').glob('episode_*.npz'))
  # Fixed split by episode id, with every fifth 50-episode block contributing
  # four validation episodes. This preserves all offsets and never touches the
  # permanent seed-35700 audit set.
  for index, source in enumerate(new):
    split = 'validation' if index % 50 in (46, 47, 48, 49) else 'train'
    np.savez_compressed(
        args.output / split / f'new_{source.name}', **converted(source))
  for split in ('train', 'validation'):
    for source in sorted((args.old / split).glob('episode_*.npz')):
      np.savez_compressed(
          args.output / split / f'old_{source.name}', **converted(source))
  for split in ('train', 'validation'):
    files = list((args.output / split).glob('*.npz'))
    outcomes = []
    for path in files:
      with np.load(path) as item:
        outcomes.extend(item['terminal_outcome'].tolist())
    values, counts = np.unique(outcomes, return_counts=True)
    print(split, len(files), dict(zip(values.tolist(), counts.tolist())))


if __name__ == '__main__':
  main()
