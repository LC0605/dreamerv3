#!/usr/bin/env python3
"""Keep DAgger recovery states while masking tails of failed episodes."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('source', type=Path)
  parser.add_argument('output', type=Path)
  parser.add_argument('--failure-tail', type=int, default=20)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  stats = {'episodes': 0, 'success': 0, 'failed': 0,
           'teacher_rows': 0, 'masked_failure_rows': 0}
  for split in ('train', 'validation'):
    (args.output / split).mkdir()
    for source in sorted((args.source / split).glob('episode_*.npz')):
      with np.load(source) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
      success = bool(data['is_terminal'][-1]) and float(data['reward'][-1]) > 50
      if success:
        stats['success'] += 1
      else:
        stats['failed'] += 1
        start = max(0, len(data['teacher']) - args.failure_tail)
        before = int(data['teacher'][start:].sum())
        data['teacher'][start:] = False
        stats['masked_failure_rows'] += before
      stats['episodes'] += 1
      stats['teacher_rows'] += int(data['teacher'].sum())
      np.savez_compressed(args.output / split / source.name, **data)
  stats.update(source=str(args.source), failure_tail=args.failure_tail)
  (args.output / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n')
  print(json.dumps(stats, indent=2))


if __name__ == '__main__':
  main()
