#!/usr/bin/env python3
"""Cache Original-Pure posterior Actor means for S2-WM-Anchor."""

import argparse
from functools import partial
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.main import make_agent


def load_config(path):
  loader = yaml.YAML(typ='safe')
  defaults = loader.load(Path('dreamerv3/dreamerv3/configs.yaml').read_text())
  saved = loader.load(path.read_text())
  return elements.Config(defaults['defaults']).update(
      defaults['quadrotor']).update(saved).update({
          'jax.precompile': False,
          'agent.anchor_scale': 0.0,
          'agent.reward_decomposition': False})


def padded(items, length=301):
  batch = len(items)
  data = {
      'vector': np.zeros((batch, length, 68), np.float32),
      'reward': np.zeros((batch, length), np.float32),
      'is_first': np.ones((batch, length), bool),
      'is_last': np.ones((batch, length), bool),
      'is_terminal': np.zeros((batch, length), bool),
      'teacher': np.zeros((batch, length), bool),
      'action_policy_action': np.zeros((batch, length, 4), np.float32),
      'diagnostic_prev_action': np.zeros((batch, length, 4), np.float32),
  }
  sizes = []
  for row, item in enumerate(items):
    size = min(len(item['vector']), length); sizes.append(size)
    for key in ('vector', 'reward', 'is_first', 'is_last', 'is_terminal',
                'teacher', 'action_policy_action'):
      data[key][row, :size] = item[key][:size]
    action = item['action'][:size]
    if size > 1:
      data['diagnostic_prev_action'][row, 1:size] = action[:-1]
  return data, sizes


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--batch-size', type=int, default=4)
  args = parser.parse_args()
  agent = make_agent(load_config(args.config))
  elements.checkpoint.load(args.checkpoint, {'agent': partial(
      agent.load, regex='^(enc|dyn|pol)/')})
  for split in ('train', 'validation'):
    sources = sorted((args.dataset / split).glob('*.npz'))
    (args.output / split).mkdir(parents=True, exist_ok=True)
    for start in range(0, len(sources), args.batch_size):
      paths = sources[start:start + args.batch_size]
      items = []
      for path in paths:
        with np.load(path) as source:
          items.append({key: source[key] for key in source.files})
      data, sizes = padded(items)
      means = np.asarray(agent.anchor_actions(data))
      for path, item, size, action_mean in zip(paths, items, sizes, means):
        item['anchor_action_mean'] = action_mean[:size].astype(np.float32)
        if 'minimum_clearance' not in item:
          raise KeyError(f'missing minimum_clearance in {path}')
        item['anchor_mask'] = (
            np.asarray(item['minimum_clearance']) < 0.70).astype(bool)
        np.savez_compressed(args.output / split / path.name, **item)
      print(split, min(start + args.batch_size, len(sources)), '/', len(sources), flush=True)


if __name__ == '__main__':
  main()
