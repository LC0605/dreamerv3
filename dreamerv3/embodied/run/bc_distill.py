from pathlib import Path
import datetime
import json
import subprocess

import elements
import embodied
import numpy as np


def _git_commit(directory):
  try:
    return subprocess.run(
        ['git', '-C', str(directory), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True).stdout.strip()
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None


def _record_candidate(checkpoint, args, update):
  """Record provenance beside an immutable checkpoint candidate."""
  checkpoint = Path(str(checkpoint)).resolve()
  logdir = Path(str(args.logdir)).resolve()
  try:
    repo = subprocess.run(
        ['git', '-C', str(logdir), 'rev-parse', '--show-toplevel'],
        check=True, capture_output=True, text=True).stdout.strip()
  except (FileNotFoundError, subprocess.CalledProcessError):
    repo = Path.cwd()
  record = {
      'update': int(update),
      'checkpoint_path': str(checkpoint),
      'parent_checkpoint': str(args.from_checkpoint),
      'config': str(logdir / 'config.yaml'),
      'replay': str(args.bc_dataset),
      'git_commit': _git_commit(repo),
      'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }
  (checkpoint / 'candidate.json').write_text(json.dumps(record, indent=2) + '\n')
  history = logdir / 'candidate_checkpoints.jsonl'
  with history.open('a') as handle:
    handle.write(json.dumps(record) + '\n')


def _episodes(directory, length):
  result = []
  for path in sorted(Path(directory).glob('*.npz')):
    with np.load(path) as source:
      data = {key: source[key] for key in source.files}
    original_size = len(data['vector'])
    if not data['is_first'][0] or not data['is_last'][-1]:
      raise ValueError(f'invalid episode boundary in {path}')
    if not np.array_equal(data['episode_step'], np.arange(original_size)):
      raise ValueError(f'non-monotonic episode_step in {path}')
    # When a shorter sequence is requested, retain the prefix from the true
    # episode start. This is a memory-safe takeover smoke test: every retained
    # posterior still sees its complete PPO action history. The final cropped
    # row is not relabelled as is_last/terminal.
    size = min(original_size, length)
    batch = {
        'vector': np.zeros((length, 68), np.float32),
        'reward': np.zeros(length, np.float32),
        'reward_dense': np.zeros(length, np.float32),
        'terminal_outcome': np.zeros(length, np.int32),
        'is_first': np.ones(length, bool),
        'is_last': np.ones(length, bool),
        'is_terminal': np.zeros(length, bool),
        'teacher': np.zeros(length, bool),
        'action_policy_action': np.zeros((length, 4), np.float32),
        'action': np.zeros((length, 4), np.float32),
        'consec': np.zeros(length, np.int32),
        'stepid': np.zeros((length, 20), np.uint8),
        'loss_mask': np.zeros(length, bool),
        # Loader-only metadata. This is removed before yielding JAX batches.
        # 2=collision-terminal episode, 1=pre-collision sequence, 0=ordinary.
        '_risk_class': np.zeros(length, np.int32),
        '_scene_class': np.zeros(length, np.int32),
        '_curriculum_class': np.zeros(length, np.int32),
        'risk_stratum': np.full(length, -1, np.int32),
        'anchor_action_mean': np.zeros((length, 4), np.float32),
        'anchor_mask': np.zeros(length, bool),
        'continue_importance': np.ones(length, np.float32),
    }
    for key in (
        'vector', 'reward', 'is_first', 'is_last', 'is_terminal', 'teacher',
        'action_policy_action', 'action'):
      batch[key][:size] = data[key][:size]
    # Reward decomposition is optional for older teacher datasets. When the
    # fields are absent, preserve the legacy total-reward target and mark all
    # outcomes nonterminal; decomposition runs validate their presence before
    # training, so they cannot silently use this fallback.
    if 'reward_dense' in data:
      batch['reward_dense'][:size] = data['reward_dense'][:size]
    else:
      batch['reward_dense'][:size] = data['reward'][:size]
    if 'terminal_outcome' in data:
      batch['terminal_outcome'][:size] = data['terminal_outcome'][:size]
    if 'anchor_action_mean' in data:
      batch['anchor_action_mean'][:size] = data['anchor_action_mean'][:size]
    if 'anchor_mask' in data:
      batch['anchor_mask'][:size] = data['anchor_mask'][:size]
    if 'loss_mask' in data:
      batch['loss_mask'][:size] = data['loss_mask'][:size].astype(bool)
    else:
      batch['loss_mask'][:size] = True
    batch['risk_stratum'][:size] = 0
    if 'pre_collision' in data:
      batch['risk_stratum'][:size][data['pre_collision'][:size].astype(bool)] = 1
    elif 'terminal_outcome' in data and np.any(data['terminal_outcome'][:size] == 2):
      terminal = int(np.flatnonzero(data['terminal_outcome'][:size] == 2)[-1])
      batch['risk_stratum'][max(0, terminal - 10):terminal] = 1
    if 'collision_terminal' in data:
      batch['risk_stratum'][:size][
          data['collision_terminal'][:size].astype(bool)] = 2
    elif 'terminal_outcome' in data:
      batch['risk_stratum'][:size][data['terminal_outcome'][:size] == 2] = 2
    collision = bool(data.get('collision_terminal', np.zeros(original_size, bool)).any())
    collision |= bool('terminal_outcome' in data and (data['terminal_outcome'] == 2).any())
    pre_collision = bool(data.get('pre_collision', np.zeros(original_size, bool)).any())
    batch['_risk_class'][:] = 2 if collision else (1 if pre_collision else 0)
    offset = float(np.asarray(data.get('offset', np.asarray([0.3]))).reshape(-1)[0])
    batch['_scene_class'][:] = 0 if offset >= .7 else (1 if offset >= .5 else 2)
    batch['_curriculum_class'][:] = int(bool(np.asarray(
        data.get('curriculum_new', np.asarray(False))).reshape(-1)[0]))
    episode_id = int(data['episode_id'][0])
    for index in range(length):
      batch['stepid'][index, :8] = np.frombuffer(
          episode_id.to_bytes(8, 'big'), np.uint8)
      batch['stepid'][index, 8:12] = np.frombuffer(
          index.to_bytes(4, 'big'), np.uint8)
    result.append(batch)
  if not result:
    raise FileNotFoundError(f'no episode files in {directory}')
  return result


def _stream(episodes, batch_size, seed, shuffle, fixed=False, stratified=False,
            scene_balanced=False, curriculum_new_fraction=0.0):
  rng = np.random.default_rng(seed)
  cursor = 0
  pools = {
      value: np.asarray([i for i, episode in enumerate(episodes)
                         if int(episode['_risk_class'][0]) == value], np.int32)
      for value in (0, 1, 2)}
  scene_pools = {
      value: np.asarray([i for i, episode in enumerate(episodes)
                         if int(episode['_scene_class'][0]) == value], np.int32)
      for value in (0, 1, 2)}
  curriculum_pools = {
      value: np.asarray([i for i, episode in enumerate(episodes)
                         if int(episode['_curriculum_class'][0]) == value], np.int32)
      for value in (0, 1)}
  if curriculum_new_fraction and any(not len(x) for x in curriculum_pools.values()):
    raise ValueError(f'missing curriculum pool: '
                     f'{ {k: len(v) for k, v in curriculum_pools.items()} }')
  if scene_balanced and any(not len(x) for x in scene_pools.values()):
    raise ValueError(f'missing scene pool: { {k: len(v) for k, v in scene_pools.items()} }')
  # A collision episode is also the only source of a complete pre-collision
  # temporal sequence. Use it for both requested strata without duplicating
  # terminal timesteps inside a sequence.
  if stratified and not len(pools[1]):
    pools[1] = pools[2]
  if stratified and (not len(pools[0]) or not len(pools[1]) or not len(pools[2])):
    raise ValueError(f'missing risk stratum: { {k: len(v) for k, v in pools.items()} }')
  while True:
    if curriculum_new_fraction:
      period = 20
      new_slots = min(max(int(round(float(curriculum_new_fraction) * period)), 1),
                      period - 1)
      classes = [1 if (cursor + i) % period < new_slots else 0
                 for i in range(batch_size)]
      indices = np.asarray([rng.choice(curriculum_pools[value]) for value in classes])
      cursor = (cursor + batch_size) % period
    elif scene_balanced:
      # Conservative takeover curriculum: +0.8/+0.6/+0.3 = 20/20/60.
      schedule = (0, 0, 1, 1, 2, 2, 2, 2, 2, 2)
      classes = [schedule[(cursor + i) % len(schedule)] for i in range(batch_size)]
      indices = np.asarray([rng.choice(scene_pools[value]) for value in classes])
      cursor = (cursor + batch_size) % len(schedule)
    elif stratified:
      # Repeating 10-slot schedule: 40% collision-terminal, 30%
      # pre-collision, 30% ordinary. With batch_size=10 every batch has the
      # exact composition; smaller batches preserve it over successive calls.
      schedule = (2, 2, 2, 2, 1, 1, 1, 0, 0, 0)
      classes = [schedule[(cursor + i) % len(schedule)] for i in range(batch_size)]
      indices = np.asarray([rng.choice(pools[value]) for value in classes])
      cursor = (cursor + batch_size) % len(schedule)
    elif fixed:
      indices = np.arange(batch_size) % len(episodes)
    elif shuffle:
      indices = rng.integers(0, len(episodes), batch_size)
    else:
      indices = np.arange(cursor, cursor + batch_size) % len(episodes)
      cursor = (cursor + batch_size) % len(episodes)
    result = {
        key: np.stack([episodes[int(index)][key] for index in indices])
        for key in episodes[0] if key not in (
            '_risk_class', '_scene_class', '_curriculum_class')}
    if stratified:
      # Importance correction maps the deliberately enriched episode sampler
      # back to the natural episode/transition mixture. Collision sequences
      # occupy schedule classes 1 and 2 (70% total); all others occupy 30%.
      n_total = len(episodes)
      n_collision = len(pools[2])
      n_other = n_total - n_collision
      weights = np.asarray([
          n_collision / max(n_total * .7, 1e-8)
          if int(episodes[int(index)]['_risk_class'][0]) == 2
          else n_other / max(n_total * .3, 1e-8)
          for index in indices], np.float32)
      result['continue_importance'] = np.broadcast_to(
          weights[:, None], result['continue_importance'].shape).copy()
    yield result


def bc_distill(make_agent, make_logger, args):
  if not args.bc_dataset:
    raise ValueError('run.bc_dataset is required')
  if args.bc_updates <= 0:
    raise ValueError('run.bc_updates must be positive')
  agent = make_agent()
  logger = make_logger()
  step = logger.step
  root = Path(args.bc_dataset)
  train_eps = _episodes(root / 'train', args.batch_length)
  valid_eps = _episodes(root / 'validation', args.report_length)
  print(f'Offline BC episodes: train={len(train_eps)} validation={len(valid_eps)}')

  keep = set(agent.spaces)
  filtered = lambda stream: (
      {key: value for key, value in batch.items() if key in keep}
      for batch in stream)
  train_stream = iter(agent.stream(filtered(_stream(
      train_eps, args.batch_size, seed=0, shuffle=True,
      fixed=args.bc_fixed_train, stratified=args.bc_risk_stratified,
      scene_balanced=args.bc_scene_balanced,
      curriculum_new_fraction=args.bc_curriculum_new_fraction))))
  valid_stream = iter(agent.stream(filtered(_stream(
      valid_eps, args.batch_size, seed=1, shuffle=False))))
  train_carry = agent.init_train(args.batch_size)
  valid_carry = agent.init_report(args.batch_size)

  logdir = elements.Path(args.logdir)
  cp = elements.Checkpoint(logdir / 'ckpt', keep=10)
  cp.step = step
  cp.agent = agent
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=lambda data: agent.load(data, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  last_saved_step = int(step)

  for update in range(int(step), int(args.bc_updates)):
    train_carry, _, metrics = agent.train(train_carry, next(train_stream))
    step.increment()
    if metrics:
      logger.add(metrics, prefix='train')
    current_step = int(step)
    should_validate = (
        current_step % args.bc_validate_every == 0 or
        current_step == args.bc_updates)
    if should_validate:
      # Reset recurrent state and cover the validation split once.
      valid_carry = agent.init_report(args.batch_size)
      agg = elements.Agg()
      batches = int(np.ceil(len(valid_eps) / args.batch_size))
      for _ in range(batches):
        valid_carry, mets = agent.report(valid_carry, next(valid_stream))
        agg.add(mets)
      logger.add(agg.result(), prefix='validation')
      logger.write()
    candidate_every = int(args.bc_candidate_every)
    should_candidate = bool(
        candidate_every and current_step % candidate_every == 0)
    if should_validate:
      cp.save()
      last_saved_step = current_step
    if should_candidate:
      candidate = (
          Path(str(args.logdir)) / 'candidates' /
          f'update_{current_step:09d}')
      cp.save(path=candidate)
      _record_candidate(candidate, args, current_step)

  # Preserve the final-checkpoint guarantee without creating a second,
  # indistinguishable checkpoint when the final update was already saved.
  if last_saved_step != int(step):
    cp.save()
  logger.write()
  logger.close()
