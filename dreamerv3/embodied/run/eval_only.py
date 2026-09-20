from collections import defaultdict
from functools import partial as bind
import os
from pathlib import Path
import csv

import elements
import embodied
import numpy as np


def eval_only(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()
  gif_output = os.environ.get('DREAMER_GIF_OUTPUT', '')
  gif_frames = []
  gif_stride = max(1, int(os.environ.get('DREAMER_GIF_STRIDE', '2')))
  gif_max_frames = max(1, int(os.environ.get('DREAMER_GIF_MAX_FRAMES', '300')))
  gif_step = 0
  gif_episode = 0
  gif_episode_step = 0
  diagnostics = []
  diagnostic_episode = 0
  diagnostic_episode_step = 0
  completed_episodes = 0
  target_episodes = int(os.environ.get('DREAMER_EVAL_EPISODES', '0'))

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    nonlocal completed_episodes
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
        if key == 'log/distance' and not tran['is_first']:
          episode.add(key + '/min', value, agg='min')
        elif key in ('log/minimum_clearance', 'log/boundary_clearance'):
          episode.add(key + '/min', value, agg='min')
    if tran['is_last']:
      completed_episodes += 1
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)
      epstats.add('episode_count', 1, agg='sum')

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)

  def record_diagnostics(tran, worker):
    nonlocal diagnostic_episode, diagnostic_episode_step
    if worker != 0:
      return
    if bool(tran['is_first']):
      diagnostic_episode += 1
      diagnostic_episode_step = 0
    diagnostics.append({
        'episode': int(diagnostic_episode),
        'episode_step': int(diagnostic_episode_step),
        'is_first': int(bool(tran['is_first'])),
        'is_last': int(bool(tran['is_last'])),
        'success': float(tran['log/success']),
        'collision': float(tran['log/collision']),
        'timeout': float(tran['log/timeout']),
        'position_x': float(tran['log/position_x']),
        'position_y': float(tran['log/position_y']),
        'position_z': float(tran['log/position_z']),
        'action_forward': float(tran['log/action_forward']),
        'action_lateral': float(tran['log/action_lateral']),
        'action_vertical': float(tran['log/action_vertical']),
        'action_yaw_rate': float(tran['log/action_yaw_rate']),
        'action_route_lateral': float(tran['log/action_route_lateral']),
        'obstacle_lateral_relative': float(
            tran['log/obstacle_lateral_relative']),
        'minimum_clearance': float(tran['log/minimum_clearance']),
        'distance': float(tran['log/distance']),
        'reward': float(tran['reward']),
        'scene_start_x': float(tran['log/scene_start_x']),
        'scene_start_y': float(tran['log/scene_start_y']),
        'scene_start_z': float(tran['log/scene_start_z']),
        'scene_goal_x': float(tran['log/scene_goal_x']),
        'scene_goal_y': float(tran['log/scene_goal_y']),
        'scene_goal_z': float(tran['log/scene_goal_z']),
        'scene_obstacle_x': float(tran['log/scene_obstacle_x']),
        'scene_obstacle_y': float(tran['log/scene_obstacle_y']),
        'scene_obstacle_z': float(tran['log/scene_obstacle_z']),
        'scene_obstacle_half_x': float(tran['log/scene_obstacle_half_x']),
        'scene_obstacle_half_y': float(tran['log/scene_obstacle_half_y']),
        'scene_obstacle_half_z': float(tran['log/scene_obstacle_half_z']),
        **{f'vector_{index:02d}': float(value)
           for index, value in enumerate(np.asarray(tran['vector']))},
    })
    diagnostic_episode_step += 1

  driver.on_step(record_diagnostics)
  if gif_output:
    if args.debug is not True:
      raise ValueError('GIF recording requires --run.debug True')

    def record_frame(tran, worker):
      nonlocal gif_step, gif_episode, gif_episode_step
      if worker == 0 and bool(tran['is_first']):
        gif_episode += 1
        gif_episode_step = 0
      if worker == 0 and len(gif_frames) < gif_max_frames:
        if gif_step % gif_stride == 0:
          status = ''
          if bool(tran['is_last']):
            if float(tran['log/success']) > 0.5:
              status = 'SUCCESS'
            elif float(tran['log/collision']) > 0.5:
              status = 'COLLISION'
            elif float(tran['log/out_of_bounds']) > 0.5:
              status = 'OUT OF BOUNDS'
            else:
              status = 'TIMEOUT'
          gif_frames.append((driver.envs[0].render(), {
              'episode': gif_episode,
              'step': gif_episode_step,
              'distance': float(tran['log/distance']),
              'speed': float(tran['log/speed']),
              'status': status,
          }))
        gif_step += 1
        gif_episode_step += 1

    driver.on_step(record_frame)

  # Distillation checkpoints can contain optimizer-only state that is absent
  # from the deployment agent. Honor the same selective checkpoint loading
  # contract as the training runner.
  elements.checkpoint.load(args.from_checkpoint, dict(
      agent=bind(agent.load, regex=args.from_checkpoint_regex)))

  print('Start evaluation')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  while step < args.steps and (not target_episodes or completed_episodes < target_episodes):
    driver(policy, steps=10)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  # Preserve the final partial logging window. Without this flush, evaluation
  # can silently omit many completed episodes when it ends between clock ticks.
  logger.add(epstats.result(), prefix='epstats')
  logger.add(usage.stats(), prefix='usage')
  logger.add({'fps/policy': policy_fps.result()})
  logger.write()

  logger.close()
  driver.close()
  if diagnostics:
    diagnostic_path = Path(str(logdir)) / 'trajectory_actions.csv'
    with diagnostic_path.open('w', newline='') as stream:
      writer = csv.DictWriter(stream, fieldnames=diagnostics[0].keys())
      writer.writeheader()
      writer.writerows(diagnostics)
    try:
      import matplotlib.pyplot as plt
      rows = [row for row in diagnostics if not row['is_first']]
      fig, axes = plt.subplots(1, 3, figsize=(15, 4))
      axes[0].plot([row['position_x'] for row in rows],
                   [row['position_y'] for row in rows], linewidth=0.8)
      axes[0].set(title='Evaluation trajectories', xlabel='x (m)', ylabel='y (m)')
      axes[0].axis('equal')
      axes[1].plot([row['action_forward'] for row in rows], label='vx world')
      axes[1].plot([row['action_lateral'] for row in rows], label='vy world')
      axes[1].plot([row['action_route_lateral'] for row in rows],
                   label='route lateral', alpha=0.8)
      axes[1].set(title='Action time series', xlabel='evaluation step')
      axes[1].legend()
      axes[2].scatter(
          [row['obstacle_lateral_relative'] for row in rows],
          [row['action_route_lateral'] for row in rows], s=3, alpha=0.25)
      axes[2].axhline(0, color='black', linewidth=0.5)
      axes[2].axvline(0, color='black', linewidth=0.5)
      axes[2].set(title='Obstacle response',
                  xlabel='obstacle lateral relative (m)',
                  ylabel='route-lateral action')
      fig.tight_layout()
      fig.savefig(Path(str(logdir)) / 'trajectory_action_diagnostics.png', dpi=160)
      plt.close(fig)
    except ImportError:
      print('matplotlib unavailable; wrote CSV diagnostics only')
  if gif_output and gif_frames:
    from PIL import Image, ImageDraw
    output = Path(gif_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame, info in gif_frames:
      image = Image.fromarray(frame).convert('RGB')
      draw = ImageDraw.Draw(image, 'RGBA')
      draw.rounded_rectangle((10, 10, 272, 82), radius=8, fill=(0, 0, 0, 175))
      draw.text((20, 18), 'Drone: small black cross   Goal: green sphere',
                fill=(255, 255, 255, 255))
      draw.text(
          (20, 40),
          f"Episode {info['episode']}  Step {info['step']}  "
          f"Distance {info['distance']:.2f} m  Speed {info['speed']:.2f} m/s",
          fill=(255, 255, 255, 255))
      if info['status']:
        success = info['status'] == 'SUCCESS'
        fill = (24, 160, 64, 225) if success else (200, 48, 40, 225)
        draw.rounded_rectangle((10, 92, 190, 126), radius=7, fill=fill)
        draw.text((20, 101), info['status'], fill=(255, 255, 255, 255))
      images.append(image)
    images[0].save(
        output, save_all=True, append_images=images[1:], duration=100 * gif_stride,
        loop=0, optimize=True)
    print(f'Wrote evaluation GIF with {len(images)} frames: {output}')
