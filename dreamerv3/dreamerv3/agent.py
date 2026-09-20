import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def last_valid_indices(mask, count):
  """Return the final `count` valid time indices for each batch row.

  Offline episodes are padded on the right for static JAX shapes. Selecting
  `[:, -count:]` therefore chooses padding for every episode shorter than the
  batch length. If a very short row has fewer than `count` valid positions,
  repeat its first valid position rather than ever returning padding.
  """
  assert mask.ndim == 2, mask.shape
  count = min(int(count), mask.shape[1])
  positions = jnp.broadcast_to(jnp.arange(mask.shape[1]), mask.shape)
  ranked = jnp.where(mask, positions, -1)
  indices = jnp.argsort(ranked, axis=1)[:, -count:]
  selected_valid = jnp.take_along_axis(mask, indices, axis=1)
  first_valid = jnp.argmax(mask, axis=1)[:, None]
  return jnp.where(selected_valid, indices, first_valid)


def take_time(tree, indices):
  """Gather a per-batch set of time indices from a tensor pytree."""
  def take(value):
    shape = (*indices.shape, *((1,) * (value.ndim - 2)))
    gather = indices.reshape(shape)
    gather = jnp.broadcast_to(gather, (*indices.shape, *value.shape[2:]))
    return jnp.take_along_axis(value, gather, axis=1)
  return jax.tree.map(take, tree)


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = (
        'is_first', 'is_last', 'is_terminal', 'reward', 'teacher',
        *[f'action_policy_{key}' for key in act_space])
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    if config.reward_decomposition:
      if config.outcome_mode == 'four_class':
        # 0 nonterminal, 1 success, 2 collision, 3 out-of-bound. Timeout
        # remains nonterminal because is_last=True but is_terminal=False.
        classes = 4
      elif config.outcome_mode == 'conditional_terminal':
        # Terminal-only identity: 0 success, 1 collision. OOB is intentionally
        # excluded until real OOB supervision exists.
        classes = 2
      else:
        raise ValueError(config.outcome_mode)
      outcome = elements.Space(np.int32, (), 0, classes)
      self.outcome = embodied.jax.MLPHead(
          outcome, **config.outcome_head, name='outcome')
    else:
      self.outcome = None
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = []
    if not config.freeze_world_model:
      if not config.freeze_dynamics:
        self.modules.append(self.dyn)
      if not config.freeze_encoder:
        self.modules.append(self.enc)
      if not config.freeze_decoder:
        self.modules.append(self.dec)
      if not config.freeze_reward:
        self.modules.append(self.rew)
      if self.outcome is not None and not config.freeze_outcome:
        self.modules.append(self.outcome)
      if not config.freeze_continue:
        self.modules.append(self.con)
    if not config.freeze_policy and not config.separate_actor_opt:
      self.modules.append(self.pol)
    if not config.freeze_value:
      self.modules.append(self.val)
    if not self.modules:
      raise ValueError('At least one trainable module is required')
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')
    self.actoropt = (
        embodied.jax.Optimizer(
            self.pol, self._make_opt(**config.actor_opt), summary_depth=1,
            name='actoropt')
        if config.separate_actor_opt and not config.freeze_policy else None)
    self.bcopt = (
        embodied.jax.Optimizer(
            self.pol, self._make_opt(**config.opt), summary_depth=1,
            name='bcopt')
        if config.bc_only else None)

    scales = self.config.loss_scales.copy()
    if config.reward_decomposition:
      scales['outcome'] = scales.get('outcome', 1.0)
    else:
      scales.pop('outcome', None)
    if config.anchor_scale:
      scales['anchor'] = (
          1.0 if config.anchor_schedule_enabled else config.anchor_scale)
    else:
      scales.pop('anchor', None)
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    spaces['loss_mask'] = elements.Space(bool)
    spaces['risk_stratum'] = elements.Space(np.int32)
    if self.config.anchor_scale:
      spaces['anchor_action_mean'] = elements.Space(np.float32, (4,))
      spaces['anchor_mask'] = elements.Space(bool)
    spaces['continue_importance'] = elements.Space(np.float32)
    if self.config.reward_decomposition:
      spaces['reward_dense'] = elements.Space(np.float32)
      spaces['terminal_outcome'] = elements.Space(np.int32)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    # Exploration belongs to data collection, not deployment evaluation.
    # Sampling here unconditionally made `mode='eval'` stochastic and caused
    # large milestone-to-milestone variance even for a frozen checkpoint.
    act = (
        jax.tree.map(lambda x: x.pred(), policy)
        if mode == 'eval' else sample(policy))
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    if self.config.bc_only:
      # The outer JAX agent discovers checkpoint parameters by tracing train().
      # Initialize the ordinary Dreamer modules once, but never execute this
      # path during actual BC updates.
      if nj.creating():
        self.opt(self.loss, carry, obs, prevact, training=False, has_aux=True)
        self.slowval.update()
      metrics, (carry, mets) = self.bcopt(
          self.bc_loss, carry, obs, prevact, has_aux=True)
      metrics.update(mets)
      carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
      return carry, {}, metrics
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, dict(obs), prevact, training=True, has_aux=True)
    metrics.update(mets)
    if self.actoropt is not None:
      actor_metrics, _ = self.actoropt(
          self.loss, carry, dict(obs), prevact, training=False, has_aux=True)
      metrics.update(prefix(actor_metrics, 'actor'))
    if not self.config.freeze_value:
      self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def bc_loss(self, carry, obs, prevact):
    """Actor-only teacher loss on frozen posterior latents."""
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    enc_carry, _, tokens = self.enc(
        enc_carry, obs, reset, training=False)
    dyn_carry, _, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    feat = jax.tree.map(sg, feat)
    policy = self.pol(self.feat2tensor(feat), 2)
    target = {k: obs[f'action_policy_{k}'] for k in self.act_space}
    mask = f32(obs['teacher'])
    denom = jnp.maximum(mask.sum(), 1.0)
    if self.config.bc_fixed_std:
      # For deterministic deployment we care about the policy mean. A learned
      # BC variance can reduce NLL merely by inflating uncertainty, so use a
      # fixed-variance Gaussian likelihood during actor-only distillation.
      nll = -sum(
          jax.scipy.stats.norm.logpdf(
              sg(target[key]), dist.pred(), f32(self.config.bc_fixed_std)).sum(-1)
          for key, dist in policy.items())
    else:
      nll = -sum(dist.logp(sg(target[key])) for key, dist in policy.items())
    loss = (nll * mask).sum() / denom
    metrics = self._bc_metrics(policy, target, mask, nll)
    return f32(self.config.bc_scale) * loss, (
        (enc_carry, dyn_carry, dec_carry), metrics)

  def _bc_metrics(self, policy, target, mask, nll):
    denom = jnp.maximum(mask.sum(), 1.0)
    metrics = {
        'bc/count': mask.sum(),
        'bc/nll': (nll * mask).sum() / denom,
        'bc/finite': f32(jnp.isfinite(nll).all()),
    }
    for key, dist in policy.items():
      pred = dist.pred()
      truth = target[key]
      weight = mask[..., None]
      axes = tuple(range(pred.ndim - 1))
      count = jnp.maximum(weight.sum(axis=axes), 1.0)
      mae = (jnp.abs(pred - truth) * weight).sum(axis=axes) / count
      pred_mean = (pred * weight).sum(axis=axes) / count
      true_mean = (truth * weight).sum(axis=axes) / count
      pred_center = pred - pred_mean
      true_center = truth - true_mean
      cov = (pred_center * true_center * weight).sum(axis=axes) / count
      pred_std = jnp.sqrt(
          (jnp.square(pred_center) * weight).sum(axis=axes) / count + 1e-8)
      true_std = jnp.sqrt(
          (jnp.square(true_center) * weight).sum(axis=axes) / count + 1e-8)
      corr = cov / (pred_std * true_std)
      actor_std = dist.output.stddev if hasattr(dist, 'output') else dist.stddev
      actor_std = (actor_std * weight).sum(axis=axes) / count
      names = ('vx', 'vy', 'vz', 'yaw_rate')
      for index, name in enumerate(names[:pred.shape[-1]]):
        metrics[f'bc/mae_{name}'] = mae[index]
        metrics[f'bc/corr_{name}'] = corr[index]
        metrics[f'bc/pred_mean_{name}'] = pred_mean[index]
        metrics[f'bc/pred_std_{name}'] = pred_std[index]
        metrics[f'bc/teacher_mean_{name}'] = true_mean[index]
        metrics[f'bc/teacher_std_{name}'] = true_std[index]
        metrics[f'bc/actor_std_{name}'] = actor_std[index]
    metrics['bc/mae'] = sum(
        metrics[f'bc/mae_{name}'] for name in ('vx', 'vy', 'vz', 'yaw_rate')) / 4
    return metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    # Offline complete-episode datasets may pad short episodes to a common
    # sequence length. Keep padding available for static JAX shapes and RSSM
    # resets, but exclude it from every world-model target. Online replay does
    # not provide this private field and therefore defaults to all-valid.
    loss_mask = f32(obs.pop('_loss_mask', jnp.ones_like(obs['reward'], bool)))
    risk_stratum = obs.pop(
        '_risk_stratum', jnp.zeros_like(obs['reward'], dtype=jnp.int32))
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    reward_target = obs.pop('_reward_dense', obs['reward'])
    losses['rew'] = self.rew(inp, 2).loss(reward_target)
    if self.outcome is not None:
      outcome_target = obs.pop('_terminal_outcome')
      if self.config.outcome_mode == 'conditional_terminal':
        # The value is ignored for nonterminal rows by the terminal-only mask.
        outcome_target = jnp.clip(outcome_target - 1, 0, 1)
      losses['outcome'] = self.outcome(inp, 2).loss(outcome_target)
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    if self.config.anchor_scale:
      anchor_target = obs.pop('_anchor_action_mean')
      anchor_mask = obs.pop('_anchor_mask')
      anchor_pred = self.pol(self.feat2tensor(repfeat), 2)['action'].pred()
      losses['anchor'] = jnp.square(anchor_pred - anchor_target).mean(-1)
      metrics['anchor/action_mae'] = jnp.abs(anchor_pred - anchor_target).mean()
      metrics['anchor/near_fraction'] = anchor_mask.mean()
      if self.config.anchor_schedule_enabled:
        step = self.opt.step.read()
        schedule = tuple(zip(
            self.config.anchor_schedule[0::2],
            self.config.anchor_schedule[1::2]))
        scale = f32(schedule[-1][1])
        for end, value in reversed(schedule[:-1]):
          scale = jnp.where(step < int(end), f32(value), scale)
        losses['anchor'] *= scale
        metrics['anchor/effective_scale'] = scale
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    model_keys = tuple(losses)
    mask_scale = (B * T) / jnp.maximum(loss_mask.sum(), 1.0)
    for key in model_keys:
      if key == 'outcome' and self.config.outcome_mode == 'conditional_terminal':
        raw = losses[key]
        terminal_mask = loss_mask * f32(obs['is_terminal'])
        count = jnp.maximum(terminal_mask.sum(), 1.0)
        losses[key] = raw * terminal_mask * (B * T / count)
        metrics['outcome/terminal_count'] = terminal_mask.sum()
        metrics['outcome/terminal_loss'] = (raw * terminal_mask).sum() / count
      elif key == 'anchor':
        mask = loss_mask * f32(anchor_mask)
        count = jnp.maximum(mask.sum(), 1.0)
        losses[key] = losses[key] * mask * (B * T / count)
        metrics['anchor/near_count'] = mask.sum()
      elif key == 'con':
        importance = obs.pop('_continue_importance', jnp.ones_like(losses[key]))
        losses[key] = losses[key] * importance * loss_mask * mask_scale
      elif key == 'rew' and any(self.config.reward_strata_weights):
        raw = losses[key]
        weighted = jnp.zeros_like(raw)
        for stratum, alpha in enumerate(self.config.reward_strata_weights):
          mask = loss_mask * f32(risk_stratum == stratum)
          count = jnp.maximum(mask.sum(), 1.0)
          weighted += raw * mask * f32(alpha) * (B * T / count)
          metrics[f'reward_strata/{stratum}_count'] = mask.sum()
          metrics[f'reward_strata/{stratum}_loss'] = (
              raw * mask).sum() / count
        losses[key] = weighted
      else:
        losses[key] = losses[key] * loss_mask * mask_scale
    metrics['model/valid_fraction'] = loss_mask.mean()

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    start_indices = last_valid_indices(loss_mask.astype(bool), K)
    selected_dyn_entries = take_time(dyn_entries, start_indices)
    selected_repfeat = take_time(repfeat, start_indices)
    starts = self.dyn.starts(selected_dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x.reshape((B * K, 1, *x.shape[2:])), selected_repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self._combined_reward(inp, 2),
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training and not (
            self.config.freeze_policy and self.config.freeze_value),
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)
    selected_mask = jnp.take_along_axis(loss_mask, start_indices, axis=1)
    metrics['imag/valid_start_fraction'] = selected_mask.mean()
    metrics['imag/min_valid_steps'] = loss_mask.sum(1).min()

    # Replay
    if self.config.repval_loss:
      feat = sg(selected_repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      last, term, rew = take_time((last, term, rew), start_indices)
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training and not self.config.freeze_value,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    # Optional teacher-action supervision. The explicit action_policy_* field
    # is the action chosen from observation at the same row: o_t -> a_t. The
    # RSSM posterior for that row was reconstructed with prevact a_(t-1).
    # Keeping the label explicit avoids the subtle one-step shift that occurs
    # when attempting to recover a_t indirectly from prevact.
    bc_loss = jnp.zeros((B, T), f32)
    if self.config.bc_scale:
      bc_feat = jax.tree.map(sg, repfeat)
      bc_policy = self.pol(self.feat2tensor(bc_feat), 2)
      bc_target = {
          key: obs[f'action_policy_{key}'] for key in self.act_space}
      if self.config.bc_fixed_std:
        bc_nll = -sum(
            jax.scipy.stats.norm.logpdf(
                sg(bc_target[key]), dist.pred(),
                f32(self.config.bc_fixed_std)).sum(-1)
            for key, dist in bc_policy.items())
      else:
        bc_nll = -sum(
            dist.logp(sg(bc_target[key]))
            for key, dist in bc_policy.items())
      bc_mask = f32(obs['teacher'])
      bc_denom = jnp.maximum(1.0, bc_mask.sum())
      bc_mean = (bc_nll * bc_mask).sum() / bc_denom
      bc_loss = bc_nll * bc_mask
      # Keep the scale independent of the number of padding/non-teacher rows.
      bc_loss = bc_loss * (B * T / bc_denom)
      metrics['loss/bc'] = bc_mean
      metrics['bc/teacher_fraction'] = bc_mask.mean()
      metrics.update(self._bc_metrics(
          bc_policy, bc_target, bc_mask, bc_nll))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    loss += bc_loss.mean() * self.config.bc_scale

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if self.config.bc_only:
      carry, obs, prevact, _ = self._apply_replay_context(carry, data)
      _, (carry, metrics) = self.bc_loss(carry, obs, prevact)
      carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
      return carry, metrics
    if not self.config.report:
      # Lightweight validation for takeover runs: evaluate teacher action
      # imitation without compiling the expensive imagination/open-loop video
      # report. This path never updates parameters.
      if self.config.bc_scale:
        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        _, (carry, metrics) = self.bc_loss(carry, obs, prevact)
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, metrics
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    metrics.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def diagnose(self, data):
    """Read-only posterior calibration and lateral counterfactual rollout."""
    obs = {key: data[key] for key in self.obs_space}
    B, T = obs['is_first'].shape
    carry = self.init_report(B)
    enc_carry, dyn_carry, dec_carry, _ = carry
    prevact = {'action': data['diagnostic_prev_action']}
    enc_carry, _, tokens = self.enc(
        enc_carry, obs, obs['is_first'], training=False)
    dyn_carry, entries, dyn_losses, feat, dyn_metrics = self.dyn.loss(
        dyn_carry, tokens, prevact, obs['is_first'], training=False)
    dec_carry, _, recons = self.dec(
        dec_carry, feat, obs['is_first'], training=False)
    inp = self.feat2tensor(feat)
    reward_dist = self.rew(inp, 2)
    continue_dist = self.con(inp, 2)
    dense_reward_pred = reward_dist.pred()
    continue_pred = continue_dist.prob(1)
    outcome_probs = jnp.zeros((*dense_reward_pred.shape, 2), f32)
    terminal_reward_pred = jnp.zeros_like(dense_reward_pred)
    if self.outcome is not None and self.config.outcome_mode == 'conditional_terminal':
      outcome_probs = jax.nn.softmax(self.outcome(inp, 2).logits, -1)
      terminal_values = jnp.asarray(
          self.config.terminal_reward_values, f32)[jnp.asarray([1, 2])]
      terminal_reward_pred = (
          (1 - continue_pred) * (outcome_probs * terminal_values).sum(-1))
    reward_pred = dense_reward_pred + terminal_reward_pred
    reward_loss = reward_dist.loss(obs['reward'])
    continue_target = f32(~obs['is_terminal'])
    if self.config.contdisc:
      continue_target *= 1 - 1 / self.config.horizon
    continue_loss = continue_dist.loss(continue_target)
    valid_mask = f32(data.get('loss_mask', jnp.ones_like(obs['is_last'], bool)))
    collision_mask = valid_mask * f32(
        obs['is_terminal'] & (obs['reward'] < -50.0))
    ordinary_mask = valid_mask * (1.0 - collision_mask)

    def masked_reward_grad(inp, target, mask):
      loss = self.rew(inp, 2).loss(target)
      return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    def masked_continue_grad(inp, target, mask):
      loss = self.con(inp, 2).loss(target)
      return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    rew_term_grad = nj.grad(masked_reward_grad, self.rew)(
        sg(inp), obs['reward'], collision_mask)[-1]
    rew_other_grad = nj.grad(masked_reward_grad, self.rew)(
        sg(inp), obs['reward'], ordinary_mask)[-1]
    con_term_grad = nj.grad(masked_continue_grad, self.con)(
        sg(inp), continue_target, collision_mask)[-1]
    con_other_grad = nj.grad(masked_continue_grad, self.con)(
        sg(inp), continue_target, ordinary_mask)[-1]
    valid_count = jnp.maximum(valid_mask.sum(), 1.0)
    collision_fraction = collision_mask.sum() / valid_count
    voffset, vscale = self.valnorm.stats()
    value_pred = self.val(inp, 2).pred() * vscale + voffset
    actor_action_mean = self.pol(inp, 2)['action'].pred()
    vector_pred = recons['vector'].pred()

    # One posterior state per batch row is selected for counterfactuals. Each
    # supplied action is a complete normalized [vx, vy, vz, yaw_rate] vector;
    # only its route-lateral component differs in the calling script.
    indices = data['diagnostic_state_index'].astype(jnp.int32)
    starts = jax.tree.map(lambda x: x[jnp.arange(B), indices], entries)
    candidates = data['diagnostic_candidate_action']
    C = candidates.shape[1]
    starts = jax.tree.map(
        lambda x: jnp.repeat(x[:, None], C, axis=1).reshape((B * C,) + x.shape[1:]),
        starts)
    first_action = {'action': candidates.reshape((B * C, -1))}
    first_carry, (first_feat, _) = self.dyn.imagine(
        starts, first_action, 1, training=False, single=True)
    policyfn = lambda state: {
        key: dist.pred() for key, dist in
        self.pol(self.feat2tensor(state), 1).items()}
    max_horizon = 15
    _, later_feat, _ = self.dyn.imagine(
        first_carry, policyfn, max_horizon - 1, training=False)
    imagined = jax.tree.map(
        lambda first, later: jnp.concatenate([first[:, None], later], 1),
        first_feat, later_feat)
    imagined_inp = self.feat2tensor(imagined)
    imagined_dense_reward = self.rew(imagined_inp, 2).pred()
    imagined_continue = self.con(imagined_inp, 2).prob(1)
    imagined_reward = imagined_dense_reward
    imagined_outcome = jnp.zeros((*imagined_dense_reward.shape, 2), f32)
    if self.outcome is not None and self.config.outcome_mode == 'conditional_terminal':
      imagined_outcome = jax.nn.softmax(self.outcome(imagined_inp, 2).logits, -1)
      terminal_values = jnp.asarray(
          self.config.terminal_reward_values, f32)[jnp.asarray([1, 2])]
      imagined_reward = imagined_dense_reward + (
          (1 - imagined_continue) *
          (imagined_outcome * terminal_values).sum(-1))
    imagined_value = self.val(imagined_inp, 2).pred() * vscale + voffset
    imag_dec_carry = self.dec.initial(B * C)
    _, _, imagined_recons = self.dec(
        imag_dec_carry, imagined,
        jnp.zeros((B * C, max_horizon), bool), training=False)
    imagined_vector = imagined_recons['vector'].pred()
    discount = 1 - 1 / self.config.horizon
    horizon_outputs = {}
    for horizon in (1, 3, 5, 15):
      rew = imagined_reward[:, :horizon]
      con = imagined_continue[:, :horizon]
      val = imagined_value[:, :horizon]
      weights = jnp.concatenate([
          jnp.ones_like(con[:, :1]),
          jnp.cumprod(discount * con[:, :-1], axis=1)], axis=1)
      boot_weight = jnp.prod(discount * con, axis=1)
      model_return = (weights * rew).sum(1) + boot_weight * val[:, -1]
      # State-local lambda return for exactly the requested number of future
      # transitions. The training helper expects an H+1 padded sequence and
      # therefore cannot represent the one-step diagnostic directly.
      lambda_ret = val[:, -1]
      lam = self.config.imag_loss.lam
      for step in reversed(range(horizon)):
        lambda_ret = rew[:, step] + discount * con[:, step] * (
            (1 - lam) * val[:, step] + lam * lambda_ret)
      horizon_outputs[f'h{horizon}_return'] = model_return.reshape(B, C)
      horizon_outputs[f'h{horizon}_lambda_return'] = lambda_ret.reshape(B, C)
      horizon_outputs[f'h{horizon}_continue'] = con.mean(1).reshape(B, C)
      horizon_outputs[f'h{horizon}_risk'] = (1 - con).max(1).reshape(B, C)
      horizon_outputs[f'h{horizon}_collision_prob'] = imagined_outcome[
          :, :horizon, 1].max(1).reshape(B, C)
      horizon_outputs[f'h{horizon}_value'] = val[:, 0].reshape(B, C)
      horizon_outputs[f'h{horizon}_lidar_min'] = imagined_vector[
          :, :horizon, 20:36].min((1, 2)).reshape(B, C)
    return {
        'encoder_token': tokens,
        'rssm_deter': feat['deter'],
        'rssm_stoch': feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1)),
        'reward_pred': reward_pred,
        'dense_reward_pred': dense_reward_pred,
        'terminal_reward_pred': terminal_reward_pred,
        'outcome_success_prob': outcome_probs[..., 0],
        'outcome_collision_prob': outcome_probs[..., 1],
        'reward_loss': reward_loss,
        'continue_pred': continue_pred,
        'continue_target': continue_target,
        'continue_loss': continue_loss,
        'valid_count': valid_mask.sum()[None],
        'collision_terminal_count': collision_mask.sum()[None],
        'collision_terminal_fraction': collision_fraction[None],
        'reward_terminal_loss_mean': (
            reward_loss * collision_mask).sum()[None] / jnp.maximum(collision_mask.sum(), 1.0),
        'reward_other_loss_mean': (
            reward_loss * ordinary_mask).sum()[None] / jnp.maximum(ordinary_mask.sum(), 1.0),
        'continue_terminal_loss_mean': (
            continue_loss * collision_mask).sum()[None] / jnp.maximum(collision_mask.sum(), 1.0),
        'continue_other_loss_mean': (
            continue_loss * ordinary_mask).sum()[None] / jnp.maximum(ordinary_mask.sum(), 1.0),
        'reward_terminal_grad_norm': optax.global_norm(rew_term_grad)[None],
        'reward_other_grad_norm': optax.global_norm(rew_other_grad)[None],
        'continue_terminal_grad_norm': optax.global_norm(con_term_grad)[None],
        'continue_other_grad_norm': optax.global_norm(con_other_grad)[None],
        'reward_terminal_weighted_grad_norm': (
            collision_fraction * optax.global_norm(rew_term_grad))[None],
        'continue_terminal_weighted_grad_norm': (
            collision_fraction * optax.global_norm(con_term_grad))[None],
        'value_pred': value_pred,
        'actor_action_mean': actor_action_mean,
        'vector_pred': vector_pred,
        'dyn_loss': dyn_losses['dyn'],
        'rep_loss': dyn_losses['rep'],
        **horizon_outputs,
    }

  def anchor_actions(self, data):
    """Old posterior-policy means for immutable interface-anchor targets."""
    obs = {key: data[key] for key in self.obs_space}
    batch = obs['is_first'].shape[0]
    enc_carry = self.enc.initial(batch)
    dyn_carry = self.dyn.initial(batch)
    prevact = {'action': data['diagnostic_prev_action']}
    enc_carry, _, tokens = self.enc(
        enc_carry, obs, obs['is_first'], training=False)
    dyn_carry, _, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, obs['is_first'], training=False)
    return self.pol(self.feat2tensor(feat), 2)['action'].pred()

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    obs['_loss_mask'] = data.get(
        'loss_mask', jnp.ones_like(data['reward'], dtype=bool))
    obs['_risk_stratum'] = data.get(
        'risk_stratum', jnp.zeros_like(data['reward'], dtype=jnp.int32))
    obs['_continue_importance'] = data.get(
        'continue_importance', jnp.ones_like(data['reward'], dtype=jnp.float32))
    if self.config.anchor_scale:
      obs['_anchor_action_mean'] = data['anchor_action_mean']
      obs['_anchor_mask'] = data['anchor_mask']
    if self.config.reward_decomposition:
      obs['_reward_dense'] = data['reward_dense']
      obs['_terminal_outcome'] = data['terminal_outcome']
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_obs['_loss_mask'] = rhs(data.get(
        'loss_mask', jnp.ones_like(data['reward'], dtype=bool)))
    rep_obs['_risk_stratum'] = rhs(data.get(
        'risk_stratum', jnp.zeros_like(data['reward'], dtype=jnp.int32)))
    rep_obs['_continue_importance'] = rhs(data.get(
        'continue_importance', jnp.ones_like(data['reward'], dtype=jnp.float32)))
    if self.config.anchor_scale:
      rep_obs['_anchor_action_mean'] = rhs(data['anchor_action_mean'])
      rep_obs['_anchor_mask'] = rhs(data['anchor_mask'])
    if self.config.reward_decomposition:
      rep_obs['_reward_dense'] = rhs(data['reward_dense'])
      rep_obs['_terminal_outcome'] = rhs(data['terminal_outcome'])
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _combined_reward(self, inp, bdims):
    dense = self.rew(inp, bdims).pred()
    if self.outcome is None:
      return dense
    probs = jax.nn.softmax(self.outcome(inp, bdims).logits, -1)
    terminal = jnp.asarray(self.config.terminal_reward_values, f32)
    if self.config.outcome_mode == 'four_class':
      return dense + (probs * terminal).sum(-1)
    # Continue predicts P(continue); it owns terminal probability, while this
    # head only owns P(success/collision | terminal).
    pterminal = 1 - self.con(inp, bdims).prob(1)
    conditional = jnp.stack([terminal[1], terminal[2]])
    return dense + pterminal * (probs * conditional).sum(-1)

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
