# UAV DreamerV3 Training Handoff

## 1. 当前项目状态

本项目仍在开发中，DreamerV3 仍在训练和诊断阶段。本 GitHub 版本用于代码版本管理、
实验追溯和训练接力，不是 final release，也不代表当前策略已经通过全部 Gate。

截至 2026-09-20，本地实验已推进到 S3A A1 Recovery2/Recovery2B。Recovery2B-1k
只有每场景 20 回合的 quick Gate，没有完整 100 回合 Gate，不能仅因时间最新就作为
新的已认证 parent。

## 2. 当前主要目标

主要任务是 PyBullet 中的四旋翼静态障碍导航：从时序 LiDAR、无人机状态和目标信息
学习闭环控制，并在不同障碍横向 offset 下兼顾成功率、碰撞率和超时率。

DreamerV3 是当前世界模型研究主线，支持在线环境交互和固定 NPZ episode 的离线更新。
PPO 是已验证的控制基线、教师数据来源和对照，不与 Dreamer 的 checkpoint 格式混用。

## 3. 当前训练阶段

### 推荐保守 baseline

当前有完整固定场景证据、适合作为新实验保守 parent 的 checkpoint 是
**Conservative-2k**：

```text
outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T154503F849521
```

它在 100 回合 fixed +0.8/+0.6/+0.3 评测中的成功率分别为 94%/92%/78%。已有分析显示，
后续 Retry A1 和 A1-Recovery 虽降低训练 loss，却在已掌握的高 offset 场景发生部分遗忘。

### 最新研究分支

- A1-Recovery final：改善 low/fixed +0.3 和随机场景碰撞，但完整 Gate 未通过。
- Recovery2：1k quick Gate 平均成功率 76%，2k 降至 68%，显示继续训练退化。
- Recovery2B-1k：quick Gate 平均成功率 76%、平均碰撞率 5%；fixed +0.3 为 85%，
  但 fixed +0.8/+0.6 均只有 80%。它尚未完成完整 Gate。

因此：新研究分支默认从 Conservative-2k 开始；若目标是精确接续 Recovery2B 诊断，
才使用 Recovery2B-1k checkpoint 和它的固定离线数据集。下一步应先完成一致种子、
100 回合的完整 Gate，再决定是否把 Recovery2B 提升为 parent，不应自动进入下一阶段。

## 4. 环境

当前机器实测环境详见 [`ENVIRONMENT.md`](ENVIRONMENT.md)。关键事实：

- Ubuntu 22.04.5 LTS，Python 3.10.20。
- Dreamer 环境：JAX/JAXLIB 0.6.2、NumPy 1.26.4、Gymnasium 1.3.0。
- PPO 环境：PyTorch 2.13.0+cu130、Stable-Baselines3 2.9.0。
- 历史硬件报告为 NVIDIA GeForce RTX 4070、驱动 580.178.04、CUDA 13.0。
- 本次审计会话中 NVIDIA 驱动不可访问，JAX 只发现 CPU；近期 S3A shell 脚本也明确使用
  `--jax.platform cpu`。不要在未验证时声称 GPU 路径可用。

## 5. 项目入口

| 功能 | 真实入口 |
|---|---|
| Dreamer 统一入口 | `dreamerv3/dreamerv3/main.py` |
| UAV 环境 | `src/uav_navigation/env.py` |
| Dreamer 环境适配 | `dreamerv3/embodied/envs/quadrotor.py` |
| PPO 训练 | `src/uav_navigation/train_ppo.py` |
| Dreamer 在线训练 | `dreamerv3/embodied/run/train.py`，即 `--script train` |
| Dreamer 离线训练 | `dreamerv3/embodied/run/bc_distill.py`，即 `--script bc_distill` |
| 在线 Replay | `dreamerv3/embodied/core/replay.py` |
| Recovery Replay 构建 | `scripts/build_recovery2_replay.py`、`scripts/build_recovery2b_replay.py` |
| 闭环评测 | `dreamerv3/embodied/run/eval_only.py` |
| Static Gate | `scripts/evaluate_static_obstacle_gate.sh` |
| S3A Gate | `scripts/evaluate_stage_s3a_gate.sh`、`scripts/evaluate_stage_s3a_recovery2.sh` |

## 6. 从零训练

以下命令创建全新在线训练目录，不加载 checkpoint，也不覆盖历史结果。它是可运行的
静态单障碍起点，不声称复现全部历史课程：

```bash
cd UAV_Dreamer
export PYTHONPATH="$PWD/src:$PWD/dreamerv3"
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"

python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/dev_from_scratch_seed0 \
  --configs quadrotor --script train \
  --run.steps 2000 --run.envs 1 --run.train_ratio 8 \
  --run.log_every 100 --run.report_every 1000 --run.save_every 500 \
  --batch_size 1 --batch_length 32 --report_length 32 --replay_context 0 \
  --agent.report False --agent.opt.lr 0.000002 --agent.opt.warmup 0 \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl \
  --env.quadrotor.seed 0 --env.quadrotor.max_steps 300 \
  --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 \
  --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_spawn_probability 1 \
  --env.quadrotor.obstacle_lateral_offset 0.8 \
  --env.quadrotor.obstacle_randomization_level 0 \
  --env.quadrotor.dynamic_obstacles False \
  --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 \
  --env.quadrotor.safety_projection False
```

在线 `train` 会把 transition 写入 `<logdir>/replay/`。崩溃后若要精确恢复在线训练，
必须同时保留该 logdir 的 checkpoint 和磁盘 Replay chunks；checkpoint 中只有 Replay
标记，不含 transitions。

## 7. 从当前进度继续训练

### 7.1 所需资产

精确延续 Recovery2B-1k 离线分支，需要把以下资产放回相对路径：

```text
outputs/dreamerv3/stageS3A_A1_Recovery2B_1000/ckpt/20260919T155040F646354/
  agent.pkl
  step.pkl
  done
outputs/dreamerv3/stageS3A_A1_Recovery2B_replay/
  train/*.npz
  validation/*.npz
  manifest.json
```

发布包中的 NPZ 必须是普通文件。当前本机 Replay 布局使用指向绝对路径的符号链接，
直接复制链接目录会在别的机器失效。

`agent.pkl` 已包含世界模型、Actor、Value、slow value、主 optimizer、Actor optimizer、
return/value normalization 及内部计数器。`step.pkl` 是外层 checkpoint step。
Dreamer 链路不需要 PPO 的 VecNormalize 文件。

### 7.2 延续命令

下面命令从 Recovery2B-1k 权重开始，在相同 100-episode 离线集合上执行额外 1000 次
更新，并写入一个全新目录。参数来自现有
`scripts/train_stage_s3a_a1_recovery2b_1000.sh`，仅将 parent 和 logdir 改为 continuation：

```bash
cd UAV_Dreamer
export PYTHONPATH="$PWD/src:$PWD/dreamerv3"
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"

python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS3A_A1_Recovery2B_continue_1000 \
  --configs quadrotor --script bc_distill \
  --run.from_checkpoint outputs/dreamerv3/stageS3A_A1_Recovery2B_1000/ckpt/20260919T155040F646354 \
  --run.from_checkpoint_regex '.*' \
  --run.bc_dataset outputs/dreamerv3/stageS3A_A1_Recovery2B_replay \
  --run.bc_updates 1000 --run.bc_validate_every 1000 \
  --run.bc_fixed_train False --run.bc_risk_stratified False \
  --run.bc_scene_balanced False --run.bc_curriculum_new_fraction 0.0 \
  --batch_size 2 --batch_length 301 --report_length 301 --replay_context 0 \
  --agent.reward_decomposition False --agent.anchor_scale 5.0 \
  --agent.anchor_schedule_enabled False --agent.separate_actor_opt True \
  --agent.bc_only False --agent.bc_scale 0 --agent.bc_fixed_std 0 \
  --agent.freeze_encoder False --agent.freeze_dynamics False \
  --agent.freeze_decoder False --agent.freeze_reward False \
  --agent.freeze_continue False --agent.freeze_policy False \
  --agent.freeze_value False --agent.freeze_world_model False \
  --agent.loss_scales.dyn 1 --agent.loss_scales.rep 0.1 \
  --agent.loss_scales.rec 1 --agent.loss_scales.policy 1 \
  --agent.loss_scales.value 1 --agent.loss_scales.repval 0.3 \
  --agent.loss_scales.rew 1 --agent.loss_scales.con 1 \
  --agent.opt.lr 0.000001 --agent.opt.warmup 0 \
  --agent.actor_opt.lr 0.0000003 --agent.actor_opt.warmup 0 \
  --agent.imag_last 16 --agent.imag_length 15 --agent.report False \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 \
  --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_lateral_offset 0.3 \
  --env.quadrotor.obstacle_randomization_level 1 \
  --env.quadrotor.random_offset_low 0.3 \
  --env.quadrotor.random_offset_high 0.8 \
  --env.quadrotor.dynamic_obstacles False \
  --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 \
  --env.quadrotor.safety_projection False
```

这条命令表达“精确延续实验”，不表示 Recovery2B 已获准成为下一阶段 parent。训练前仍应
确定 Gate；训练后先运行 `scripts/evaluate_stage_s3a_recovery2.sh` 的完整 100 回合评测。

## 8. 当前 checkpoint 信息

| 名称 | 相对路径 | 用途 | 推荐作为 continuation parent |
|---|---|---|---|
| Conservative-2k | `outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T154503F849521` | 最可靠的已评测固定场景 baseline | 是，默认保守 parent |
| A1-Recovery final | `outputs/dreamerv3/stageS3A_A1_Recovery_4000/ckpt/20260830T153553F674727` | Recovery2/2B 的实际 parent；低 offset 恢复研究 | 仅用于复现该分支；完整 Gate 未通过 |
| Recovery2B-1k | `outputs/dreamerv3/stageS3A_A1_Recovery2B_1000/ckpt/20260919T155040F646354` | 最新研究进度、继续 Recovery2B 诊断 | 否，完成完整 Gate 前不得升级 |

每个上述 checkpoint 目录含 `agent.pkl`（8,036,157 字节）、`step.pkl`（15 字节）和
空的完成标记 `done`，合计 8,036,172 字节。

## 9. 数据与模型文件

普通 Git 不包含完整 `outputs/`、Replay、checkpoint、metrics、trajectory 或 profiler。
原因是它们合计约 9 GB、生成频繁，并会使代码历史不可维护。本地资产没有被删除。

Recovery2B 精确 continuation 最小包为：

| 内容 | 大小 |
|---|---:|
| Recovery2B-1k checkpoint | 8,036,172 bytes |
| 100 个唯一 NPZ 物化后的 train/validation 数据集及布局 | 约 1,684,233 bytes |
| 对应 `config.yaml` | 7,488 bytes |
| 合计 | 约 9,727,893 bytes（9.28 MiB） |

这类小型、人工选择的 continuation 包适合 GitHub Release，并附 SHA-256 清单；无需把
全部 outputs 放进 Git LFS。若以后发布大量 Replay 或多阶段 checkpoint，使用对象存储/
数据仓库更合适。首次源码提交不包含这个包。

## 10. 已知问题

- Recovery2B 只有 quick Gate，没有完整 Gate，不能称为最佳模型。
- Recovery2 从 1k 到 2k quick Gate 明显退化，继续训练可能加重遗忘。
- Conservative-2k 对固定场景最稳，但缺少同口径随机场景评测。
- 较晚阶段 loss 下降并未稳定转化为导航性能提升，存在灾难性/部分遗忘。
- collision terminal/continue 建模仍是项目级风险；当前日志缺少各 checkpoint 对齐的
  terminal-stratified 和多步预测验证。
- Recovery Replay 目录目前包含本机绝对 symlink，发布时必须物化。
- 多数历史 shell 脚本硬编码 `/home/user/UAV_Dreamer` 和 Conda 解释器路径；换机器需替换。
- `dreamerv3/requirements.txt` 与当前实际 JAX 0.6.2 环境不一致，不能把它当作锁文件。
