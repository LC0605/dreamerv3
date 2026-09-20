# UAV DreamerV3 项目续接交接（2026-08-29）

## 1. 最终研究目标与硬约束

最终部署链路保持纯 DreamerV3：

`Observation (3-frame LiDAR + goal/self state) -> Encoder -> RSSM -> Actor -> continuous action`

- 动作：`[vx_world, vy_world, vz_world, yaw_rate]`，policy space 为 `[-1, 1]`。
- PPO 只曾作为离线教师；当前及最终推理均不使用 PPO。
- `bc_scale=0`，无 PPO ensemble、无 safety projection、无 planner、无人工绕障规则。
- 当前只研究单静态障碍位置泛化；不要进入镜像、随机、多障碍、动态障碍或8-frame LiDAR，除非当前阶段完成并明确决定。
- 不再继续 reward decomposition、terminal hazard、Continue LR sweep 或 Full-WM 微型诊断。

## 2. 已确认的基础结果

- PPO -> Dreamer Actor 显式蒸馏有效，RSSM teacher-action 时序与 observation/action 对齐已修复。
- Teacher validation action MAE 约0.16，相关性约 `0.98/0.97/0.90/0.91`。
- 固定 S1 曾达到 `114/114 success`，零碰撞/越界/超时。
- +0.3 公平 zero-shot：
  - BC-only：80.46% success / 12.64% collision。
  - Mixed：81.05% / 13.68%。
  - Original Pure：73.63% / 20.88%。
- 核心长期现象：Pure Dreamer 更新后近障 forward 增强、lateral avoidance 减弱、clearance 下降，导致 collision 上升。

## 3. 关键 checkpoint（禁止覆盖）

### 原始消融分支

- BC-only：
  `outputs/dreamerv3/stageS1fA3_temporal3_actor_mean_bc/ckpt/20260825T163314F148682`
- Mixed（当前行为参考）：
  `outputs/dreamerv3/stageS1fB2_temporal3_teacher_takeover_midbc/ckpt/20260825T222659F216085`
- Original Pure：
  `outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/20260825T223426F313624`

注册表：`outputs/dreamerv3/checkpoint_registry_s1f.json`

### Conservative 分支

- **Conservative-2k（当前已评测最佳模型）**：
  `outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T154503F849521`
- Conservative-4k（衰减到 anchor=2 后失败，仅作对照）：
  `outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T173921F074846`
- **S2-Consolidate 最终 checkpoint（刚完成，尚未评测）**：
  `outputs/dreamerv3/stageS2_Consolidate_4000/ckpt/20260829T001544F726916`
- 同时保存的阶段末 checkpoint：
  `outputs/dreamerv3/stageS2_Consolidate_4000/ckpt/20260829T001544F261750`

不要使用失败的 S2-WM、S2-WM-Anchor、S2-Joint 或 Conservative-4k 作为新 parent。

## 4. 公平 paired baseline（相同100回合种子）

固定 seeds：

- +0.8：70000
- +0.6：71000
- +0.3：72000

| 模型 | +0.8 S/C/T | +0.6 S/C/T | +0.3 S/C/T | +0.3 clearance |
|---|---:|---:|---:|---:|
| Mixed | 92/1/7 | 94/0/6 | 83/12/5 | 0.227 m |
| Original Pure | 92/2/6 | 88/0/12 | 71/20/9 | 0.186 m |
| Conservative-2k | 94/1/5 | 92/1/7 | **78/14/8** | **0.227 m** |
| Conservative-4k | 92/0/8 | 89/1/10 | 74/15/11 | 0.220 m |

所有表中 OOB 均为0。

Conservative-2k 相对 Mixed：overall Actor MAE 0.0689，near-obstacle MAE 0.0615。

Conservative-4k 失败原因：anchor 由5降到2后，+0.3 success 78->74，近障 forward 由 -0.029 变为 -0.015，lateral 幅度约0.030降至0.012，clearance 0.227->0.220。

## 5. 当前阶段：S2-Consolidate

目的：验证固定强 anchor=5.0 长时间训练，能否让 WM/Value/Actor 在安全行为约束下充分适应并真正超过 Conservative-2k。

配置：

- Parent：Conservative-2k。
- 恢复了模型权重和 Adam optimizer state。
- +0.8/+0.6/+0.3 replay = 20/20/60。
- near-obstacle 定义：真实 `minimum_clearance < 0.70 m`。
- anchor 固定5.0，不衰减。
- 标准 DreamerV3 reward/Continue。
- WM + Value + Actor 正常联合训练。
- `bc_scale=0`，PPO=0。
- 新增4000 updates 已于 2026-08-29 00:15 完成，无 NaN/爆炸。
- 阶段末：actor optimizer update count 5999（含 parent 原有2000），anchor loss约0.0061，near batch count 71，WM/Value/Actor训练数值稳定。

训练脚本：`scripts/train_stage_s2_consolidate_4000.sh`

## 6. 下次打开后的第一项工作（不要重新训练）

直接评测 Consolidate 最终 checkpoint：

`outputs/dreamerv3/stageS2_Consolidate_4000/ckpt/20260829T001544F726916`

对 +0.8/+0.6/+0.3 各做100回合 paired deterministic evaluation，必须复用 seeds `70000/71000/72000` 和原有环境参数。

输出并与 Mixed、Original Pure、Conservative-2k 对比：

- Success / Collision / Timeout / OOB
- mean minimum clearance
- navigation time（control_dt=0.1 s）
- near-obstacle forward/lateral action（clearance <0.70 m）
- 相对 Conservative-2k 的 overall / near-obstacle Actor action MAE

可复制 `scripts/evaluate_s2_conservative_4k_paired.sh`，只需替换 checkpoint 与输出目录；不要覆盖已有评测目录。

## 7. S2-Consolidate Gate

若同时满足：

- +0.8 success >=90%
- +0.6 success 约>=90%
- +0.3 success >=80%
- +0.3 collision <=14%
- +0.3 clearance >=约0.227 m

则 `S2-Consolidate PASS`，保留 checkpoint，下一阶段设计更慢 anchor 衰减（例如5->4->3，不要直接到2或0）。

若与 Conservative-2k 基本持平：不再固定 +0.3 训练，直接采用 Conservative-2k 进入随机单静态障碍课程，用场景多样性代替继续过拟合该 offset。

若明显退化：停止 Consolidate 分支，Conservative-2k 继续作为最佳 parent。

## 8. 本轮代码改动

- `dreamerv3/dreamerv3/agent.py`
  - 支持 Actor 独立 optimizer。
  - 支持 frozen policy action-mean anchor。
  - 新增 `anchor_mask`，anchor loss 只在 near-obstacle transition 内归一化。
  - 支持 anchor schedule（Consolidate 当前禁用，固定5.0）。
- `dreamerv3/dreamerv3/configs.yaml`
  - `separate_actor_opt`、`actor_opt`、`anchor_scale`、`anchor_schedule_enabled`、`anchor_schedule` 等配置。
- `dreamerv3/embodied/run/bc_distill.py`
  - 离线完整 episode sequence loader。
  - 场景采样当前为20/20/60。
  - 加载 `anchor_action_mean` 与 `anchor_mask`。
- `scripts/cache_s2_wm_anchor_actions.py`
  - 缓存 Mixed frozen RSSM+Actor action mean。
  - 根据真实 clearance 构造 near-obstacle mask。
- Mixed anchor replay：
  `outputs/dreamerv3/stageS2_Conservative_mixed_anchor_replay`

## 9. 操作注意事项

- 不要重复启动 S2-Consolidate；它已经完成。
- 不要把报告里的近似旧 zero-shot 比率与固定100-seed paired结果混用。
- 统计 CSV 时排除 episode 0 的非终止初始化记录，只统计 `last.is_last=true` 的100个完整 episode。
- 不要覆盖2k、4k、Consolidate checkpoint或既有评测输出。
- 下一步只做最终评测和 Gate 判断，不再增加微型诊断。
