# UAV_Dreamer 学习地图

核对日期：2026-09-18。范围：本机 `/home/user/UAV_Dreamer` 的源码、脚本、配置和已有报告。本次仅整理文档，不修改训练逻辑、不启动训练、不移动或清理实验文件。

## 1. 先认清四部分

| 部分 | 实际位置 | 应怎样理解 |
|---|---|---|
| 源码 | `src/uav_navigation/`、`dreamerv3/dreamerv3/`、`dreamerv3/embodied/` | 环境与一套共用 Dreamer 实现；PPO 是另一个基线分支 |
| 配置 | `dreamerv3/dreamerv3/configs.yaml` | Dreamer 默认值、quadrotor preset、环境参数、模型参数、更新与评测参数 |
| 配置/历史基线说明 | `configs/stage27_best.yaml` | PPO stage27 参数记录，不是 Dreamer main 自动读取的配置 |
| 训练/评测入口 | `dreamerv3/dreamerv3/main.py` 及 `scripts/*.sh` | main 分派运行模式；shell 文件主要绑定一次实验的参数、数据与输出位置 |
| 实验输出 | `outputs/` | checkpoint、replay、解析后的配置、指标、评测、图表、报告；不是按目录复制出的算法 |

`quadrotor_env/` 是旧仿真/测试代码，当前 Dreamer adapter 导入的是 `uav_navigation.env.UAVNavigationEnv`，学习当前训练链路先不用读它。

根 README 仍以 PPO 结果为主；`dreamerv3/UAV_BRANCH.md` 中“尚无 Dreamer checkpoint”“算法未修改”的表述已不能代表当前工作树。当前 Agent、训练循环等已有本地扩展，且 outputs 中存在 Dreamer checkpoint。旧文档只作为历史背景；定位实现以源码、启动脚本和对应运行的 `config.yaml` 为准，实验结论以对应 Gate 报告为准。

## 2. 核心阅读清单：13 个文件（含 1 个配置）

“需要理解”不等于“每轮都修改”。普通场景实验通常只需调整配置。

| # | 文件 | 作用 / 主要类、函数 | 通常修改什么 | 什么情况下不要动 |
|---|---|---|---|---|
| 1 | `dreamerv3/dreamerv3/configs.yaml` | 配置；`defaults`、`quadrotor` 等 preset，无类/函数 | `env.quadrotor.*` 场景；`agent.*` 学习率、冻结项、anchor；`run.*` 数据与更新次数 | 不为单次实验随意改全局 defaults；不回写历史输出 config |
| 2 | `src/uav_navigation/env.py` | 真正 UAV 环境；`UAVNavigationEnv`、`reset`、`step`、`_observation`、`_lidar`、`_simulate_command` | 新观测/奖励定义、终止规则、尚未支持的场景机制 | 仅改障碍数量/offset/速度等已有参数时不动；不要为新场景复制 env |
| 3 | `src/uav_navigation/dynamics.py` | CF2X 参数与速度—姿态控制；`CF2XParameters`、`VelocityAttitudeController.compute` | 有依据的物理参数或底层控制器修正 | 调学习率、场景或 Dreamer loss 时不动；会改变动力学基准 |
| 4 | `dreamerv3/embodied/envs/quadrotor.py` | Gym 环境到 Embodied 接口；`Quadrotor`、`obs_space`、`act_space`、`step`、`_observation` | 新环境参数透传、日志字段、接口适配 | 不在这里另写环境动力学或另算一套奖励；保留 terminal/truncation 区别 |
| 5 | `dreamerv3/dreamerv3/main.py` | 共用入口、配置解析与工厂；`main`、`make_agent`、`make_env`、`make_replay`、`make_stream` | 注册真正的新环境 suite/运行模式 | 换 seed、checkpoint、logdir、场景时不动 |
| 6 | `dreamerv3/embodied/run/train.py` | 在线采样—更新循环；`train` | 采样与训练调度、checkpoint 生命周期的必要修复 | 近期 `bc_distill` 离线实验不走这个循环；普通超参数调整不动 |
| 7 | `dreamerv3/embodied/run/bc_distill.py` | 离线 episode 加载与更新；`_episodes`、`_stream`、`bc_distill` | 数据字段兼容、序列/mask 对齐、采样机制 | 不因名称含 BC 就假定只训练 Actor；仅调已有 run 参数不动 |
| 8 | `dreamerv3/embodied/core/replay.py` | 在线序列缓存；`Replay.add/sample/update/save/load` | 有明确需要时调整通用采样/持久化机制 | 离线 NPZ 数据集不通过这个缓存；不要把课程目录当 Replay 类 |
| 9 | `dreamerv3/embodied/core/driver.py` | policy 与 env 的交互及回调；`Driver.reset/on_step/_step` | 交互协议、并行环境或回调错误修复 | 场景变化不动；不能打乱 reset 和上一步动作时序 |
| 10 | `dreamerv3/dreamerv3/rssm.py` | 编码、潜状态动力学、重建；`Encoder`、`RSSM.observe/imagine/loss`、`Decoder` | 真正的世界模型结构研究 | 只改地图/障碍/训练步数不动；结构变化影响旧权重兼容性 |
| 11 | `dreamerv3/dreamerv3/agent.py` | 算法组装与目标；`Agent.policy/train/loss/bc_loss`、`imag_loss`、`repl_loss`、`lambda_return` | 明确提出的损失、梯度流、策略更新研究 | 普通课程实验不改 loss；不能把已有 anchor/冻结/BC 配置又写成新算法副本 |
| 12 | `dreamerv3/embodied/jax/heads.py` | Actor、Value 等共用输出头；`MLPHead`、`DictHead`、`Head` | 新输出分布或 head 实现 | 只调隐藏层/学习率优先配置；这里的改变可能同时影响多个预测头 |
| 13 | `dreamerv3/embodied/run/eval_only.py` | 恢复模型、闭环评测和轨迹诊断；`eval_only` 及内部统计回调 | 评测日志字段、明确的统计错误修复 | 不为改善分数改 seed、成功判据或动作模式；不在评测中训练模型 |

补充文件按需查阅，不作为第一轮必读：`embodied/jax/agent.py` 是 JAX 执行、参数加载与设备封装；`src/uav_navigation/temporal_features.py` 是 PPO 特征提取器，不是 Dreamer Encoder。上述 embodied 路径均位于外层 `dreamerv3/` 下。

## 3. 实际入口与配置如何生效

### 统一入口

`dreamerv3/dreamerv3/main.py` 按 `--script` 分派：

- `train` → `embodied/run/train.py`：在线交互、在线 replay、更新。
- `bc_distill` → `embodied/run/bc_distill.py`：读取 `run.bc_dataset/train/*.npz` 与 `validation/*.npz`，调用 `agent.train`。当 `agent.bc_only=False`、`agent.bc_scale=0` 且各模块未冻结时，可以进行 WM + Value + Actor 更新，并非仅 BC 蒸馏。
- `eval_only` → `embodied/run/eval_only.py`：加载 checkpoint 后评测；Actor 在 `Agent.policy(mode='eval')` 使用分布 `pred()`，而非训练时动作采样。
- 还支持 `train_eval`、`parallel` 等运行模式，但不能将“存在入口”当作自动执行许可。

现有近期训练包装脚本：`scripts/train_stage_s2_conservative_6000.sh`、`scripts/train_stage_s2_consolidate_4000.sh`、`scripts/train_stage_s3a_retry_a1_4000.sh`、`scripts/train_stage_s3a_a1_recovery_4000.sh`，均指向共用 main；后两者通过 `--script bc_distill` 进行离线更新。这里的“实际入口”指现有实现和可追溯实验调用链，不表示当前有训练进程正在运行。

Dreamer 评测包装入口包括 `scripts/evaluate_stage_s3a_gate.sh`、`scripts/evaluate_stage_s3a_retry_a1_gate.sh`。文件名带 gate 不代表自动完成 Gate 决策：仍要读取指标并逐项比较阈值。

容易误认的入口：

- `scripts/evaluate_checkpoints.py` 评测 PPO `.zip` 与 `VecNormalize`，不是 Dreamer。
- `src/uav_navigation/train_ppo.py` 是 PPO 训练入口。
- `scripts/continue_dreamer_curriculum.py` 是历史课程编排器，包含 `train_stage/train_and_gate` 和自动启动行为，不能作为只读查看工具直接运行。

### 配置优先级

`configs.yaml: defaults` → `--configs` 指定的 preset（依次覆盖）→ 命令行 `--env.* / --agent.* / --run.*` 覆盖 → main 保存最终值至 `logdir/config.yaml`。

当前入口没有直接把任意 `configs/*.yaml` 当作实验配置载入的通用参数；不要虚构 `--config-file`。当前优先在现有 `configs.yaml` 添加有说明的 preset，并复用 main。将来如需独立实验 YAML，再单独设计加载方式、展示 diff 并验证等价性。本轮未做配置迁移。

场景参数看 `env.quadrotor.*`：`obstacle_count`、`obstacle_lateral_offset`、`obstacle_randomization_level`、`random_offset_low/high`、`dynamic_obstacles`、`temporal_history`、`max_steps` 等。训练参数看 `batch_size/batch_length`、`agent.opt`、`agent.actor_opt`、`agent.freeze_*`、`agent.anchor_*`、`run.bc_*`。

离线训练中，修改 env 参数不会重生成磁盘上的 episode，也不会自动改变已有训练数据的场景分布。场景变化还要核对数据生成参数、manifest、采样比例以及 train/validation 分割。当前 `_episodes` 中 vector 宽度写死为 68，`_stream` 中部分场景/风险采样比例仍硬编码；这是未来配置化的候选位置，本次保留原样。不能声称现在所有参数都已经配置化。

## 4. 输出目录不是独立算法

以下路径均以 `outputs/dreamerv3/` 为前缀：

| 现有目录/模式 | 内容与意义 |
|---|---|
| `stageS1*`、`stageS2*`、`stageS3*` | 课程/消融实验标签；同一套源码的不同参数、数据和训练产物 |
| `stageS2_Conservative_6000`、`stageS2_Consolidate_4000` | 训练结果；内部 `ckpt/`、`config.yaml`、`metrics.jsonl` 等不是算法源码 |
| `stageS3A_random_lateral_5000`、`stageS3A_Retry_A1_4000`、`stageS3A_A1_Recovery_4000` | S3/A1 训练结果；A1/Recovery 是实验标签，不是新的 Agent/RSSM |
| `stageS2_Conservative_mixed_anchor_replay` | 带 anchor 字段的训练数据 |
| `stageS3A_Retry_A1_parent_replay`、`stageS3A_Retry_A1_joint_replay` | parent 数据审计/新旧课程混合数据 |
| `stageS3A_A1_Recovery_replay`、`stageS3A_A1_Recovery_low_replay`、`stageS3A_A1_Recovery_mid_replay`、`stageS3A_A1_Recovery_high_replay` | 恢复实验与不同 offset 区间的数据集 |
| `stageS3A_A1_Recovery_low_split`、`stageS3A_A1_Recovery_low_anchored` | 数据分割/anchor 加工产物 |
| `eval_Conservative*`、`eval_S3A*`、其他 `eval_*` | 评测配置、分数、轨迹与诊断结果 |
| `diagnostics/`、`STAGE_*REPORT*.md`、注册表 JSON | 分析产物、结论与 checkpoint 索引 |

本次扫描 `outputs/dreamerv3/` 未发现 `.py` / `.sh` 文件。不要按 stage 目录数量计算算法数量。相反，`scripts/train_stage*.sh` 确实是启动代码，只是它们复用算法，不能把它们当成 checkpoint。历史重复启动包装保留供追溯，本次不再增加。

`outputs/curriculum/`、`outputs/ablations/`、`outputs/evaluation/` 主要是 PPO 历史实验；`outputs/report/` 和 `outputs/visualization/` 是报告与图像。数据和权重不是源码，但复现实验需要保留，不应因此删除。

### 当前状态的证据边界

8 月 29 日交接文档之后已有 8 月 30 日报告：

- [Retry A1 Gate 报告](outputs/dreamerv3/STAGE_S3A_RETRY_A1_GATE_REPORT_2026-08-30.md)：未通过，不能自动进入 A2。
- [A1 Recovery Gate 报告](outputs/dreamerv3/STAGE_S3A_A1_RECOVERY_GATE_REPORT_2026-08-30.md)：完整 Gate 未通过，A2 未启动。

保留的 Conservative-2k 参考路径：`outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T154503F849521`。Retry A1 与 Recovery 的具体 parent 关系以各自脚本和 config 为准：Retry 来自 Conservative-2k，Recovery 来自 Retry A1 final。文件名中的 6000 不能代替具体 checkpoint 的实际更新数；不能把最近输出自动当作最佳 parent。本次未重新评测或认证模型性能。

## 5. 按顺序学习数据流

1. **config**：从 `configs.yaml` 的 `defaults`、`quadrotor` 和一个已有实验 `config.yaml` 入手，列出场景、观测、动作、训练模式与冻结项。目标：解释为什么同一份源码能跑不同 stage。
2. **env**：读 `UAVNavigationEnv.reset → _observation → step`，再看动力学控制器和 `Quadrotor.step` 适配。目标：说清动作如何变为物理控制，奖励何时生成，以及 `is_first/is_last/is_terminal` 的区别。策略动作经 main 的环境 wrapper 映射到物理动作接口，不是直接设置坐标。
3. **train entry**：读 `main` 的配置合并和分派，再分别看 `train`、`bc_distill`。目标：判断这次是在线采样还是读固定数据；分清环境步数与离线 update 数。
4. **replay**：在线读 `Driver → Replay.add → Replay.sample`；近期离线实验读 `_episodes → _stream → agent.stream → agent.train`。目标：跟踪一个 episode 的观测、动作、reward、终止标志、padding/loss_mask、anchor_mask；保持 observation 与 previous action 对齐。
5. **encoder/RSSM**：读 `Encoder` 得到 tokens，再跟 `RSSM.observe` 得到 posterior；`RSSM.imagine` 在潜状态里按 Actor 动作展开；`RSSM.loss` 训练动态/表示。目标：区别真实观测更新与模型想象，不把 PPO temporal_features 当成 Dreamer Encoder。
6. **actor/value**：回到 `Agent.__init__`：`enc/dyn/dec` 是世界模型组件，`rew/con` 预测奖励与继续概率，`pol` 是 Actor，`val/slowval` 是 Value/Critic。继续读 `loss → imag_loss/repl_loss → lambda_return`，必要时看 `heads.py`。这里没有必须另找的 `actor.py` 或 `critic.py`。目标：解释 imagined return、价值目标、策略目标与可选 BC/anchor 的区别。
7. **evaluation**：读 `eval_only` 的 checkpoint 加载、`mode='eval'` 调用与 episode 统计，对照 Gate 报告。目标：只统计完整终止 episode，排除初始化记录；同样的 seed、场景、回合数和动作口径下比较 Success/Collision/Timeout/OOB、每回合最小 clearance 的均值、navigation time、近障动作与 Actor drift。

一个完整闭环是：观测 → Encoder → RSSM posterior → Actor → 环境动作 → 下一观测/奖励 → replay；训练另从 posterior 出发，在 RSSM 内想象未来，用 reward/continue/value 形成目标更新策略。

## 6. 以后每次训练前的必填汇报

```text
实验名称/目的：
parent checkpoint：精确目录；来源 Gate；恢复模型/优化器/计数器的范围和 regex
修改文件：逐文件列出；无改动也写“无”
修改内容：配置旧值→新值；数据来源、分割与比例；代码改动理由
保持不变内容：观测/动作/奖励/终止/动力学/模型结构/损失/冻结项/评测口径
启动命令：工作目录、Python 路径、环境变量、完整参数（不含省略号）
输出目录：训练、数据、评测分别列出；不能覆盖已有产物
Gate：对照模型、固定种子、完整回合数、指标、每项数值阈值、退化容限
失败处理：停止本分支；保留结果；不自动续训、换 parent 或进入下一阶段
变更证据：git diff + git status；未纳入 Git 的文件另附 before/after diff
确认状态：等待用户明确确认，确认前不启动训练
```

Gate 必须在实验前确定，不能看完结果再移动阈值。“明显改善”等表述需要事先变成数值标准；各阶段 Gate 不可混用。已有历史命令只作追溯，不能直接复跑并覆盖旧 logdir。

每轮优先增加/调整有说明的 config preset，复用 main 和现有训练循环。只有现有机制确实表达不了需求时才提出最小源码变更，解释必要性与测试；不为新场景复制 Agent/RSSM/Actor/Value/replay。

## 7. Git 审阅与本次变更

项目根目录没有 `.git`，因此根目录 `git status` / `git diff` 会报告“不是 git 仓库”。只有 `dreamerv3/` 在 Git 管理下。不能把子仓库状态说成整个项目状态，也不能用空 diff 声称根目录文件未改变。

```bash
git -C /home/user/UAV_Dreamer/dreamerv3 status --short
git -C /home/user/UAV_Dreamer/dreamerv3 diff
git -C /home/user/UAV_Dreamer/dreamerv3 diff --cached
# 本次两个新增文件不受子仓库跟踪；no-index 返回 1 表示发现差异：
git diff --no-index -- /dev/null /home/user/UAV_Dreamer/LEARNING_MAP.md
git diff --no-index -- /dev/null /home/user/UAV_Dreamer/AGENTS.md
```

本次新增 `LEARNING_MAP.md`（地图与学习顺序）和 `AGENTS.md`（将用户要求保留为后续协作规则），未修改已有源码、配置或脚本。子仓库已有 9 个 tracked 文件修改，另有 `UAV_BRANCH.md`、`embodied/envs/quadrotor.py`、`embodied/run/bc_distill.py` 未跟踪；它们在本次开始时已经存在。未初始化根仓库，未暂存或提交。

上述规则通过协作执行，尚不是训练程序中的强制确认机制。今后修改后、训练前必须展示差异和状态，等待用户确认。
