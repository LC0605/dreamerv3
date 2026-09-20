# UAV Dreamer Navigation

> **Status: Work in Progress** — 项目仍在开发和训练中，当前仓库是用于版本管理、
> 实验追溯和训练接力的开发快照，不是 final release。

基于 PyBullet 的四旋翼导航与避障研究项目，包含两条实验链路：PPO
基线，以及适配 UAV 环境的 DreamerV3 在线训练、离线 replay 更新和闭环评测。
策略动作经过速度—姿态控制、X 型电机分配和刚体动力学，不直接设置无人机位置。

代码可直接 clone；checkpoint 和 Replay 不在普通 Git 中。继续当前训练前，请先阅读
[`docs/TRAINING_HANDOFF.md`](docs/TRAINING_HANDOFF.md)，按其中路径放置 continuation
资产。实际环境版本及 CPU/GPU 状态见
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)。训练资产尚未随首次源码上传发布。

本仓库保留了大量阶段训练脚本，用于追溯论文实验。脚本名称代表一次实验配置，
不代表一份独立算法实现，也不应在未核对 parent checkpoint、输出目录和 Gate
条件时直接重跑。项目结构和当前实验关系详见 `LEARNING_MAP.md`。

## 目录结构

- `src/uav_navigation/`：当前 UAV 环境、CF2X 动力学、PPO 训练与可视化。
- `dreamerv3/dreamerv3/`：DreamerV3 Agent、RSSM、统一入口和配置。
- `dreamerv3/embodied/`：训练循环、replay、环境适配和 JAX 执行层。
- `quadrotor_env/`：较早的仿真与测试实现；当前 Dreamer adapter 不从这里导入环境。
- `configs/`：PPO 历史配置记录；Dreamer 的有效 preset 位于
  `dreamerv3/dreamerv3/configs.yaml`。
- `scripts/`：训练、评测、replay 构建和诊断包装脚本。当前为保持历史路径稳定，
  未按子目录移动。
- `tests/`：环境、Dreamer adapter 和课程控制器测试。
- `outputs/`：本地 checkpoint、replay、评测、日志、图表和报告；默认不提交普通 Git。

## 环境与安装

基础 UAV/PPO 包要求 Python 3.10 或更高版本：

```bash
conda activate drone_rl
python -m pip install -e . --no-deps --no-build-isolation
PYTHONPATH=src python -m pytest -q
```

DreamerV3 使用独立依赖和环境。已有实验脚本固定使用本机
`/home/user/miniconda3/envs/dreamer_uav/`；在其他机器复现时应先按
`dreamerv3/requirements.txt` 和 `dreamerv3/setup.py` 创建等价环境，并将脚本中的
解释器路径参数化或替换为实际路径。

注意：当前机器的实际 Dreamer 环境是 JAX/JAXLIB 0.6.2，而上游
`dreamerv3/requirements.txt` 仍固定 JAX 0.4.33 CUDA 12。该文件单独不足以精确复现
本项目训练环境，接手者应以环境文档记录的已验证版本为准。

## 主要入口

- PPO 训练：`src/uav_navigation/train_ppo.py`（安装后可使用 `uav-train-ppo`）。
- PPO checkpoint 评测：`scripts/evaluate_checkpoints.py`。
- DreamerV3 统一入口：`dreamerv3/dreamerv3/main.py`。
  - `--script train`：在线环境交互和 replay 训练。
  - `--script bc_distill`：从固定 NPZ episode 数据集更新；配置允许时可更新完整模型，
    不应仅凭名称理解为 Actor BC。
  - `--script eval_only`：加载 checkpoint 做闭环评测。
- UAV 环境：`src/uav_navigation/env.py`；Dreamer 接口适配位于
  `dreamerv3/embodied/envs/quadrotor.py`。
- replay 构建与实验入口：保留在 `scripts/build_*_replay.py`、
  `scripts/collect_*.py` 和 `scripts/train_*.sh` 中。
- 静态障碍、paired 和 Gate 评测：`scripts/evaluate_static_obstacle_gate.sh`、
  `scripts/evaluate_*_paired.sh`、`scripts/evaluate_stage_*_gate.sh`。

所有 Dreamer 命令都应从项目根目录运行，并设置：

```bash
export PYTHONPATH="$PWD/src:$PWD/dreamerv3"
```

## Checkpoint 与实验复现

Dreamer checkpoint 是包含 `agent.pkl` 等文件的目录，通过
`--run.from_checkpoint <checkpoint-directory>` 恢复；具体恢复范围由
`--run.from_checkpoint_regex` 决定。PPO checkpoint 通常由成对的 `model_*.zip`
和 `vec_normalize_*.pkl` 组成，不能只复制其中一个文件。

`outputs/` 当前是约 9 GB 的本地实验档案，包含关键权重和 replay，不应删除或整体
上传普通 Git。发布模型时应从 Gate 报告和实验配置中明确选择 checkpoint；小型、最终
发布权重适合 GitHub Release，大量阶段权重和 replay 更适合外部对象存储。不要把
“时间最新”自动当作“最佳模型”。

当前已充分评测、适合作为保守 continuation baseline 的 Dreamer checkpoint 是：

```text
outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T154503F849521
```

最新研究分支 Recovery2B-1k 仅完成 20-episode quick Gate，尚未完成/通过完整 Gate，
因此没有替代上述 baseline。精确延续 Recovery2B 还需要对应离线 Replay；详细资产清单、
命令和已知退化见训练交接文档。

## 已验证的 PPO 历史基线

PPO stage27 的参数记录位于 `configs/stage27_best.yaml`，对应本地输出为
`outputs/ablations/stage27_shield_g060_m070/`。完整历史报告位于
`outputs/report/UAV_Dreamer阶段汇报.md`。DreamerV3 的当前阶段、parent 关系和 Gate
结论以 `outputs/dreamerv3/` 下的报告、各运行的 `config.yaml` 及
`LEARNING_MAP.md` 为准。
