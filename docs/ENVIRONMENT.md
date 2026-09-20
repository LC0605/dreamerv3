# Development Environment

本页记录 2026-09-20 审计时的实际机器与两个 Conda 环境。它是可复现参考，不是把整台
机器的 `pip freeze` 当成项目依赖锁。

## Host

- Ubuntu 22.04.5 LTS，x86_64
- Python 3.10.20（两个项目 Conda 环境）
- 历史 `nvidia-smi` 快照：NVIDIA GeForce RTX 4070、驱动 580.178.04、CUDA 13.0
- 本次非交互审计会话无法连接 NVIDIA 驱动；GPU 可用性必须在实际训练 shell 中重验

验证 GPU：

```bash
nvidia-smi
python -c 'import jax; print(jax.devices())'
python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

## Dreamer 环境

现有环境路径：`/home/user/miniconda3/envs/dreamer_uav`。

| 包 | 实测版本 |
|---|---|
| Python | 3.10.20 |
| JAX | 0.6.2 |
| JAXLIB | 0.6.2 |
| NumPy | 1.26.4 |
| Gymnasium | 1.3.0 |
| PyBullet | 3.2.7 构建 |

该环境本次只枚举到 CPU。近期 S3A/Recovery 脚本显式指定 `--jax.platform cpu`，因此其
复现不要求 NVIDIA GPU，但速度会受 CPU 限制。要启用 GPU，应先安装与本机驱动匹配的
JAX wheel，并移除/修改该参数后做 smoke test。

上游 `dreamerv3/requirements.txt` 固定 `jax[cuda12]==0.4.33`，与当前实际 JAX 0.6.2
不一致。因此它目前是依赖参考，不是已验证 lock file。接手者应先创建隔离环境并安装
项目本身，再验证 import 和测试；不要在现有训练环境中盲目升级 JAX。

典型环境激活和源码路径：

```bash
conda activate dreamer_uav
cd UAV_Dreamer
export PYTHONPATH="$PWD/src:$PWD/dreamerv3"
python -c 'import jax; print(jax.__version__, jax.devices())'
```

## PPO 环境

现有环境路径：`/home/user/miniconda3/envs/drone_rl`。

| 包 | 实测版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.9 |

本次审计中 `torch.cuda.is_available()` 为 False。PPO 环境测试可在 CPU 运行；大规模
训练前应在正常驱动会话中重新确认 CUDA。当前环境的 Pandas 因缺少 `pytz` 无法 import，
这是报告/绘图环境的已知缺口，不影响 36 个根项目测试。

## 安装与验证边界

根 `pyproject.toml` 只声明基础 UAV 环境依赖：Gymnasium、NumPy 和 PyBullet；它不足以
安装 Dreamer/JAX 或 PPO/Stable-Baselines3 训练栈。推荐暂时保留两个隔离环境：

1. `dreamer_uav`：Dreamer/JAX 与 `dreamerv3/requirements.txt` 所列组件。
2. `drone_rl`：PyTorch、Stable-Baselines3 和 PPO 工具。

基础包和测试：

```bash
conda activate drone_rl
cd UAV_Dreamer
python -m pip install -e . --no-deps --no-build-isolation
PYTHONPATH=src python -m pytest -q
```

CPU 可运行：根测试、环境 smoke test、近期显式 CPU 的 Dreamer 离线脚本。GPU 建议用于
大规模 Dreamer/PPO 训练，但必须先验证驱动、CUDA 与 JAX/PyTorch wheel 的兼容性。
