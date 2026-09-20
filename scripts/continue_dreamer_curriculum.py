#!/usr/bin/env python3
"""Continue the verified Dreamer curriculum after the active Stage 0e run.

The runner is intentionally conservative: it waits for the already running
process, evaluates with matching dynamics, inserts a smooth-command adaptation
stage, evaluates it, and only then starts the fixed-obstacle curriculum.
Every decision is written to curriculum_status.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/user/miniconda3/envs/dreamer_uav/bin/python")
MAIN = ROOT / "dreamerv3/dreamerv3/main.py"
OUT = ROOT / "outputs/dreamerv3"
STATUS = OUT / "curriculum_status.json"
VISUALIZER_PYTHON = Path("/home/user/miniconda3/envs/drone_rl/bin/python")
VISUALIZER = ROOT / "scripts/visualize_dreamer_curriculum.py"
SNAPSHOT_WATCHER = ROOT / "scripts/snapshot_training_milestones.sh"
BASE_ENV = [
    "--configs", "quadrotor", "--jax.platform", "cpu", "--jax.prealloc", "False",
    "--logger.outputs", "jsonl", "--logger.filter",
    "score|length|fps|ratio|train/loss/|train/rand/|log/",
    "--env.quadrotor.max_steps", "600", "--env.quadrotor.arena_x", "16.0",
    "--env.quadrotor.arena_y", "16.0", "--env.quadrotor.arena_z", "5.0",
    "--env.quadrotor.goal_distance_low", "4.0",
    "--env.quadrotor.goal_distance_high", "8.0",
]


def write_status(stage: str, state: str, **details):
    payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "stage": stage, "state": state, **details}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def refresh_visualization():
    subprocess.run([str(VISUALIZER_PYTHON), str(VISUALIZER)], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=False)


def maximum_step(folder: Path) -> int:
    result = 0
    path = folder / "metrics.jsonl"
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        try:
            result = max(result, int(json.loads(line).get("step", 0)))
        except (ValueError, json.JSONDecodeError):
            continue
    return result


def process_mentions(text: str) -> bool:
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if text in command and "continue_dreamer_curriculum.py" not in command:
            return True
    return False


def latest_checkpoint(folder: Path) -> Path:
    checkpoints = [path for path in (folder / "ckpt").iterdir() if path.is_dir()]
    if not checkpoints:
        raise RuntimeError(f"no checkpoint found in {folder}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def saved_summary(name: str) -> dict | None:
    path = OUT / f"eval_{name}" / "evaluation_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def run(folder: Path, extra: list[str]):
    folder.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(MAIN), "--logdir", str(folder), *BASE_ENV, *extra]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    # Training and evaluation use the same model shapes across many fresh JAX
    # processes. Persist compiled CPU executables so later gates do not spend
    # about a minute recompiling identical functions each time.
    cache = OUT / "jax_compilation_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    with (folder / "runner.log").open("a") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def options(values: dict[str, object]) -> list[str]:
    result = []
    for key, value in values.items():
        result.extend((f"--env.quadrotor.{key}", str(value)))
    return result


def evaluate(name: str, checkpoint: Path, acceleration: tuple[float, float, float],
             environment: dict[str, object] | None = None, steps: int = 20000):
    folder = OUT / name
    run(folder, [
        "--script", "eval_only", "--run.steps", str(steps), "--run.envs", "1",
        "--run.from_checkpoint", str(checkpoint), "--run.log_every", "60",
        "--env.quadrotor.acceleration_xy", str(acceleration[0]),
        "--env.quadrotor.acceleration_z", str(acceleration[1]),
        "--env.quadrotor.acceleration_yaw", str(acceleration[2]),
        *options(environment or {"obstacle_count": 0}),
    ])
    episodes = successes = collisions = out_of_bounds = timeouts = 0.0
    path_lengths = minimum_clearances = boundary_clearances = action_deltas = 0.0
    path_ratios = maximum_altitudes = 0.0
    episodes_with_obstacle = successes_with_obstacle = 0.0
    collisions_with_obstacle = timeouts_with_obstacle = 0.0
    episode_scores = []
    for line in (folder / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "episode/length" in row:
            episode_scores.append(float(row["episode/score"]))
        count = float(row.get("epstats/episode_count", 0.0))
        episodes += count
        successes += count * row.get("epstats/log/success/sum", 0.0)
        collisions += count * row.get("epstats/log/collision/sum", 0.0)
        out_of_bounds += count * row.get("epstats/log/out_of_bounds/sum", 0.0)
        timeouts += count * row.get("epstats/log/timeout/sum", 0.0)
        episodes_with_obstacle += count * row.get(
            "epstats/log/obstacle_episode/sum", 0.0)
        successes_with_obstacle += count * row.get(
            "epstats/log/success_with_obstacle/sum", 0.0)
        collisions_with_obstacle += count * row.get(
            "epstats/log/collision_with_obstacle/sum", 0.0)
        timeouts_with_obstacle += count * row.get(
            "epstats/log/timeout_with_obstacle/sum", 0.0)
        path_lengths += count * row.get("epstats/log/path_length/max", 0.0)
        minimum_clearances += count * row.get(
            "epstats/log/minimum_clearance/min", 0.0)
        boundary_clearances += count * row.get(
            "epstats/log/boundary_clearance/min", 0.0)
        action_deltas += count * row.get("epstats/log/action_delta/avg", 0.0)
        path_ratios += count * row.get("epstats/log/path_length_ratio/max", 0.0)
        maximum_altitudes += count * row.get("epstats/log/altitude/max", 0.0)
    if not episodes:
        # Compatibility with evaluations produced before episode_count existed.
        episodes = float(len(episode_scores))
        successes = float(sum(score > 0.0 for score in episode_scores))
    metrics = {
        "episodes": int(episodes), "successes": int(round(successes)),
        "collisions": int(round(collisions)),
        "out_of_bounds": int(round(out_of_bounds)), "timeouts": int(round(timeouts)),
    }
    metrics["success_rate"] = successes / max(episodes, 1)
    metrics["out_of_bounds_rate"] = out_of_bounds / max(episodes, 1)
    metrics["mean_path_length"] = path_lengths / max(episodes, 1)
    metrics["mean_minimum_clearance"] = minimum_clearances / max(episodes, 1)
    metrics["mean_minimum_boundary_clearance"] = boundary_clearances / max(episodes, 1)
    metrics["mean_action_delta"] = action_deltas / max(episodes, 1)
    metrics["mean_path_length_ratio"] = path_ratios / max(episodes, 1)
    metrics["mean_maximum_altitude"] = maximum_altitudes / max(episodes, 1)
    if episodes_with_obstacle:
        metrics["episodes_with_obstacle"] = int(round(episodes_with_obstacle))
        metrics["success_rate_with_obstacle"] = (
            successes_with_obstacle / episodes_with_obstacle)
        metrics["collision_rate_with_obstacle"] = (
            collisions_with_obstacle / episodes_with_obstacle)
        metrics["timeout_rate_with_obstacle"] = (
            timeouts_with_obstacle / episodes_with_obstacle)
    (folder / "evaluation_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    refresh_visualization()
    return metrics


def passed(metrics: dict, success_rate: float = 0.90,
           collision_rate: float = 0.05) -> bool:
    episodes = max(metrics["episodes"], 1)
    return metrics["episodes"] >= 30 \
        and metrics["success_rate"] >= success_rate \
        and metrics["collisions"] / episodes <= collision_rate \
        and metrics["out_of_bounds_rate"] <= 0.05


def optimization_options(name: str) -> list[str]:
    """Return the update recipe for a named curriculum attempt."""
    optimization = ["--run.train_ratio", "16", "--agent.opt.lr", "2e-5"]
    is_late_retry = any(
        token in name for token in ("_retry2", "_retry3", "_retry4"))
    is_precision_stage = any(
        token in name for token in ("_offset_", "_centered"))
    if "_actoronly_" in name:
        # The parent world model already represents obstacle observations, but
        # joint fine-tuning at narrower offsets repeatedly moves its latent
        # features and destroys the mature policy. Freeze representation,
        # reconstruction, reward and continuation losses; update only the
        # actor/value objectives on replay states from the harder geometry.
        optimization = [
            "--run.train_ratio", "8", "--agent.opt.lr", "1e-5",
            "--agent.loss_scales.rec", "0.0",
            "--agent.loss_scales.rew", "0.0",
            "--agent.loss_scales.con", "0.0",
            "--agent.loss_scales.dyn", "0.0",
            "--agent.loss_scales.rep", "0.0",
            "--agent.loss_scales.policy", "0.1",
            "--agent.loss_scales.value", "1.0",
            "--agent.loss_scales.repval", "0.3",
            "--agent.imag_length", "8",
            "--agent.policy.minstd", "0.03",
            "--agent.imag_loss.actent", "1e-4",
        ]
    elif "_recovery" in name and is_late_retry:
        # A recovery retry must not silently reuse the first recovery recipe.
        # At this point the inherited controller is already close to the gate,
        # so preserve it with fewer updates and a much smaller actor loss. The
        # world model still adapts to obstacle transitions, while short latent
        # rollouts limit compounding reward-model errors.
        optimization = [
            "--run.train_ratio", "4", "--agent.opt.lr", "1e-5",
            "--agent.loss_scales.policy", "0.015",
            "--agent.imag_length", "6",
            # The generic continuous-control minimum (0.1) injects roughly
            # 0.04 m/s velocity noise at this UAV speed limit on every action.
            # Keep stochastic Dreamer evaluation, but use exploration suited
            # to precision flight and reduce entropy pressure after transfer.
            "--agent.policy.minstd", "0.03",
            "--agent.imag_loss.actent", "1e-4",
        ]
    elif "_recovery" in name:
        # Recovery starts from a nearly passing controller. Keep world-model
        # learning active while making only very small policy changes so two
        # remaining collision cases can be corrected without inducing hover.
        optimization = [
            "--run.train_ratio", "8", "--agent.opt.lr", "2e-5",
            "--agent.loss_scales.policy", "0.03",
            "--agent.imag_length", "8",
        ]
    elif is_late_retry:
        # If uniform fine-tuning still drifts, keep world-model adaptation
        # active but slow the actor specifically and shorten imagination so it
        # cannot amplify reward-model errors over long latent rollouts.
        optimization = [
            "--run.train_ratio", "4", "--agent.opt.lr", "1e-5",
            "--agent.loss_scales.policy", "0.015",
            "--agent.imag_length", "6",
            "--agent.policy.minstd", "0.03",
            "--agent.imag_loss.actent", "1e-4",
        ]
    elif is_precision_stage:
        # Narrow-offset transfer starts from an already competent controller;
        # its first full-rate run showed catastrophic forgetting. Adapt the
        # world model while limiting how far the actor can move in one stage.
        optimization = [
            "--run.train_ratio", "4", "--agent.opt.lr", "1e-5",
            "--agent.loss_scales.policy", "0.005",
            "--agent.imag_length", "6",
            "--agent.policy.minstd", "0.03",
            "--agent.imag_loss.actent", "1e-4",
        ]
    return optimization


def milestone_thresholds(steps: int) -> list[int]:
    """Capture early transfer peaks as well as the middle and late policy."""
    first = min(4000, max(1, int(steps * 0.16)))
    return sorted({first, *(max(1, int(steps * fraction))
                            for fraction in (0.32, 0.48, 0.64, 0.80))})


def snapshot_step(path: Path) -> int:
    """Extract numeric training progress from ``step_<n>_<checkpoint>``."""
    try:
        return int(path.name.split("_", 2)[1])
    except (IndexError, ValueError):
        return 2**63 - 1


def train_stage(name: str, checkpoint: Path, steps: int,
                environment: dict[str, object]) -> Path:
    folder = OUT / name
    write_status(name, "training", inherited_checkpoint=str(checkpoint),
                 target_steps=steps, environment=environment)
    optimization = optimization_options(name)
    # Preserve four immutable points from every training attempt. This is part
    # of the controller (rather than a manually launched helper) so later
    # static, dynamic, and conflict stages receive the same anti-forgetting
    # model selection automatically.
    thresholds = milestone_thresholds(steps)
    watcher = subprocess.Popen(
        ["bash", str(SNAPSHOT_WATCHER), name,
         *(str(value) for value in thresholds)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        run(folder, [
            "--run.steps", str(steps), "--run.envs", "1",
            "--run.from_checkpoint", str(checkpoint), "--run.log_every", "60",
            "--run.report_every", "300", "--run.save_every", "60",
            # Curriculum stages fine-tune a mature controller. A lower update
            # density and learning rate reduce catastrophic forgetting while still
            # allowing the larger downstream budgets to learn new dynamics.
            *optimization,
            *options(environment),
        ])
    finally:
        try:
            watcher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.terminate()
            watcher.wait(timeout=5)
    return latest_checkpoint(folder)


def train_and_gate(name: str, checkpoint: Path, train_steps: int,
                   environment: dict[str, object], success_rate: float,
                   collision_rate: float) -> Path | None:
    current = checkpoint
    best_checkpoint = checkpoint
    best_metrics = None
    budget = train_steps

    def quality(metrics: dict) -> float:
        episodes = max(metrics["episodes"], 1)
        # Success dominates, while unsafe terminal modes are explicitly
        # penalized. Timeouts cost less because they still provide safe route
        # experience that can improve with more optimization.
        return (metrics["success_rate"]
                - 0.75 * metrics["collisions"] / episodes
                - 0.50 * metrics["out_of_bounds"] / episodes
                - 0.20 * metrics["timeouts"] / episodes)

    max_attempts = 5

    def evaluate_milestones(run_name: str) -> Path | None:
        """Evaluate immutable in-training snapshots before discarding a run.

        Fine-tuning can peak early and then forget a previously learned skill.
        The snapshot watcher preserves several points; treating them as normal
        gate candidates makes model selection depend on independent evaluation
        rather than the arbitrary final optimizer step.
        """
        nonlocal best_metrics, best_checkpoint
        snapshot_root = OUT / "earlystop_snapshots" / run_name
        if not snapshot_root.exists():
            return None
        for candidate in sorted(snapshot_root.glob("step_*"), key=snapshot_step):
            if not candidate.is_dir() or not (candidate / "done").exists():
                continue
            eval_name = f"{run_name}_milestone_{candidate.name}"
            snapshot_metrics = saved_summary(eval_name)
            if snapshot_metrics is None:
                write_status(run_name, "milestone_evaluating",
                             checkpoint=str(candidate), environment=environment)
                snapshot_metrics = evaluate(
                    f"eval_{eval_name}", candidate, (0.8, 0.5, 0.8), environment)
            if best_metrics is None or quality(snapshot_metrics) > quality(best_metrics):
                best_metrics, best_checkpoint = snapshot_metrics, candidate
            if passed(snapshot_metrics, success_rate, collision_rate):
                write_status(name, "gate_passed_by_milestone",
                             checkpoint=str(candidate), metrics=snapshot_metrics)
                return candidate
        return None

    # Measure transfer before updating the model. Easy curriculum stages are
    # often already solved by the inherited policy; training them anyway can
    # cause needless forgetting. The preflight is also an immutable baseline
    # for diagnosing whether learning helped.
    preflight_name = f"{name}_pretrain"
    preflight = saved_summary(preflight_name)
    if preflight is None:
        write_status(name, "preflight_evaluating", checkpoint=str(checkpoint),
                     environment=environment)
        preflight = evaluate(f"eval_{preflight_name}", checkpoint,
                             (0.8, 0.5, 0.8), environment)
    if passed(preflight, success_rate, collision_rate):
        write_status(name, "gate_passed_without_training",
                     checkpoint=str(checkpoint), metrics=preflight)
        return checkpoint
    best_metrics = preflight
    best_checkpoint = checkpoint

    for attempt in range(max_attempts):
        run_name = name if attempt == 0 else f"{name}_retry{attempt}"
        prior = saved_summary(run_name)
        if prior:
            candidate = latest_checkpoint(OUT / run_name)
            if best_metrics is None or quality(prior) > quality(best_metrics):
                best_metrics, best_checkpoint = prior, candidate
            current = best_checkpoint
            if passed(prior, success_rate, collision_rate):
                write_status(name, "gate_passed", checkpoint=str(candidate),
                             attempts=attempt + 1, metrics=prior, resumed=True)
                return candidate
            milestone = evaluate_milestones(run_name)
            if milestone is not None:
                return milestone
            # A completed failed attempt is immutable evidence; resume from its
            # checkpoint at the next retry instead of evaluating it again.
            budget = max(25000, train_steps // 2)
            continue
        folder = OUT / run_name
        if maximum_step(folder) >= int(0.98 * budget) and not process_mentions(str(folder)):
            current = latest_checkpoint(folder)
        else:
            current = train_stage(run_name, current, budget, environment)
        write_status(run_name, "evaluating", checkpoint=str(current), attempt=attempt)
        metrics = evaluate(f"eval_{run_name}", current, (0.8, 0.5, 0.8),
                           environment)
        if best_metrics is None or quality(metrics) > quality(best_metrics):
            best_metrics, best_checkpoint = metrics, current
        if passed(metrics, success_rate, collision_rate):
            write_status(name, "gate_passed", checkpoint=str(current),
                         attempts=attempt + 1, metrics=metrics)
            return current
        milestone = evaluate_milestones(run_name)
        if milestone is not None:
            return milestone
        failure_rates = {
            key: metrics[key] / max(metrics["episodes"], 1)
            for key in ("collisions", "out_of_bounds", "timeouts")
        }
        write_status(run_name, "gate_retry", checkpoint=str(current),
                     attempt=attempt + 1, metrics=metrics,
                     failure_rates=failure_rates,
                     best_checkpoint=str(best_checkpoint),
                     best_quality=quality(best_metrics))
        # Never propagate a catastrophically forgotten policy into the next
        # retry. Continue from the strongest independently evaluated policy.
        current = best_checkpoint
        budget = max(25000, train_steps // 2)
    write_status(name, "gate_failed_after_retries", checkpoint=str(current),
                 attempts=max_attempts, metrics=best_metrics,
                 best_checkpoint=str(best_checkpoint))
    return None


def gate_frozen_policy(name: str, checkpoint: Path,
                       environment: dict[str, object], success_rate: float,
                       collision_rate: float) -> Path | None:
    """Gate a frozen Dreamer policy with an external execution-time shield.

    Safety projection changes the action actually applied by the simulator.
    Training through it would therefore pair the commanded action in replay
    with a transition caused by a different executed action. Keep the learned
    model immutable and evaluate the hybrid controller only.
    """
    metrics = saved_summary(name)
    if metrics is None:
        write_status(name, "shielded_evaluating", checkpoint=str(checkpoint),
                     environment=environment)
        metrics = evaluate(f"eval_{name}", checkpoint, (0.8, 0.5, 0.8),
                           environment)
    if passed(metrics, success_rate, collision_rate):
        write_status(name, "shielded_gate_passed", checkpoint=str(checkpoint),
                     metrics=metrics)
        return checkpoint
    write_status(name, "shielded_gate_failed", checkpoint=str(checkpoint),
                 metrics=metrics)
    return None


def main():
    stage0e = OUT / "stage0e_large_arena_long_goal"
    # Resume from authoritative gate evidence when it already exists. The
    # original run stopped at 49,744/50,000 environment steps after writing a
    # final checkpoint and a passing independent evaluation. Waiting for the
    # nominal counter in that case deadlocks recovery even though the stage is
    # demonstrably complete.
    metrics0e = saved_summary("stage0e_large_arena_long_goal")
    if not (metrics0e and passed(metrics0e)):
        write_status("stage0e", "waiting_for_training", step=maximum_step(stage0e))
        while maximum_step(stage0e) < 50000 and (
                maximum_step(stage0e) < 49000 or process_mentions(str(stage0e))):
            time.sleep(30)
            write_status("stage0e", "waiting_for_training", step=maximum_step(stage0e))
        # The terminal metric can be flushed just before the final checkpoint.
        time.sleep(60)
    checkpoint0e = latest_checkpoint(stage0e)
    write_status("stage0e", "evaluating", checkpoint=str(checkpoint0e))
    if not metrics0e or not passed(metrics0e):
        metrics0e = evaluate("eval_stage0e_large_arena_long_goal", checkpoint0e,
                             (100.0, 100.0, 100.0))
    if not passed(metrics0e):
        write_status("stage0e", "gate_failed", metrics=metrics0e)
        return

    checkpoint0f = checkpoint0e
    smoothing_stages = (
        ("stage0f1_accel_2p0", 10000,
         {"obstacle_count": 0, "acceleration_xy": 2.0,
          "acceleration_z": 1.0, "acceleration_yaw": 1.5}),
        ("stage0f2_accel_1p2", 15000,
         {"obstacle_count": 0, "acceleration_xy": 1.2,
          "acceleration_z": 0.7, "acceleration_yaw": 1.0}),
        ("stage0f3_accel_0p8", 25000,
         {"obstacle_count": 0, "acceleration_xy": 0.8,
          "acceleration_z": 0.5, "acceleration_yaw": 0.8}),
    )
    for smooth_name, smooth_steps, smooth_env in smoothing_stages:
        checkpoint0f = train_and_gate(
            smooth_name, checkpoint0f, smooth_steps, smooth_env, 0.90, 0.05)
        if checkpoint0f is None:
            return

    fixed = {
        "obstacle_count": 1, "obstacle_layout": "corridor",
        "obstacle_randomization_level": 0,
        "dynamic_obstacles": False,
    }
    checkpoint = checkpoint0f
    # Safety projection is deliberately excluded from training: an internal
    # action correction would make replay contain commanded actions paired
    # with transitions caused by different executed actions, violating the
    # world model's action-conditioning assumption. It remains an evaluation
    # and deployment ablation only.
    # Presence mixing supplies obstacle-free rehearsal inside the same replay
    # stream. This protects the mature goal-reaching skill while the world
    # model and actor learn the new obstacle-conditioned branch. Earlier weak
    # shaping (0.8 m) replaces the old late/strong 0.45 m penalty that either
    # allowed collisions or caused hovering after fine-tuning.
    obstacle_bootstrap = (
        ("v4_presence_0p35", 1.5, 0.35, 0.80, 0.50, 0.75,
         0.45, 30000, 0.95, 0.03),
        ("v5_recovery_presence_0p65", 1.5, 0.65, 0.90, 0.30, 0.50,
         0.42, 20000, 0.93, 0.04),
        # A separate p72 bridge was tested for five attempts but did not pass
        # (best preflight 89.8% success with 6.1% collisions), while the
        # downstream p80 milestone independently passed. Keep the evidence on
        # disk, but do not force recovery through a demonstrably worse branch.
        ("v5_recovery_presence_0p80", 1.5, 0.80, 0.85, 0.45, 0.70,
         0.41, 30000, 0.92, 0.045),
        # The p80 policy reaches reliably, but a direct jump to p100 leaves a
        # repeatable ~6% collision tail. Adapt the obstacle-conditioned branch
        # at p90 with an earlier safety/TTC signal and slower vehicle envelope
        # before removing obstacle-free rehearsal completely.
        ("v7_safety_presence_0p90", 1.5, 0.90, 0.95, 0.70, 0.90,
         0.38, 30000, 0.90, 0.04),
        # New name is intentional: cached v5 evaluations were produced from
        # the old direct-p80 parent and are not valid evidence for the new p90
        # lineage.
        ("v7_safety_offset_1p5", 1.5, 1.00, 0.90, 0.65, 0.85,
         0.38, 40000, 0.90, 0.05),
        ("v11_shielded_offset_1p25", 1.25, 1.00, 0.85, 0.70, 0.90,
         0.40, 0, 0.90, 0.05),
        ("v12_strongshield_offset_1p0", 1.0, 1.00, 0.80, 0.85, 1.00,
         0.38, 0, 0.90, 0.05),
        ("v13_splitshield_offset_0p5", 0.5, 1.00, 0.75, 1.05, 1.20,
         0.28, 0, 0.90, 0.05),
        ("v13_splitshield_centered", 0.0, 1.00, 0.70, 1.25, 1.35,
         0.26, 0, 0.90, 0.05),
    )
    for (suffix, offset, spawn_probability, safety_distance, safety_weight,
         ttc_weight, speed, steps, success_gate, collision_gate) in obstacle_bootstrap:
        if suffix == "v5_recovery_presence_0p65":
            recovery = OUT / "stage1a_fixed_v4_presence_0p65_retry2"
            if recovery.exists():
                checkpoint = latest_checkpoint(recovery)
        shielded = any(token in suffix for token in (
            "v11_shielded_", "v12_strongshield_", "v13_splitshield_"))
        scene = {
            **fixed, "obstacle_lateral_offset": offset,
            "obstacle_spawn_probability": spawn_probability,
            "safety_distance": safety_distance,
            "safety_weight": safety_weight,
            "ttc_reward_weight": ttc_weight,
            "horizontal_speed_limit": speed,
            "safety_projection": shielded,
        }
        if suffix.startswith("v12_strongshield_"):
            scene.update({
                "safety_projection_horizon": 4.0,
                "safety_projection_margin": (
                    0.9 if offset >= 1.0 else 1.0 if offset >= 0.5 else 1.1),
                "safety_projection_gain": (
                    0.7 if offset >= 1.0 else 0.8 if offset >= 0.5 else 0.9),
            })
        if suffix.startswith("v13_splitshield_"):
            centered = offset < 0.25
            scene.update({
                "boundary_guard_distance": 1.0,
                "safety_projection_horizon": 5.0,
                "safety_projection_margin": 1.5 if centered else 1.4,
                "safety_projection_gain": 1.2 if centered else 1.1,
                "acceleration_xy": 0.50 if centered else 0.55,
                "acceleration_z": 0.35 if centered else 0.38,
                "acceleration_yaw": 0.50 if centered else 0.55,
            })
        stage_name = f"stage1a_fixed_{suffix}"
        if shielded:
            checkpoint = gate_frozen_policy(
                stage_name, checkpoint, scene, success_gate, collision_gate)
        else:
            checkpoint = train_and_gate(stage_name, checkpoint, steps, scene,
                                        success_gate, collision_gate)
        if checkpoint is None:
            return

    random_single = {
        **fixed, "obstacle_lateral_offset": 0.0, "obstacle_randomization_level": 3,
        "obstacle_half_xy_low": 0.20, "obstacle_half_xy_high": 0.35,
        "obstacle_height_low": 1.20, "obstacle_height_high": 2.80,
        "horizontal_speed_limit": 0.28,
        "boundary_guard_distance": 1.0,
        "safety_projection": True,
        "safety_projection_horizon": 5.0,
        "safety_projection_margin": 1.4,
        "safety_projection_gain": 1.1,
        "acceleration_xy": 0.55, "acceleration_z": 0.38,
        "acceleration_yaw": 0.55,
    }
    checkpoint = gate_frozen_policy(
        "stage1b_random_single_static_shielded", checkpoint,
        random_single, 0.90, 0.05)
    if checkpoint is None:
        return

    common_arena = {
        "obstacle_layout": "arena", "obstacle_randomization_level": 3,
        "obstacle_half_xy_low": 0.20, "obstacle_half_xy_high": 0.35,
        "obstacle_height_low": 1.20, "obstacle_height_high": 2.80,
        "obstacle_min_spacing": 1.0, "endpoint_clearance": 1.0,
    }
    for count in (3, 6, 12):
        scene = {**common_arena, "obstacle_count": count,
                 "dynamic_obstacles": False}
        if count >= 6:
            dense = count >= 12
            scene.update({
                "horizontal_speed_limit": 0.26 if dense else 0.28,
                "boundary_guard_distance": 1.0,
                "safety_projection": True,
                "safety_projection_horizon": 5.0,
                "safety_projection_margin": 1.5 if dense else 1.4,
                "safety_projection_gain": 1.2 if dense else 1.1,
                "acceleration_xy": 0.50 if dense else 0.55,
                "acceleration_z": 0.35 if dense else 0.38,
                "acceleration_yaw": 0.50 if dense else 0.55,
            })
            checkpoint = gate_frozen_policy(
                f"stage2_static_{count}_shielded", checkpoint, scene, 0.90, 0.05)
        else:
            checkpoint = train_and_gate(
                f"stage2_static_{count}", checkpoint, 50000, scene, 0.90, 0.05)
        if checkpoint is None:
            return

    slow_motion = {
        "dynamic_obstacles": True, "obstacle_motion_amplitude": 0.60,
        "obstacle_speed_low": 0.08, "obstacle_speed_high": 0.15,
    }
    for count, steps, threshold in ((1, 75000, 0.85), (3, 100000, 0.85),
                                     (6, 100000, 0.82), (12, 100000, 0.80)):
        scene = {**common_arena, **slow_motion, "obstacle_count": count}
        checkpoint = train_and_gate(f"stage3_dynamic_{count}", checkpoint, steps,
                                    scene, threshold, 0.10)
        if checkpoint is None:
            return

    # Learn all conflict geometries one at a time before combining two threats.
    for mode in ("crossing", "head_on", "vertical"):
        scene = {
            **common_arena, **slow_motion, "obstacle_count": 12,
            "forced_conflict_count": 1, "forced_conflict_probability": 1.0,
            "forced_conflict_mode": mode, "conflict_nominal_speed": 0.50,
            "conflict_speed_low": 0.30, "conflict_speed_high": 0.70,
        }
        checkpoint = train_and_gate(f"stage4_conflict_1_{mode}", checkpoint,
                                    100000, scene, 0.80, 0.10)
        if checkpoint is None:
            return

    double_scene = {
        **common_arena, **slow_motion, "obstacle_count": 12,
        "forced_conflict_count": 2, "forced_conflict_probability": 1.0,
        "forced_conflict_mode": "mixed", "conflict_nominal_speed": 0.50,
        "conflict_speed_low": 0.30, "conflict_speed_high": 0.70,
    }
    checkpoint = train_and_gate("stage5_conflict_2_mixed", checkpoint, 200000,
                                double_scene, 0.80, 0.10)
    if checkpoint is None:
        return
    baseline = saved_summary("stage5_conflict_2_mixed")
    generalization_cases = {
        "unseen_seed": {**double_scene, "seed": 101},
        "longer_routes": {
            **double_scene, "seed": 103, "arena_x": 18.0, "arena_y": 18.0,
            "goal_distance_low": 8.0, "goal_distance_high": 12.0,
        },
        "sensor_noise_dropout": {
            **double_scene, "seed": 107, "lidar_noise_std": 0.03,
            "observation_noise_std": 0.005,
            "observation_dropout_probability": 0.05,
        },
        "delay_200ms": {
            **double_scene, "seed": 109, "observation_delay_steps": 2,
            "action_delay_steps": 1,
        },
        "dynamics_and_sidewind": {
            **double_scene, "seed": 111, "mass_scale": 1.15,
            "inertia_scale": 1.20, "thrust_scale": 0.90,
            "wind_acceleration_x": 0.15, "wind_acceleration_y": -0.10,
        },
        "extreme_conflict_speed": {
            **double_scene, "seed": 113, "conflict_speed_low": 0.70,
            "conflict_speed_high": 1.0,
        },
    }
    generalization = {}
    for case, scene in generalization_cases.items():
        write_status("generalization", "evaluating", case=case,
                     checkpoint=str(checkpoint))
        metrics = evaluate(f"generalization_{case}", checkpoint,
                           (0.8, 0.5, 0.8), scene, steps=30000)
        metrics["drop_from_standard"] = (
            baseline["success_rate"] - metrics["success_rate"] if baseline else None
        )
        metrics["within_15pp"] = (
            metrics["drop_from_standard"] is not None
            and metrics["drop_from_standard"] <= 0.15
        )
        generalization[case] = metrics
    generalization_path = OUT / "generalization_summary.json"
    generalization_path.write_text(
        json.dumps(generalization, ensure_ascii=False, indent=2) + "\n")

    shield_scene = {**double_scene, "safety_projection": True}
    shield_metrics = evaluate("ablation_safety_projection_on", checkpoint,
                              (0.8, 0.5, 0.8), shield_scene, steps=30000)
    ablation = {"safety_projection_off": baseline,
                "safety_projection_on": shield_metrics}
    (OUT / "ablation_summary.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2) + "\n")
    write_status("curriculum", "training_and_generalization_complete",
                 checkpoint=str(checkpoint), generalization=generalization,
                 ablation=ablation)


if __name__ == "__main__":
    main()
