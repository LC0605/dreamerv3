import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continue_dreamer_curriculum.py"
SPEC = importlib.util.spec_from_file_location("continue_dreamer_curriculum", SCRIPT)
CURRICULUM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURRICULUM)


def _option(options, key):
    return options[options.index(key) + 1]


def test_late_recovery_retry_uses_anti_forgetting_recipe():
    options = CURRICULUM.optimization_options(
        "stage1a_fixed_v5_recovery_presence_0p65_retry2")

    assert _option(options, "--run.train_ratio") == "4"
    assert _option(options, "--agent.opt.lr") == "1e-5"
    assert _option(options, "--agent.loss_scales.policy") == "0.015"
    assert _option(options, "--agent.imag_length") == "6"
    assert _option(options, "--agent.policy.minstd") == "0.03"
    assert _option(options, "--agent.imag_loss.actent") == "1e-4"


def test_first_recovery_attempt_retains_adaptation_recipe():
    options = CURRICULUM.optimization_options(
        "stage1a_fixed_v5_recovery_presence_0p65")

    assert _option(options, "--run.train_ratio") == "8"
    assert _option(options, "--agent.loss_scales.policy") == "0.03"


def test_milestones_include_early_transfer_peak():
    assert CURRICULUM.milestone_thresholds(25000) == [
        4000, 8000, 12000, 16000, 20000]
    assert CURRICULUM.milestone_thresholds(30000) == [
        4000, 9600, 14400, 19200, 24000]


def test_snapshot_steps_sort_numerically():
    snapshots = [Path("step_13000_a"), Path("step_4410_b"), Path("step_8630_c")]
    assert sorted(snapshots, key=CURRICULUM.snapshot_step) == [
        Path("step_4410_b"), Path("step_8630_c"), Path("step_13000_a")]


def test_quadrotor_config_uses_precision_flight_exploration():
    config = (Path(__file__).resolve().parents[1] /
              "dreamerv3" / "dreamerv3" / "configs.yaml").read_text()
    quadrotor = config.split("\nquadrotor:\n", 1)[1]
    assert "agent.policy.minstd: 0.03" in quadrotor
    assert "agent.imag_loss.actent: 1e-4" in quadrotor
