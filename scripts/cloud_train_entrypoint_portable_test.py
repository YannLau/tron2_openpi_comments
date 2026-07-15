import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).with_name("cloud_train_entrypoint_portable.sh")


def _run_dry_run(tmp_path: pathlib.Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    data_dir = tmp_path / "data"
    (data_dir / "input" / "data").mkdir(parents=True)
    (data_dir / "input" / "meta").mkdir()
    weight_path = tmp_path / "checkpoint" / "params"
    weight_path.mkdir(parents=True)
    output_dir = tmp_path / "output"

    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-dir",
            str(data_dir),
            "--repo-id",
            "input",
            "--weight",
            str(weight_path),
            "--output-dir",
            str(output_dir),
            "--exp",
            "public_test",
            "--dry-run",
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_dry_run_generates_platform_task_config(tmp_path: pathlib.Path):
    result = _run_dry_run(tmp_path, "--prompt-from-task", "--rtc-delay", "10", "--action-horizon", "30")

    assert result.returncode == 0, result.stderr
    assert '"repo_id": "input"' in result.stdout
    assert '"prompt_from_task": true' in result.stdout
    assert '"rtc_training_simulated_delay": 10' in result.stdout
    assert "scripts/compute_norm_stats.py --task-config" in result.stdout
    assert "scripts/train_tron2_task.py --task-config" in result.stdout


def test_resume_does_not_overwrite_checkpoint(tmp_path: pathlib.Path):
    result = _run_dry_run(tmp_path, "--resume")

    assert result.returncode == 0, result.stderr
    train_command = next(line for line in result.stdout.splitlines() if "scripts/train_tron2_task.py" in line)
    assert "--resume" in train_command
    assert "--overwrite" not in train_command


def test_task_config_rejects_unsafe_experiment_name(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text("name: test\nrepo_id: input\nprompt: test\n")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--task-config",
            str(task_path),
            "--exp",
            "../outside",
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported characters" in result.stderr
