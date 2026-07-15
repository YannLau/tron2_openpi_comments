import os
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


def test_hf_lerobot_home_is_used_as_default_data_dir(tmp_path: pathlib.Path):
    data_dir = tmp_path / "datasets"
    (data_dir / "input" / "data").mkdir(parents=True)
    (data_dir / "input" / "meta").mkdir()
    weight_path = tmp_path / "checkpoint" / "params"
    weight_path.mkdir(parents=True)

    env = os.environ.copy()
    env["HF_LEROBOT_HOME"] = str(data_dir)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo-id",
            "input",
            "--weight",
            str(weight_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--exp",
            "public_test",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"Dataset root:     {data_dir}" in result.stdout


def test_platform_mount_environment_paths_are_used(tmp_path: pathlib.Path):
    platform_root = tmp_path / "data"
    (platform_root / "input" / "data").mkdir(parents=True)
    (platform_root / "input" / "meta").mkdir()
    weight_path = platform_root / "checkpoint" / "params"
    weight_path.mkdir(parents=True)

    env = os.environ.copy()
    env["DATA_DIR"] = str(platform_root)
    env["WEIGHT_PATH"] = str(weight_path)
    env["OUTPUT_DIR"] = str(tmp_path / "output")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo-id",
            "input",
            "--exp",
            "cloud_test",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"Dataset root:     {platform_root}" in result.stdout
    assert f"Model params:     {weight_path}" in result.stdout
    assert '"repo_id": "input"' in result.stdout


def test_custom_data_and_weight_paths_generate_task_config(tmp_path: pathlib.Path):
    data_dir = tmp_path / "datasets"
    (data_dir / "my_dataset" / "data").mkdir(parents=True)
    (data_dir / "my_dataset" / "meta").mkdir()
    weight_path = tmp_path / "weights" / "params"
    weight_path.mkdir(parents=True)
    output_dir = tmp_path / "outputs"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-dir",
            str(data_dir),
            "--repo-id",
            "my_dataset",
            "--weight",
            str(weight_path),
            "--output-dir",
            str(output_dir),
            "--exp",
            "local_test",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Dataset root:     {data_dir}" in result.stdout
    assert f"Model params:     {weight_path}" in result.stdout
    assert f"Output root:      {output_dir}" in result.stdout
    assert '"repo_id": "my_dataset"' in result.stdout
    assert f'"weight_loader": "{weight_path}"' in result.stdout


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
