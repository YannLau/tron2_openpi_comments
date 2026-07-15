import pathlib

from openpi.training import tron2_task_config


def _write_task(path: pathlib.Path, *, prompt_from_task: bool = False) -> None:
    path.write_text(
        "\n".join(
            [
                "name: pi05_tron2_test",
                "repo_id: test_dataset",
                "prompt: fallback prompt",
                "weight_loader: /tmp/weights/params",
                f"prompt_from_task: {str(prompt_from_task).lower()}",
            ]
        )
    )


def test_load_task_supports_prompt_from_task(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    _write_task(task_path, prompt_from_task=True)

    task = tron2_task_config.load_task(task_path)

    assert task.prompt_from_task is True


def test_create_train_config_propagates_prompt_from_task(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    _write_task(task_path, prompt_from_task=True)

    config = tron2_task_config.create_train_config(task_path)

    assert config.data.base_config is not None
    assert config.data.base_config.prompt_from_task is True
