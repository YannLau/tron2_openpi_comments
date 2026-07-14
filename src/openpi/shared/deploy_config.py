from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_config_path(path: str | Path) -> Path:
    profile_path = Path(path).expanduser()
    if profile_path.is_absolute():
        return profile_path

    cwd_path = Path.cwd() / profile_path
    if cwd_path.exists():
        return cwd_path

    return REPO_ROOT / profile_path


def load_deploy_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    resolved_path = resolve_config_path(path)
    with resolved_path.open() as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Deploy config must be a mapping: {resolved_path}")
    return data


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Deploy config section must be a mapping: {name}")
    return value


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
