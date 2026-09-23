from __future__ import annotations

from pathlib import Path

import deploy_config
import numpy as np
import pytest

from openpi.training import config as training_config

DEPLOY_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "deploy"


@pytest.mark.parametrize(
    ("sections", "dimension", "components"),
    [
        (
            {"mobile_base": {"enabled": True}},
            19,
            ("left_arm", "left_gripper", "right_arm", "right_gripper", "chassis"),
        ),
        (
            {"end_effector": {"type": "brainco2", "command_time": 2.0}},
            26,
            ("left_arm", "left_hand", "right_arm", "right_hand"),
        ),
        (
            {
                "arm": {"mode": "servop"},
                "end_effector": {"type": "brainco2", "command_time": [2.0] * 6},
                "head": {"enabled": True},
                "mobile_base": {"enabled": True, "lifter_state_source": "position_mm"},
            },
            31,
            ("left_arm", "left_hand", "right_arm", "right_hand", "head", "chassis"),
        ),
    ],
)
def test_resolve_deploy_config_derives_modular_layout(sections, dimension, components):
    resolved = deploy_config.resolve_deploy_config(
        {
            "client": {"state_dim": dimension},
            "robot": {"ip": "ROBOT_IP", "init_joints": None, "init_head": None},
            "camera": {"serial_to_name": {}},
            **sections,
        }
    )

    assert resolved.layout.dim == dimension
    assert resolved.layout.components == components
    assert resolved.env_config.state_dim == dimension


def test_brainco2_requires_explicit_command_time():
    with pytest.raises(ValueError, match="command_time"):
        deploy_config.resolve_deploy_config({"end_effector": {"type": "brainco2"}})


def test_public_package_rejects_lowlevel_backend():
    with pytest.raises(ValueError, match="websocket"):
        deploy_config.resolve_deploy_config({"client": {"control_backend": "lowlevel"}})


def test_brainco2_with_bridge_images_requires_direct_robot_state():
    with pytest.raises(ValueError, match="bridge_state_source='legacy'"):
        deploy_config.resolve_deploy_config(
            {
                "client": {"observation_source": "bridge"},
                "bridge": {"state_source": "bridge"},
                "end_effector": {"type": "brainco2", "command_time": 2.0},
            }
        )


def test_lifter_control_requires_position_mm_state():
    with pytest.raises(ValueError, match="lifter_state_source='position_mm'"):
        deploy_config.resolve_deploy_config(
            {
                "mobile_base": {
                    "enabled": True,
                    "lifter_state_source": "raw_q",
                    "lifter_control_enabled": True,
                }
            }
        )


def test_component_indices_follow_brainco2_layout():
    resolved = deploy_config.resolve_deploy_config(
        {
            "client": {"state_dim": 26},
            "end_effector": {"type": "brainco2", "command_time": 2.0},
        }
    )

    np.testing.assert_array_equal(deploy_config.arm_indices(resolved.layout), np.r_[0:7, 13:20])
    np.testing.assert_array_equal(
        deploy_config.end_effector_indices(resolved.layout),
        np.r_[7:13, 20:26],
    )


def test_server_dimension_mismatch_is_rejected():
    resolved = deploy_config.resolve_deploy_config({"mobile_base": {"enabled": True}})

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        deploy_config.validate_server_physical_dimensions(
            {"state_dim": 16, "action_dim": 16},
            resolved.layout,
        )


@pytest.mark.parametrize(
    ("profile_name", "dimension", "components", "arm", "end_effector", "head", "mobile_base"),
    [
        (
            "brainco_client.yaml",
            26,
            ("left_arm", "left_hand", "right_arm", "right_hand"),
            "servop",
            "brainco2",
            False,
            False,
        ),
        (
            "chassis_client.yaml",
            21,
            ("left_arm", "left_gripper", "right_arm", "right_gripper", "head", "chassis"),
            "servoj",
            "gripper",
            True,
            True,
        ),
    ],
)
def test_public_modular_client_profiles_match_physical_layout(
    profile_name,
    dimension,
    components,
    arm,
    end_effector,
    head,
    mobile_base,
):
    profile = deploy_config.load_deploy_profile(DEPLOY_CONFIG_DIR / profile_name)
    resolved = deploy_config.resolve_deploy_config(profile)

    assert resolved.layout.dim == dimension
    assert resolved.layout.components == components
    assert resolved.modules is not None
    assert resolved.modules.arm == arm
    assert resolved.modules.end_effector == end_effector
    assert resolved.modules.head is head
    assert resolved.modules.mobile_base is mobile_base

    if end_effector == "brainco2":
        assert resolved.brainco2_config is not None
        assert resolved.brainco2_config.command_time == (1.0,) * 6
        assert resolved.env_config.bridge_state_source == "legacy"
    if mobile_base:
        assert resolved.env_config.lifter_state_source == "position_mm"
        assert resolved.env_config.lifter_control_enabled is True


@pytest.mark.parametrize(
    ("profile_name", "dimension"),
    [("brainco_server.yaml", 26), ("chassis_server.yaml", 21)],
)
def test_public_modular_server_profiles_use_registered_generic_config(profile_name, dimension):
    profile = deploy_config.load_deploy_profile(DEPLOY_CONFIG_DIR / profile_name)
    policy = deploy_config.section(profile, "policy")

    assert policy["config"] == "pi05_tron2_example"
    assert training_config.get_config(policy["config"]).name == policy["config"]
    assert policy["state_dim"] == dimension
    assert policy["action_horizon"] == 30
    assert policy["use_delta_joint_actions"] is False
