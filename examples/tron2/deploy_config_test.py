from __future__ import annotations

import deploy_config
import numpy as np
import pytest


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
