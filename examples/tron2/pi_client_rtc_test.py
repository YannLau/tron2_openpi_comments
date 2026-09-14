from __future__ import annotations

import deploy_config
import numpy as np
import pi_client_rtc
import pytest


def test_action_postprocess_uses_brainco2_end_effector_slices():
    resolved = deploy_config.resolve_deploy_config({"end_effector": {"type": "brainco2", "command_time": 2.0}})
    processor = pi_client_rtc._resolve_action_postprocess(  # noqa: SLF001
        {
            "rtc_action_postprocess": {
                "enabled": True,
                "ema_alpha": 0.5,
                "ema_scope": "end_effector",
            }
        },
        resolved,
    )
    actions = np.ones((1, resolved.layout.dim), dtype=np.float32)
    previous = np.zeros_like(actions)

    processed, _ = processor.apply(actions, previous, merge_delay=0)

    np.testing.assert_array_equal(processed[:, deploy_config.arm_indices(resolved.layout)], 1.0)
    np.testing.assert_array_equal(
        processed[:, deploy_config.end_effector_indices(resolved.layout)],
        0.5,
    )


def test_servop_rejects_arm_postprocessing():
    resolved = deploy_config.resolve_deploy_config({"arm": {"mode": "servop"}})

    with pytest.raises(ValueError, match="ServoP.*cannot linearly blend"):
        pi_client_rtc._resolve_action_postprocess(  # noqa: SLF001
            {
                "rtc_action_postprocess": {
                    "enabled": True,
                    "boundary_blend_frames": 2,
                    "boundary_blend_scope": "arm",
                }
            },
            resolved,
        )


def test_arm_jump_uses_layout_aware_arm_indices():
    resolved = deploy_config.resolve_deploy_config({"end_effector": {"type": "brainco2", "command_time": 2.0}})
    previous = np.zeros(resolved.layout.dim, dtype=np.float32)
    current = previous.copy()
    current[8] = 0.9  # Left-hand value, not an arm joint.
    current[13] = 0.6  # First right-arm joint in the BrainCo2 layout.

    jump = pi_client_rtc._max_arm_jump(  # noqa: SLF001
        current,
        previous,
        resolved.layout,
        arm_mode="servoj",
    )

    assert jump == (7, pytest.approx(0.6))


def test_arm_jump_is_skipped_for_servop():
    resolved = deploy_config.resolve_deploy_config({"arm": {"mode": "servop"}})

    assert (
        pi_client_rtc._max_arm_jump(  # noqa: SLF001
            np.ones(resolved.layout.dim),
            np.zeros(resolved.layout.dim),
            resolved.layout,
            arm_mode="servop",
        )
        is None
    )


def test_record_headers_follow_modular_layout(tmp_path):
    resolved = deploy_config.resolve_deploy_config({"end_effector": {"type": "brainco2", "command_time": 2.0}})
    action_path = tmp_path / "actions.csv"
    state_path = tmp_path / "states.csv"
    profile = {
        "client": {
            "rtc_action_output_path": str(action_path),
            "rtc_state_output_path": str(state_path),
        }
    }

    pi_client_rtc._save_records(  # noqa: SLF001
        profile,
        resolved.layout,
        [{"state": np.zeros(resolved.layout.dim)}],
        [{"action": np.zeros(resolved.layout.dim)}],
    )

    assert state_path.read_text().splitlines()[0].endswith(",right_hand_5")
    assert action_path.read_text().splitlines()[0].endswith(",right_hand_5")
