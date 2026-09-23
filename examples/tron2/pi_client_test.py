from __future__ import annotations

import deploy_config
import numpy as np
import pi_client


def test_arm_values_follow_brainco2_layout():
    layout = deploy_config.resolve_deploy_config({"end_effector": {"type": "brainco2", "command_time": 2.0}}).layout
    values = np.arange(layout.dim)

    np.testing.assert_array_equal(pi_client._arm_values(values, layout), np.r_[0:7, 13:20])  # noqa: SLF001
