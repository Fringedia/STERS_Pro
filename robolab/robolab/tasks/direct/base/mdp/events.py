# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from robolab.tasks.direct.base.base_env import BaseEnv


def randomize_tendon_parameters_scale(
    env: "BaseEnv",
    env_ids: torch.Tensor | None,
    k_distribution_params: tuple[float, float] | None = None,
    theta_a0_distribution_params: tuple[float, float] | None = None,
    theta_k0_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """域随机化 k、theta_a0 和 theta_k0，写入对应的逐环境 buffer。"""
    if not getattr(env, "use_extra_torque", False):
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    for name, params in (
        ("k", k_distribution_params),
        ("theta_a0", theta_a0_distribution_params),
        ("theta_k0", theta_k0_distribution_params),
    ):
        if params is None:
            continue
        default = getattr(env.cfg.robot, f"spring_tendon_{name}_default")
        values = torch.full((env.num_envs, 1), default, device=env.device)
        values = _randomize_prop_by_op(
            values,
            params,
            env_ids,
            slice(None),
            operation=operation,
            distribution=distribution,
        )
        getattr(env, f"randomized_{name}")[env_ids] = values[env_ids].squeeze(-1)
