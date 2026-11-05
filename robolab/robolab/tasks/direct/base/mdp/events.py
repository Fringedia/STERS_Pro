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
    stiffness_distribution_params: tuple[float, float] | None = None,
    theta30_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """域随机化弹簧肌腱参数 k 与 theta30，写入 env.randomized_k / env.randomized_theta30。"""
    if not getattr(env, "use_extra_torque", False):
        return
    if stiffness_distribution_params is None and theta30_distribution_params is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    k_default = env.cfg.robot.spring_tendon_k_default
    theta30_default = env.cfg.robot.spring_tendon_theta30_default

    if stiffness_distribution_params is not None:
        k_values = torch.full((env.num_envs, 1), k_default, device=env.device)
        k_values = _randomize_prop_by_op(
            k_values,
            stiffness_distribution_params,
            env_ids,
            slice(None),
            operation=operation,
            distribution=distribution,
        )
        env.randomized_k[env_ids] = k_values[env_ids].squeeze(-1)

    if theta30_distribution_params is not None:
        theta30_values = torch.full((env.num_envs, 1), theta30_default, device=env.device)
        theta30_values = _randomize_prop_by_op(
            theta30_values,
            theta30_distribution_params,
            env_ids,
            slice(None),
            operation=operation,
            distribution=distribution,
        )
        env.randomized_theta30[env_ids] = theta30_values[env_ids].squeeze(-1)
