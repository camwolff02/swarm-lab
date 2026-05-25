# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paper swarm manager-based MARL task package.

Importing this module registers the Gymnasium task ids used by Isaac Lab and
SKRL multi-agent runners.
"""

from __future__ import annotations

import gymnasium as gym

from .agents.runner import _install_runner_patch

_install_runner_patch()

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-IPPO-v0",
    entry_point="environments.envs:ManagerBasedMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmIppoEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-v0",
    entry_point="environments.envs:ManagerBasedMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-Eval-v0",
    entry_point="environments.envs:ManagerBasedMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmEvalEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-Eval-v0",
    entry_point="environments.envs:ManagerBasedMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoEvalEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
    },
)
