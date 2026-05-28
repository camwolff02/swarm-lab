# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paper swarm manager-based MARL task package.

Importing this module registers the Gymnasium task ids used by Isaac Lab and
SKRL multi-agent runners.

3-stage curriculum:
    1. Single-agent pretraining with passive hovering drones.
    2. MARL interaction learning (variable agent count, sparse obstacles).
    3. Target fine-tuning (target-N, dense obstacles, strong DR).
"""

from __future__ import annotations

import gymnasium as gym

from .agents.runner import _install_runner_patch

_install_runner_patch()

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-IPPO-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmIppoEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoStage1EnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage1_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage1_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-Eval-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoStage1EvalCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage1_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage1_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-Stage2-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoStage2EnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage2_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage2_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-Stage3-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoStage3EnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage3_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_stage3_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-Eval-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmEvalEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_ippo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Paper-Swarm-Waypoint-MAPPO-Eval-v0",
    entry_point="environments.tasks.paper_swarm.paper_swarm_env:PaperSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.paper_swarm.paper_swarm_env_cfg:PaperSwarmMappoEvalEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.paper_swarm:config/skrl_mappo_cfg.yaml",
    },
)
