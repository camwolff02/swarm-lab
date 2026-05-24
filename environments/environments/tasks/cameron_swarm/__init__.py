# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cameron swarm manager-based MARL task package.

Importing this module registers the Gymnasium task ids used by Isaac Lab and
SKRL multi-agent runners.
"""

from __future__ import annotations

import gymnasium as gym


gym.register(
    id="Isaac-Cameron-Drone-Waypoint-IPPO-v0",
    entry_point="environments.envs:ManagerBasedMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.cameron_swarm.drone_waypoint_marl_env_cfg:DroneWaypointIppoEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.cameron_swarm:config/skrl_ippo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "environments.tasks.cameron_swarm:config/skrl_ippo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Cameron-Drone-Waypoint-MAPPO-v0",
    entry_point="environments.envs:ManagerBasedMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.cameron_swarm.drone_waypoint_marl_env_cfg:DroneWaypointMappoEnvCfg",
        "skrl_cfg_entry_point": "environments.tasks.cameron_swarm:config/skrl_mappo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "environments.tasks.cameron_swarm:config/skrl_mappo_cfg.yaml",
    },
)
