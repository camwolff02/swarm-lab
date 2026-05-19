"""Registration for the Xie et al. formation swarm task."""

from __future__ import annotations

import gymnasium as gym

from .agents.runner import install_formation_swarm_runner_patch

install_formation_swarm_runner_patch()

gym.register(
    id="Isaac-Formation-Swarm-Crazyflie-v3",
    entry_point=f"{__name__}.env:FormationSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FormationSwarmEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{__name__}.agents:skrl_mappo_cfg.yaml",
    },
)

