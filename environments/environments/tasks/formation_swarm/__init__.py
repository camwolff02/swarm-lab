"""Registration for the Xie et al. formation swarm task."""

from __future__ import annotations

import gymnasium as gym

from .agents.runner import install_formation_swarm_runner_patch

install_formation_swarm_runner_patch()

# Legacy DirectMARLEnv registration
gym.register(
    id="Isaac-Formation-Swarm-Crazyflie-v3",
    entry_point=f"{__name__}.env:FormationSwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FormationSwarmEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{__name__}.agents:skrl_mappo_cfg.yaml",
    },
)

# --- ManagerBasedMarlEnv registrations ---

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-Stage1-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmStage1EnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-Stage1-Video-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmStage1VideoEnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-Stage1-Legacy-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmStage1LegacyEnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_legacy_shared_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-Stage1-Legacy-Video-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmStage1LegacyVideoEnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_legacy_shared_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-Stage2-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmStage2EnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Formation-Swarm-MAPPO-Stage3-v0",
    entry_point="environments.tasks.formation_swarm.formation_marl_env:FormationSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "environments.tasks.formation_swarm.formation_marl_env_cfg:FormationSwarmStage3EnvCfg",
        "skrl_mappo_cfg_entry_point": "environments.tasks.formation_swarm.agents:skrl_mappo_cfg.yaml",
    },
)
