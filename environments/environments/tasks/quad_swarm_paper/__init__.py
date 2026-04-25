"""Registration for the quad swarm paper task."""

from __future__ import annotations

import gymnasium as gym

from .agents.runner import install_quad_swarm_runner_patch


install_quad_swarm_runner_patch()

gym.register(
    id="Isaac-Quad-Swarm-Paper-Crazyflie-v0",
    entry_point=f"{__name__}.env:QuadSwarmPaperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:QuadSwarmPaperEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{__name__}.agents:skrl_ippo_cfg.yaml",
    },
)
