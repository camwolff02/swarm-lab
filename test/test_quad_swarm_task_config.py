from __future__ import annotations

import gymnasium as gym

import environments.tasks.quad_swarm_paper  # noqa: F401
from environments.tasks.quad_swarm_paper import paper_spec
from environments.tasks.quad_swarm_paper.env_cfg import QuadSwarmPaperEnvCfg


def test_quad_swarm_task_registers_gym_id() -> None:
    spec = gym.spec("Isaac-Quad-Swarm-Paper-Crazyflie-v0")

    assert spec.id == "Isaac-Quad-Swarm-Paper-Crazyflie-v0"
    assert "skrl_ippo_cfg_entry_point" in spec.kwargs


def test_quad_swarm_env_cfg_spaces_match_paper_observation_order() -> None:
    cfg = QuadSwarmPaperEnvCfg()

    assert len(cfg.possible_agents) == paper_spec.NUM_DRONES
    assert cfg.state_space == -1
    assert cfg.observation_spaces["drone_0"].shape == (paper_spec.OBS_SIZE,)
    assert cfg.action_spaces["drone_0"].shape == (paper_spec.ACTION_SIZE,)
