from __future__ import annotations

import gymnasium as gym
import torch
import yaml

import environments.tasks.quad_swarm_paper  # noqa: F401
from environments.tasks.quad_swarm_paper import paper_spec
from environments.tasks.quad_swarm_paper.env_cfg import QuadSwarmPaperEnvCfg
from environments.tasks.quad_swarm_paper.obstacle_room import (
    sample_obstacle_aware_start_goal_pairs,
    sample_obstacle_field,
)


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


def test_quad_swarm_cfg_defaults_match_reference_episode_semantics() -> None:
    cfg = QuadSwarmPaperEnvCfg()

    assert cfg.robot_collision_radius == 0.10
    assert cfg.robot_proximity_radius == 0.20
    assert cfg.obstacle_collision_robot_radius == 0.05
    assert cfg.terminate_on_collision is False
    assert cfg.terminate_on_crash is False
    assert cfg.terminate_on_success is False


def test_obstacle_aware_sampling_uses_free_grid_cells() -> None:
    torch.manual_seed(1)
    obstacle_positions, obstacle_mask = sample_obstacle_field(4, device="cpu")

    starts, goals, orientations = sample_obstacle_aware_start_goal_pairs(
        obstacle_mask,
        obstacle_positions,
        paper_spec.NUM_DRONES,
        device="cpu",
    )

    assert starts.shape == (4, paper_spec.NUM_DRONES, 3)
    assert goals.shape == starts.shape
    assert orientations.shape == (4, paper_spec.NUM_DRONES, 4)

    for env_id in range(obstacle_mask.shape[0]):
        occupied_xy = obstacle_positions[env_id, :, :2][obstacle_mask[env_id]]
        sampled_xy = torch.cat((starts[env_id, :, :2], goals[env_id, :, :2]), dim=0)
        distances = torch.linalg.norm(sampled_xy[:, None, :] - occupied_xy[None, :, :], dim=-1)
        assert torch.all(distances.min(dim=-1).values > 1.0e-6)


def test_skrl_config_uses_effective_reference_gae_key() -> None:
    with open(
        "environments/environments/tasks/quad_swarm_paper/agents/skrl_ippo_cfg.yaml",
        encoding="utf-8",
    ) as file:
        cfg = yaml.safe_load(file)

    assert "gae_lambda" not in cfg["agent"]
    assert cfg["agent"]["lambda"] == 1.0
    assert cfg["agent"]["grad_norm_clip"] == 5.0
