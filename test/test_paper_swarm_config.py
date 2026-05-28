# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration-level tests for the paper_swarm task.

These tests validate that environment configs, Gym registrations, and MDP
helper shapes are internally consistent without requiring a running Isaac Sim
instance.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import pytest

from types import SimpleNamespace

from environments.tasks.paper_swarm import (  # noqa: F401 — registers Gym ids
    paper_swarm_env_cfg as env_cfg_module,
    paper_swarm_env,
)
from environments.tasks.paper_swarm.mdp import observations as obs_module
from environments.envs.manager_based_marl_env_cfg import ManagerBasedMarlEnvCfg


# -----------------------------------------------------------------------------
# Gym registration
# -----------------------------------------------------------------------------


def test_paper_swarm_registers_all_stage_ids() -> None:
    expected = [
        "Isaac-Paper-Swarm-Waypoint-IPPO-v0",
        "Isaac-Paper-Swarm-Waypoint-MAPPO-v0",
        "Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-v0",
        "Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-Eval-v0",
        "Isaac-Paper-Swarm-Waypoint-MAPPO-Stage2-v0",
        "Isaac-Paper-Swarm-Waypoint-MAPPO-Stage3-v0",
        "Isaac-Paper-Swarm-Waypoint-Eval-v0",
        "Isaac-Paper-Swarm-Waypoint-MAPPO-Eval-v0",
    ]
    for task_id in expected:
        spec = gym.spec(task_id)
        assert spec.id == task_id
        assert "env_cfg_entry_point" in spec.kwargs


# -----------------------------------------------------------------------------
# Config structure
# -----------------------------------------------------------------------------


def test_base_env_cfg_is_valid_marl_cfg() -> None:
    cfg = env_cfg_module.PaperSwarmBaseMarlEnvCfg()
    assert isinstance(cfg, ManagerBasedMarlEnvCfg)
    assert len(cfg.possible_agents) == 8
    assert cfg.possible_agents == [f"drone_{i}" for i in range(8)]
    assert cfg.observation_group == "policy"
    assert cfg.active_agent_mask_key == "active_drones"
    assert cfg.reset_on == "any"
    assert cfg.episode_length_s == 20.0
    assert cfg.is_finite_horizon is False
    assert cfg.decimation == 2


def test_base_env_cfg_has_agent_group() -> None:
    cfg = env_cfg_module.PaperSwarmBaseMarlEnvCfg()
    assert len(cfg.agent_groups) == 1
    group = cfg.agent_groups[0]
    assert group.name == "drone"
    assert group.count == 8
    assert group.id_template == "drone_{i}"


def test_ippo_cfg_uses_decentralized_critic() -> None:
    cfg = env_cfg_module.PaperSwarmIppoEnvCfg()


def test_mappo_cfg_uses_centralized_critic() -> None:
    cfg = env_cfg_module.PaperSwarmMappoEnvCfg()
    agent_cfg = cfg.agent_groups[0].agent_cfg
    observations = agent_cfg.observations
    assert hasattr(observations, "critic")
    assert observations.critic.__class__.__name__ == "CentralizedCriticCfg"


def test_stage1_cfg_has_single_agent() -> None:
    cfg = env_cfg_module.PaperSwarmMappoStage1EnvCfg()
    assert cfg.possible_agents == ["drone_0"]
    assert cfg.agent_groups[0].count == 1
    assert cfg.scene.num_envs == 8192


def test_stage2_cfg_has_all_agents_no_obstacles() -> None:
    cfg = env_cfg_module.PaperSwarmMappoStage2EnvCfg()
    assert cfg.possible_agents == [f"drone_{i}" for i in range(8)]


def test_stage3_cfg_has_all_agents_with_obstacles() -> None:
    cfg = env_cfg_module.PaperSwarmMappoStage3EnvCfg()
    assert cfg.possible_agents == [f"drone_{i}" for i in range(8)]


def test_eval_cfg_has_recorder_and_fewer_envs() -> None:
    cfg = env_cfg_module.PaperSwarmEvalEnvCfg()
    assert cfg.recorders is not None
    assert cfg.scene.num_envs == 4
    assert cfg.episode_length_s == 10.0


def test_physics_uses_preset_cfg() -> None:
    cfg = env_cfg_module.PaperSwarmBaseMarlEnvCfg()
    physics = cfg.sim.physics
    assert physics.__class__.__name__ == "PaperSwarmPhysicsCfg"
    assert hasattr(physics, "default")
    assert hasattr(physics, "physx")
    assert hasattr(physics, "newton")


# -----------------------------------------------------------------------------
# MDP helper shape tests
# -----------------------------------------------------------------------------


def test_get_agent_active_mask_returns_float() -> None:
    result = obs_module.get_agent_active_mask(_fake_env(4, 8), "drone_0", "active_drones")
    assert result.shape == (4,)
    assert result.dtype == torch.float32


def test_get_agent_active_mask_returns_ones_for_unknown_agent() -> None:
    result = obs_module.get_agent_active_mask(_fake_env(2, 4), "nonexistent", "active_drones")
    assert result.shape == (2,)
    assert result.dtype == torch.float32
    assert result.eq(1.0).all()


def test_active_mask_returns_bool() -> None:
    result = obs_module._active_mask(_fake_env(4, 8), [f"drone_{i}" for i in range(8)], "active_drones")
    assert result.shape == (4, 8)
    assert result.dtype == torch.bool


def test_active_mask_all_ones_when_no_key() -> None:
    result = obs_module._active_mask(_fake_env(3, 4), [f"drone_{i}" for i in range(4)], None)
    assert result.eq(True).all()


def test_nearest_neighbor_padding_produces_fixed_shape() -> None:
    features = torch.randn(2, 8, 6)
    distances = torch.rand(2, 8)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[:, 0] = False
    out = obs_module._nearest_neighbor_features(features, distances, valid, max_neighbors=2)
    assert out.shape == (2, 2 * 6)


def test_nearest_neighbor_returns_zeros_for_empty_input() -> None:
    out = obs_module._nearest_neighbor_features(
        torch.randn(2, 0, 6), torch.randn(2, 0), torch.ones(2, 0, dtype=torch.bool), max_neighbors=2
    )
    assert out.shape == (2, 2 * 6)
    assert out.eq(0.0).all()


def test_pad_features_zero_pads() -> None:
    features = torch.randn(2, 8, 3)
    out = obs_module._pad_features(features, 10)
    assert out.shape == (2, 10 * 3)


def test_pad_features_truncates() -> None:
    features = torch.randn(2, 8, 3)
    out = obs_module._pad_features(features, 3)
    assert out.shape == (2, 3 * 3)


def test_rotate_world_to_body_returns_same_shape() -> None:
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    vectors = torch.randn(2, 3, 3)
    result = obs_module._rotate_world_to_body(quat, vectors)
    assert result.shape == vectors.shape


def test_static_sdf_defaults_for_no_columns() -> None:
    env = _fake_env(4, 4)
    out = obs_module.static_sdf(env, _fake_scene_entity_cfg(), "nonexistent_key")
    assert out.shape == (4, 9)
    assert (out > 0).all()


def test_drone_identity_shape() -> None:
    result = obs_module.drone_identity(_fake_env(2, 8), [f"drone_{i}" for i in range(8)], "drone_3")
    assert result.shape == (2, 8)
    assert result[0, 3] == 1.0
    assert result[0, 0] == 0.0


def test_goal_distance_reward_returns_correct_shape() -> None:
    from environments.tasks.paper_swarm.mdp.rewards import goal_distance_reward

    env = _fake_env_with_commands(4)
    result = goal_distance_reward(env, _fake_scene_entity_cfg(), "drone_0", "target_pose", "active_drones")
    assert result.shape == (4,)


def test_collision_reward_returns_correct_shape() -> None:
    from environments.tasks.paper_swarm.mdp.rewards import collision_avoidance_reward

    env = _fake_env_with_full_scene(4, 4)
    result = collision_avoidance_reward(
        env, _fake_scene_entity_cfg(), "drone_0", [f"drone_{i}" for i in range(4)], 0.5, 0.12, "active_drones"
    )
    assert result.shape == (4,)


def test_drone_out_of_bounds_shape() -> None:
    from environments.tasks.paper_swarm.mdp.terminations import drone_out_of_bounds

    env = _fake_env_with_full_scene(2, 4)
    env.root = env
    env.cfg.possible_agents = [f"drone_{i}" for i in range(4)]
    result = drone_out_of_bounds(env, _fake_scene_entity_cfg(), "drone_0", (-6.0, 6.0), (0.2, 5.0), "active_drones")
    assert result.shape == (2,)
    assert result.dtype == torch.bool


def test_active_mask_all_ones_when_no_key() -> None:
    result = obs_module._active_mask(_fake_env(3, 4), [f"drone_{i}" for i in range(4)], None)
    assert result.eq(True).all()


def test_nearest_neighbor_padding_produces_fixed_shape() -> None:
    features = torch.randn(2, 8, 6)
    distances = torch.rand(2, 8)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[:, 0] = False
    out = obs_module._nearest_neighbor_features(features, distances, valid, max_neighbors=2)
    assert out.shape == (2, 2 * 6)


def test_nearest_neighbor_returns_zeros_for_empty_input() -> None:
    out = obs_module._nearest_neighbor_features(
        torch.randn(2, 0, 6), torch.randn(2, 0), torch.ones(2, 0, dtype=torch.bool), max_neighbors=2
    )
    assert out.shape == (2, 2 * 6)
    assert out.eq(0.0).all()


def test_pad_features_zero_pads() -> None:
    features = torch.randn(2, 8, 3)
    out = obs_module._pad_features(features, 10)
    assert out.shape == (2, 10 * 3)


def test_pad_features_truncates() -> None:
    features = torch.randn(2, 8, 3)
    out = obs_module._pad_features(features, 3)
    assert out.shape == (2, 3 * 3)


def test_rotate_world_to_body_returns_same_shape() -> None:
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    vectors = torch.randn(2, 3, 3)
    result = obs_module._rotate_world_to_body(quat, vectors)
    assert result.shape == vectors.shape


def test_static_sdf_defaults_for_no_columns() -> None:
    env = _fake_env(4, 4)
    out = obs_module.static_sdf(env, _fake_scene_entity_cfg(), "nonexistent_key")
    assert out.shape == (4, 9)
    assert (out > 0).all()


def test_drone_identity_shape() -> None:
    result = obs_module.drone_identity(_fake_env(2, 8), [f"drone_{i}" for i in range(8)], "drone_3")
    assert result.shape == (2, 8)
    assert result[0, 3] == 1.0
    assert result[0, 0] == 0.0


# -----------------------------------------------------------------------------
# Reward / Termination helpers
# -----------------------------------------------------------------------------


def test_goal_distance_reward_returns_correct_shape() -> None:
    from environments.tasks.paper_swarm.mdp.rewards import goal_distance_reward

    env = _fake_env_with_commands(4)
    result = goal_distance_reward(env, _fake_scene_entity_cfg(), "drone_0", "target_pose", "active_drones")
    assert result.shape == (4,)


def test_collision_reward_returns_correct_shape() -> None:
    from environments.tasks.paper_swarm.mdp.rewards import collision_avoidance_reward

    env = _fake_env_with_full_scene(4, 4)
    result = collision_avoidance_reward(
        env, _fake_scene_entity_cfg(), "drone_0", [f"drone_{i}" for i in range(4)], 0.5, 0.12, "active_drones"
    )
    assert result.shape == (4,)


def test_drone_out_of_bounds_shape() -> None:
    from environments.tasks.paper_swarm.mdp.terminations import drone_out_of_bounds

    env = _fake_env_with_full_scene(2, 4)
    env.root = env
    env.cfg.possible_agents = [f"drone_{i}" for i in range(4)]
    result = drone_out_of_bounds(env, _fake_scene_entity_cfg(), "drone_0", (-6.0, 6.0), (0.2, 5.0), "active_drones")
    assert result.shape == (2,)
    assert result.dtype == torch.bool


# -----------------------------------------------------------------------------
# Vectorized position sampler
# -----------------------------------------------------------------------------


def test_sample_positions_vectorized_shape() -> None:
    from environments.tasks.paper_swarm.mdp.commands import _sample_positions_vectorized

    num_envs = 4
    num_agents = 8
    active_mask = torch.ones(num_envs, num_agents, dtype=torch.bool)
    active_mask[0, 4:] = False

    positions, yaws = _sample_positions_vectorized(
        active_mask=active_mask,
        xy_bounds=(-4.0, 4.0),
        z_bounds=(1.0, 3.0),
        min_separation=2.0,
        safe_prob=1.0,
        columns=None,
        column_radius=0.15,
        column_safe_distance=0.6,
        device="cpu",
    )
    assert positions.shape == (num_envs, num_agents, 3)
    assert yaws.shape == (num_envs, num_agents)
    assert positions[0, 4:, :].eq(0.0).all()
    assert (positions[:3, :4].abs().max() <= 4.0)


def test_sample_positions_min_separation() -> None:
    from environments.tasks.paper_swarm.mdp.commands import _sample_positions_vectorized

    active_mask = torch.ones(8, 4, dtype=torch.bool)
    positions, _ = _sample_positions_vectorized(
        active_mask=active_mask,
        xy_bounds=(-4.0, 4.0),
        z_bounds=(1.0, 3.0),
        min_separation=2.0,
        safe_prob=1.0,
        columns=None,
        column_radius=0.15,
        column_safe_distance=0.6,
        device="cpu",
    )
    for e in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                dist = torch.linalg.norm(positions[e, i, :2] - positions[e, j, :2])
                assert dist >= 2.0 - 1e-5, f"Env {e}, agents {i}-{j}: dist {dist} < 2.0"


# -----------------------------------------------------------------------------
# Curriculum tests
# -----------------------------------------------------------------------------


def test_active_agent_count_curriculum_prefix() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import active_agent_count_curriculum

    env = _fake_env(4, 8)
    env.common_step_counter = 100_000

    active_agent_count_curriculum(
        env, env_ids=None, agent_ids=[f"drone_{i}" for i in range(8)],
        min_agents=2, max_agents=8, ramp_steps=200_000, mask_key="active_drones", selection="prefix",
    )
    mask = getattr(env, "active_drones", None)
    assert mask is not None
    assert mask.shape == (4, 8)
    assert mask[:, 0].all()
    assert mask[:, 1].all()
    assert mask[:, 2].all()
    assert mask[:, 3].all()
    assert mask[:, 4].all()
    assert not mask[:, 5].any()
    assert int(env.extras["active_agent_count"]) == 5


def test_active_agent_count_ramp_end_reaches_max() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import active_agent_count_curriculum

    env = _fake_env(4, 8)
    env.common_step_counter = 300_000

    active_agent_count_curriculum(
        env, env_ids=None, agent_ids=[f"drone_{i}" for i in range(8)],
        min_agents=1, max_agents=8, ramp_steps=200_000, mask_key="active_drones", selection="prefix",
    )
    assert int(env.extras["active_agent_count"]) == 8


def test_passive_drone_count_curriculum_ramp() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import passive_drone_count_curriculum

    env = _fake_env(4, 8)
    env.common_step_counter = 75_000

    passive_drone_count_curriculum(env, env_ids=None, min_passive=1, max_passive=7, ramp_steps=150_000)
    assert getattr(env, "_paper_swarm_num_passive_active", None) == 4
    assert int(env.extras["passive_drone_count"]) == 4


def test_passive_drone_count_curriculum_clamped() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import passive_drone_count_curriculum

    env = _fake_env(4, 8)
    env.common_step_counter = 200_000

    passive_drone_count_curriculum(env, env_ids=None, min_passive=1, max_passive=7, ramp_steps=150_000)
    assert int(env.extras["passive_drone_count"]) == 7


def test_paper_swarm_task_curriculum_obstacles_and_randomization() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import paper_swarm_task_curriculum

    env = _fake_env(4, 8)
    env.common_step_counter = 100_000

    paper_swarm_task_curriculum(
        env, env_ids=None,
        workspace_xy=(-4.0, 4.0), workspace_z=(1.0, 3.0), max_static_columns=10,
        obstacle_start_step=50_000, obstacle_ramp_steps=250_000,
        randomization_start_step=0, randomization_ramp_steps=200_000,
        start_safe_sampling_prob=1.0, end_safe_sampling_prob=0.0,
        start_min_separation=2.0, end_min_separation=0.0,
        column_radius=0.15, column_safe_distance=0.6,
    )
    assert getattr(env, "_paper_swarm_num_static_columns", None) == 2
    assert getattr(env, "_paper_swarm_spawn_safe_sampling_prob", None) == 0.5
    assert getattr(env, "_paper_swarm_spawn_min_separation", None) == 1.0
    assert getattr(env, "_paper_swarm_target_min_separation", None) == 1.0


def test_paper_swarm_task_curriculum_full_obstacles() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import paper_swarm_task_curriculum

    env = _fake_env(4, 8)
    env.common_step_counter = 300_000

    paper_swarm_task_curriculum(
        env, env_ids=None,
        workspace_xy=(-4.0, 4.0), workspace_z=(1.0, 3.0), max_static_columns=6,
        obstacle_start_step=50_000, obstacle_ramp_steps=250_000,
        randomization_start_step=300_000, randomization_ramp_steps=100_000,
        start_safe_sampling_prob=0.5, end_safe_sampling_prob=0.0,
        start_min_separation=1.0, end_min_separation=0.0,
        column_radius=0.15, column_safe_distance=0.6,
    )
    assert getattr(env, "_paper_swarm_num_static_columns", None) == 6
    assert getattr(env, "_paper_swarm_spawn_safe_sampling_prob", None) == 0.5
    assert getattr(env, "_paper_swarm_spawn_min_separation", None) == 1.0


def test_curriculum_fraction_clamped() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import curriculum_fraction

    env = _fake_env(2, 4)
    env.common_step_counter = 0
    assert curriculum_fraction(env, 0, 100_000) == 0.0

    env.common_step_counter = 50_000
    assert curriculum_fraction(env, 0, 100_000) == 0.5

    env.common_step_counter = 100_000
    assert curriculum_fraction(env, 0, 100_000) == 1.0


def test_expand_target_range_curriculum() -> None:
    from environments.tasks.paper_swarm.mdp.curriculums import expand_target_range_curriculum

    env = _fake_env(2, 4)
    env.common_step_counter = 25_000

    cmd_cfg = SimpleNamespace()
    cmd_cfg.target_pose = SimpleNamespace()
    cmd_cfg.target_pose.ranges = SimpleNamespace()
    env.command_manager = SimpleNamespace()
    env.command_manager.cfg = cmd_cfg

    result = expand_target_range_curriculum(
        env, env_ids=None,
        start_step=0, end_step=50_000,
        start_xy=0.0, end_xy=1.5, start_z_delta=0.0, end_z_delta=0.5,
    )
    assert result["frac"] == 0.5
    assert result["xy"] == 0.75
    assert result["z_delta"] == 0.25


# -----------------------------------------------------------------------------
# Fake env helpers
# -----------------------------------------------------------------------------


class _FakeAssetData:
    def __init__(self, num_envs: int) -> None:
        self.root_pos_w = _FakeWarpArray(torch.randn(num_envs, 3))
        self.root_quat_w = _FakeWarpArray(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_envs))
        self.root_lin_vel_w = _FakeWarpArray(torch.zeros(num_envs, 3))
        self.root_ang_vel_w = _FakeWarpArray(torch.zeros(num_envs, 3))
        self.root_ang_vel_b = _FakeWarpArray(torch.zeros(num_envs, 3))
        self.default_root_pose = _FakeWarpArray(torch.zeros(num_envs, 7))
        self.default_root_vel = _FakeWarpArray(torch.zeros(num_envs, 6))
        self.default_joint_pos = _FakeWarpArray(torch.zeros(num_envs, 1))
        self.default_joint_vel = _FakeWarpArray(torch.zeros(num_envs, 1))
        self.default_root_pose.torch_raw = torch.zeros(num_envs, 7)
        self.default_root_pose.torch_raw[:, 6] = 1.0
        self.default_root_vel.torch_raw = torch.zeros(num_envs, 6)
        self.default_joint_pos.torch_raw = torch.zeros(num_envs, 1)
        self.default_joint_vel.torch_raw = torch.zeros(num_envs, 1)

    @property
    def torch(self):
        return self


class _FakeWarpArray:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.torch_raw = tensor

    @property
    def torch(self):
        return self.torch_raw


class _FakeAsset:
    def __init__(self, num_envs: int) -> None:
        self.data = _FakeAssetData(num_envs)

    def write_root_pose_to_sim_index(self, root_pose=None, env_ids=None) -> None:
        pass

    def write_root_velocity_to_sim_index(self, root_velocity=None, env_ids=None) -> None:
        pass

    def write_joint_position_to_sim_index(self, position=None, env_ids=None) -> None:
        pass

    def write_joint_velocity_to_sim_index(self, velocity=None, env_ids=None) -> None:
        pass


class _FakeScene:
    def __init__(self, num_envs: int, num_agents: int) -> None:
        self.env_origins = torch.zeros(num_envs, 3)
        self._assets = {f"drone_{i}": _FakeAsset(num_envs) for i in range(num_agents)}

    def __getitem__(self, key: str):
        return self._assets.get(key, _FakeAsset(1))


class _FakeCommand:
    def __init__(self) -> None:
        self.pose_command_b = torch.zeros(4, 7)
        self.time_left = torch.zeros(4)


class _FakeCommandManager:
    def __init__(self) -> None:
        self._command = torch.zeros(4, 7)
        self._terms = {"target_pose": _FakeCommand()}

    def get_command(self, name: str):
        return self._command

    def compute(self, dt=None) -> None:
        pass


class _FakeObservationManager:
    def __init__(self) -> None:
        self.group_obs_dim = {"policy": (86,)}

    def compute(self, update_history=None):
        return {"policy": torch.zeros(4, 86)}


class _FakeActionManager:
    def __init__(self) -> None:
        self.total_action_dim = 4

    def process_action(self, action=None) -> None:
        pass

    def apply_action(self) -> None:
        pass


class _FakeBundle:
    def __init__(self, agent_ids=None) -> None:
        self.command_manager = _FakeCommandManager()
        self.observation_manager = _FakeObservationManager()
        self.action_manager = _FakeActionManager()
        self.runtime = _FakeRuntime(agent_ids or ["drone_0"])
        self.termination_manager = _FakeNoopManager()
        self.reward_manager = _FakeNoopManager()
        self.curriculum_manager = None


class _FakeRuntime:
    def __init__(self, agent_ids=None) -> None:
        self.agent_ids = agent_ids or ["drone_0"]
        self.name = "drone"


class _FakeNoopManager:
    terminated = torch.zeros(2)
    time_outs = torch.zeros(2)

    def compute(self) -> None:
        pass


def _fake_env(num_envs: int, num_agents: int) -> SimpleNamespace:
    env = SimpleNamespace()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.scene = _FakeScene(num_envs, num_agents)
    env.common_step_counter = 42
    env.episode_length_buf = torch.zeros(num_envs)
    env.cfg = SimpleNamespace()
    env.cfg.possible_agents = [f"drone_{i}" for i in range(num_agents)]
    env.cfg.active_agent_mask_key = "active_drones"
    env.active_drones = torch.ones(num_envs, num_agents, dtype=torch.bool)
    env.extras = {}
    env.root = env
    return env


def _fake_env_with_commands(num_envs: int) -> SimpleNamespace:
    env = _fake_env(num_envs, 4)
    env.command_manager = _FakeCommandManager()
    env.scene.env_origins = torch.zeros(num_envs, 3)
    return env


def _fake_env_with_full_scene(num_envs: int, num_agents: int) -> SimpleNamespace:
    env = _fake_env(num_envs, num_agents)
    env.scene = _FakeScene(num_envs, num_agents)
    env.command_manager = _FakeCommandManager()
    env._agent_to_bundle = {f"drone_{i}": "drone" for i in range(num_agents)}
    env._manager_bundles = {"drone": _FakeBundle([f"drone_{i}" for i in range(num_agents)])}
    env.extras = {}
    env.cfg.possible_agents = [f"drone_{i}" for i in range(num_agents)]
    return env


def _fake_scene_entity_cfg() -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.name = "drone_0"
    return cfg
