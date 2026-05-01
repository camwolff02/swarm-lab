from __future__ import annotations

import types

import torch
from environments.tasks.quad_swarm_paper.env_cache import EnvCache


class FakeScene:
    def __init__(self, assets: dict[str, object], env_origins: torch.Tensor) -> None:
        self.assets = assets
        self.articulations = assets
        self.env_origins = env_origins

    def __getitem__(self, name: str) -> object:
        return self.assets[name]


def _fake_asset(root_pos_w: torch.Tensor) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=root_pos_w,
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(root_pos_w.shape[0], 1),
            root_lin_vel_w=torch.zeros_like(root_pos_w),
            root_ang_vel_b=torch.zeros_like(root_pos_w),
        )
    )


def _fake_env() -> types.SimpleNamespace:
    env_origins = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float32)
    assets = {
        "drone_0": _fake_asset(torch.tensor([[11.0, 2.0, 3.0], [21.0, 2.0, 3.0]], dtype=torch.float32)),
        "drone_1": _fake_asset(torch.tensor([[14.0, 2.0, 3.0], [24.0, 2.0, 3.0]], dtype=torch.float32)),
    }
    return types.SimpleNamespace(
        scene=FakeScene(assets, env_origins),
        _agent_ids=["drone_0", "drone_1"],
        _goals=torch.tensor(
            [
                [[2.0, 2.0, 3.0], [4.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0], [7.0, 2.0, 3.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_env_cache_reuses_swarm_tracking_within_phase() -> None:
    env = _fake_env()
    cache = EnvCache(env)

    first = cache.swarm_tracking(env._agent_ids, phase="reward")
    second = cache.swarm_tracking(env._agent_ids, phase="reward")

    assert first is second
    assert torch.equal(first.kinematics.root_pos_env[:, :, 0], torch.tensor([[1.0, 4.0], [1.0, 4.0]]))
    assert torch.equal(first.distance, torch.tensor([[1.0, 0.0], [0.0, 3.0]]))
    assert cache.stats["asset_misses"] == 2
    assert cache.stats["reward_swarm_tracking_hits"] == 1


def test_env_cache_separates_reward_and_observation_phases_and_reset_invalidation() -> None:
    env = _fake_env()
    cache = EnvCache(env)

    reward_state = cache.swarm_kinematics(env._agent_ids, phase="reward")
    env.scene.assets["drone_0"].data.root_pos_w += torch.tensor([5.0, 0.0, 0.0])

    cached_reward_state = cache.swarm_kinematics(env._agent_ids, phase="reward")
    observation_state = cache.swarm_kinematics(env._agent_ids, phase="observation")
    cache.on_reset()
    refreshed_reward_state = cache.swarm_kinematics(env._agent_ids, phase="reward")

    assert cached_reward_state is reward_state
    assert torch.equal(cached_reward_state.root_pos_env[:, 0, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(observation_state.root_pos_env[:, 0, 0], torch.tensor([6.0, 6.0]))
    assert torch.equal(refreshed_reward_state.root_pos_env[:, 0, 0], torch.tensor([6.0, 6.0]))
