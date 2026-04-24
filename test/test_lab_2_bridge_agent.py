from __future__ import annotations

from types import SimpleNamespace

import torch

from environments.tasks.lab_2_classical_control.agents.hover_agent import ClassicalHoverAgent
from environments.tasks.lab_2_classical_control.agents.hover_controller import parse_controller_observation


class FakeAgentBridge:
    def __init__(self) -> None:
        self.keys = {
            "schema_key": "bridge/demo/schema",
            "observations_key": "bridge/demo/observations",
            "actions_key": "bridge/demo/actions",
            "status_key": "bridge/demo/status",
        }
        self.published_actions: list[tuple[object, dict | None]] = []
        self._schema = {"task": "Isaac-SimpleFlight-Classical-Crazyflie-v0", "step_dt": 0.01, "num_envs": 1}

    def wait_for_schema(self, timeout_s: float | None = None):
        return self._schema

    def put_actions(self, actions, metadata=None):
        self.published_actions.append((actions, metadata))
        return len(self.published_actions)

    def close(self):
        return None


def test_parse_controller_observation_splits_expected_fields():
    observation = torch.tensor(
        [[0.0, 0.0, 1.0, 0.1, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0, 0.02, -0.03, 0.01, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    parsed = parse_controller_observation(observation)

    assert tuple(parsed["target_position"].shape) == (1, 3)
    assert tuple(parsed["root_quaternion"].shape) == (1, 4)
    assert tuple(parsed["body_angular_velocity"].shape) == (1, 3)
    assert torch.allclose(parsed["target_position"], torch.tensor([[0.0, 0.0, 1.0]]))


def test_classical_hover_agent_computes_and_publishes_actions():
    bridge = FakeAgentBridge()
    agent = ClassicalHoverAgent(bridge=bridge, print_every=1)

    observation_payload = {
        "step": 0,
        "observations": {
            "controller": [
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.8,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        },
    }

    sequence, actions, diagnostics = agent.step(observation_payload)

    assert sequence == 1
    assert tuple(actions.shape) == (1, 4)
    assert "thrust_ratio" in diagnostics
    assert len(bridge.published_actions) == 1
    published_actions, metadata = bridge.published_actions[0]
    assert isinstance(published_actions, list)
    assert len(published_actions[0]) == 4
    assert metadata is not None
    assert metadata["observation_step"] == 0
