from __future__ import annotations

import json


def test_agent_bridge_receives_schema_observations_and_status(agent_bridge_module, fake_ros2_sdk):
    bridge = agent_bridge_module.AgentBridge(task="Isaac Demo")

    schema = {
        "task": "Isaac Demo",
        "step_dt": 0.02,
        "num_envs": 1,
        "observation_spec": {
            "root_type": "dict",
            "total_size": 2,
            "leaves": [{"path": ["policy"], "shape": [1, 2], "size": 2}],
        },
        "action_spec": {
            "root_type": "tensor",
            "total_size": 2,
            "leaves": [{"path": [], "shape": [1, 2], "size": 2}],
        },
        "topics": bridge.keys,
    }

    fake_ros2_sdk.emit(bridge.keys["schema_key"], data=json.dumps(schema))
    observed_schema = bridge.wait_for_schema(timeout_s=0.1)
    assert observed_schema == schema

    fake_ros2_sdk.emit(bridge.keys["observations_key"], data=[1.0, 2.0])
    packet = bridge.wait_for_observation_packet(timeout_s=0.1)
    assert packet is not None
    assert packet.sequence == 1
    assert packet.payload["step"] == 1
    assert packet.payload["observations"]["policy"].tolist() == [[1.0, 2.0]]

    fake_ros2_sdk.emit(bridge.keys["status_key"], data=json.dumps({"event": "reset"}))
    assert bridge.wait_for_status(timeout_s=0.1, event="reset") == {"event": "reset"}

    sequence = bridge.put_actions([[0.25, -0.5]])
    assert sequence == 1
    assert fake_ros2_sdk.publishers[bridge.keys["actions_key"]].messages[-1]["data"].tolist() == [0.25, -0.5]

    bridge.close()


def test_agent_bridge_context_manager_closes_resources(agent_bridge_module, fake_ros2_sdk):
    with agent_bridge_module.AgentBridge(task="Isaac Demo") as bridge:
        schema = {
            "task": "Isaac Demo",
            "step_dt": 0.02,
            "num_envs": 1,
            "observation_spec": {
                "root_type": "tensor",
                "total_size": 1,
                "leaves": [{"path": [], "shape": [1], "size": 1}],
            },
            "action_spec": {
                "root_type": "tensor",
                "total_size": 1,
                "leaves": [{"path": [], "shape": [1], "size": 1}],
            },
            "topics": bridge.keys,
        }
        fake_ros2_sdk.emit(bridge.keys["schema_key"], data=json.dumps(schema))
        assert bridge.wait_for_schema(timeout_s=0.1) is not None

    assert fake_ros2_sdk.publishers[bridge.keys["actions_key"]].closed is True
