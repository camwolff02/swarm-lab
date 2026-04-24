from __future__ import annotations

import argparse
import json
import time

import numpy as np

from cpsquare_lab.ros2.common import flatten_with_spec, infer_array_spec, unflatten_with_spec


def test_flatten_round_trip_for_nested_observations():
    observations = {
        "controller": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "policy": np.asarray([[3.0, 4.0, 5.0]], dtype=np.float32),
    }

    spec = infer_array_spec(observations)
    flat = flatten_with_spec(observations, spec)
    rebuilt = unflatten_with_spec(flat, spec)

    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rebuilt["controller"].tolist() == [[1.0, 2.0]]
    assert rebuilt["policy"].tolist() == [[3.0, 4.0, 5.0]]


def test_environment_bridge_publish_and_receive(ros2_bridge_module, fake_ros2_sdk):
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
        "topics": {
            "schema_key": "/bridge/demo/schema",
            "observations_key": "/bridge/demo/observations",
            "actions_key": "/bridge/demo/actions",
            "status_key": "/bridge/demo/status",
        },
    }

    bridge = ros2_bridge_module.EnvironmentBridge(schema=schema)
    bridge.publish_schema()
    bridge.publish_status({"event": "started"})
    bridge.publish_observations({"policy": np.asarray([[1.0, 2.0]], dtype=np.float32)})

    assert json.loads(fake_ros2_sdk.publishers["/bridge/demo/schema"].messages[-1]["data"])["task"] == "Isaac Demo"
    assert json.loads(fake_ros2_sdk.publishers["/bridge/demo/status"].messages[-1]["data"]) == {"event": "started"}
    assert fake_ros2_sdk.publishers["/bridge/demo/observations"].messages[-1]["data"].tolist() == [1.0, 2.0]

    fake_ros2_sdk.emit("/bridge/demo/actions", data=[0.25, -0.5])
    packet = bridge.wait_for_action(timeout_s=0.1)
    assert packet is not None
    assert packet.payload.tolist() == [[0.25, -0.5]]

    fake_ros2_sdk.emit("/bridge/demo/actions", data=[1.0])
    assert bridge.invalid_action_count == 1

    bridge.close()


def test_select_action_returns_zero_when_timeout_behavior_is_zero(ros2_bridge_module):
    zero_action = np.zeros((1, 2), dtype=np.float32)
    args = argparse.Namespace(wait_for_first_action=False, action_timeout=0.1, timeout_behavior="zero")

    class FakeBridge:
        def __init__(self, packet):
            self.packet = packet

        def get_latest_action(self):
            return self.packet

        def wait_for_action(self, timeout_s=None, after_sequence=None):
            return None

    stale_packet = ros2_bridge_module.PacketStore(history_size=1).put(np.asarray([[1.0, 2.0]], dtype=np.float32))
    object.__setattr__(stale_packet, "received_monotonic_s", time.monotonic() - 1.0)
    selected, _, age = ros2_bridge_module.select_action(
        bridge=FakeBridge(stale_packet),
        zero_action=zero_action,
        last_action=None,
        args=args,
    )

    assert selected.tolist() == [[0.0, 0.0]]
    assert age is not None and age > 0.1
