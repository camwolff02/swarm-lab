from __future__ import annotations

import importlib.util
import importlib
import sys
import types
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeRosMessage:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeROS2Publisher:
    def __init__(self, sdk, topic: str, msg_type: str) -> None:
        self.sdk = sdk
        self.topic = topic
        self.msg_type = msg_type
        self.messages: list[dict] = []
        self.closed = False

    def publish(self, **kwargs) -> None:
        self.messages.append(kwargs)
        self.sdk.emit(self.topic, **kwargs)

    def close(self) -> None:
        self.closed = True


class FakeROS2Subscriber:
    def __init__(self, sdk, topic: str, msg_type: str, callback) -> None:
        self.sdk = sdk
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback
        self.closed = False
        self.sdk.subscribers.setdefault(topic, []).append(self)

    def close(self) -> None:
        self.closed = True


class FakeZenohRos2Sdk(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("zenoh_ros2_sdk")
        self.publishers: dict[str, FakeROS2Publisher] = {}
        self.subscribers: dict[str, list[FakeROS2Subscriber]] = {}

    def ROS2Publisher(self, topic: str, msg_type: str):
        publisher = FakeROS2Publisher(self, topic, msg_type)
        self.publishers[topic] = publisher
        return publisher

    def ROS2Subscriber(self, topic: str, msg_type: str, callback):
        return FakeROS2Subscriber(self, topic, msg_type, callback)

    def load_message_type(self, msg_type: str, messages_dir: str | None = None) -> bool:
        return True

    def get_message_class(self, msg_type: str):
        class FakeMessage:
            def __init__(self, **kwargs) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return FakeMessage

    def emit(self, topic: str, **kwargs) -> None:
        message = FakeRosMessage(**kwargs)
        for subscriber in list(self.subscribers.get(topic, [])):
            subscriber.callback(message)


@pytest.fixture
def fake_ros2_sdk(monkeypatch) -> FakeZenohRos2Sdk:
    module = FakeZenohRos2Sdk()
    monkeypatch.setitem(sys.modules, "zenoh_ros2_sdk", module)
    return module


@pytest.fixture
def agent_bridge_module():
    sys.modules.pop("cpsquare_lab.ros2.agent_bridge", None)
    return importlib.import_module("cpsquare_lab.ros2.agent_bridge")


@pytest.fixture
def ros2_bridge_module():
    sys.modules.pop("cpsquare_lab.ros2.environment_bridge", None)
    return importlib.import_module("cpsquare_lab.ros2.environment_bridge")


@pytest.fixture
def run_env_module(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []

    utils_pkg = types.ModuleType("isaaclab.utils")
    utils_pkg.__path__ = []
    utils_io = types.ModuleType("isaaclab.utils.io")

    def dump_yaml(path, payload):
        Path(path).write_text("stub\n", encoding="utf-8")

    utils_io.dump_yaml = dump_yaml

    isaaclab_tasks = types.ModuleType("isaaclab_tasks")
    isaaclab_tasks_utils = types.ModuleType("isaaclab_tasks.utils")

    def add_launcher_args(_parser) -> None:
        return None

    def resolve_task_config(_task, _agent):
        cfg = types.SimpleNamespace(
            scene=types.SimpleNamespace(num_envs=1),
            sim=types.SimpleNamespace(device="cpu", use_fabric=True),
        )
        return cfg, None

    class _NullContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def launch_simulation(_env_cfg, _args):
        return _NullContext()

    isaaclab_tasks_utils.add_launcher_args = add_launcher_args
    isaaclab_tasks_utils.resolve_task_config = resolve_task_config
    isaaclab_tasks_utils.launch_simulation = launch_simulation

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.io", utils_io)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", isaaclab_tasks)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", isaaclab_tasks_utils)
    monkeypatch.setattr(sys, "argv", ["run_env.py", "--task", "Isaac-Demo-v0"])

    module_name = f"test_run_env_{uuid.uuid4().hex}"
    return load_module_from_path(ROOT / "scripts" / "ros2" / "run_env.py", module_name)


@pytest.fixture
def run_agent_module(monkeypatch):
    isaaclab_tasks = types.ModuleType("isaaclab_tasks")
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", isaaclab_tasks)
    monkeypatch.setattr(sys, "argv", ["run_agent.py"])

    module_name = f"test_run_agent_{uuid.uuid4().hex}"
    return load_module_from_path(ROOT / "scripts" / "ros2" / "run_agent.py", module_name)
