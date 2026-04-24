from __future__ import annotations


def test_run_agent_uses_task_agent_resolution(monkeypatch, run_agent_module):
    calls: dict[str, object] = {}

    class FakeAgent:
        def __init__(self) -> None:
            self.closed = False

        def run(self, max_steps=None):
            calls["max_steps"] = max_steps

        def close(self):
            self.closed = True
            calls["closed"] = True

    def fake_resolve(task: str, agent_name: str | None = None) -> str:
        calls["resolve"] = (task, agent_name)
        return "demo.module:FakeAgent"

    def fake_instantiate(*, task: str, agent_name: str | None = None, init_kwargs=None):
        calls["instantiate"] = (task, agent_name, dict(init_kwargs or {}))
        return FakeAgent()

    monkeypatch.setattr(run_agent_module, "resolve_agent_entry_point", fake_resolve)
    monkeypatch.setattr(run_agent_module, "instantiate_agent", fake_instantiate)
    monkeypatch.setattr(
        run_agent_module.sys,
        "argv",
        [
            "run_agent.py",
            "--task",
            "Isaac-Demo-v0",
            "--agent",
            "hover",
            "--max_steps",
            "7",
        ],
    )

    run_agent_module.main()

    assert calls["resolve"] == ("Isaac-Demo-v0", "hover")
    assert calls["instantiate"][0] == "Isaac-Demo-v0"
    assert calls["instantiate"][1] == "hover"
    assert calls["instantiate"][2]["task"] == "Isaac-Demo-v0"
    assert calls["max_steps"] == 7
    assert calls["closed"] is True
