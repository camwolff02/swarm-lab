"""Opt-in startup hook for running local tasks through Isaac Lab's unified CLI."""

from __future__ import annotations

import contextlib
import sys

with contextlib.suppress(ImportError):
    from environments import tasks as _local_tasks

    _task_name = None
    for _index, _arg in enumerate(sys.argv):
        if _arg == "--task" and _index + 1 < len(sys.argv):
            _task_name = sys.argv[_index + 1]
            break
        if _arg.startswith("--task="):
            _task_name = _arg.split("=", 1)[1]
            break
    if _task_name:
        _local_tasks.register_tasks_for(_task_name)

with contextlib.suppress(ImportError):
    import isaaclab_rl.skrl as _skrl_wrapper

    _original_skrl_vec_env_wrapper = _skrl_wrapper.SkrlVecEnvWrapper

    def _manager_marl_aware_skrl_wrapper(env, ml_framework="torch", wrapper="isaaclab"):
        if wrapper == "isaaclab" and hasattr(env.unwrapped, "possible_agents"):
            wrapper = "isaaclab-multi-agent"
        return _original_skrl_vec_env_wrapper(env, ml_framework=ml_framework, wrapper=wrapper)

    _skrl_wrapper.SkrlVecEnvWrapper = _manager_marl_aware_skrl_wrapper
