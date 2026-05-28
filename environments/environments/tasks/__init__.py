# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

from __future__ import annotations

import importlib
import pkgutil
import warnings


def _import_task_packages() -> None:
    """Register task packages without letting one broken import hide the rest.

    This is intentionally opt-in. Several task packages depend on Omniverse
    modules, and importing them before IsaacLab starts ``SimulationApp`` can
    preload ``pxr`` and destabilize Kit startup.
    """
    for module_info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if not module_info.ispkg:
            continue
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            warnings.warn(
                f"Skipping task package {module_info.name!r} during registration: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def register_tasks_for(task_name: str | None) -> None:
    """Register the package that owns ``task_name`` without importing all tasks."""

    if task_name and task_name.startswith("Isaac-Cameron-Drone-Waypoint-"):
        importlib.import_module(f"{__name__}.cameron_swarm")
        return
    if task_name and task_name.startswith("Isaac-Paper-Swarm-Waypoint-"):
        importlib.import_module(f"{__name__}.paper_swarm")
        return
    if task_name and task_name.startswith("Isaac-Formation-Swarm-"):
        importlib.import_module(f"{__name__}.formation_swarm")
        return
    _import_task_packages()


__all__ = ["register_tasks_for", "_import_task_packages"]
