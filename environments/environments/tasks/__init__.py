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
    """Register task packages without letting one broken import hide the rest."""
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


_import_task_packages()
