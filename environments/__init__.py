# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge module: re-exports from the inner ``environments.environments`` package
so that ``from environments import tasks`` (and similar) work without needing
a separate pip install of the nested package.
"""

from environments.environments import tasks  # noqa: F401 — makes ``from environments import tasks`` work
