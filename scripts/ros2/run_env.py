# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin launcher for running an Isaac Lab task behind the cpsquare_lab ROS2 bridge."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

with contextlib.suppress(ImportError):
    import environments.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from cpsquare_lab.ros2.environment_bridge import run_environment_bridge


parser = argparse.ArgumentParser(description="Run an Isaac Lab environment through the cpsquare_lab ROS2 bridge.")
parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--publish_every", type=int, default=1, help="Publish observations every N environment steps.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of environment steps to run.")
parser.add_argument("--wait_for_first_action", action="store_true", default=False, help="Block until the first action.")
parser.add_argument(
    "--action_timeout",
    type=float,
    default=None,
    help="Maximum age in seconds of the latest action before timeout behavior applies.",
)
parser.add_argument(
    "--timeout_behavior",
    type=str,
    default="hold-last",
    choices=["hold-last", "zero"],
    help="Behavior when the newest action is older than --action_timeout.",
)
parser.add_argument("--report_interval", type=int, default=200, help="Print a runtime summary every N steps.")
parser.add_argument("--session_name", type=str, default=None, help="Optional suffix for the run directory.")
parser.add_argument("--key_prefix", type=str, default=None, help="Override the ROS topic prefix.")
parser.add_argument("--schema_key", type=str, default=None, help="Override the schema topic.")
parser.add_argument("--observations_key", type=str, default=None, help="Override the observations topic.")
parser.add_argument("--actions_key", type=str, default=None, help="Override the actions topic.")
parser.add_argument("--status_key", type=str, default=None, help="Override the status topic.")
parser.add_argument("--real_time", action="store_true", default=False, help="Sleep to match the env step time.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


dump_yaml = None


def ensure_runtime_imports() -> None:
    """Import runtime helpers only after Isaac Sim has started."""
    global dump_yaml
    if dump_yaml is None:
        from isaaclab.utils.io import dump_yaml as _dump_yaml

        dump_yaml = _dump_yaml


def main() -> None:
    """Launch Isaac Lab and hand the live environment off to the shared ROS2 bridge."""
    env_cfg, _ = resolve_task_config(args_cli.task, None)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    with launch_simulation(env_cfg, args_cli):
        ensure_runtime_imports()
        run_environment_bridge(task=args_cli.task, args=args_cli, env_cfg=env_cfg, dump_yaml=dump_yaml)


if __name__ == "__main__":
    main()
