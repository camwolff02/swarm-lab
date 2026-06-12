# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin launcher for running a task-specific ROS2 agent."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import environments.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from cpsquare_lab.ros2.environment_bridge import instantiate_agent, resolve_agent_entry_point


def build_parser() -> argparse.ArgumentParser:
    """Build the requested object from runtime metadata."""
    parser = argparse.ArgumentParser(description="Run a task-specific ROS2 agent.")
    parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task.")
    parser.add_argument(
        "--agent",
        type=str,
        default="default",
        help="Named agent registered by the task, or a raw module:object entry point.",
    )
    parser.add_argument("--key_prefix", type=str, default=None, help="Override the ROS topic prefix.")
    parser.add_argument("--schema_key", type=str, default=None, help="Override the schema topic.")
    parser.add_argument("--observations_key", type=str, default=None, help="Override the observations topic.")
    parser.add_argument("--actions_key", type=str, default=None, help="Override the actions topic.")
    parser.add_argument("--status_key", type=str, default=None, help="Override the status topic.")
    parser.add_argument("--timeout_s", type=float, default=5.0, help="Timeout when waiting for bridge messages.")
    parser.add_argument("--print_every", type=int, default=50, help="How often to print agent diagnostics.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of agent steps to run.")
    return parser


def main() -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args()
    entry_point = resolve_agent_entry_point(args.task, args.agent)
    agent = instantiate_agent(
        task=args.task,
        agent_name=args.agent,
        init_kwargs={
            "task": args.task,
            "key_prefix": args.key_prefix,
            "schema_key": args.schema_key,
            "observations_key": args.observations_key,
            "actions_key": args.actions_key,
            "status_key": args.status_key,
            "timeout_s": args.timeout_s,
            "print_every": args.print_every,
        },
    )

    try:
        print(f"[INFO] Loaded agent: {entry_point}")
        agent.run(max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("[INFO] Agent interrupted by user.")
    finally:
        with contextlib.suppress(Exception):
            agent.close()


if __name__ == "__main__":
    main()
