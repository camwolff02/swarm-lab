# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate paper-swarm Stage 1 checkpoints with physical rollout metrics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

from environments import tasks as local_tasks

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


SKRL_VERSION = "1.4.3"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-LegacyInDistributionEval-v0",
        help="Paper-swarm Stage 1 task id.",
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Repeat to compare.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", default="MAPPO", choices=["MAPPO"])
    parser.add_argument("--ml_framework", default="torch", choices=["torch"])
    parser.add_argument("--success_distance", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path("logs/evaluations/paper_swarm_stage1_search.csv"))
    parser.add_argument("--trace_output", type=Path, default=None, help="Optional NPZ path for env-0 rollout traces.")
    add_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args, hydra_args


def _agent_goal(root, agent_id: str) -> torch.Tensor:
    bundle_name = root._agent_to_bundle[agent_id]
    command_manager = root._manager_bundles[bundle_name].command_manager
    command = command_manager.get_command("target_pose")
    return command[:, :3] + root.scene.env_origins


def _positions(root, agent_ids: list[str]) -> torch.Tensor:
    return torch.stack([root.scene[aid].data.root_pos_w.torch for aid in agent_ids], dim=1)


def _paper_swarm_metrics(root, success_distance: float) -> dict[str, torch.Tensor]:
    active_id = root.possible_agents[0]
    passive_ids = list(getattr(root, "_passive_drone_ids", []))
    active_pos = root.scene[active_id].data.root_pos_w.torch
    goal_pos = _agent_goal(root, active_id)
    active_goal_distance = torch.linalg.norm(goal_pos - active_pos, dim=-1)

    passive_drift = torch.zeros(root.num_envs, 0, device=root.device)
    active_passive_distance = torch.full((root.num_envs,), torch.inf, device=root.device)
    if passive_ids:
        passive_pos = _positions(root, passive_ids)
        hover = getattr(root, "_passive_drone_hover_positions", None)
        if hover is not None:
            hover_w = hover[:, : len(passive_ids), :3] + root.scene.env_origins.unsqueeze(1)
            passive_drift = torch.linalg.norm(passive_pos - hover_w, dim=-1)
        active_passive_distance = torch.linalg.norm(passive_pos - active_pos.unsqueeze(1), dim=-1).amin(dim=1)

    robot_collision = getattr(root, "_paper_swarm_robot_collision_events", None)
    obstacle_collision = getattr(root, "_paper_swarm_obstacle_collision_events", None)
    robot_collision_any = (
        robot_collision.any(dim=1) if robot_collision is not None else torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)
    )
    obstacle_collision_any = (
        obstacle_collision.any(dim=1)
        if obstacle_collision is not None
        else torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)
    )

    active_local = active_pos - root.scene.env_origins
    active_crash = active_local[:, 2] < 0.2

    return {
        "active_goal_distance": active_goal_distance,
        "success": active_goal_distance < success_distance,
        "passive_drift_mean": passive_drift.mean(dim=1) if passive_drift.numel() else torch.zeros(root.num_envs, device=root.device),
        "passive_drift_max": passive_drift.amax(dim=1) if passive_drift.numel() else torch.zeros(root.num_envs, device=root.device),
        "active_passive_distance": active_passive_distance,
        "robot_collision": robot_collision_any,
        "obstacle_collision": obstacle_collision_any,
        "active_crash": active_crash,
    }


def _to_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().mean().cpu().item())


def _true_count(value) -> int:
    if isinstance(value, dict):
        return sum(_true_count(item) for item in value.values())
    return int(value.bool().sum().item())


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: str) -> dict[str, float | int | str]:
    local_tasks.register_tasks_for(args.task)
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

    agent_cfg_entry_point = "skrl_mappo_cfg_entry_point"
    env_cfg, experiment_cfg = resolve_task_config(args.task, agent_cfg_entry_point)
    with launch_simulation(env_cfg, args):
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from skrl.utils.runner.torch import Runner

        if args.seed == -1:
            args.seed = random.randint(0, 10000)
        experiment_cfg["seed"] = args.seed if args.seed is not None else experiment_cfg["seed"]
        env_cfg.seed = experiment_cfg["seed"]
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
        env_cfg.log_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))

        print(f"[INFO] Creating env for {checkpoint}", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        root = env.unwrapped
        env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework, wrapper="isaaclab-multi-agent")

        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        print("[INFO] Creating SKRL runner", flush=True)
        runner = Runner(env, experiment_cfg)
        print(f"[INFO] Loading checkpoint: {os.path.abspath(checkpoint)}", flush=True)
        runner.agent.load(os.path.abspath(checkpoint))
        if hasattr(runner.agent, "enable_training_mode"):
            runner.agent.enable_training_mode(False, apply_to_models=True)
        elif hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("eval")

        print("[INFO] Resetting env", flush=True)
        obs, _ = env.reset()
        states = env.state()
        active_id = root.possible_agents[0]
        first = _paper_swarm_metrics(root, args.success_distance)
        initial_goal_distance = first["active_goal_distance"].clone()
        min_goal_distance = initial_goal_distance.clone()
        final_metrics = first
        trace_positions: list[np.ndarray] = []
        trace_goals: list[np.ndarray] = []
        trace_passive_hover: list[np.ndarray] = []
        reached = first["success"].clone()
        ever_robot_collision = first["robot_collision"].clone()
        ever_obstacle_collision = first["obstacle_collision"].clone()
        ever_crash = first["active_crash"].clone()
        max_passive_drift = first["passive_drift_max"].clone()
        min_active_passive_distance = first["active_passive_distance"].clone()
        terminated_count = 0
        truncated_count = 0
        reward_sum = 0.0

        print("[INFO] Starting rollout", flush=True)
        for timestep in range(args.num_steps):
            with torch.inference_mode():
                if args.trace_output is not None:
                    agent_ids = [active_id, *list(getattr(root, "_passive_drone_ids", []))]
                    trace_positions.append(_positions(root, agent_ids)[0].detach().cpu().numpy())
                    trace_goals.append(_agent_goal(root, active_id)[0].detach().cpu().numpy())
                    hover = getattr(root, "_passive_drone_hover_positions", None)
                    if hover is not None:
                        passive_ids = list(getattr(root, "_passive_drone_ids", []))
                        trace_passive_hover.append(
                            (hover[0, : len(passive_ids), :3] + root.scene.env_origins[0]).detach().cpu().numpy()
                        )
                outputs = runner.agent.act(obs, states, timestep=timestep, timesteps=args.num_steps)
                actions = {agent: outputs[-1][agent].get("mean_actions", outputs[0][agent]) for agent in env.possible_agents}
                obs, rewards, terminated, truncated, _ = env.step(actions)
                states = env.state()
                reward_sum += sum(float(value.float().mean().item()) for value in rewards.values()) / max(1, len(rewards))
                terminated_count += _true_count(terminated)
                truncated_count += _true_count(truncated)

                metrics = _paper_swarm_metrics(root, args.success_distance)
                final_metrics = metrics
                min_goal_distance = torch.minimum(min_goal_distance, metrics["active_goal_distance"])
                reached |= metrics["success"]
                ever_robot_collision |= metrics["robot_collision"]
                ever_obstacle_collision |= metrics["obstacle_collision"]
                ever_crash |= metrics["active_crash"]
                max_passive_drift = torch.maximum(max_passive_drift, metrics["passive_drift_max"])
                min_active_passive_distance = torch.minimum(
                    min_active_passive_distance, metrics["active_passive_distance"]
                )

        summary = {
            "checkpoint": os.path.abspath(checkpoint),
            "seed": int(args.seed),
            "num_envs": int(root.num_envs),
            "num_steps": int(args.num_steps),
            "initial_goal_distance_mean": _to_float(initial_goal_distance),
            "final_goal_distance_mean": _to_float(final_metrics["active_goal_distance"]),
            "min_goal_distance_mean": _to_float(min_goal_distance),
            "progress_to_min_mean": _to_float(initial_goal_distance - min_goal_distance),
            "progress_to_final_mean": _to_float(initial_goal_distance - final_metrics["active_goal_distance"]),
            "success_rate": _to_float(reached),
            "final_success_rate": _to_float(final_metrics["success"]),
            "max_passive_drift_mean": _to_float(max_passive_drift),
            "max_passive_drift_p95": float(torch.quantile(max_passive_drift.detach().float(), 0.95).cpu().item()),
            "min_active_passive_distance_mean": _to_float(min_active_passive_distance),
            "robot_collision_rate": _to_float(ever_robot_collision),
            "obstacle_collision_rate": _to_float(ever_obstacle_collision),
            "active_crash_rate": _to_float(ever_crash),
            "terminated_count": int(terminated_count),
            "truncated_count": int(truncated_count),
            "mean_reward_per_step": float(reward_sum / max(1, args.num_steps)),
        }
        print(json.dumps(summary, indent=2), flush=True)
        if args.trace_output is not None and trace_positions:
            args.trace_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.trace_output,
                positions=np.asarray(trace_positions, dtype=np.float32),
                goals=np.asarray(trace_goals, dtype=np.float32),
                passive_hover=np.asarray(trace_passive_hover, dtype=np.float32),
                active_idx=np.asarray([0], dtype=np.int64),
                agent_ids=np.asarray([active_id, *list(getattr(root, "_passive_drone_ids", []))]),
                checkpoint=np.asarray([os.path.abspath(checkpoint)]),
                task=np.asarray([args.task]),
                seed=np.asarray([int(args.seed)]),
            )
            print(f"[INFO] Wrote trace to: {args.trace_output.resolve()}", flush=True)
        env.close()
        return summary


def main() -> None:
    args, _ = parse_args()
    results = []
    for checkpoint in args.checkpoint:
        result = evaluate_checkpoint(args, checkpoint)
        print(json.dumps(result, indent=2))
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Wrote results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
