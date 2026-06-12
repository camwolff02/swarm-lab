# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a formation-swarm MAPPO checkpoint and persist rollout metrics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, resolve_task_config

from environments import tasks as local_tasks

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


SKRL_VERSION = "1.4.3"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate a formation-swarm MAPPO checkpoint.")
    parser.add_argument("--task", default="Isaac-Formation-Swarm-MAPPO-Stage1-v0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=455)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", default="MAPPO", choices=["MAPPO"])
    parser.add_argument("--ml_framework", default="torch", choices=["torch"])
    parser.add_argument("--record_stride", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/evaluations/thesis_sim/formation_swarm_stage1"))
    parser.add_argument("--dataset_name", default="formation_swarm_stage1_seed42")
    add_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args, hydra_args


def _stack_positions(root) -> torch.Tensor:
    asset_names = tuple(getattr(root, "_formation_asset_names", root.possible_agents))
    return torch.stack(
        [root.scene[asset_name].data.root_pos_w.torch - root.scene.env_origins for asset_name in asset_names],
        dim=1,
    )


def _formation_metrics(root) -> dict[str, torch.Tensor]:
    positions = _stack_positions(root)
    pairwise = torch.cdist(positions, positions)
    target_offsets = getattr(root, "_formation_offsets")
    target_pairwise = torch.cdist(target_offsets, target_offsets)
    agent_count = len(root.possible_agents)
    upper = torch.triu(torch.ones(agent_count, agent_count, device=root.device, dtype=torch.bool), diagonal=1)
    formation_error = (pairwise - target_pairwise.unsqueeze(0)).abs()[:, upper].mean(dim=-1)

    hard_safe = float(getattr(root.cfg, "hard_safe_distance", 0.15))
    collision = pairwise.masked_fill(torch.eye(agent_count, device=root.device, dtype=torch.bool), torch.inf).amin(
        dim=(1, 2)
    ) < hard_safe

    crash_min = float(getattr(root.cfg, "crash_min_height", 0.2))
    crash_max = float(getattr(root.cfg, "crash_max_height", 2.8))
    crash = ((positions[..., 2] < crash_min) | (positions[..., 2] > crash_max)).any(dim=1)

    ball_hit = torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)
    balls = getattr(root, "_formation_ball_positions", None)
    ball_active = getattr(root, "_formation_ball_active", None)
    if balls is not None and ball_active is not None and balls.numel() > 0:
        ball_dist = torch.linalg.norm(balls.unsqueeze(1) - positions.unsqueeze(2), dim=-1)
        ball_hit = ((ball_dist < float(getattr(root.cfg, "ball_radius", 0.15))) & ball_active.unsqueeze(1)).any(
            dim=(1, 2)
        )

    column_hit = torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)
    columns = getattr(root, "_formation_column_positions", None)
    active_static = int(getattr(root, "_formation_active_static_obstacles", 0))
    if columns is not None and active_static > 0:
        col_dist = torch.linalg.norm(columns[:, :active_static].unsqueeze(1)[..., :2] - positions.unsqueeze(2)[..., :2], dim=-1)
        column_hit = (col_dist < float(getattr(root.cfg, "column_radius", 0.15))).any(dim=(1, 2))

    center = positions.mean(dim=1)
    final_pos = getattr(root, "_formation_final_pos", torch.zeros(3, device=root.device))
    center_goal_error = torch.linalg.norm(center - final_pos.view(1, 3), dim=-1)

    return {
        "positions": positions,
        "formation_error": formation_error,
        "collision": collision,
        "crash": crash,
        "ball_hit": ball_hit,
        "column_hit": column_hit,
        "center_goal_error": center_goal_error,
    }


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _write_outputs(output_dir: Path, dataset_name: str, records: dict[str, list[np.ndarray]], summary: dict[str, float | int | str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{dataset_name}.npz", **{key: np.asarray(value) for key, value in records.items()})
    with (output_dir / f"{dataset_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / f"{dataset_name}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    positions = np.asarray(records["positions"])
    times = np.asarray(records["time_s"]).reshape(-1)
    with (output_dir / f"{dataset_name}_trajectory_env0.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["time_s", "agent", "x", "y", "z"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, time_s in enumerate(times):
            for agent_idx in range(positions.shape[2]):
                x, y, z = positions[sample_idx, 0, agent_idx]
                writer.writerow({"time_s": time_s, "agent": f"robot_{agent_idx}", "x": x, "y": y, "z": z})


def main() -> None:
    args, _ = parse_args()
    local_tasks.register_tasks_for(args.task)
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

    env_cfg, experiment_cfg = resolve_task_config(args.task, "skrl_mappo_cfg_entry_point")
    with launch_simulation(env_cfg, args):
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        if args.seed == -1:
            args.seed = random.randint(0, 10000)
        experiment_cfg["seed"] = args.seed if args.seed is not None else experiment_cfg["seed"]
        env_cfg.seed = experiment_cfg["seed"]
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

        log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))
        if args.checkpoint:
            resume_path = os.path.abspath(args.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, run_dir=".*_mappo_torch", other_dirs=["checkpoints"])
        env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

        env = gym.make(args.task, cfg=env_cfg)
        root = env.unwrapped
        dt = root.step_dt
        env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework, wrapper="isaaclab-multi-agent")

        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        if hasattr(runner.agent, "enable_training_mode"):
            runner.agent.enable_training_mode(False, apply_to_models=True)
        elif hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("eval")

        obs, _ = env.reset()
        states = env.state()
        records: dict[str, list[np.ndarray]] = {
            "time_s": [],
            "positions": [],
            "formation_error": [],
            "center_goal_error": [],
            "collision": [],
            "crash": [],
            "ball_hit": [],
            "column_hit": [],
        }
        ever_failed = torch.zeros(root.num_envs, device=root.device, dtype=torch.bool)
        last_metrics = None
        reward_sum = 0.0

        for timestep in range(args.num_steps):
            start_time = time.time()
            with torch.inference_mode():
                outputs = runner.agent.act(obs, states, timestep=timestep, timesteps=args.num_steps)
                actions = {agent: outputs[-1][agent].get("mean_actions", outputs[0][agent]) for agent in env.possible_agents}
                obs, rewards, _, _, _ = env.step(actions)
                states = env.state()
                reward_sum += sum(float(value.float().mean().item()) for value in rewards.values()) / max(1, len(rewards))

                metrics = _formation_metrics(root)
                last_metrics = metrics
                failed = metrics["collision"] | metrics["crash"] | metrics["ball_hit"] | metrics["column_hit"]
                ever_failed |= failed

                if timestep % max(1, args.record_stride) == 0 or timestep == args.num_steps - 1:
                    records["time_s"].append(np.asarray(timestep * dt, dtype=np.float32))
                    for key in records:
                        if key == "time_s":
                            continue
                        records[key].append(_to_numpy(metrics[key]))

            sleep_time = dt - (time.time() - start_time)
            if getattr(args, "real_time", False) and sleep_time > 0:
                time.sleep(sleep_time)

        assert last_metrics is not None
        scenario = args.task.removeprefix("Isaac-Formation-Swarm-MAPPO-").removesuffix("-v0").lower()
        summary = {
            "method": "formation_swarm_stage1",
            "scenario": scenario,
            "num_envs": int(root.num_envs),
            "num_steps": int(args.num_steps),
            "duration_s": float(args.num_steps * dt),
            "cfr": float((~ever_failed).float().mean().item()),
            "formation_error_mean": float(last_metrics["formation_error"].mean().item()),
            "formation_error_std": float(last_metrics["formation_error"].std(unbiased=False).item()),
            "center_goal_error_mean": float(last_metrics["center_goal_error"].mean().item()),
            "collision_rate": float(ever_failed.float().mean().item()),
            "mean_reward_per_step": float(reward_sum / max(1, args.num_steps)),
            "checkpoint": resume_path,
        }
        _write_outputs(args.output_dir, args.dataset_name, records, summary)
        print(f"[INFO] Wrote formation evaluation outputs to: {args.output_dir.resolve()}")
        env.close()


if __name__ == "__main__":
    main()
