#!/usr/bin/env python
"""Probe the paper_swarm environment: spawn positions, passive drone control, obstacles.

Runs a short-sniff environment (no RL agent) so Isaac Sim is needed.
Writes per-step state to a local HDF5 snapshot.

Usage:
    uv run scripts/analysis/probe_env.py --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-Eval-v0
    uv run scripts/analysis/probe_env.py --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage3-Eval-v0
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager

import h5py
import numpy as np
import torch


def probe(
    task_id: str,
    num_envs: int = 1,
    max_steps: int = 100,
    hdf5_path: str = "/tmp/isaaclab/logs/probe_snapshot.hdf5",
) -> None:
    # ------------------------------------------------------------------
    # Late imports (Isaac Lab is heavy)
    # ------------------------------------------------------------------
    import gymnasium as gym
    from isaaclab_tasks.utils import resolve_task_config
    from environments import tasks as local_tasks

    sys.argv = [sys.argv[0]]
    local_tasks.register_tasks_for(task_id)

    # Determine algorithm from task ID
    is_mappo = "MAPPO" in task_id
    entry = "skrl_mappo_cfg_entry_point" if is_mappo else "skrl_cfg_entry_point"
    env_cfg, _experiment_cfg = resolve_task_config(task_id, entry)

    env_cfg.scene.num_envs = num_envs
    env_cfg.log_dir = None
    print(f"[probe] Creating env: {task_id} (num_envs={num_envs})")

    env = gym.make(task_id, cfg=env_cfg)
    env.reset()

    root_env = env.unwrapped
    possible = root_env.possible_agents
    passive_ids = getattr(root_env, "_passive_drone_ids", [])
    all_drone_ids = list(passive_ids) + [a for a in possible if a not in passive_ids]
    num_drones = len(all_drone_ids)
    device = root_env.device

    print(f"[probe]   possible_agents: {possible}")
    print(f"[probe]   passive_drone_ids: {passive_ids}")
    print(f"[probe]   all_drone_ids (recorder order): {all_drone_ids}")
    print(f"[probe]   episode_length_s: {getattr(root_env.cfg, 'episode_length_s', '?')}")

    # Check hover positions
    hover = getattr(root_env, "_passive_drone_hover_positions", None)
    if hover is not None:
        print(f"[probe]   passive_hover_positions shape: {hover.shape}")
        for env_idx in range(num_envs):
            for pid, pname in enumerate(passive_ids):
                h = hover[env_idx, pid].cpu()
                print(f"[probe]     env_{env_idx} {pname}: hover_setpoint=[{h[0]:+.2f},{h[1]:+.2f},{h[2]:.2f}]")

    # Check active drone mask
    mask_key = getattr(root_env.cfg, "active_agent_mask_key", "active_drones")
    mask = getattr(root_env, mask_key, None)
    if mask is not None:
        print(f"[probe]   {mask_key} mask shape: {mask.shape}")
        print(f"[probe]   {mask_key} env_0: {mask[0].int().tolist()}")

    # Check columns
    columns = getattr(root_env, "column_positions", None)
    if columns is not None:
        active_cols = (torch.linalg.norm(columns[:, :, :2], dim=-1) < 100.0).sum(dim=-1)
        print(f"[probe]   column_positions shape: {columns.shape}")
        for env_idx in range(num_envs):
            print(f"[probe]     env_{env_idx}: active_columns={int(active_cols[env_idx].item())}")

    # Check num_static_columns curriculum state
    nc = getattr(root_env, "_paper_swarm_num_static_columns", None)
    print(f"[probe]   _paper_swarm_num_static_columns: {nc}")

    # ------------------------------------------------------------------
    # Run steps and record state
    # ------------------------------------------------------------------
    print(f"\n[probe] Running {max_steps} steps (action = all zeros)...")
    # All-zero actions for each agent
    action_dim = root_env.action_spaces[possible[0]].shape[0]
    zero_action = torch.zeros(num_envs, action_dim, device=device)
    actions = {a: zero_action for a in possible}

    positions_log = []
    goals_log = []
    hover_setpoints_log = []
    obs_env = env.unwrapped

    for step in range(max_steps):
        # Record before stepping
        pos = torch.zeros(num_envs, num_drones, 3, device=device)
        goal = torch.zeros(num_envs, num_drones, 3, device=device)
        for di, drone_id in enumerate(all_drone_ids):
            pos[:, di, :] = obs_env.scene[drone_id].data.root_pos_w.torch
            origin = obs_env.scene.env_origins
            if drone_id in possible:
                bname = obs_env._agent_to_bundle.get(drone_id)
                if bname:
                    cmd = obs_env._manager_bundles[bname].command_manager.get_command("target_pose")
                    goal[:, di, :3] = cmd[:, :3] + origin
            elif hover is not None and drone_id in passive_ids:
                pi = passive_ids.index(drone_id)
                goal[:, di, :3] = hover[:, pi, :3] + origin
        positions_log.append(pos.cpu().numpy().copy())
        goals_log.append(goal.cpu().numpy().copy())
        if hover is not None:
            hover_setpoints_log.append(hover.cpu().numpy().copy())

        # Step with zero actions
        obs, rewards, terminated, truncated, infos = env.step(actions)
        # Reset if any env terminated
        if any(terminated.values()) or any(truncated.values()):
            env.reset()
            hover = getattr(root_env, "_passive_drone_hover_positions", None)

    env.close()

    # ------------------------------------------------------------------
    # Save snapshot
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        g = f.create_group("probe")
        g.create_dataset("drone_ids", data=np.array(all_drone_ids, dtype="S"))
        g.create_dataset("positions", data=np.stack(positions_log))
        g.create_dataset("goals", data=np.stack(goals_log))
        if hover_setpoints_log:
            g.create_dataset("hover_setpoints", data=np.stack(hover_setpoints_log))
        g.attrs["task_id"] = task_id
        g.attrs["num_envs"] = num_envs
        g.attrs["max_steps"] = max_steps

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    positions = np.stack(positions_log)
    goals = np.stack(goals_log)
    print(f"\n[probe] Snapshot saved to {hdf5_path}")
    print(f"[probe]   shape: {positions.shape} (steps, envs, drones, xyz)")

    for di, drone_id in enumerate(all_drone_ids):
        p = positions[0, 0, di, :]
        g = goals[0, 0, di, :]
        origin = np.array([0.0, 0.0, 0.0])  # env_0
        p_end = positions[-1, 0, di, :]
        drift = float(np.linalg.norm(p_end - p))
        dist_to_goal_0 = float(np.linalg.norm(p - g))
        dist_to_goal_end = float(np.linalg.norm(p_end - g))
        role = "active" if drone_id in possible else "passive"
        print(f"  {drone_id:>8s} ({role:>7s}): "
              f"start=[{p[0]:+.2f},{p[1]:+.2f},{p[2]:.2f}]  "
              f"goal=[{g[0]:+.2f},{g[1]:+.2f},{g[2]:.2f}]  "
              f"end_z={p_end[2]:.2f}  drift={drift:.3f}m  "
              f"dist_goal: {dist_to_goal_0:.2f}→{dist_to_goal_end:.2f}")


def main() -> None:
    # CLI must work with the Isaac Lab launcher — re-use play.py's argument pattern
    parser = argparse.ArgumentParser(description="Probe paper_swarm environment.")
    parser.add_argument("--task", required=True, help="Gym task ID.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", default="/tmp/isaaclab/logs/probe_snapshot.hdf5")
    args, _hydra = parser.parse_known_args()

    # The Isaac Lab launcher parses sys.argv, so we must set it up properly
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation

    sys.argv = [sys.argv[0]] + _hydra

    from environments import tasks as local_tasks

    local_tasks.register_tasks_for(args.task)
    entry = "skrl_mappo_cfg_entry_point" if "MAPPO" in args.task else "skrl_cfg_entry_point"
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, entry)
    env_cfg.scene.num_envs = args.num_envs

    with launch_simulation(env_cfg, args):
        probe(args.task, num_envs=args.num_envs, max_steps=args.max_steps, hdf5_path=args.output)


if __name__ == "__main__":
    main()
