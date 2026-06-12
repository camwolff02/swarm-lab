#!/usr/bin/env python
"""Analyze paper_swarm HDF5 evaluation logs produced by the recorder manager.

Usage:
    uv run scripts/analysis/analyze_paper_swarm_eval.py [--file /tmp/isaaclab/logs/paper_swarm_dataset.hdf5]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import h5py
except ImportError:
    print("h5py not available — install with: uv pip install h5py", file=sys.stderr)
    sys.exit(1)

DEFAULT_HDF5 = "/tmp/isaaclab/logs/paper_swarm_dataset.hdf5"
TARGET_REACHED_DIST = 0.35

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_origin(demo_idx: int, env_spacing: float = 8.0) -> np.ndarray:
    """Return env origin for a demo index (assumes sequential env assignment)."""
    return np.array([demo_idx % 4 * env_spacing, 0.0, 0.0], dtype=np.float32)


@dataclass
class EvalSummary:
    """Aggregate metrics across all evaluation episodes."""

    num_episodes: int = 0
    total_steps: int = 0
    max_step_dt: float = 0.02  # decimation=2, dt=0.01

    episodes_with_goal_reached: int = 0
    goal_reached_steps: list[int] = field(default_factory=list)
    mean_goal_dist_end: float = 0.0
    mean_goal_dist_start: float = 0.0
    crash_count: int = 0  # episodes where z < 0.3 for any drone

    def pct_goal_reached(self) -> float:
        return self.episodes_with_goal_reached / max(1, self.num_episodes) * 100

    def mean_episode_len_s(self) -> float:
        steps = self.total_steps / max(1, self.num_episodes)
        return steps * self.max_step_dt


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_hdf5(hdf5_path: str) -> EvalSummary:
    f = h5py.File(hdf5_path, "r")

    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    summary = EvalSummary(num_episodes=len(demos))

    print(f"{'Demo':>6}  {'Steps':>5}  {'Start dist':>10}  {'End dist':>10}  "
          f"{'Min dist':>10}  {'Drone hit':>10}  {'Any crash':>7}")

    for demo_key in demos:
        ds = f[f"data/{demo_key}/drone_state/positions"]
        gd = f[f"data/{demo_key}/goal/goal_distances"]
        steps = ds.shape[0]
        summary.total_steps += steps

        any_reached = False
        any_crash = False
        min_end_dist = float("inf")
        reached_at = -1
        crash_drone = -1

        for drone_idx in range(8):
            dists = gd[:, drone_idx]
            zs = ds[:, drone_idx, 2]

            if dists[-1] < TARGET_REACHED_DIST:
                any_reached = True
                if reached_at < 0:
                    reached = (dists < TARGET_REACHED_DIST).nonzero()[0]
                    reached_at = int(reached[0]) if len(reached) > 0 else -1

            if zs.min() < 0.3 and crash_drone < 0:
                crash_drone = drone_idx
                any_crash = True

            min_end_dist = min(min_end_dist, float(dists[-1]))

        if any_reached:
            summary.episodes_with_goal_reached += 1
            summary.goal_reached_steps.append(reached_at)
        if any_crash:
            summary.crash_count += 1

        print(f"{demo_key:>6}  {steps:>5}  {float(gd[0, :].mean()):>10.2f}  "
              f"{float(gd[-1, :].mean()):>10.2f}  {float(gd[:, :].min()):>10.2f}  "
              f"{crash_drone if any_crash else '--':>10}  {'YES' if any_crash else 'no':>7}")

    # Aggregate
    all_end_dists = []
    all_start_dists = []
    for demo_key in demos:
        gd = f[f"data/{demo_key}/goal/goal_distances"]
        all_start_dists.append(float(gd[0, :].mean()))
        all_end_dists.append(float(gd[-1, :].mean()))

    summary.mean_goal_dist_start = np.mean(all_start_dists)
    summary.mean_goal_dist_end = np.mean(all_end_dists)

    f.close()
    return summary


# ---------------------------------------------------------------------------
# Per-drone trajectory dump (table)
# ---------------------------------------------------------------------------


def print_drone_trajectories(hdf5_path: str, demo_key: str = "demo_0") -> None:
    f = h5py.File(hdf5_path, "r")
    ds = f[f"data/{demo_key}/drone_state/positions"]
    gd = f[f"data/{demo_key}/goal/goal_distances"]
    gpos = f[f"data/{demo_key}/goal/goal_positions"]
    steps = ds.shape[0]

    demo_idx = int(demo_key.split("_")[1])
    origin = _env_origin(demo_idx)

    print(f"\n=== {demo_key} ({steps} steps) ===")
    for drone_idx in range(8):
        p0 = ds[0, drone_idx, :3] - origin
        pend = ds[-1, drone_idx, :3] - origin
        goal = gpos[0, drone_idx, :3] - origin
        g0 = gd[0, drone_idx]
        gend = gd[-1, drone_idx]
        gmin = gd[:, drone_idx].min()
        reached = (gd[:, drone_idx] < TARGET_REACHED_DIST).sum()
        zs = ds[:, drone_idx, 2]
        print(f"  drone_{drone_idx}: "
              f"local_start=[{p0[0]:+.2f},{p0[1]:+.2f},{p0[2]:.2f}]  "
              f"goal=[{goal[0]:+.2f},{goal[1]:+.2f},{goal[2]:.2f}]  "
              f"end=[{pend[0]:+.2f},{pend[1]:+.2f},{pend[2]:.2f}]  "
              f"dist={g0:.2f}→{gend:.2f}(min={gmin:.2f})  "
              f"z={zs[0]:.2f}→{zs[-1]:.2f}  "
              f"reached={reached}/{steps}")
    f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paper_swarm HDF5 eval logs.")
    parser.add_argument("--file", default=DEFAULT_HDF5, help="Path to HDF5 dataset.")
    parser.add_argument("--demo", default=None, help="Print detailed per-drone trajectory for a specific demo.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        print_drone_trajectories(args.file, args.demo)
        return

    summary = analyze_hdf5(args.file)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes:                           {summary.num_episodes}")
    print(f"  Total steps:                        {summary.total_steps}")
    print(f"  Mean episode length:                {summary.total_steps / max(1, summary.num_episodes):.1f} steps "
          f"({summary.mean_episode_len_s():.1f}s)")
    print(f"  Mean goal dist (start):             {summary.mean_goal_dist_start:.2f} m")
    print(f"  Mean goal dist (end):               {summary.mean_goal_dist_end:.2f} m")
    print(f"  Episodes with goal reached:         {summary.episodes_with_goal_reached} / {summary.num_episodes} "
          f"({summary.pct_goal_reached():.0f}%)")
    if summary.goal_reached_steps:
        print(f"  Mean steps to reach:                {np.mean(summary.goal_reached_steps):.0f}")
    print(f"  Episodes with crash (z < 0.3m):     {summary.crash_count} / {summary.num_episodes}")

    # Verdict
    if summary.pct_goal_reached() >= 75 and summary.crash_count <= summary.num_episodes * 0.1:
        print("\nVERDICT: PASS — most drones reached their goals, few crashes.")
    elif summary.pct_goal_reached() >= 50:
        print("\nVERDICT: MARGINAL — some goals reached but improvement needed.")
    else:
        print("\nVERDICT: FAIL — goal success rate is too low or crash rate too high.")


if __name__ == "__main__":
    main()
