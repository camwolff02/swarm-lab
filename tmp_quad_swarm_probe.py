"""Temporary render/scene probe for the quad swarm paper task."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

import environments  # noqa: F401

with contextlib.suppress(ImportError):
    import environments.tasks  # noqa: F401


parser = argparse.ArgumentParser(description="Probe quad swarm task scene and render output.")
parser.add_argument("--task", type=str, default="Isaac-Quad-Swarm-Paper-Crazyflie-v0")
parser.add_argument("--agent", type=str, default="skrl_ippo_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--frames", type=int, default=12)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
RESULT_PATH = Path(__file__).resolve().parent / "tmp_quad_swarm_probe_result.txt"


def _count_meshes(prim):
    count = 0
    for child in prim.GetAllChildren():
        if child.GetTypeName() == "Mesh":
            count += 1
        count += _count_meshes(child)
    return count


def main() -> None:
    lines: list[str] = []

    def emit(message: str) -> None:
        print(message)
        lines.append(message)

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        unwrapped = env.unwrapped
        emit(f"viewer.cam_prim_path={unwrapped.cfg.viewer.cam_prim_path}")
        stage = unwrapped.sim.stage
        camera = stage.GetPrimAtPath(unwrapped.cfg.viewer.cam_prim_path)
        emit(f"camera.valid={camera.IsValid()} type={camera.GetTypeName() if camera.IsValid() else None}")
        for index in range(unwrapped.cfg.num_drones):
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/drone_{index}")
            emit(
                f"drone_{index}.valid={prim.IsValid()} type={prim.GetTypeName() if prim.IsValid() else None} "
                f"meshes={_count_meshes(prim) if prim.IsValid() else 0}"
            )
        obs, _ = env.reset()
        emit(f"reset.agents={sorted(obs.keys()) if isinstance(obs, dict) else type(obs)}")
        state = unwrapped._collect_swarm_state()
        emit(f"positions.env0={state['positions'][0].detach().cpu().numpy().round(3).tolist()}")
        emit(f"obstacles.active.env0={int(unwrapped._obstacle_mask[0].sum().detach().cpu())}")
        for frame in range(args_cli.frames):
            image = env.render()
            if image is None:
                emit(f"frame {frame}: none")
                continue
            arr = np.asarray(image)
            emit(
                f"frame {frame}: shape={arr.shape} min={int(arr.min())} max={int(arr.max())} "
                f"mean={float(arr.mean()):.3f} nonzero={int(np.count_nonzero(arr))}"
            )
        RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        env.close()


if __name__ == "__main__":
    main()
