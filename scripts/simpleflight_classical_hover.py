# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the classical SimpleFlight environment with a simple PID hover controller."""

import argparse
import sys

from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_PARAMS
import environments.tasks  # noqa: F401
import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config
from environments.tasks.lab_2_classical_control.agents.hover_controller import HoverPidController

# add argparse arguments
parser = argparse.ArgumentParser(description="PID hover controller for the classical SimpleFlight environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SimpleFlight-Classical-Crazyflie-v0",
    help="Name of the task.",
)
parser.add_argument("--max_steps", type=int, default=600, help="Number of controller steps to run.")
parser.add_argument("--print_every", type=int, default=50, help="How often to print hover diagnostics.")
parser.add_argument(
    "--target_position",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override the hover target in the environment frame.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    print("[INFO]: Resolved task config.", flush=True)

    with launch_simulation(env_cfg, args_cli):
        print("[INFO]: Simulation app launched.", flush=True)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        print("[INFO]: Creating environment...", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        print("[INFO]: Environment created.", flush=True)

        if args_cli.target_position is not None:
            base_env.set_reference_origin(args_cli.target_position)

        action_term = base_env.action_manager.get_term("flight_command")
        robot = base_env.scene["robot"]
        body_masses = wp.to_torch(robot.data.body_mass).sum(dim=1)
        max_total_thrust = CRAZYFLIE_PARAMS.max_thrusts[0] * CRAZYFLIE_PARAMS.num_rotors
        print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
        print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
        print(f"[INFO]: Observation groups: {base_env.observation_manager.group_obs_dim}", flush=True)
        print(f"[INFO]: Reference origin: {base_env.reference_origin.tolist()}", flush=True)
        print(f"[INFO]: Simulated total body mass: {body_masses.mean().item():.6f} kg", flush=True)
        sim_hover_ratio = body_masses * 9.81 / max_total_thrust
        print(f"[INFO]: Simulated hover thrust ratio: {sim_hover_ratio.mean().item():.4f}", flush=True)
        print(
            f"[INFO]: Param hover thrust ratio: {CRAZYFLIE_PARAMS.hover_thrust / CRAZYFLIE_PARAMS.max_thrusts[0]:.4f}",
            flush=True,
        )

        print("[INFO]: Resetting environment...", flush=True)
        env.reset()
        print("[INFO]: Environment reset complete.", flush=True)

        controller = HoverPidController(
            num_envs=base_env.num_envs,
            device=base_env.device,
            dt=base_env.step_dt,
            hover_ratio=sim_hover_ratio,
        )
        controller.reset()
        print("[INFO]: Controller initialized.", flush=True)

        diagnostics = None
        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                control_state = base_env.get_control_state()
                actions, diagnostics = controller.compute_actions(control_state)
                env.step(actions)

            if step == 0 or (step + 1) % args_cli.print_every == 0 or step + 1 == args_cli.max_steps:
                current_state = base_env.get_control_state()
                position_error = current_state["target_position"] - current_state["root_position"]
                mean_position = current_state["root_position"].mean(dim=0)
                mean_error = position_error.abs().mean(dim=0)
                max_error = position_error.norm(dim=-1).max()
                mean_target_thrust_ratio = action_term.target_thrust_ratio.mean()
                mean_applied_thrust_ratio = ((action_term.processed_actions + 1.0) * 0.5).mean()
                sim_applied_thrust_ratio = robot.data.applied_thrust.sum(dim=1).mean() / max_total_thrust
                print(
                    f"[STEP {step + 1:04d}] "
                    f"mean_position={mean_position.tolist()} "
                    f"mean_abs_error={mean_error.tolist()} "
                    f"max_norm_error={max_error.item():.4f} "
                    f"target_thrust_ratio={mean_target_thrust_ratio.item():.4f} "
                    f"mixer_thrust_ratio={mean_applied_thrust_ratio.item():.4f} "
                    f"sim_thrust_ratio={sim_applied_thrust_ratio.item():.4f}",
                    flush=True,
                )

        if diagnostics is not None:
            final_state = base_env.get_control_state()
            final_error = final_state["target_position"] - final_state["root_position"]
            final_mean_error = final_error.abs().mean(dim=0)
            final_max_error = final_error.norm(dim=-1).max()
            final_applied_thrust_ratio = ((action_term.processed_actions + 1.0) * 0.5).mean()
            final_sim_thrust_ratio = robot.data.applied_thrust.sum(dim=1).mean() / max_total_thrust
            print(
                "[RESULT] "
                f"final_mean_abs_error={final_mean_error.tolist()} "
                f"final_max_norm_error={final_max_error.item():.4f} "
                f"commanded_thrust_ratio={diagnostics['thrust_ratio'].mean().item():.4f} "
                f"mixer_thrust_ratio={final_applied_thrust_ratio.item():.4f} "
                f"sim_thrust_ratio={final_sim_thrust_ratio.item():.4f}",
                flush=True,
            )

        env.close()
        print("[INFO]: Environment closed.", flush=True)


if __name__ == "__main__":
    main()
