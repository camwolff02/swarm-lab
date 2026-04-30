#!/usr/bin/env python3
"""Compute hover ratio and hover action from an OmniDrones-style multirotor YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def compute_metrics(param_path: Path) -> dict[str, float | int | list[float]]:
    with param_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    rotor_cfg = raw["rotor_configuration"]
    num_rotors = int(rotor_cfg["num_rotors"])
    mass = float(raw["mass"])
    gravity = 9.81
    hover_thrust = mass * gravity / num_rotors

    force_constants = [float(value) for value in rotor_cfg["force_constants"]]
    max_rotation_velocities = [float(value) for value in rotor_cfg["max_rotation_velocities"]]
    max_thrusts = [k_f * (omega_max ** 2) for k_f, omega_max in zip(force_constants, max_rotation_velocities, strict=True)]
    hover_ratios = [hover_thrust / max_thrust for max_thrust in max_thrusts]
    hover_actions = [2.0 * ratio - 1.0 for ratio in hover_ratios]
    total_thrust_to_weight = sum(max_thrusts) / (mass * gravity)

    return {
        "name": str(raw.get("name", param_path.stem)),
        "mass": mass,
        "num_rotors": num_rotors,
        "hover_thrust_per_rotor": hover_thrust,
        "max_thrusts": max_thrusts,
        "hover_ratios": hover_ratios,
        "hover_actions": hover_actions,
        "mean_hover_ratio": sum(hover_ratios) / len(hover_ratios),
        "mean_hover_action": sum(hover_actions) / len(hover_actions),
        "total_thrust_to_weight": total_thrust_to_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file", type=Path, help="Path to cf2x.yaml or another OmniDrones-style multirotor YAML")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON instead of pretty text")
    args = parser.parse_args()

    metrics = compute_metrics(args.param_file)
    if args.as_json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    print(f"vehicle: {metrics['name']}")
    print(f"mass: {metrics['mass']:.6f} kg")
    print(f"num_rotors: {metrics['num_rotors']}")
    print(f"hover_thrust_per_rotor: {metrics['hover_thrust_per_rotor']:.6f} N")
    print("max_thrusts:", ", ".join(f"{value:.6f}" for value in metrics["max_thrusts"]))
    print("hover_ratios:", ", ".join(f"{value:.6f}" for value in metrics["hover_ratios"]))
    print("hover_actions:", ", ".join(f"{value:.6f}" for value in metrics["hover_actions"]))
    print(f"mean_hover_ratio: {metrics['mean_hover_ratio']:.6f}")
    print(f"mean_hover_action: {metrics['mean_hover_action']:.6f}")
    print(f"total_thrust_to_weight: {metrics['total_thrust_to_weight']:.6f}")


if __name__ == "__main__":
    main()
