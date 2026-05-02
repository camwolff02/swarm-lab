# Quad swarm Isaac Lab convergence package

This bundle turns the current diagnosis into an implementation and validation handoff for an automated coding agent.

## Scope

The package assumes the codebase is split like this:

- `swarm-rl/` contains the task registration, environment, model wrappers, and skrl runner/config.
- `cpsquare-lab/` contains the Crazyflie parameters, multirotor helpers, action mapping, controllers, and swarm utility functions.

## What this package is trying to fix first

1. The policy starts too close to a below-hover operating point for the translated Crazyflie.
2. Replay activation is effectively blocked by a raw floor-contact counter.
3. The environment is missing a few low-risk stabilizers that make early PPO learning less brittle.
4. The current default validation loop does not expose the fastest signals that would prove the fixes are working.

## Contents

- `FILE_MAP.md` — likely repo-relative locations for the uploaded flat files.
- `agent_manifest.yaml` — machine-readable backlog for an implementation agent.
- `PATCH_RECIPE.md` — file-by-file change plan with concrete code-level guidance.
- `VALIDATION_PROTOCOL.md` — smoke tests and training checks.
- `INSTRUMENTATION_SPEC.md` — extra metrics and rollout diagnostics to add.
- `scripts/compute_hover_action.py` — standalone helper to compute hover ratio and action bias from a vehicle YAML file.

## Recommended execution order

1. Implement task `T01_hover_bias_and_exploration`.
2. Implement task `T02_replay_activation_fix`.
3. Implement task `T03_reward_and_observation_stabilizers`.
4. Implement task `T04_diagnostics_and_validation_hooks`.
5. Run the validation protocol before attempting longer training.

## Important non-goals

- Do **not** change `cpsquare-lab/.../action_mapping.py` first. The direct `[-1, 1] -> [0, max_thrust]` mapping is already the correct affine map for the current direct-rotor path.
- Do **not** re-enable `share_parameters: true` for stock skrl IPPO training unless a true shared-optimizer update path is implemented.

