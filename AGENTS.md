# IsaacLab Guidelines

## API design rules (naming + structure)

- **Group by common prefix for discoverability (autocomplete).**
  - **Classes**: group by domain concept — `ActuatorNetLSTM`, `ActuatorNetMLP` (not `LSTMActuatorNet`, `MLPActuatorNet`).
  - **Methods**: group by noun before modifier — `set_joint_position_target()` (not `set_target_joint_position()`).
- **Method names are `snake_case`.**
- **CLI arguments are `snake_case`.**
- **Prefer nested classes when self-contained.**
- **Follow PEP 8.**
- **Use modern Python type-hint syntax.**
  - Prefer PEP 604 unions: `x | y`, `x | None`. Do not use `typing.Union` or `typing.Optional`.
- **Use Google-style docstrings.**
  - Keep argument/return types in function annotations, not inline in docstrings.
  - In `Args:` entries, use `name: description` (not `name (Type): description`).
- **State SI units for all physical quantities in docstrings.**
  - Use inline `[unit]` notation, e.g. `"""Particle positions [m], shape [particle_count, 3], float."""`.

## Dependencies

- **Avoid adding new required dependencies.** IsaacLab's core should remain lightweight.
- **Strongly prefer not adding new optional dependencies.** Use existing deps (Warp, NumPy, stdlib) when possible.

## Tooling: prefer `uv run` for running, testing, and benchmarking

## File headers and copyright

- New files must use the current year (2026) in the SPDX copyright header:

  ```
  # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```

- Do not change the year in existing file headers.

## Testing Guidelines

- **Always verify regression tests fail without the fix.** Temporarily revert the fix and run the test to confirm it fails, then reapply.

## Debugging Warp kernels

**Do not add `wp.printf` to kernels in production code.** Debug prints affect performance and can produce noisy test output.

To debug Warp kernel behavior:
1. Write a standalone reproduction script and run with `uv run python -c "..."` or `uv run script.py`.
2. Use high-precision format strings (e.g., `wp.printf("val=%.15e\n", x)`).
3. Remove all `wp.printf` calls before committing.

# Swarm-lab Guidelines

Swarm lab guidelines should always be superceded by IsaacLab guidelines.

## Architecture

- **IsaacLab is vendored** at `../IsaacLab` (branch `release/3.0.0-beta2`). **Never modify it** to fix this repo.
- **`cpsquare-lab` is an editable dependency** at `../cpsquare-lab` (see `pyproject.toml` `[tool.uv.sources]`).
- **`environments/`** is a nested Python package with its own `pyproject.toml`. It registers Gym environments and tasks.
- Python 3.12 only (pinned in `.python-version` and `pyproject.toml`).

## Commands

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Typecheck
uv run pyright

# Pre-commit (runs from IsaacLab, not this repo)
cd ../IsaacLab && ./isaaclab.sh -f

# Run all tests
uv run pytest test/

# Run a single test file
uv run pytest test/test_formation_swarm_task.py

# Run a single test function
uv run pytest test/test_formation_swarm_task.py::test_formation_laplacian_cost_is_zero_for_target_formation

# Docs (live-reload)
uv run --group docs mkdocs serve

# Docs (build)
uv run --group docs mkdocs build --strict
```

## Test directory: `test/`, NOT `tests/`

Tests live in `test/` (singular). The `tests/` directory is **gitignored** and contains old XML test reports — do not put test files there.

## Training

Training uses SKRL with MAPPO via `scripts/skrl/train.py`. The `justfile` has curated recipes:

```bash
# Single stage
just formation-train-stage 1

# Resume from checkpoint
just formation-train-stage 2 checkpoint=/path/to/agent_*.pt

# Full 3-stage curriculum
just formation-curriculum

# Play (evaluate) a trained model
just formation-play
```

Direct invocation:

```bash
uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Formation-Swarm-Crazyflie-v0 env.curriculum_stage=1
uv run scripts/skrl/play.py --algorithm MAPPO --task Isaac-Formation-Swarm-Crazyflie-v0 --checkpoint /path/to/agent_*.pt
```

### Paper Swarm training stages

```bash
# Stage 1 — single drone waypoint control
uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-v0

# Stage 2 — resume from Stage 1 checkpoint
uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage2-v0 \
  --checkpoint logs/skrl/paper_swarm/mappo_stage1/<run>/checkpoints/agent_*.pt \
  --reset_optimizer_on_resume

# Stage 3 — resume from Stage 2 checkpoint
uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage3-v0 \
  --checkpoint logs/skrl/paper_swarm/mappo_stage2/<run>/checkpoints/agent_*.pt \
  --reset_optimizer_on_resume
```

## Reviewing Training Progress

Whenever the user asks to review a training run, follow this procedure:

1. **Read TensorBoard logs** — use the `training-metrics` skill to pull up episode length, reward, policy std, value loss, learning rate, and env stepping time. Look for NaN, diverging std, flat reward, or collapsing episode lengths.

2. **Run an eval with the latest checkpoint** — use the Eval environment (`Isaac-Paper-Swarm-Waypoint-Eval-v0` or `Isaac-Paper-Swarm-Waypoint-MAPPO-Eval-v0`) with the recorder enabled. This writes HDF5 data to `/tmp/isaaclab/logs/paper_swarm_dataset.hdf5`. The recorder captures per-step drone positions, quaternions, velocities, goal distances, and the `InitialStateCheckRecorder` validates upright/spacing/bounds on first step.

3. **Inspect the HDF5 recorder output** — open the dataset and check:
   - `record_drone_state/step_*`: drone positions, quaternions, velocities over time (does the drone fly toward its target or just go up?)
   - `record_goal_distance/step_*`: goal positions and distances (are waypoints being reached?)
   - `check_initial_state/step_0`: all_upright, all_in_bounds, all_separated, inactive_parked (did any drone start in a bad state?)

4. **Cross-reference with the TensorBoard metrics** — correlate physical behavior from HDF5 with reward/loss curves. If rewards are flat but the drone is reaching targets, the reward computation may be buggy. If episode lengths are 1.0, a termination is firing immediately.

Task IDs are registered by importing `environments.tasks` (the `environments` package's `__init__.py` does `from .tasks import *`).

## Repository Rules

- Robot-specific assets belong under `../cpsquare-lab/src/cpsquare_lab/embodiments/`.
- Shared robot-generic code belongs under `../cpsquare-lab/src/cpsquare_lab/embodiments/common/`.
- Shared multirotor code belongs under `../cpsquare-lab/src/cpsquare_lab/embodiments/multirotor/common/`.
- Robot-specific logic belongs in that robot's embodiment subtree.
- Do not create or restore a top-level `../cpsquare-lab/src/cpsquare_lab/mdp`, `assets`, or `utils` package.
- Action terms belong with embodiments. Observations belong with the embodiment or task. Rewards belong with the task.
- Task-specific geometry/grid helpers belong with the relevant task, not in a shared `utils` package.

## Working Conventions

- Prefer fixing imports and call sites to match the current structure instead of adding compatibility packages.
- Keep public imports shallow and explicit. Avoid implicit cross-package magic.
- When moving files, update tests to the new module paths.
- Keep lint and test configuration scoped to this repository; vendored code is not collected.
- Before adding abstractions, check whether the code can reuse an existing embodiment-common or task-local module.
- Do not overwrite unrelated user changes in the worktree.

## Code Style

- Ruff config: line-length 120, Python 3.12 target, Google-style docstrings, isort with IsaacLab-aware section ordering.
- Pyright config: `typeCheckingMode = "basic"`, includes only `environments/`.
