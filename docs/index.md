# swarm-lab

`swarm-lab` is the Isaac Sim and Isaac Lab workspace for CPSquare swarm experiments and thesis work. It ties the reusable `cpsquare-lab` robotics library to task packages, training scripts, ROS 2 bridge processes, and documentation for the public repository.

The repository is intentionally split into two layers:

- `cpsquare-lab` provides robot embodiments, controllers, reusable task helpers, ROS 2 bridge schemas, and policy interfaces.
- `swarm-lab` provides executable task packages under `environments/`, training and play scripts under `scripts/`, and workspace-level dependency management.

## Developer Workflow

Use the root `pyproject.toml` and `uv.lock` as the workspace source of truth. The root environment installs `cpsquare-lab` from `../cpsquare-lab` in editable mode and installs this repository's `environments` workspace member.

Common entry points are:

- `scripts/skrl/train.py` and `scripts/skrl/play.py` for SKRL-based training and evaluation.
- `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` for RSL-RL workflows.
- `scripts/ros2/run_env.py` and `scripts/ros2/run_agent.py` for detached environment/agent bridge runs.
- `scripts/list_envs.py` for inspecting registered Isaac Lab environments.

Use the generated API Reference section for package and script symbols. MkDocs generates
that section from Google-style docstrings at build time, so public Python documentation
stays attached to the modules that define the API.

## Documentation

The documentation is designed for mkdocs and mkdocstrings. Narrative pages explain task structure and runtime workflows, while the API reference is generated from Google-style Python docstrings.

Run local checks before publishing documentation:

```bash
uv run ruff check environments/environments scripts
uv run --group docs mkdocs build --strict
```
