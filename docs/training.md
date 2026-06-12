# Training

Training scripts are thin Isaac Lab launchers that keep task-specific behavior in task packages and agent configuration files.

## SKRL

Use `scripts/skrl/train.py` for training and `scripts/skrl/play.py` for evaluation. These scripts load Isaac Lab app-launcher arguments, resolve the registered task, and then defer to task-local SKRL config entries.

Current custom runner hooks are task-local:

- Formation swarm installs a model factory for `FormationAttentionEncoder` and `FormationSharedMAPPO`.
- Paper swarm installs a model factory for `PaperAttentionEncoder` and its shared-policy wrappers.

## RSL-RL

Use `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` for RSL-RL workflows. Shared command-line parsing lives in `scripts/rsl_rl/cli_args.py`.

## Experiment Data

Generated training output should stay in runtime output directories such as `logs/` and `outputs/`. Source modules and docs should describe expected metrics, dimensions, and constants rather than relying on generated artifacts.
