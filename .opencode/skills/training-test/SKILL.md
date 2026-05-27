---
name: training-test
description: Run a quick paper_swarm training smoke test. Use when user wants to verify training starts, test env changes, or run a short training iteration before a full run.
---

# Training Test

Run a quick smoke test to verify the environment and training pipeline work before starting a full run.

## Quick test (3 rollouts, ~2 min)

Replace `TASK_ID` with the target stage:

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 120 uv run scripts/skrl/train.py --algorithm MAPPO --task TASK_ID --max_iterations 3 --headless 2>&1 | tail -5
```

**Valid task IDs:**
- `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-v0`
- `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage2-v0`
- `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage3-v0`
- `Isaac-Paper-Swarm-Waypoint-Eval-v0`

## Check throughput

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 90 uv run scripts/skrl/train.py --algorithm MAPPO --task TASK_ID --max_iterations 3 --headless 2>&1 | grep "it/s|100%" | tail -3
```

## Full pipeline verification

Before a long training run, verify:
1. Config loads: `uv run python -c "from environments import tasks as local_tasks; from isaaclab_tasks.utils import resolve_task_config; local_tasks.register_tasks_for('TASK_ID'); ec, tc = resolve_task_config('TASK_ID', 'skrl_mappo_cfg_entry_point'); print('OK')"`
2. Pre-commit passes: `cd /home/cam/Development/cpsquare/IsaacLab && ./isaaclab.sh -f`
3. Smoke test runs clean (above)

## Run pre-commit

```bash
cd /home/cam/Development/cpsquare/IsaacLab && ./isaaclab.sh -f 2>&1 | tail -5
```

## Common issues and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: Expected observation dim 86, got 232` | Policy receives critic state instead of observations | Check `_states()` in skrl_models.py prefers `"observations"` |
| `ep_len = 1.0` for all episodes | Termination fires immediately on reset | Check termination thresholds vs reset ranges |
| `env stepping > 500ms` | Episodes resetting every step | Same as above |
| `Specified keys (set())` | Empty dict in SKRL config | Add `_kwarg: null` entries or populated defaults |
| `NameError: name 'positions' is not defined` | Stray return statement in event function | Check events.py for orphaned returns |
