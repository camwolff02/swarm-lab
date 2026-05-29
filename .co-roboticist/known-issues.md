# Known Issues — Paper Swarm Curriculum

Last updated: 2026-05-29

---

## Stage 1 — Passive Drone Drift

**Severity**: Medium
**Status**: Open

Passive hovering drones (drones 1-7 in Stage 1 eval) drift significantly — 5-6m horizontal + z=2.0→1.0 over 10s.  Lee position controllers (`LeePosController` with gains `K_pos=[3,3,2]`, `K_vel=[2.5,2.5,1.5]`) appear to lose altitude authority or receive incorrect setpoints when running alongside the RL-controlled drone.

### Observations
- HDF5 eval from `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-Eval-v0` (2026-05-29)
- Passive drones: start at z=2.0m, end at z≈1.0m, horizontal drift 5-6m
- Active drone_0: reaches goals successfully despite drift

### Impact
- Neighbor-attention stream sees moving instead of hovering neighbors — may impair collision avoidance learning
- Does not affect active drone's own waypoint navigation

### Next step
Check whether `_apply_passive_drone_control` receives correct `_passive_drone_hover_positions` setpoints during eval (vs training where curriculum sets them).

---

## Stage 1 — Eval Spawn Position Offset

**Severity**: Low
**Status**: Open

When evaluated without curriculum (`curriculum=None`), `drone_0` spawns at column-grid-like positions (e.g. `[4, -4, 1]`) instead of `[0, 0, 1]`.  Caused by `active_drones` mask not being initialised before the first reset event fires.

### Impact
- Cosmetic for evaluation (drone still reaches goals) but incorrect spawn may mask out-of-bounds detection

---

## Stage 3 — Policy Collapse / Training Failure

**Severity**: Critical
**Status**: Root cause found in Stage 2 (see below)

Stage 3 (8-agent MARL with dense obstacles, 400k steps) collapsed.  However, investigation of Stage 2 revealed the collapse **originated in Stage 2**, not Stage 3.  Stage 3 inherited a dead policy from Stage 2.

### Metrics (Stage 3 eval of `best_agent.pt`, 18 episodes)
| Metric | Value |
|---|---|
| Episodes with any goal reached (<0.35m) | ≤ 2 / 18 (≤11%) |
| Episodes with crash (z < 0.3m) | 8 / 18 (44%) |
| Mean episode length | 115 steps / 2.3s (max 500 / 10s) |
| Mean reward per step | −0.078 |

### Training metrics (final)
| Metric | Value |
|---|---|
| Mean total reward | −49.1 (worsening) |
| Mean episode length | 124 / 1000 |
| Learning rate | 5e-5 (floor) |
| Policy std | 0.073 (collapsed) |
| Entropy loss | 0.0 |
| Value/Policy loss | 0.0 |

---

## Stage 2 — Policy Collapse Root Cause

**Severity**: Critical
**Status**: Root cause identified — missing `clip_log_std` in config

The policy collapsed **during Stage 2 training**, not Stage 3.  Eval metrics are identical to Stage 3 (same 11% goal success, 44% crash rate, −0.078 reward/step, 112 terminations in 520 steps).  Stage 3 was doomed from the start.

### Root cause

The Stage 2 config (`skrl_mappo_stage2_cfg.yaml`) dropped `clip_log_std`, `min_log_std`, and `max_log_std` from the model config.  Stage 1 had:

```yaml
clip_log_std: true
min_log_std: -3.0
max_log_std: -0.5
```

These bounds kept std ∈ [0.05, 0.61].  Stage 1 finished with std ≈ 0.17 (healthy).

Stage 2 config omitted all three keys:
```yaml
models:
  factory: paper_swarm_attention
  ...
  initial_log_std: 0.0
  # MISSING: clip_log_std, min_log_std, max_log_std
```

In `_generate_models` (runner.py), the defaults are:
```python
clip_log_std = bool(model_cfg.get("clip_log_std", False))  # False!
min_log_std  = float(model_cfg.get("min_log_std", -20.0))   # exp(-20) ≈ 2e-9!
max_log_std  = float(model_cfg.get("max_log_std", 2.0))
```

With no clipping, the PPO update could freely drive log_std down to −2.62 (std = 0.073) and beyond.  All losses flattened to zero.  KLAdaptiveLR drove LR to its floor (5e-5).  Training stopped learning.

The Stage 3 config (`skrl_mappo_stage3_cfg.yaml`) inherits the same missing bounds, so even if Stage 2 had survived, Stage 3 would have been vulnerable.

### Fix
Add to `skrl_mappo_stage2_cfg.yaml` and `skrl_mappo_stage3_cfg.yaml`:
```yaml
  clip_log_std: true
  min_log_std: -3.0
  max_log_std: -0.5
```

### Training metrics (Stage 2, final vs early)
| Metric | Early | Final |
|---|---|---|
| Mean total reward | −28.2 | −50.1 |
| Mean episode length | 125.8 | 131.8 |
| Policy std | 0.073 | 0.073 |
| Entropy loss | 0.0 | 0.0 |
| Learning rate | floor | floor |
