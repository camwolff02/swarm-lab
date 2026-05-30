# Known Issues — Paper Swarm Curriculum

Last updated: 2026-05-30

---

## Stage 1 — Passive Drone Drift

**Severity**: Medium
**Status**: Open

Passive hovering drones (drones 1-7 in Stage 1 eval) drift significantly — 5-6m horizontal + z=2.0→1.0 over 10s. Lee position controllers (`LeePosController` with gains `K_pos=[3,3,2]`, `K_vel=[2.5,2.5,1.5]`) appear to lose altitude authority or receive incorrect setpoints when running alongside the RL-controlled drone.

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

When evaluated without curriculum (`curriculum=None`), `drone_0` spawns at column-grid-like positions (e.g. `[4, -4, 1]`) instead of `[0, 0, 1]`. Caused by `active_drones` mask not being initialised before the first reset event fires.

### Impact
- Cosmetic for evaluation (drone still reaches goals) but incorrect spawn may mask out-of-bounds detection

---

## Stage 2 — Zero Gradient Updates (Critical)

**Severity**: Critical
**Status**: Root cause unidentified — see analysis below

Two Stage 2 training runs (`paper_swarm_train_stage2` and `paper_swarm_train_stage2_v2`) both produced bit-identical policy weights to the Stage 1 checkpoint. No gradient updates occurred during training.

### Run comparison

| Metric | Stage 2 v1 (no clip_log_std) | Stage 2 v2 (with clip_log_std) |
|---|---|---|
| Policy weights vs Stage 1 | Identical | Identical |
| log_std | -2.68 (from S1) | -2.68 (from S1) |
| All losses | 0.0 (all 100 events) | 0.0 (all 100 events) |
| Std at step 3000 | 0.073 | 0.073 |
| Std at step 300000 | 0.073 | 0.073 |
| LR at step 3000 | floor (5e-5) | floor (5e-5) |
| LR at step 300000 | floor (5e-5) | floor (5e-5) |

### Previous incorrect diagnosis

Earlier analysis blamed missing `clip_log_std` as the root cause of Stage 2 collapse. This was wrong:
- `clip_log_std` defaults to `False` when absent, and `min_log_std = -20.0`
- But Stage 1's checkpoint already had log_std ≈ -2.6, within bounds
- Since zero gradient updates occurred, log_std never changed regardless of clip setting
- Both runs (with and without clip_log_std) are identical

### Observed behavior
- Physical: one drone (drone_0) flies toward goals, others fly up and away
- TensorBoard: all losses zero, std frozen at 0.073, LR at floor from first logged step
- `best_agent.pt` is bit-identical to Stage 1 checkpoint weights

### Known facts
- Checkpoint loading uses shared policy/value objects (runner.py:68-73)
- Stage 1 checkpoint (`drone_0` only) populates all 8 agents via object identity
- `--reset_optimizer_on_resume` was passed (confirmed by reset optimizer step)
- The training loop runs (env stepping time 58ms, algorithm update 139ms, checkpoint files every 30k steps)
- The PPO update produces zero for all three loss components from step 3000 onward

### Hypothesis for zero gradients
The PPO value loss is `MSE(returns, predicted_values)`. Advantages use GAE. For all components to be zero, the value prediction must exactly match the GAE-smoothed returns. This could indicate:

1. **Value function doesn't update**: The KLAdaptiveLR drops LR to floor immediately (first KL check exceeds threshold), subsequent gradient steps are too small to change weights
2. **Preprocessor mismatch**: RunningStandardScaler statistics from Stage 1 don't match Stage 2 state distribution, producing normalized observations that lead to constant value outputs
3. **Observation distribution shift**: The attention-based policy receives multi-agent observations (neighbors, SDF) that it was never trained on → produces near-constant actions → collected returns are predictable → value network's Stage 1 weights happen to predict them

### Next steps
1. Run a forward pass diagnostic: load model + dummy Stage 2 state → check if value output varies or is constant
2. Check if the KLAdaptiveLR fires on the very first update: why does KL exceed threshold immediately when weights haven't changed?
3. Consider fixing the LR scheduling (higher min_lr, warmup, or fixed LR for initial phase)
4. Consider resetting log_std at stage transition (re-initialize to higher value)

---

## Stage 2 — clip_log_std Fix (Not Root Cause, Still Correct)

**Severity**: Low (already fixed)
**Status**: Applied but insufficient

The missing `clip_log_std` in Stage 2/3 configs was identified and fixed. However, the Stage 2 collapse predates this fix — log_std was already at -2.6 from Stage 1, and zero gradient updates kept it there. The fix is still correct as a guard against future unbounded variance drift, but it does not address the primary failure mode.

See detailed stage transition analysis in `.co-roboticist/stage2-training-analysis.md`.

---

## Stage 3 — Downstream of Stage 2

**Severity**: Critical
**Status**: Blocked on Stage 2 fix

Stage 3 was never reached with a healthy checkpoint. Both Stage 3 runs inherited the collapsed policy from Stage 2. Stage 3 eval metrics (11% goal success, 44% crash rate) are identical to Stage 2 eval because the policy weights are the same.
