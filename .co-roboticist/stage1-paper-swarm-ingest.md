# Stage 1: Active Task Source Ingest — paper_swarm

Generated: 2026-05-26

---

## 1. Architecture Summary

### 1.1 Task: Paper Swarm Waypoint Navigation
Hybrid of two papers: ICRA 2024 collision avoidance + Xie et al. formation control.

**Core params** (from `paper_swarm_env_cfg.py`):
| Param | Value |
|-------|-------|
| Drones | 8 Crazyflie (CTBR-controlled) |
| Workspace | XY: ±4m, Z: 1–3m |
| Episode length | 20s (train), 10s (eval) |
| Sim dt | 0.01s |
| Policy decimation | 2 (50Hz control) |
| Action space | 4D `[c, ωx, ωy, ωz]` ∈ [-1,1], mapped to thrust |
| Replay | Enabled, 75% probability, 1.5s lag, 1.5s collision grace |

### 1.2 Three-Stage Curriculum

| Stage | Active Drones | Obstacles | Sampling | Target Separation | Steps |
|-------|-------------|-----------|----------|--------------------|-------|
| **1** | 1 (drone_0 only) | None | Fixed origin → expand | Safe (2.0m) | 75K |
| **2** | 8 (all) | None | Random intersecting | 0.0m | 300K |
| **3** | 8 (all) | 0 → 10 ramp | Random intersecting | 0.0m | 300K |

### 1.3 Observation Space (per drone)
```
Self block:     root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3)
                + root_pos(3) + root_rotation_matrix(9) = 21
                + active_flag(1) + drone_identity(8) + last_action(4) = 13
Neighbors:      2 × [rel_pos(3) + distance(1) + rel_vel(3)] = 14 (padded when < 2)
Static SDF:     3×3 grid = 9
Goal block:     target_pos_b(3) + yaw_error(1) + distance(1) + goal_reached(1) = 6
-------------------------------------------------------------------
Policy total:   ~63 (variable, via concatenation)
Critic (MAPPO): global swarm state (all drones + columns)
```

### 1.4 Reward Structure (full stage 2/3)
| Term | Weight | Signal |
|------|--------|--------|
| goal_distance | 1.0 | Smooth distance-to-goal |
| waypoint_tracking | 1.0 | exp(-‖pos_error‖²/σ²) |
| heading_tracking | 0.2 | exp(-(yaw_error)²/σ²) |
| reached_target_bonus | 2.0 | Binary @ <0.35m + <0.35rad |
| collision_avoidance | 2.0 | Linear penalty < safe_dist |
| obstacle_avoidance | 1.0 | Linear penalty < safe_dist |
| action_rate_l2 | -0.05 | Action smoothness |
| body_rate_l2 | -0.01 | Angular rate penalty |
| upright | 0.2 | Upright orientation |
| robot_collision_event | -5.0 | Binary collision penalty |
| obstacle_collision_event | -5.0 | Binary obstacle hit penalty |

### 1.5 Terminations (full stage 2/3)
- `time_out` (20s)
- `out_of_bounds` (XY > 6m, Z < 0.2m or > 5m)
- `drone_collision` (< 0.12m)
- `column_collision` (< 0.2m)

Training uses replay (`ReplayTrainingTerminationsCfg`): only time_out and out_of_bounds terminate; collisions are replayed.

### 1.6 Model Architecture
- `PaperAttentionEncoder`: Split embeddings → self-attention → cross-attention → ELU MLP `[256,256,256]`
- `PaperGaussianPolicy`: encoder → mean_head + log_std parameter
- `PaperDeterministicValue`: encoder → value_head
- Shared MAPPO: one policy, one value, one optimizer each

---

## 2. Working Tree Diff Classification

11 files modified, all in the `paper_swarm` task area. Classified below.

### 2.1 `method-change` — Core curriculum & config rework

| File | Change | Impact |
|------|--------|--------|
| `paper_swarm_env_cfg.py` | Stage 1 rewrite: fixed-origin resets, `expand_target_range` curriculum, `num_envs 256→512`, `possible_agents` reduced to `["drone_0"]`, CLIP actions (was TANH), new `Stage1EventsCfg`, removed `too_far_from_command` termination | **Substantial**: Stage 1 now mimics lab_5 hover curriculum — drone starts at origin (0,0,1), target initially at same spot, curriculum expands XY range from 0→1.5m over 50K steps. This is a one-drone hover → short-flight curriculum, not the old "tight random targets" approach |
| `mdp/curriculums.py` | New `expand_target_range_curriculum()` + `curriculum_fraction()` helper | Dynamically expands `pos_x`, `pos_y`, `pos_z` ranges on the command manager based on step counter |
| `models/skrl_models.py` | Added `clip_log_std`, `min_log_std`, `max_log_std` support | Policy log std is now clamped, preventing std divergence |
| `config/skrl_mappo_stage1_cfg.yaml` | Rollouts 64→32, mini_batches 32→64, `clip_log_std:true` with range [-4.0, -0.5] | Training hyperparameters tightened: shorter rollouts, more minibatch granularity, std clamped tight |
| `agents/runner.py` | Pass `clip_log_std`, `min_log_std`, `max_log_std` to policy constructor | Wires yaml config through to model |

### 2.2 `bugfix` — Environment correctness

| File | Change | Impact |
|------|--------|--------|
| `mdp/events.py` | Inactive drones spread on 3×N grid at z=0.05; hover thrust only for `possible_agent_ids` | Inactive drones no longer clip into each other at origin or receive conflicting thrust commands |

### 2.3 `research-infra` — Performance & debugging

| File | Change | Impact |
|------|--------|--------|
| `manager_based_ma_env.py` | Observation caching: `compute_group(name)` → `compute()` + `dict[name]` | Allobs groups computed once; lookup for policy group avoids redundant forward passes |
| `manager_based_marl_env.py` | `state()` now fans out per-agent critic obs (was global singleton) | Critic state is properly per-agent, not one tensor broadcast to all |
| `models/encoder.py` | NaN detection guard in encoder forward | Debugging: raises ValueError with specific NaN dimensions |

### 2.4 `refactor` — Documentation consolidation

| File | Change | Impact |
|------|--------|--------|
| `AGENTS.md` | Rewritten from IsaacLab-generic to swarm-lab-specific; added training commands, test conventions, review procedures | Agent now gets swarm-lab operational context, not just IsaacLab API rules |
| `GEMINI.md` | Deleted | Folded into AGENTS.md |

---

## 3. Key Design Observations

### 3.1 Stage 1 Reboot Pattern
The old Stage 1 used tight random targets (`x,y ∈ ±0.25m`, `z ∈ [0.9, 1.1]`) with TANH action clamping. The new Stage 1:
- Resets drone at **fixed origin** `(0, 0, 1.0)`
- Target starts at **same position** `(0, 0, 1.0)` → zero error
- `expand_target_range` curriculum linearly grows target XY range 0→1.5m, Z delta 0→0.5m over 50K steps
- Uses CLIP (hard boundary) action handling instead of TANH
- Only `drone_0` is managed; 7 others parked inert at z=0.05 in a grid
- 512 parallel envs (was 256)
- This is explicitly modeled after a "lab_5 hover curriculum" pattern

### 3.2 Std Clamping
`clip_log_std: true` with range `[-4.0, -0.5]` on log_std means:
- exp(-4.0) ≈ 0.018 → policy std cannot fall below ~0.018
- exp(-0.5) ≈ 0.607 → policy std cannot rise above ~0.607
- This prevents the common MAPPO failure mode of std collapsing to 0 (deterministic collapse) or exploding

### 3.3 Observation Caching
The environment now computes all observation groups together in `compute()`, then extracts the needed group by name. This avoids redundant forward passes when multiple observers depend on the same underlying terms — a performance optimization (commit 9cd6133).

### 3.4 NaN Guard
The encoder now checks for NaN in its input and raises with dimension information. This is a debugging aid — if the training produces NaN observations, you get an immediate error with the offending dimensions rather than silent garbage.

---

## 4. Configuration Sources Not Yet Read

These files were touched in the diff but not yet read in full:

- [ ] `mdp/curriculums.py` — full curriculum functions (was read via diff, but not the entire existing `paper_swarm_task_curriculum`)
- [ ] `mdp/events.py` — full event functions
- [ ] `mdp/observations.py` — observation implementations
- [ ] `mdp/rewards.py` — reward implementations
- [ ] `mdp/terminations.py` — termination implementations
- [ ] `mdp/commands.py` — PaperSwarmPoseCommand
- [ ] `models/encoder.py` — full attention encoder
- [ ] `models/skrl_models.py` — full SKRL wrappers
- [ ] `agents/shared_mappo.py` — shared MAPPO trainer logic
- [ ] `agents/runner.py` — SKRL runner hook
- [ ] `paper_swarm_env.py` — env class (vs config)
- [ ] `paper_swarm_recorders.py` — HDF5 recorder
