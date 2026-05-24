# Paper Swarm: Hybrid Waypoint Navigation with Static Obstacle Avoidance

## Overview

A hybrid task combining techniques from two papers:

1. **"Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end DRL"** (collision-swarm.pdf)
   - Task: Each drone flies to unique waypoints
   - Reward: Goal tracking + collision avoidance + control effort
   - SDF-based obstacle observations (3x3 grid)

2. **"Multi-UAV Formation Control with Static and Dynamic Obstacle Avoidance via RL"** (dynamic-static-formation-swarm.pdf)
   - Model: FormationAttentionEncoder (split embeddings + self-attention + cross-attention)
   - Actions: CTBR (collective thrust + body rates)
   - Reward: Weighted multi-objective sum
   - MAPPO with shared parameters
   - Curriculum learning

## Hybrid Approach

### Task
Each drone in a swarm of up to 8 Crazyflie quadrotors navigates to its own unique waypoint while avoiding static obstacles (columns) and other drones.

### Model Architecture (from formation_swarm)
- `PaperAttentionEncoder` using PyTorch's builtin `nn.MultiheadAttention`
- Split embeddings: self, neighbors, static SDF, goal info → tokens
- Self-attention across all tokens
- Self-query cross-attention (query=self, key/value=others)
- MLP trunk → action mean + log_std

### Action Space (from formation_swarm)
- CTBR: `[c, wx, wy, wz]` with `c` = collective thrust, `ω` = body rates
- Output scaled by `CtbrActionCfg` with TANH clamping

### Observation Space (blend of both papers)
Per-agent observation vector:
1. Self state (21 dims): pos (3), vel (3), quat (4), omega (3), heading (3), up (3), rel_vel_to_target (3)
2. Goal info (6 dims): target_pos_b (3), target_yaw_error (1), distance_to_goal (1), goal_reached_flag (1)
3. Neighbors (N-1 * 7 dims): relative_pos (3), distance (1), relative_vel (3) per neighbor
4. Static SDF (9 dims): 3x3 signed distance grid
5. Active flag (1 dim)
6. Last action (4 dims)

### Reward Structure (blend of both papers)
Weighted sum of:
1. **Goal tracking** (3.0): `exp(-||pos_error||^2 / std^2)` — smooth distance-to-goal reward
2. **Heading tracking** (0.5): `exp(-(yaw_error)^2 / std^2)` — align with target yaw
3. **Goal reached bonus** (1.0): Binary bonus when position < 0.25m AND yaw < 0.35rad
4. **Collision avoidance** (2.0): Linear penalty from collision_dist to safe_dist
5. **Obstacle avoidance** (1.0): Linear penalty within safe distance from static obstacles
6. **Action smoothness** (-0.05): L2 penalty on action delta
7. **Body rate penalty** (-0.01): L2 penalty on angular velocity

### Termination Conditions
- Episode timeout (20s)
- Out of bounds (XY > 6m, Z < 0.2m or > 5m)
- Drone-drone collision (< 0.45m)

### Curriculum
1. Active agent ramp: 1 → 8 over 500K steps (prefix selection)
2. Waypoint randomization: starts at step 500K, anneals separation constraints over 300K steps

### Static Obstacles
- Cylindrical columns (radius 0.15m, height 3m)
- 10 columns placed in a zigzag grid
- SDF sampled at 3x3 grid points around each drone (±0.1m spacing)

### Training Modes
- **IPPO**: Decentralized, each drone independently
- **MAPPO**: Shared parameters + centralized critic

## Files

```
environments/environments/tasks/paper_swarm/
├── __init__.py                    # Gym registration
├── paper_swarm_env_cfg.py         # Environment config
├── paper_swarm_recorders.py       # Recorder terms for debugging
├── mdp/
│   ├── __init__.py
│   ├── observations.py            # Observation functions
│   ├── rewards.py                 # Reward functions
│   ├── terminations.py            # Termination conditions
│   ├── events.py                  # Reset/event functions
│   └── curriculums.py             # Curriculum terms
├── agents/
│   ├── __init__.py
│   ├── shared_mappo.py            # Shared MAPPO agent
│   ├── runner.py                  # SKRL Runner hook
│   └── skrl_mappo_cfg.yaml        # Training hyperparams
├── models/
│   ├── __init__.py
│   ├── encoder.py                 # Attention encoder (PyTorch builtin)
│   └── skrl_models.py             # SKRL model wrappers
└── config/
    ├── skrl_ippo_cfg.yaml         # IPPO training config
    └── skrl_mappo_cfg.yaml        # MAPPO training config
```

## Debugging with RecorderManager

A custom `PaperSwarmRecorder` records per-step data to HDF5:
- Drone positions, velocities, quaternions
- Actions and rewards
- Goal positions and distances
- Obstacle positions
- Episode metrics

This enables offline analysis and visualization of agent behavior.

## Testing Plan

1. **Visual inspection**: Run with `--viz viser` to verify:
   - Drones spawn upright (identity quaternion)
   - Physics is physically realistic
   - Waypoints are correctly generated
   - Obstacles are visible
   - Drones navigate toward waypoints

2. **Training check**: Run with `--num_envs 64` and verify:
   - No crashes during initialization
   - Observation/action dimensions are consistent
   - Rewards are non-zero and learning progresses
   - TensorBoard logs show training curves

3. **Code review**: Run pre-commit checks, verify AGENTS.md compliance
