# Stage 2: Training History Analysis — paper_swarm Stage 1

Generated: 2026-05-26

---

## 1. Run Classification Summary

### Timeline of runs (last 20)

| Run | ep_len start→end | Status | Notes |
|-----|-----------------|--------|-------|
| 00-18-11 | 1→1 | ok@103 | **Old config**: drones crash immediately |
| 00-24-15 | 110→173 | ok@173 | **Old config**: very short training, ep_len improving |
| 00-27-45 | 110→173 | ok@173 | Same pattern |
| 00-31-49 | — | **CRASHED** | No metrics written |
| 00-34-34 | 146→148 | ok@1500 | Old config, brief training |
| 00-51-51 | 107→169 | ok@169 | Old config |
| 00-52-47 | 107→169 | ok@169 | Old config |
| 00-55-13 | 146→121 | **ok@21750** | **Best result**: ran 21K steps without NaN |
| 01-23-49 | — | **CRASHED** | Transition to new config |
| 01-24-44 | — | **CRASHED** | |
| 01-26-54 | 147→129 | **NaN@15750** | New config: NaN after 15.7K steps |
| 11-13-18 | 136→130 | **NaN@16500** | Same: NaN at 16.5K |
| 14-25-56 | — | **CRASHED** | |
| 14-28-10 | — | **CRASHED** | |
| 14-31-13 | — | **CRASHED** | |
| 14-33-39 | 159→130 | **NaN@15000** | Recompiled, same NaN pattern |
| 16-06-14 | — | **CRASHED** | |
| 16-08-08 | — | **CRASHED** | |
| 16-08-39 | — | **CRASHED** | |
| 16-15-15 | 159→130 | **NaN@15000** | Same NaN pattern |

### Classification:
- **7 CRASHED** (35%) — init failure, no metrics written
- **3 NaN-diverged** (15%) — training starts but NaN at step ~15K
- **8 OK-but-short** (50%) — old config, training stopped early or ep_len=1

---

## 2. Root Cause #1: Centralized Critic Returns All Zeros → NaN

### Evidence Chain

**A. Stage 1 config uses MAPPO with centralized critic:**
```python
# PaperSwarmMappoStage1EnvCfg
possible_agents = ["drone_0"]  # Only 1 managed agent
observations = MappoObservationsCfg()  # Includes CentralizedCriticCfg
```

**B. CentralizedCriticCfg calls `paper_swarm_global_state` with all 8 drone IDs:**
```python
swarm_state = ObsTerm(
    func=mdp.paper_swarm_global_state,
    params={
        "agent_ids": DRONE_AGENT_IDS,  # ["drone_0",...,"drone_7"]
        ...
    },
)
```

**C. `paper_swarm_global_state` fails when unmanaged drones are requested:**
```python
def paper_swarm_global_state(env, agent_ids, ...):
    if not all(agent_id in getattr(root, "_agent_to_bundle", {}) for agent_id in agent_ids):
        return torch.zeros(root.num_envs, state_dim, device=root.device)
    # ↑ drone_1..drone_7 NOT in _agent_to_bundle → returns zeros!
```

**D. State dimension = 272 zeros:**
```
mask(8) + root_states(8×13=104) + commands(8×7=56) + pairwise(64) + columns(30) + col_mask(10) = 272
```

**E. RunningStandardScaler on constant zeros → NaN:**
```
Running mean → 0.0
Running std  → 0.0
(x - 0.0) / 0.0 = NaN
```

This takes ~15K steps because the running scaler needs enough updates to converge the variance estimate to zero. Before that, the initial estimates (mean=0, std=1) keep things finite.

**F. NaN propagates:**
```
state_preprocessor → NaN
  → value network forward → NaN prediction
  → value loss = NaN
  → GAE advantages = NaN  
  → policy loss = NaN
  → policy gradients = NaN
  → entire model corrupted
```

### Verification

- Lab_5 (working) is single-agent PPO, NO centralized critic, NO global state
- Lab_5 uses `state_preprocessor: RunningStandardScaler` but on valid per-agent states
- Paper_swarm Stage 1 should NOT be using centralized MAPPO critic — there's only 1 agent

---

## 3. Root Cause #2: Init Crashes (7/20 runs)

The new Stage 1 has `possible_agents = ["drone_0"]` but event reset functions iterate over `agent_ids: DRONE_AGENT_IDS` (all 8):

```python
# Stage1EventsCfg
reset_drone_root_state = EventTerm(
    func=mdp.reset_drone_root_state_uniform,
    params={"agent_ids": DRONE_AGENT_IDS, ...},  # 8 drones
)
```

The `reset_drone_root_state_uniform` function accesses per-agent masks:
```python
def reset_drone_root_state_uniform(env, env_ids, agent_ids, ...):
    for i, agent_id in enumerate(agent_ids):
        ...
        is_active = active_agents[:, i]  # Index into active mask
```

With `possible_agents = ["drone_0"]`, the `active_agent_count_curriculum` likely creates a mask of shape `(num_envs, 1)` for drone_0 only. When the loop hits `i=1` (drone_1), `active_agents[:, 1]` is an **index out of bounds** → crash.

This explains the intermittent crashes: the curriculum mask may be created with variable widths depending on timing/init order.

---

## 4. Comparison: lab_5 (working) vs paper_swarm (failing)

| Aspect | lab_5_hover ✅ | paper_swarm Stage 1 ❌ |
|--------|--------------|---------------------|
| Agent type | Single-agent PPO | Multi-agent MAPPO |
| Critic | Per-agent critic (same dim as policy) | Centralized global state (272 dim) |
| Observation | ~22 dims (body-frame target + multirotor state) | ~61 dims (attention encoder including neighbors, SDF, one-hot ID) |
| Model | MLP [256,256,256] | Attention encoder (self-attn + cross-attn + MLP) |
| Episode length | 5.0s | 20.0s |
| Decimation | 1 (100Hz) | 2 (50Hz) |
| Num envs | 8192 | 512 |
| Target range start | ±0.25m XY | 0.0m XY (point target) |
| Target range end | ±2.0m | ±1.5m |
| Terminations | time_out, crash(z<0.2), too_far(>4m) | time_out, out_of_bounds(z<0.2 or z>5 or xy>6) |
| Actions clamp | CLIP | CLIP (new) / TANH (old) |

The fundamental issue is not the observation/action space — it's that **Stage 1 is treated as a multi-agent MAPPO problem when it's actually a single-agent hover task**.

---

## 5. Implemented Fix

### ✅ Fix: `paper_swarm_global_state` now handles unmanaged drones

**File**: `environments/environments/tasks/paper_swarm/mdp/observations.py`

**Before**: The function checked if all `agent_ids` were in `_agent_to_bundle`. If any were missing (e.g., drone_1..7 in Stage 1 where only drone_0 is managed), it returned `torch.zeros(...)` — a 272-dim vector of zeros.

**After**: 
1. Removed the early-exit guard
2. For each agent, physics state is read directly from `root.scene[agent_id].data` (works for all drones)
3. For managed agents, commands come from the bundle's command manager
4. For unmanaged agents, a zero-command tensor (7 dims) is used

**Why this preserves Stage 2/3 compatibility**: The value network architecture is unchanged — always 272-dim input. The policy architecture is also unchanged — always 61-dim input. Checkpoints transfer cleanly.

**Why the scaler won't produce NaN**: Unmanaged drone dimensions have near-constant values (parked at fixed grid positions, zero velocity). The RunningStandardScaler normalizes them as `(0 - 0) / (sqrt(ε) + 1e-8) ≈ 0` — stable and finite. Managed drone dimensions vary normally.

### Additional recommendations (lower priority)

| Param | lab_5 | paper_swarm | Recommendation |
|-------|-------|-------------|----------------|
| Episode length | 5s | 20s | Try 5-10s for Stage 1 |
| Decimation | 1 | 2 | Try 1 for faster control response |
| Num envs | 8192 | 512 | Increase if GPU allows |
| Rollouts | 64 | 32 | Match lab_5 (64) |
| Mini batches | 32 | 64 | Match lab_5 (32) |

---

## 6. Verification Plan

After applying Fix A + Fix B:
1. Smoke test: `uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-v0 --max_iterations 3 --headless`
2. Check no init crash
3. Check episode_length > 1
4. Run short training (10K steps) and verify no NaN
5. Run full Stage 1 (75K steps) and check reward improves, std decreases
