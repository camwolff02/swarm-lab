# Agent Handoff: Formation V2 Stage 1 — Successful Train

Date: 2026-05-28 | Branch: `new-paper` | Classification: `research-experiment`

---

## What Was Trained

**Task**: `Isaac-Formation-Swarm-MAPPO-Stage1-v0` — obstacle-free formation flight with 3 Crazyflie drones

**Architecture**: ManagerBasedMarlEnv (V2 refactor from DirectMARLEnv)

**Run ID**: `2026-05-27_23-33-59_mappo_torch`

**Command**:
```bash
uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Formation-Swarm-MAPPO-Stage1-v0 --headless
```

**Env/Agent specs**:
- 1024 parallel envs, 3 drones (robot_0, robot_1, robot_2)
- Observation: 72-dim (self=29 + others=14 + balls=20 + sdf=9)
- Action: 4-dim CTBR via `CtbrActionCfg`
- Model: `FormationAttentionEncoder` + MLP (256-256-256), shared policy/value
- Agent: `FormationMAPPO(MAPPO)` — thin override that passes observations to policy
- Curriculum: `curriculum_stage=1` (no static obstacles, no dynamic balls)
- Formation: Equilateral triangle, 1.0m size, target velocity (0, 2.0, 0) m/s

---

## Result

**Status**: Successful — trained to 700K timesteps without NaN or divergence

**Checkpoints**:
| Step | Path |
|------|------|
| 300K | `logs/skrl/formation_swarm/2026-05-27_23-33-59_mappo_torch/checkpoints/agent_300000.pt` |
| 400K | `.../agent_400000.pt` |
| 500K | `.../agent_500000.pt` |
| 600K | `.../agent_600000.pt` |
| 700K | `.../agent_700000.pt` |

**Throughput**: ~5-9 env steps/s on RTX 3090 Ti (1024 envs × 3 agents = 3072 drone steps/s)

---

## Significance

This is the **first successful training of the V2 (ManagerBasedMarlEnv) formation swarm**. It validates that the transformation from DirectMARLEnv to the manager-based spec works end-to-end:

- Observations, rewards, terminations, events, curriculum all delegate to managers
- Action flow through `CtbrActionCfg` integrated with `FormationMAPPO`
- Ball dynamics and obstacle visuals handled in env class
- SKRL MAPPO integration works with the thin `FormationMAPPO` wrapper

---

## V2 Architecture Notes

**What changed from V1 (DirectMARLEnv)**:
- `env.py` (786 lines) → `formation_marl_env.py` (415 lines) + `mdp/` modules
- `env_cfg.py` (96 lines) → `formation_marl_env_cfg.py` (414 lines) with `ManagerBasedMarlEnvCfg`
- Direct `_get_observations`/~`_get_rewards`/~`_get_dones` → `ObsTerm`/`RewTerm`/`DoneTerm` MDP functions
- `DirectMARLEnvCfg` → `ManagerBasedMarlEnvCfg` with `AgentGroupCfg`/`AgentRlCfg`
- Custom `FormationSharedMAPPO` (383 lines) → Thin `FormationMAPPO(MAPPO)` (38 lines)

**Observation layout preserved**: `[ego_state(29) | other_drones(14) | balls(20) | sdf(9)]` — attention encoder compatible

**Action reordering**: Policy emits `[wx, wy, wz, c]` (CTBR native), observation term `formation_last_action` reorders to `[c, wx, wy, wz]` for reward computation

---

## Next Step: Stage 2

Resume from Stage 1 checkpoint with static obstacles:

```bash
uv run scripts/skrl/train.py \
  --algorithm MAPPO \
  --task Isaac-Formation-Swarm-MAPPO-Stage2-v0 \
  --checkpoint logs/skrl/formation_swarm/2026-05-27_23-33-59_mappo_torch/checkpoints/agent_700000.pt \
  --reset_optimizer_on_resume --headless
```

Or in tmux:
```bash
tmux new-session -d -s formation-stage2 \
  -c /home/cam/Development/cpsquare/swarm-lab \
  "uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Formation-Swarm-MAPPO-Stage2-v0 --checkpoint logs/skrl/formation_swarm/2026-05-27_23-33-59_mappo_torch/checkpoints/agent_700000.pt --reset_optimizer_on_resume --headless 2>&1"
```

`--reset_optimizer_on_resume` resets Adam moment estimates (keeps weights) — important since Stage 2 introduces static column obstacles.

---

## Gym IDs Registered

| ID | Stage | Obstacles |
|---|---|---|
| `Isaac-Formation-Swarm-MAPPO-v0` | 3 (default) | Static + balls |
| `Isaac-Formation-Swarm-MAPPO-Stage1-v0` | 1 | None |
| `Isaac-Formation-Swarm-MAPPO-Stage2-v0` | 2 | Static only |
| `Isaac-Formation-Swarm-MAPPO-Stage3-v0` | 3 | Static + balls |
| `Isaac-Formation-Swarm-Crazyflie-v3` | Legacy (V1) | Per curriculum_stage |
