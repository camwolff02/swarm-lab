# Swarm-Lab Project Context Map

Generated: 2026-05-26 | Branch: `new-paper` | Classification: `research-experiment`

---

## 1. Repository Identity

| Property | Value |
|----------|-------|
| **Path** | `/home/cam/Development/cpsquare/swarm-lab` |
| **Role** | IsaacLab-based multi-drone swarm RL experiments |
| **Parent (editable dep)** | `../cpsquare-lab` — reusable robotics code |
| **Vendored (never modify)** | `../IsaacLab` — branch `release/3.0.0-beta2` |
| **Language** | Python 3.12 |
| **Package manager** | `uv` |
| **Environment registry** | `environments/tasks/` (Gym-style register via `__init__.py`) |
| **Test dir** | `test/` (singular; `tests/` is gitignored trash) |

---

## 2. Git State

| Item | Value |
|------|-------|
| **Current branch** | `new-paper` (tracking `origin/new-paper`) |
| **Other branch** | `main` (behind; not active) |
| **Uncommitted changes** | 11 files modified in `paper_swarm` task + AGENTS.md + manager envs |
| **Last commit** | `9cd6133` — "Added optimization patch with observation caching" |

---

## 3. Task / Environment Inventory

### 3.1 `paper_swarm` — ACTIVE (current work)
- **Directory**: `environments/environments/tasks/paper_swarm/`
- **Purpose**: Hybrid waypoint navigation + static obstacle avoidance (merges 2 papers)
- **Agent**: Homogeneous MAPPO with shared parameters, attention encoder
- **Action**: CTBR `[c, wx, wy, wz]`
- **Observation**: Self(21) + Goal(6) + Neighbors(N-1×7) + SDF(9) + Active(1) + LastAction(4)
- **Model**: `PaperAttentionEncoder` (PyTorch builtin `MultiheadAttention`)
- **Stages**: 3-stage curriculum (stage1 = hover → waypoint expansion → full swarm)
- **Docs**: `PAPER_SWARM_PLAN.md` (main plan), `docs/` entries
- **Uncommitted changes**: Stage1 curriculum rewrite (fixed reset at origin, expand_target_range), hover thrust reset, observation caching
- **Recent training**: `logs/skrl/paper_swarm/mappo_stage1/` — dozens of runs today and yesterday

### 3.2 `formation_swarm` — ACTIVE (V2 ManagerBasedMarlEnv refactor)
- **Directory**: `environments/environments/tasks/formation_swarm/`
- **Purpose**: Xie et al. multi-UAV formation control (Laplacian-based)
- **Agent**: FormationMAPPO (thin MAPPO wrapper) with FormationAttentionEncoder
- **V2**: Refactored from DirectMARLEnv to ManagerBasedMarlEnv (see `.co-roboticist/formation-v2-stage1.md`)
- **V1 (legacy)**: `env.py` (786-line DirectMARLEnv), registered as `Isaac-Formation-Swarm-Crazyflie-v3`
- **V2 (current)**: `formation_marl_env.py` + `mdp/` modules, registered as `Isaac-Formation-Swarm-MAPPO-v0` and `Isaac-Formation-Swarm-MAPPO-Stage{1,2,3}-v0`
- **Last training (V2)**: `logs/skrl/formation_swarm/2026-05-27_23-33-59_mappo_torch/` — Stage 1 successful to 700K steps
- **Last training (V1)**: `logs/skrl/formation_swarm/` — May 19
- **Docs**: `docs/xie_formation_swarm_plan.md`, `.co-roboticist/formation-v2-stage1.md`

### 3.3 `quad_swarm_paper` — STABLE (shared IPPO work done)
- **Directory**: `environments/environments/tasks/quad_swarm_paper/`
- **Purpose**: ICRA 2024 collision avoidance paper port
- **Agent**: Shared IPPO (one policy, one value, one optimizer each)
- **Last training**: `logs/skrl/quad_swarm_paper/` — April 30
- **Docs**: `docs/quad_swarm_port_notes.md`, `docs/shared_ippo_phase0.md`, `artifacts/collision-swarm/codex_shared_ippo_plan.md`
- **Registration**: `Isaac-Quad-Swarm-Paper-Crazyflie-v0`

### 3.4 `cameron_swarm` — EARLY EXPERIMENT
- **Directory**: `environments/environments/tasks/cameron_swarm/`
- **Purpose**: Original Cameron swarm task prototype
- **Agent**: IPPO and MAPPO (shared params)
- **Training**: `logs/skrl/cameron_swarm/`

### 3.5 `manager_swarm` — COMMON ENV BASE
- **Directory**: `environments/environments/tasks/manager_swarm/` + `environments/environments/envs/`
- **Purpose**: Manager-based MA/MARL environment base classes
- **Uncommitted changes**: manager_based_ma_env.py, manager_based_marl_env.py (observation caching)

---

## 4. Documentation Inventory

| File | Purpose | Status |
|------|---------|--------|
| `AGENTS.md` | Project-level agent instructions (large, has local edits) | Modified in WT |
| `PAPER_SWARM_PLAN.md` | Current main plan: hybrid waypoint + obstacle task | Current |
| `README.md` | Minimal repo readme | Stale |
| `docs/development.md` | Dev standards, boundaries | Stable |
| `docs/training.md` | Training entrypoints (SKRL, RSL-RL) | Stable |
| `docs/env_cache_validation.md` | Cache correctness checklist | Stable |
| `docs/quad_swarm_port_notes.md` | Quad swarm port parity report | Stable |
| `docs/shared_ippo_phase0.md` | Shared IPPO implementation note | Stable |
| `docs/xie_formation_swarm_plan.md` | Formation swarm implementation plan | Stable |
| `docs/viser_isaacsim6_compatibility.md` | Viser is broken with IsaacSim 6 | Stable |
| `artifacts/collision-swarm/codex_shared_ippo_plan.md` | Agent-authored shared IPPO plan (lower confidence) | Historical |
| `environments/README.md` | Environments package overview | Stable |

---

## 5. Training Data Inventory

### 5.1 SKRL Logs
- `logs/skrl/paper_swarm/mappo_stage1/` — **ACTIVE**: 40+ runs, latest today (16:15)
- `logs/skrl/paper_swarm/mappo/` — May 25, 9 runs
- `logs/skrl/paper_swarm/ippo/` — (exists, dates unknown)
- `logs/skrl/formation_swarm/` — May 1-19, ~45 runs (last: May 19 13:38)
- `logs/skrl/quad_swarm_paper/` — April 24-30, ~40 runs (last: April 30)
- `logs/skrl/cameron_swarm/` — IPPO and MAPPO variants
- **TensorBoard events**: Present in individual run dirs as `*.tfevents.*`

### 5.2 RSL-RL Logs
- `logs/rsl_rl/simpleflight_hover_crazyflie/` — 2 runs (April 17, April 21)

### 5.3 Connector/ROS2 Logs
- `logs/connector/` — Zenoh and ROS2 connector experiments (April 23-24)
- `logs/ros2/` — ROS2 single-agent (April 24)

### 5.4 Outputs
- `outputs/` — Hydra config dumps, one dir per run (April 17 through May 19)

### 5.5 Artifacts
- `artifacts/collision-swarm/quad_swarm_agent_package/` — package export
- `artifacts/collision-swarm/quad_swarm_isaaclab_agent_handoff/` — handoff artifacts

---

## 6. Test Inventory

| File | Purpose |
|------|---------|
| `test/conftest.py` | Pytest fixtures |
| `test/test_formation_swarm_task.py` | Formation Laplacian/geometry tests |
| `test/test_quad_swarm_env_cache.py` | Env cache correctness |
| `test/test_quad_swarm_model_forward.py` | Quad swarm model forward pass |
| `test/test_quad_swarm_task_config.py` | Quad swarm task config shapes |
| `test/test_shared_swarm_ippo_phase1.py` | Shared IPPO shapes/optimizers |
| `test/test_agent_bridge.py` | Agent bridge tests |
| `test/test_environment_bridge.py` | Environment bridge tests |
| `test/test_lab_2_bridge_agent.py` | Lab-2 bridge tests |
| `test/test_lab_2_imports.py` | Import validation |
| `test/test_run_agent.py` | Agent run tests |

---

## 7. Key Dependencies

- `cpsquare-lab` (editable path dep) — reusable robot logic
- `torch==2.10.0`, `torchvision==0.25.0` (CUDA 12.8)
- `iceoryx2>=0.9.0` — inter-process comms
- `ruff`, `pytest`, `mypy`, `pyright` — dev tools
- IsaacSim 6.0.0 (implicit, from IsaacLab)
- SKRL (implicit, from IsaacLab)

---

## 8. Known Constraints & Notes

1. **Viser is broken** with IsaacSim 6 — do not use `--viz viser`
2. **No top-level `mdp`, `assets`, `utils`** in cpsquare-lab
3. **Do not modify vendored IsaacLab** at `../IsaacLab`
4. **Python 3.12 only** — pinned in `.python-version`
5. **Test dir is `test/`** (singular); `tests/` is gitignored
6. **Shared IPPO work complete** for quad_swarm; all agents now use true shared params
7. **Current work** is Stage 1 curriculum merge: fixed-origin resets + progressive target range expansion
8. **Hardware runs** are gated; do not auto-start robot commands

---

## 9. Evidence Confidence Tiers

| Confidence | Sources |
|------------|---------|
| **High** (code + docs confirm) | Task configs, test files, committed code, AGENTS.md, PAPER_SWARM_PLAN.md |
| **Medium** (docs exist but unverified) | `docs/*.md` implementation notes, parity reports |
| **Low** (agent-authored plans) | `artifacts/collision-swarm/codex_shared_ippo_plan.md` — agent plan, may not match implementation |
| **Inferred** (directory names only) | Log dir names, output dir names, git messages |
