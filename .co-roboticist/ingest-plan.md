# Swarm-Lab Knowledge Bootstrap — Ingest Plan

Generated: 2026-05-26 | Branch: `new-paper`

---

## Staged Ingest Overview

This plan bootstraps project knowledge across 5 stages. Each stage produces durable summaries; none loads large logs blindly. The plan progresses from high-confidence code and docs toward lower-confidence agent artifacts and gap analysis.

---

## Stage 0: Project Identity & Layout ← COMPLETE

**Output**: `.co-roboticist/context-map.md`

**Actions done**:
- [x] Git state: branch, commits, uncommitted diff inventory
- [x] Directory structure: tasks, tests, logs, outputs, docs, papers
- [x] Task inventory: 5 tasks (paper_swarm★, formation_swarm, quad_swarm_paper, cameron_swarm, manager_swarm)
- [x] Documentation inventory: 12 markdown files classified by topic/status
- [x] Training data inventory: log dirs mapped, date ranges estimated
- [x] Evidence confidence tiers assigned (High/Medium/Low/Inferred)

---

## Stage 1: Active Task Source Ingest ← NEXT

**Goal**: Understand the `paper_swarm` task codebase deeply enough to propose experiments or fixes.

### 1.1 Core Config & Env (HIGH priority)
- [ ] Read `environments/environments/tasks/paper_swarm/paper_swarm_env_cfg.py` — full config (large file, ~900 lines)
- [ ] Read `environments/environments/tasks/paper_swarm/paper_swarm_env.py` — env setup
- [ ] Read `environments/environments/tasks/paper_swarm/paper_swarm_recorders.py` — debug recorders
- **Output**: Summary of stage configs, what differs between Stage1/2/3, observation dims, action space, termination conditions

### 1.2 MDP Modules (HIGH priority)
- [ ] Read `environments/environments/tasks/paper_swarm/mdp/observations.py` — observation functions
- [ ] Read `environments/environments/tasks/paper_swarm/mdp/rewards.py` — reward functions + weights
- [ ] Read `environments/environments/tasks/paper_swarm/mdp/terminations.py` — termination conditions
- [ ] Read `environments/environments/tasks/paper_swarm/mdp/events.py` — reset/event functions
- [ ] Read `environments/environments/tasks/paper_swarm/mdp/curriculums.py` — curriculum terms
- [ ] Read `environments/environments/tasks/paper_swarm/mdp/commands.py` — waypoint command gen
- **Output**: Paper-spec alignment audit (reward weights, collision distances, obstacle params)

### 1.3 Models & Agents (MEDIUM priority)
- [ ] Read `environments/environments/tasks/paper_swarm/models/encoder.py` — attention encoder
- [ ] Read `environments/environments/tasks/paper_swarm/models/skrl_models.py` — SKRL model wrappers
- [ ] Read `environments/environments/tasks/paper_swarm/agents/shared_mappo.py` — shared MAPPO trainer
- [ ] Read `environments/environments/tasks/paper_swarm/agents/runner.py` — SKRL runner hook
- **Output**: Model architecture summary, shared parameter confirmation

### 1.4 Working Tree Diff Audit
- [ ] Read the full uncommitted diff (`git diff HEAD`) and classify each change
- **Output**: Classification of each change as `method-change`, `research-infra`, `bugfix`, `refactor`

### 1.5 Test Coverage
- [ ] Run `uv run pytest test/ -k paper_swarm` (if any paper_swarm tests exist) or identify gaps
- **Output**: Test gap report for active task

---

## Stage 2: Training History Analysis

**Goal**: Understand what training runs have happened, what patterns of failure/success exist.

### 2.1 Recent Stage 1 Runs (today)
- [ ] Read TensorBoard metrics for most recent `mappo_stage1` run (`2026-05-26_16-15-15`)
- [ ] Compare with 2nd and 3rd most recent
- **Output**: Episode length, reward, policy std, value loss trends

### 2.2 Stage 1 Run Classification
- [ ] For each run in `logs/skrl/paper_swarm/mappo_stage1/`:
  - Quick check: does `params/` folder contain agent config?
  - Quick check: does `checkpoints/` directory exist and have files?
  - Quick check: last episode length from events file
- Classify runs as: `completed`, `early-termination`, `nan-diverged`, `failing-to-learn`
- **Output**: Run manifest with classification

### 2.3 Cross-timeline Correlation
- [ ] Map git commits to the timeline of training runs
- [ ] Identify: which code changes preceded which training behaviors
- **Output**: Timeline of interventions and their training effects

### 2.4 Prior Task Comparisons
- [ ] Summarize final state of `formation_swarm` experiments (last run: May 19)
- [ ] Summarize final state of `quad_swarm_paper` experiments (last run: April 30)
- [ ] Document any known failure modes from those tasks
- **Output**: Cross-task lessons transferable to paper_swarm

---

## Stage 3: Prior Task Consolidation

**Goal**: Preserve what was learned from earlier tasks before it's forgotten.

### 3.1 Quad Swarm Paper Lessons
- Source: `docs/quad_swarm_port_notes.md` (parity report), `docs/shared_ippo_phase0.md`, log dirs
- [ ] Extract: what worked, what didn't, open issues
- [ ] Document: shared IPPO path status (confirmed working?)
- **Output**: Quad swarm post-mortem (or if still viable: baseline metrics)

### 3.2 Formation Swarm Lessons
- Source: `docs/xie_formation_swarm_plan.md`, log dirs, test files
- [ ] Extract: Laplacian formation cost correctness, training behavior
- [ ] Document: did formations hold? did obstacles matter?
- **Output**: Formation swarm post-mortem

### 3.3 Connector / ROS2 Experiments
- Source: `logs/connector/`, `logs/ros2/`, `scripts/ros2/`
- [ ] Classify: was this prep for hardware deployment? results?
- **Output**: Connector experiment summary

---

## Stage 4: Agent Conversation Ingestion

**Goal**: Ingest previous agent-created artifacts with appropriate confidence.

### 4.1 OpenCode Skills (MEDIUM confidence)
- [x] Read `.opencode/skills/*/SKILL.md` (4 skills ingested)
- Classify:
  - `env-info`: operational, high-confidence (tested commands)
  - `training-metrics`: operational, high-confidence
  - `training-test`: operational, high-confidence
  - `pre-commit`: operational, medium-confidence (may be stale)
- [ ] Validate each skill command works today
- **Output**: Validated command cookbook

### 4.2 Agent-authored Plans (LOW confidence)
- [x] Read `artifacts/collision-swarm/codex_shared_ippo_plan.md` — full shared IPPO design
- [ ] Cross-reference with actual implementation at `quad_swarm_paper/agents/shared_ippo.py`
- [ ] Document: does plan match implementation? what diverged?
- **Output**: Plan-to-code fidelity report

### 4.3 AGENTS.md Differences
- [ ] Read diff between commited AGENTS.md and working tree AGENTS.md
- [ ] Classify: which instructions were added/removed and why?
- **Output**: AGENTS.md change classification

---

## Stage 5: Knowledge Gap Analysis

**Goal**: Identify what we don't know that matters for the active task.

### 5.1 Paper Ingestion Gaps
- [ ] Papers present at `papers/`:
  - `collision-swarm.pdf` — ICRA 2024 swarm collision avoidance
  - `dynamic-static-formation-swarm.pdf` — Xie et al. formation control
- [ ] Check: are key paper parameters extracted and documented?
- [ ] Check: are there parameter discrepancies between paper and code?
- **Output**: Paper-to-code parameter audit

### 5.2 Missing Experiment Baselines
- [ ] Identify: what baselines should exist before running new experiments?
- [ ] Identify: what ablations are implied by the PAPER_SWARM_PLAN but not yet run?
- **Output**: Missing-baseline list

### 5.3 Missing Tests
- [ ] For `paper_swarm`: what test coverage exists?
- [ ] Gap: Stage1 curriculum tests? Reward shape tests? Attention encoder shape tests?
- **Output**: Test gap list

### 5.4 Hardware Readiness
- [ ] Does `paper_swarm` have a hardware deployment path?
- [ ] Check connector configs, ROS2 bridges for multi-drone dispatch
- **Output**: Hardware readiness assessment

---

## Ingest Order Recommendation

```
Stage 0 ──► Stage 1.1-1.2 ──► Stage 2.1-2.2 ──► Stage 3.1-3.2 ──► Stage 5
                │                    │
                └─ Stage 1.4        └─ Stage 2.3
                     │
                     └─ Stage 4.3

Stage 1.3-1.5: defer until model/agent changes are needed
Stage 4.1-4.2: useful background, not blocking
```

**Immediate next action**: Stage 1.1 (read paper_swarm_env_cfg.py) followed by Stage 1.4 (classify working tree diff). These give us a clear picture of what the current uncommitted work is doing before we look at training results.

---

## Artifact Storage Rules

- Stage outputs go in `.co-roboticist/` as markdown files
- Never load full TensorBoard event files or large logs into context
- Prefer `training-metrics` skill for metric extraction
- Training run manifests are `.co-roboticist/run-manifest-<task>.md`
- Experiment cards go in `.co-roboticist/experiments/<id>.md`
- Known failures go in `.co-roboticist/known-failures.md`
- Configuration snapshots go in `.co-roboticist/config-snapshots/`
