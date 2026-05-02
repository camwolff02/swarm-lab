# Phased implementation plan

## Phase 0 - Source-of-truth mapping

### Goal
Create a single mapping from paper concepts to release-code locations and to target Isaac Lab locations.

### Tasks
- Read the paper and the released repo files that define:
  - environment constants
  - observation layout
  - attention encoder
  - reward logic
  - replay logic
  - baseline hyperparameters
- Write a small internal note mapping each paper concept to:
  - original repo source
  - new `cpsquare-lab` location
  - new main-workspace location
- Resolve paper-vs-repo discrepancies before coding.

### Exit criteria
- Every major feature is assigned a target file.
- The team agrees on the action-space convention and the shared-policy plan.

## Phase 1 - Extract reusable primitives into cpsquare-lab

### Goal
Build the reusable foundation first so the paper task stays thin.

### Tasks
1. **Control / actuation layer**
   - add reusable action mapping helpers
   - expose a single conversion path from policy action to rotor thrust command
   - keep this independent of the paper task

2. **Observation primitives**
   - self-state observation helper
   - goal-relative observation helper
   - KNN neighbor relative position/velocity helper
   - local 3 x 3 obstacle SDF helper

3. **Collision/event primitives**
   - robot-robot collision tracking
   - robot-obstacle collision tracking
   - floor / wall / out-of-bounds event tracking if useful
   - support "penalize collision once" bookkeeping

4. **Reward primitives**
   - distance-to-goal reward
   - close-proximity penalty
   - binary collision penalties
   - floor/crash penalty
   - angular-velocity penalty
   - control-effort penalty
   - orientation/tilt penalty

5. **Replay manager**
   - snapshot environment state
   - save the state from 1.5 seconds before collision
   - replay with configurable probability
   - retire overused replay states

### Exit criteria
- Generic utilities are unit-tested in `cpsquare-lab`.
- No paper-task code duplicates these generic utilities.

## Phase 2 - Thin paper environment in the main workspace

### Goal
Build the paper task as a composition layer over reusable cpsquare-lab pieces.

### Tasks
1. Create `paper_spec.py` with task defaults and parity notes.
2. Create `env_cfg.py` for the `DirectMARLEnv` configuration.
3. Create `obstacle_room.py` for paper-specific room and obstacle sampling.
4. Create `env.py` as a thin composition layer that:
   - spawns agents and goals
   - samples obstacle layouts
   - uses cpsquare-lab helpers to build observations
   - uses cpsquare-lab helpers to compute rewards and terminations
   - exports metrics and per-agent spaces
5. Add task registration.

### Exit criteria
- The environment launches and resets.
- Observation and action spaces are correct.
- A smoke test can step random actions without shape/runtime failures.

## Phase 3 - Model and training path

### Goal
Implement the custom paper encoder and connect it to skrl IPPO.

### Tasks
1. Implement the custom attention block in `models/attention.py`.
2. Implement the paper encoder in `models/quad_swarm_encoder.py`.
3. Implement explicit skrl actor/value wrappers in `models/quad_swarm_skrl_models.py`.
4. Create `train_quad_swarm_ippo.py` using a custom Python setup rather than the stock generic runner path.
5. Add an evaluation script.
6. Resolve the shared-policy checkpoint:
   - verify whether skrl can reuse the same module instances across agents
   - if not, build the thinnest possible shared-parameter training adapter

### Exit criteria
- Forward-pass tests succeed.
- The policy can sample actions and the critic can produce values.
- A short training run executes end-to-end.

## Phase 4 - Paper-specific replay and curriculum behavior

### Goal
Match the paper features that materially improve training.

### Tasks
- enable collision replay with task-appropriate reset semantics
- port collision penalty annealing if present in the release baseline
- expose replay statistics and curriculum statistics in logging
- verify that replay does not duplicate invalid or already-exhausted states

### Exit criteria
- Replay states are saved, sampled, and retired correctly.
- Episode resets can come from either fresh initial states or replay states.

## Phase 5 - Validation and parity passes

### Goal
Prove the port is technically sound and document remaining gaps.

### Tasks
- run shape and smoke tests
- run short training sanity checks
- run ablations:
  - no attention
  - no replay
  - no obstacle SDF
- compare task metrics to expected trends from the paper
- write a brief parity report listing all confirmed matches and remaining deviations

### Exit criteria
- The implementation passes all acceptance criteria in `06_VALIDATION_AND_ACCEPTANCE.md`.
