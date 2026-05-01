# Codex Implementation Plan: True Shared-Policy / Shared-Optimizer IPPO for `quad_swarm_paper`

## Objective

Replace the current stock-skrl multi-agent training path for `Isaac-Quad-Swarm-Paper-Crazyflie-v0` with a **paper-faithful homogeneous decentralized IPPO** path:

- one shared policy network for all drones
- one shared value network for all drones
- one optimizer for the shared policy
- one optimizer for the shared value
- rollout and PPO updates pooled across all drones and all environments

Do **not** change the environment observation layout, attention architecture, action mapping, or `cpsquare-lab` embodiment code in this task.

---

## Why this change is needed

The current runner patch supports two behaviors:

1. `share_parameters: false`
   - builds one separate policy/value pair per named drone
   - stable under stock skrl IPPO, but **not faithful** to the paper's homogeneous decentralized controller

2. `share_parameters: true`
   - aliases the same model objects into multiple agent slots
   - **unsafe for training** under stock skrl IPPO because skrl still updates per-agent, which implies multiple optimizer/update streams touching the same parameters

The fix must therefore happen in the **training/update path**, not just the model factory.

---

## Non-goals

Do not do any of the following in this task:

- do not switch to MAPPO or a centralized critic
- do not add SimpleFlight / "What Matters" changes
- do not change `PaperMultiheadAttention`
- do not change direct rotor-thrust action semantics
- do not refactor `cpsquare-lab` unless required by import hygiene
- do not add per-agent specialization or agent-ID embeddings

This task is specifically about making IPPO parameter sharing **real** and **safe**.

---

## Current code touchpoints

Primary files involved:

- `tasks/quad_swarm_paper/agents/runner.py`
- `tasks/quad_swarm_paper/agents/skrl_ippo_cfg.yaml`
- `tasks/quad_swarm_paper/models/quad_swarm_encoder.py`
- `tasks/quad_swarm_paper/models/quad_swarm_skrl_models.py`
- `environments/tasks/quad_swarm_paper/env.py`
- `environments/tasks/quad_swarm_paper/env_cfg.py`

The policy/value model code can largely stay as-is. The main work belongs in a **new custom shared trainer / agent path**.

---

## High-level design

### Desired semantics

At every environment step:

1. Collect local observations for all drones
2. Stack them into a single pooled batch of shape like:
   - `[num_envs, num_drones, obs_dim]`
   - flattened to `[num_envs * num_drones, obs_dim]`
3. Run the shared policy and shared value on that pooled batch
4. Reshape outputs back to per-agent dicts for the env
5. During learning, store rollout tensors as `[T, E, N, ...]`
6. Flatten rollout tensors to `[T * E * N, ...]` for PPO minibatching
7. Apply one shared policy optimizer step and one shared value optimizer step

### Keep the policy decentralized

- policy input: each drone's local observation only
- policy output: that drone's 4D action only
- value input: same local observation only

This remains decentralized training / decentralized execution with homogeneous parameter sharing.

---

## Recommended implementation strategy

## Phase 1: Add a new explicit shared-training mode

### 1. Add a new config switch

In `tasks/quad_swarm_paper/agents/skrl_ippo_cfg.yaml` add an explicit mode flag, for example:

```yaml
models:
  factory: quad_swarm_paper_attention
  share_parameters: true
training:
  shared_homogeneous_ippo: true
```

Or equivalent naming. The important part is that this mode must route away from stock skrl multi-agent per-agent updates.

### 2. Keep `share_parameters` warning in stock path

In `runner.py`, preserve the current warning for stock skrl multi-agent training. Do **not** silently make `share_parameters: true` safe in the existing path.

Instead, if shared training is requested, route into a **new custom implementation**.

---

## Phase 2: Build a dedicated shared trainer/agent

Create a new module, for example:

- `tasks/quad_swarm_paper/agents/shared_ippo.py`

This module should own the shared-policy/shared-optimizer logic.

### 3. Instantiate exactly one policy and one value model

Reuse existing classes:

- `QuadSwarmGaussianPolicy`
- `QuadSwarmDeterministicValue`

Instantiate exactly one of each on the chosen device.

### 4. Instantiate exactly one optimizer per role

Create:

- one optimizer for shared policy params
- one optimizer for shared value params

These optimizers must own the complete parameter sets of the shared modules.

### 5. Create a rollout buffer for pooled swarm samples

Add a rollout storage object, either custom or thin wrapper, storing at least:

- observations: `[T, E, N, obs_dim]`
- actions: `[T, E, N, act_dim]`
- log_probs: `[T, E, N, 1]`
- values: `[T, E, N, 1]`
- rewards: `[T, E, N, 1]`
- terminated: `[T, E, N, 1]`
- truncated: `[T, E, N, 1]`
- next_values or bootstrap values as needed

Keep storage contiguous and torch-native.

---

## Phase 3: Shared action path

### 6. Add helpers to collate env dictionaries

Implement helpers like:

- `stack_agent_observations(obs_dict, agent_ids) -> Tensor[E, N, obs_dim]`
- `unstack_agent_actions(action_tensor, agent_ids) -> dict[str, Tensor[E, act_dim]]`

The agent order must always follow `env.possible_agents` exactly.

### 7. Shared forward pass for acting

At action time:

1. stack obs dict to `[E, N, obs_dim]`
2. flatten to `[E*N, obs_dim]`
3. compute policy distribution / sample actions
4. compute value predictions on same flattened obs batch
5. reshape outputs back to `[E, N, ...]`
6. convert actions back to env agent dict

### 8. Preserve existing action semantics

Do not alter the env-side action format. The env should continue receiving:

```python
{ "drone_0": tensor[E, 4], ..., "drone_7": tensor[E, 4] }
```

The env already maps these to direct thrust targets.

---

## Phase 4: Shared update path

### 9. Compute GAE over pooled swarm data

Advantages and returns should be computed per drone sample, but under the same shared value function.

Store or derive:

- returns: `[T, E, N, 1]`
- advantages: `[T, E, N, 1]`

Then flatten to `[T*E*N, 1]`.

### 10. Normalize advantages globally over the pooled batch

Do **not** normalize separately per drone. The whole point is that all drone experience contributes to one shared policy gradient estimate.

### 11. PPO loss over pooled minibatches

Construct minibatches from the flattened pooled batch and compute standard PPO clipped losses:

- policy surrogate loss
- value loss
- entropy bonus

Then perform:

- one backward/update sequence for policy optimizer
- one backward/update sequence for value optimizer

If using shared encoder parameters between policy and value in the future, handle that explicitly. For now, keep them as separate modules as in current code.

### 12. Logging

Track at least:

- shared policy loss
n- shared value loss
- entropy / std
- KL
- gradient norm
- pooled advantage mean/std

Also retain env-side metrics already tracked in `env.py`.

---

## Phase 5: Runner integration

### 13. Add a runner entry path for shared swarm PPO

In `tasks/quad_swarm_paper/agents/runner.py`:

- keep the current model-factory patch for stock path
- add a new code path that detects the explicit shared-training flag
- instantiate the custom shared agent/trainer instead of stock skrl IPPO

Avoid trying to patch skrl internals in-place if a thin custom training loop is simpler.

### 14. Preserve evaluation compatibility

The resulting shared model should still be loadable for inference and used to act on per-agent dict observations.

Evaluation can still reshape local observations to pooled form internally.

---

## Phase 6: Config and checkpoint behavior

### 15. Make debugging checkpoints explicit

In `skrl_ippo_cfg.yaml` (or custom shared config), set explicit save intervals while debugging. Do not use only `auto`.

Example:

```yaml
experiment:
  write_interval: 5000
  checkpoint_interval: 5000
```

### 16. Keep env config source-of-truth clean

Do not change env behavior in this task, but ensure the training run uses one clear env count source. If an override is required, make it explicit and visible in logs.

---

## Acceptance criteria

Codex should not stop after implementation; it should verify the following.

### A. Structural checks

1. Exactly one shared policy module exists in the shared-training mode
2. Exactly one shared value module exists in the shared-training mode
3. Exactly one policy optimizer exists
4. Exactly one value optimizer exists
5. All drones' rollout data contributes to the same pooled PPO update

### B. Behavioral checks

1. Actions can still be produced for all eight drones without changing env API
2. Forward pass works with batch flatten/unflatten without shape bugs
3. Training runs without per-agent optimizer duplication
4. Policy/value losses remain finite for at least the first 50k trainer steps
5. `policy std`, `policy loss`, and `value loss` should not go NaN during early training

### C. Regression checks

1. Hover-biased init still prints sensible startup stats
2. Replay logic still works unchanged
3. No changes to observation dimensionality
4. No changes to direct rotor-thrust mapping
5. No changes to attention architecture

---

## Tests Codex should add

### Unit-style tests

Create tests for:

1. **Observation collation**
   - input: dict of 8 `[E, obs_dim]` tensors
   - output: `[E, 8, obs_dim]`
   - flatten/unflatten round-trip preserves agent ordering

2. **Action unstacking**
   - input: `[E, 8, act_dim]`
   - output dict has correct per-agent tensors

3. **Shared module identity**
   - confirm shared mode constructs exactly one policy object and one value object

4. **Single optimizer ownership**
   - confirm each parameter tensor appears in exactly one policy optimizer param group and one value optimizer param group, not duplicated per agent

5. **Pooled minibatch shapes**
   - confirm rollout flattening maps `[T, E, N, ...] -> [T*E*N, ...]`

### Smoke test

Run a very short train, e.g. 2k–5k trainer steps, and verify:

- no NaNs in policy/value loss
- no NaNs in policy std
- checkpoints are written
- actions are finite
- startup hover diagnostics still look correct

---

## Suggested file additions

Recommended new files:

- `tasks/quad_swarm_paper/agents/shared_ippo.py`
- `tasks/quad_swarm_paper/agents/shared_rollout.py`
- `tests/test_shared_swarm_ippo_shapes.py`
- `tests/test_shared_swarm_ippo_optimizers.py`

Optional:

- `tasks/quad_swarm_paper/agents/shared_utils.py`

---

## Suggested file modifications

### `tasks/quad_swarm_paper/agents/runner.py`

- keep existing stock-skrl path
- add explicit routing to shared custom trainer path
- keep warning for unsafe stock aliasing mode

### `tasks/quad_swarm_paper/agents/skrl_ippo_cfg.yaml`

- add explicit shared-training mode flag
- add explicit checkpoint/write intervals for debugging

### `tasks/quad_swarm_paper/models/quad_swarm_skrl_models.py`

- likely no functional changes needed
- only minor adapter methods if shared trainer needs a different call signature

### `tasks/quad_swarm_paper/models/quad_swarm_encoder.py`

- no planned architectural changes

### `environments/tasks/quad_swarm_paper/env.py`

- no semantic changes required for this task
- optional: add a small log line confirming shared-trainer mode at startup if useful

---

## Important implementation invariants

These are hard constraints. Codex should not violate them.

1. **One optimizer path only** for shared parameters
2. **No per-agent Adam states** for the same shared tensors
3. **No change to env observation or action API**
4. **No change to the paper attention block**
5. **No centralized critic**
6. **No agent ID appended to observations**
7. **No dependence on `cpsquare-lab` changes** unless absolutely required for imports only

---

## Stop conditions / definition of done

The task is done when all of the following are true:

- shared mode uses one policy and one value module only
- shared mode uses one optimizer per role only
- pooled swarm rollout/update path is implemented
- short smoke training runs complete without NaN policy/value losses
- checkpoints are emitted during debugging runs
- stock non-shared path still works unchanged

---

## Nice-to-have follow-up after this task

Do not implement in this PR unless required, but note as next-step candidates:

- unify reward scaling to use one consistent time scale (`step_dt` vs `sim.dt`)
- clean up config source-of-truth for `scene.num_envs`
- later, compare shared-IPPO against a SimpleFlight-inspired privileged-critic branch

