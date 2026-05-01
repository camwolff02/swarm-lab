# Environment Cache Validation

Use this checklist when validating cache changes in an Isaac Lab session.

## Functional Parity

- Run a fixed-seed rollout before and after the cache refactor with the same action tensors.
- Compare per-agent observation shapes and values with `torch.allclose(..., atol=1e-6, rtol=1e-5)`.
- Compare per-agent reward shapes and values with the same tolerance.
- Force at least one reset and confirm post-reset observations reflect the reset state, not pre-reset reward state.

## Cache Behavior

- Inspect `env.cache.stats` after a rollout.
- Asset misses should occur on first access only; later accesses should be hits or phase-local derived-cache hits.
- Reward-phase `swarm_tracking` should be built once and reused by dones and rewards in the same step.
- Observation-phase `swarm_tracking` should build separately after reset handling.

## Timing

- Time `_get_dones`, `_get_rewards`, `_get_observations`, and full `env.step(...)` over a warm rollout.
- Compare against a pre-refactor baseline using the same `num_envs`, seed, policy/actions, and render settings.
- If Python overhead still dominates, profile whether always-coaccessed observation terms should be consolidated.
