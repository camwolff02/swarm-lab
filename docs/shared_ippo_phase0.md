# Shared IPPO Phase 0 Implementation Note

## Repo And Architecture Confirmation

- The shared-IPPO plan file was read from `swarm-lab/codex_shared_ippo_plan.md`.
- The plan's task touchpoints resolve in this repository under `environments/environments/tasks/quad_swarm_paper`.
- No directory literally named `swarm-rl` was present next to `swarm-lab`; the sibling `quad-swarm-rl` does not contain the plan's `quad_swarm_paper` task paths.
- `cpsquare-lab` is consumed through imports only and does not need changes for Phase 0 or Phase 1.

## Current Training Path

- Task registration imports `environments.tasks.quad_swarm_paper.agents.runner.install_quad_swarm_runner_patch`.
- That patch only overrides `skrl.utils.runner.torch.Runner._generate_models`.
- The stock skrl IPPO runner still owns the trainer, memory, and per-agent optimizer/update path.
- `share_parameters: true` in the stock path aliases the same model instances into multiple skrl agent slots, which is unsafe for training because stock IPPO creates per-agent optimizer state.

## Shared Trainer Update

- The dedicated shared path now lives in `environments/environments/tasks/quad_swarm_paper/agents/shared_ippo.py`.
- It keeps one shared policy, one shared value function, one policy optimizer, and one value optimizer.
- Rollouts are pooled over `[time, env, drone]` and flattened for shared PPO minibatches.
- The shared PPO minibatch cap is an implementation detail to keep each backward pass at the existing skrl per-agent batch scale after pooling drones. It does not change rollout length, learning epochs, learning rate, discounting, observation structure, reward terms, or the shared homogeneous controller assumption.
- The stock skrl path remains available when `training.shared_homogeneous_ippo: false`.

## Paper Parameter Alignment

- The local handoff lists the starting parity target as learning rate `1e-4`, rollout length `128`, hidden size `256`, separate actor/critic weights, no recurrence, replay probability `0.75`, episode duration `15.0 s`, visible neighbors `2`, obstacle density `0.2`, and obstacle size `0.6`.
- The active shared-IPPO config keeps those values where they are represented in this port.
- `paper_spec.HIDDEN_SIZE` has been corrected to `256` so the central constants match the active model config and handoff documentation.
