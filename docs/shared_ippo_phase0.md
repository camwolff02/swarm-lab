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

## Phase 1 Boundary

- Phase 1 adds explicit shared-policy/shared-value ownership helpers and tensor collation helpers.
- Phase 1 does not implement PPO rollout storage, GAE, minibatching, optimizer stepping, checkpointing, or a replacement trainer.
- Enabling the explicit shared mode with the stock skrl runner must fail closed until the dedicated shared trainer/update path is implemented.
