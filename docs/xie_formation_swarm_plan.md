# Xie et al. Formation Swarm Implementation Plan

Source paper: Yuqing Xie et al., "Multi-UAV Formation Control with Static and Dynamic Obstacle Avoidance via Reinforcement Learning", arXiv:2410.18495v2.

Local source implementation: `../papers/multi-UAV-formation`.

## Scope

Implement a minimal IsaacLab 3.0 beta `DirectMARLEnv` task in `environments/environments/tasks/formation_swarm` for the paper's directed multi-UAV formation maintenance problem with static columns and dynamic balls.

The implementation keeps the paper's MDP components:

- Agents: homogeneous Crazyflie UAVs initialized in the target formation.
- Action: CTBR command `(collective, roll_rate, pitch_rate, yaw_rate)`, converted to rotor thrust through `cpsquare_lab.embodiments.multirotor.common.actions.CtbrAction`.
- Observations: self state, other-drone relative state, static obstacle 3x3 distance field, and dynamic obstacle relative/absolute velocity features.
- Rewards: weighted scalarization of flight, formation, obstacle, and action-smoothness objectives using the paper/source weights.
- Obstacles: grid-sampled static columns and parabolic dynamic balls targeted toward the formation.
- Learning: a task-local SKRL-compatible MAPPO subclass with the paper/source shared actor, observation critic, PPO hyperparameters, and attention encoder.

## Implementation Steps

1. Add `FormationSwarmEnvCfg` and register `Isaac-Formation-Swarm-Crazyflie-v0`.
2. Spawn one Crazyflie articulation per drone and instantiate one `CtbrAction` term per drone.
3. Implement the paper-order action adapter from `(c, roll, pitch, yaw)` to the shared CTBR component's `(roll, pitch, yaw, c)` layout.
4. Implement reset logic:
   - initialize drones at the configured target formation,
   - sample static obstacle columns from the zigzag grid,
   - schedule dynamic ball launches.
5. Implement observations:
   - self: position, quaternion, linear/angular velocity, rotated heading/up vectors, normalized rotor throttle, velocity error, and 3D drone id,
   - other drones: relative position, distance, and relative velocity,
   - static obstacles: 9-cell local distance field,
   - dynamic balls: distance, relative position, relative velocity, and ball velocity.
6. Implement reward and done logic from the released source:
   - formation Laplacian and size rewards,
   - target velocity/heading/height/position rewards,
   - ball/column collision and proximity rewards,
   - action/network/throttle/spin smoothness rewards,
   - scalarized reward weights from `FormationUnified.yaml`.
7. Add SKRL model factory for the paper/source attention encoder:
   - per-entity linear embeddings,
   - self-attention over self, other drones, balls, and static SDF token,
   - cross-attention from self token to non-self tokens,
   - ELU MLP trunk.
8. Add a MAPPO YAML entry point compatible with `uv run scripts/skrl/train.py --algorithm MAPPO`.
9. Patch SKRL's runner only for this task's model factory so the training command instantiates `FormationSharedMAPPO` instead of stock SKRL MAPPO.

## MAPPO Compatibility Notes

The paper's released config sets `share_actor: true` and `critic_input: obs`. Stock SKRL MAPPO treats each DirectMARLEnv agent as a separate policy/value update and uses `shared_states` for the critic. For this task, that would train three separate drone actors and a global-state critic, which does not match the released implementation.

`FormationSharedMAPPO` closes that gap while keeping SKRL's trainer, wrappers, memories, logging, and checkpoint path:

- the model factory creates one `FormationGaussianPolicy` and one `FormationDeterministicValue`, then assigns those same module objects to every drone id;
- the runner passes per-drone observation spaces as `shared_observation_spaces`, so the value model consumes the same observation vector style as the paper's `critic_input: obs`;
- transition recording stores each drone's own observation in the `shared_states` memory slot used by SKRL's MAPPO storage schema;
- each PPO update computes GAE per drone, then concatenates all drone rollouts into one shared minibatch stream;
- actor and critic use separate Adam optimizers and ExponentialLR schedulers;
- value targets use the paper's `ValueNorm1(beta=0.995)`;
- the critic loss is clipped Huber loss with `delta=10`;
- the actor loss is scaled by action dimension, matching the released MAPPO update.

The implementation intentionally remains task-local under `environments/environments/tasks/formation_swarm/agents/` and does not modify vendored SKRL or IsaacLab code.

## Verification Checks

- Import and registration: importing `environments.tasks` registers `Isaac-Formation-Swarm-Crazyflie-v0`.
- Config shape checks:
  - observation size equals self + other-drone + dynamic-obstacle + static-SDF dimensions,
  - action size is 4 and documented as paper-order CTBR.
- Model checks:
  - policy accepts a batch of flat observations and returns 4 action means plus log standard deviations,
  - attention encoder uses `torch.nn.MultiheadAttention` for both self-attention and cross-attention.
- Reward helper checks:
  - Laplacian formation cost is near zero for the target formation,
  - static SDF returns 9 distances and decreases near a column.
- Runtime smoke checks:
  - instantiate the task with a small number of environments,
  - reset and step with zero actions without shape errors,
  - run a short MAPPO launch command far enough to validate model construction and the shared MAPPO update path.
