# Paper Swarm Stage 2 Update Plan

Generated: 2026-05-30

## Current success estimate

With the current worktree and `--reset_optimizer_on_resume`, estimated probability of a successful Stage 2 train is
approximately 35-45%. Without the current transition reset path, probability is approximately 0-5% because the preserved
Stage 2 runs made no effective policy/value updates.

## Evidence to preserve

- Stage 1 learned in `paper_swarm_train_stage1/2026-05-27_19-52-32_mappo_torch`:
  - mean reward: 32.95 -> 86.34
  - value loss: 0.0696 -> 0.0056
  - policy std: 0.2066 -> 0.1696
- Stage 2 and Stage 2 v2 failed in `paper_swarm_train_stage2*`:
  - policy, value, and entropy losses logged as 0.0 for all events
  - policy std stayed fixed at 0.0729685
  - learning rate fell to about 5e-5 immediately
  - mean reward degraded from about -29.8 to -50.1
- Checkpoint comparison showed `paper_swarm_train_stage2_v2/.../agent_300000.pt` policy and value weights were bit-identical
  to Stage 1 `best_agent.pt`. Only preprocessors changed.

## Updates likely needed

1. Remove, neutralize, or redesign `drone_identity` in the actor observation.
   - 2026-05-30 update: `drone_identity` was removed from the `paper_swarm` actor observation, and
     `PaperAttentionEncoderCfg.self_obs_dim` was reduced from 34 to 26. With the current `max_neighbors: 2` configs, this
     changes the actor observation from 61 to 53 dims.
   - A shared homogeneous policy should not depend on absolute agent ID unless every ID appears during pretraining.
   - Stage 1 currently trains only `drone_0`, so the one-hot identity for `drone_1..drone_7` is out-of-distribution in
     Stage 2 even though the weights are shared.
   - Preferred direction: make the policy permutation-equivariant by using ego-centric state, relative neighbors, relative
     goals, and optional active flags, but no absolute one-hot ID.

2. Keep the transition reset path and validate it.
   - `PaperMAPPO.reset_for_transition()` should reset the critic, log standard deviation, preprocessors, optimizer, and
     scheduler while preserving the Stage 1 actor trunk/attention weights.
   - Add a smoke diagnostic that runs a few Stage 2 updates and asserts policy/value weights differ from the loaded
     checkpoint.

3. Avoid `best_agent.pt` for Stage 1 -> Stage 2 promotion unless the best criterion is transfer-aware.
   - The preserved Stage 1 `best_agent.pt` was an early low-entropy checkpoint with log std around -2.6.
   - The final Stage 1 checkpoint had higher entropy and better reward.

4. Replace or delay `KLAdaptiveLR` during the initial Stage 2 transition.
   - Historical Stage 2 immediately hit the LR floor.
   - Use fixed LR or ExponentialLR for the first pilot, then reintroduce KL scheduling only after update health is
     confirmed.

5. Make the first Stage 2 phase gentler.
   - Start with exactly 2 active learning drones, no static obstacles, no domain randomization, and generous spawn/goal
     separation.
   - Ramp to more agents only after nonzero losses, changing weights, stable value learning, and sane flight behavior are
     observed.

6. Fix/evaluate passive-drone and spawn issues before treating Stage 1 as robust collision-avoidance pretraining.
   - Existing HDF5 eval showed `all_in_bounds = 0.0` on initial-state checks and passive-drift concerns.
   - 2026-05-30 update: Stage 1 now starts with one passive hovering drone and ramps the passive mask to seven drones.
     The active-agent curriculum is clamped at one learning drone so Stage 1 remains single-agent pretraining.
   - 2026-05-30 update: collision shaping now follows the `quad-swarm-rl` pattern more closely: a binary robot-collision
     penalty is separated from a smooth linear proximity falloff, and both active and passive drones count as collision
     objects. Static obstacle collision penalties are direct binary distance checks rather than only replay/event signals.

## Architecture clarification

SKRL MAPPO is compatible with the intended high-level scheme only if the observation function is homogeneous. In this
repo's custom model factory, every possible agent receives the same Python policy object and the same Python value object.
That means Stage 2 agents can start with exactly the same Stage 1 actor weights. However, SKRL still stores a separate
memory and runs an update pass per `uid`; with shared model objects, each per-agent update mutates the same underlying
parameters sequentially.

The current problem is not "weights are not shared." They are shared. The problem is that Stage 1 trains the shared policy
only under the `drone_0` observation distribution. The actor observation includes an absolute one-hot drone identity, so
`drone_1..drone_7` receive fixed identity bits that the Stage 1 policy never saw. For arbitrary-N inference, absolute
index identity should not be part of the actor input.

## Recommended target behavior

Stage 1 should be a single learning-agent task that uses the same actor architecture as later stages:

- one active learning drone
- passive drone-shaped obstacles visible through the neighbor-attention stream
- no absolute drone identity in the actor observation
- no conflicting multi-agent goals
- MAPPO-compatible training allowed, but effectively equivalent to PPO when `possible_agents = ["drone_0"]`

Stage 2 should instantiate all homogeneous agents with the Stage 1 actor weights, reset the critic/preprocessors/log std,
and train reciprocal interaction from a gentle 2-agent curriculum.
