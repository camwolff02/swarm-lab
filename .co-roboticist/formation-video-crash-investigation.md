# Formation Swarm Video Crash Investigation

Date: 2026-06-03

Question: the thesis formation video frames the drones well, but the swarm appears to hit the floor instead of flying away in stable formation. Determine whether this is a recording-environment artifact or a checkpoint-selection issue.

## Current Evidence

- Video task: `Isaac-Formation-Swarm-MAPPO-Stage1-Video-v0`.
- Normal eval task: `Isaac-Formation-Swarm-MAPPO-Stage1-v0`.
- Thesis eval summary points to checkpoint:
  `logs/skrl/formation_swarm/2026-05-27_23-33-59_mappo_torch/checkpoints/agent_1000000.pt`.
- Existing thesis eval summary:
  - `cfr`: 1.0
  - `formation_error_mean`: 0.0161 m
  - `center_goal_error_mean`: 15.12 m
  - `collision_rate`: 0.0
  - horizon: 455 steps / 9.1 s
- Existing thesis trajectory shows the swarm preserves relative formation well but descends near crash height before the normal task resets:
  - robot 0 z range: 0.401 to 1.512 m
  - robot 1 z range: 0.359 to 1.552 m
  - robot 2 z range: 0.231 to 1.528 m
  - at about 8.2 s robot 2 was near 0.299 m, then the trajectory reset near 8.3 s.

## Environment Finding

`FormationSwarmStage1VideoEnvCfg` uses video-only timeout terminations. The normal Stage 1 eval task also terminates on crash/too-close/ball/column safety events. Therefore the video task can show a continuous floor impact where the normal evaluation task would reset.

This means the visible crash is partly a recording-preset effect, but the altitude loss itself is policy/checkpoint behavior. The video preset did not create the descent; it exposed it.

## Checkpoint Search Log

## Compatibility Environment

Old May 3 formation checkpoints were not compatible with the current Stage 1 task as-is:

- May 3 checkpoints save modules under public agent ids `drone_0`, `drone_1`, `drone_2`.
- The current Stage 1 task exposes `robot_0`, `robot_1`, `robot_2`.
- Running a May 3 checkpoint directly through `Isaac-Formation-Swarm-MAPPO-Stage1-v0` produced SKRL skipped-module warnings for the robot agents, so the policy weights were not being applied correctly.

Implemented compatibility support:

- Added task id `Isaac-Formation-Swarm-MAPPO-Stage1-Legacy-v0`.
- Added `FormationSwarmStage1LegacyEnvCfg`, which exposes legacy public agent ids `drone_*` while binding managers to current scene assets `robot_*`.
- Added `skrl_mappo_legacy_shared_cfg.yaml`, matching the old no-scaler shared-attention checkpoint layout while using the current SKRL MAPPO schema.
- Updated formation MDP helpers and evaluator code to distinguish public checkpoint agent ids from physical scene asset names.

Notes:

- Optimizer and value-normalizer modules are skipped when loading old checkpoints into the current eval agent. This is acceptable for playback/evaluation because actor/critic policy weights load under `drone_*`; optimizer state is not used in eval.
- The current video/eval tasks remain usable for May 27 `robot_*` checkpoints.

### May 27 Stage 1 `agent_100000.pt`

Command:

```bash
uv run scripts/analysis/evaluate_formation_swarm.py --task Isaac-Formation-Swarm-MAPPO-Stage1-v0 --checkpoint logs/skrl/formation_swarm/2026-05-27_23-33-59_mappo_torch/checkpoints/agent_100000.pt --num_envs 16 --num_steps 455 --seed 42 --record_stride 5 --output_dir logs/evaluations/formation_checkpoint_search --dataset_name may27_agent_100000
```

Summary:

- `cfr`: 1.0
- `formation_error_mean`: 0.0829 m
- `center_goal_error_mean`: 15.08 m
- `collision_rate`: 0.0
- `mean_reward_per_step`: 0.0956

Trajectory:

- robot 0 z range: 0.217 to 1.500 m, last 0.596 m
- robot 1 z range: 0.308 to 1.500 m, last 0.676 m
- robot 2 z range: 0.362 to 1.500 m, last 0.702 m
- reset occurred around 8.3 s after z approached crash height.

Verdict: not the stable formation-flight checkpoint. It flies forward and keeps formation better than random, but also descends into the reset boundary.

### May 3 Legacy `agent_100000.pt` / `best_agent.pt`

Task: `Isaac-Formation-Swarm-MAPPO-Stage1-Legacy-v0`

Checkpoint:

- `logs/skrl/formation_swarm/2026-05-03_11-14-41_mappo_torch/checkpoints/agent_100000.pt`
- `best_agent.pt` produced the same rollout metrics as `agent_100000.pt`.

Summary:

- `cfr`: 1.0
- `formation_error_mean`: 0.2480 m
- `center_goal_error_mean`: 11.27 m
- `mean_reward_per_step`: 0.0892

Trajectory:

- robot 0 z range: 0.239 to 2.543 m, last 0.445 m, y max 5.313 m
- robot 1 z range: 0.308 to 2.512 m, last 0.537 m, y max 6.298 m
- robot 2 z range: 0.475 to 2.453 m, last 0.650 m, y max 4.511 m

Verdict: flies farther than May 27 100k, but still descends near the floor by the end. Not the best stable-flight candidate.

### May 3 Legacy `agent_1000000.pt`

Task: `Isaac-Formation-Swarm-MAPPO-Stage1-Legacy-v0`

Checkpoint:

`logs/skrl/formation_swarm/2026-05-03_11-14-41_mappo_torch/checkpoints/agent_1000000.pt`

Summary:

- `cfr`: 1.0
- `formation_error_mean`: 0.000009 m
- `center_goal_error_mean`: 16.20 m
- `mean_reward_per_step`: 0.0843

Trajectory:

- robot 0 z range: 0.562 to 2.378 m, last 1.501 m, y max 4.927 m
- robot 1 z range: 0.799 to 2.345 m, last 1.501 m, y max 5.852 m
- robot 2 z range: 0.486 to 2.293 m, last 1.501 m, y max 3.372 m

Verdict: best match so far for the remembered behavior. The swarm flies downrange in tight formation and returns to the 1.5 m target altitude by the end of the 9.1 s rollout. It does not match the final-position metric, but visually it should read as stable formation flight moving away from the start.

## Next Candidates

- If recording a replacement thesis video, use `agent_1000000.pt` with a legacy-video task/config so the camera follows the May 3 compatible rollout.
- Optional extra check: evaluate May 27 `best_agent.pt` and `agent_700000.pt`, but current evidence indicates May 3 final is the stronger visual demo.
