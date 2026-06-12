# Paper Swarm Stage 1 Checkpoint Search

Date: 2026-06-04

Goal: recover a strong paper-swarm Stage 1 single-active-agent checkpoint, especially one where the active drone reaches its waypoint while the passive drones hold position with slight drift.

## Search Sources

- `remote_skrl_logs.zip` at the cpsquare repo root.
- Extracted Stage 1 runs to `/tmp/cpsquare_remote_skrl_stage1`.
- Remote Stage 1 runs found:
  - `/tmp/cpsquare_remote_skrl_stage1/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch`
  - `/tmp/cpsquare_remote_skrl_stage1/logs/skrl/paper_swarm/mappo_stage1/2026-05-30_16-52-47_mappo_torch`

## Evaluation Setup

- Evaluator: `scripts/analysis/evaluate_paper_swarm_stage1.py`
- Main legacy task for the May 27 checkpoints:
  `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-LegacyInDistributionEval-v0`
- May 30 checkpoints were trained with the non-legacy observation shape, so they must use:
  `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-Eval-v0`
- Common rollout settings:
  - `num_envs=32`
  - `num_steps=300`
  - `seed=42`
  - `success_distance=0.35`

Stage 1 has passive drones as the relevant obstacles. Static columns are not active in the Stage 1 eval used here.

## Best Candidate

Checkpoint:

`/tmp/cpsquare_remote_skrl_stage1/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch/checkpoints/agent_80000.pt`

Preserved log copy:

`/home/cam/Development/cpsquare/swarm-lab/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch`

Preserved checkpoint:

`/home/cam/Development/cpsquare/swarm-lab/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch/checkpoints/agent_80000.pt`

Metrics:

- `success_rate`: 1.0000
- `final_success_rate`: 0.09375
- `initial_goal_distance_mean`: 1.2904 m
- `min_goal_distance_mean`: 0.1703 m
- `final_goal_distance_mean`: 1.2187 m
- `progress_to_min_mean`: 1.1201 m
- `robot_collision_rate`: 0.0
- `obstacle_collision_rate`: 0.0
- `active_crash_rate`: 0.0
- `max_passive_drift_mean`: 0.1343 m
- `max_passive_drift_p95`: 0.1375 m
- `min_active_passive_distance_mean`: 1.2156 m
- `terminated_count`: 81

Interpretation: this is the strongest recovered checkpoint for the remembered Stage 1 behavior. Across all 32 eval envs it reached the waypoint at least once, avoided collisions, avoided crashing, and passive drones drifted only about 13 cm from their hover checkpoints. The high termination count is consistent with repeated success/reset events during the fixed 300-step rollout, so this checkpoint should be evaluated with episode-level success metrics for final paper tables.

## Other Checked Candidates

May 27 legacy run:

| Checkpoint | Ever Success | Final Success | Min Goal Dist Mean | Final Goal Dist Mean | Collisions/Crash |
| --- | ---: | ---: | ---: | ---: | --- |
| `best_agent.pt` | 0.78125 | 0.03125 | 0.2714 | 1.2300 | none |
| `agent_20000.pt` | 0.78125 | 0.03125 | 0.2714 | 1.2300 | none |
| `agent_60000.pt` | 0.5625 | 0.09375 | 0.3439 | 1.3052 | none |
| `agent_80000.pt` | 1.0000 | 0.09375 | 0.1703 | 1.2187 | none |
| `agent_100000.pt` | 0.9375 | 0.0 | 0.2018 | 1.1766 | none |
| `agent_120000.pt` | 0.78125 | 0.03125 | 0.2724 | 1.2665 | none |
| `agent_140000.pt` | 0.46875 | 0.0625 | 0.3544 | 1.0277 | none |
| `agent_200000.pt` | 0.78125 | 0.0 | 0.2465 | 0.9714 | none |

May 30 non-legacy run:

| Checkpoint | Task | Ever Success | Final Goal Dist Mean | Notes |
| --- | --- | ---: | ---: | --- |
| `best_agent.pt` | `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-Eval-v0` | 0.0625 | 2.5257 | Not the remembered successful policy. |

The May 30 checkpoint fails under the legacy eval because the model observation encoder expects a 26-dimension self observation, while the legacy task uses 34.

## Visual Validation

Initial validation clip:

`/tmp/cpsquare_remote_skrl_stage1/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch/videos/play/rl-video-step-0.mp4`

Command:

```bash
./.venv/bin/python scripts/skrl/play.py \
  --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-LegacyInDistributionEval-v0 \
  --algorithm MAPPO \
  --checkpoint /tmp/cpsquare_remote_skrl_stage1/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch/checkpoints/agent_80000.pt \
  --num_envs 1 \
  --seed 42 \
  --video \
  --video_length 300 \
  --eval_metrics
```

MP4 check:

- 300 frames
- 1280x720
- 30 FPS
- 10.0 seconds
- first frame is black warmup; later sampled frames are nonblank

This clip is useful validation evidence but not thesis-polished. The default camera is distant, and success terminations reset the single environment during the clip.

## Thesis Video

Final close-camera clip:

`/home/cam/Development/cpsquare/swarm-lab/logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch/videos/play/thesis_paper_swarm_stage1_legacy_agent80000_v1_close.mp4`

Nextcloud copy:

`/home/cam/Nextcloud/Documents/Cal Poly/Thesis/Videos/thesis_paper_swarm_stage1_legacy_agent80000_v1_close.mp4`

Command:

```bash
./.venv/bin/python scripts/skrl/play.py \
  --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-LegacyVideo-v0 \
  --algorithm MAPPO \
  --checkpoint logs/skrl/paper_swarm/mappo_stage1/2026-05-27_19-52-32_mappo_torch/checkpoints/agent_80000.pt \
  --num_envs 1 \
  --seed 42 \
  --video \
  --video_length 220 \
  --eval_metrics
```

MP4 check:

- 220 frames
- 1280x720
- 30 FPS
- 7.33 seconds
- first frame is black warmup; later sampled frames are nonblank

The video task is a recording preset with one environment, timeout-only termination, and a close follow camera. It does not replace the metrics task above; use `Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-LegacyInDistributionEval-v0` for reported evaluation metrics.

## Reproducibility Caveat

The thesis CSVs and figures were updated directly from the recovered evaluator metrics and TensorBoard event file. The full thesis compiler, `thesis_ee599_cwolff/evaluation/compile_sim_results.py`, was not rerun end-to-end because its default paper-swarm input still points at stale HDF5 data under `/tmp/isaaclab/logs/paper_swarm_dataset.hdf5`. Running that compiler unchanged may overwrite the corrected paper-swarm CSVs with old results.

Future fix: adapt the compiler so the paper-swarm Stage 1 rows are generated from the recovered checkpoint evaluation output, or update the evaluator to emit the HDF5/trace format expected by `compile_sim_results.py`. After that, regenerate all thesis CSVs and figures from one reproducible command.
