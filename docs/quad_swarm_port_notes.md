# Quad Swarm Paper Port Notes

This task ports the ICRA 2024 quadrotor swarm collision-avoidance setup into an
Isaac Lab 3.0 `DirectMARLEnv` task registered as:

```text
Isaac-Quad-Swarm-Paper-Crazyflie-v0
```

## Layout

- Task: `environments/environments/tasks/quad_swarm_paper`
- Reusable swarm helpers: `../cpsquare-lab/src/cpsquare_lab/tasks/swarm`
- Reusable multirotor action mapping:
  `../cpsquare-lab/src/cpsquare_lab/embodiments/multirotor/common/action_mapping.py`
- Training entrypoint: stock `scripts/skrl/train.py`
- Playback entrypoint: stock `scripts/skrl/play.py`

## Observation And Action Contract

Each drone is a separate homogeneous agent named `drone_0` through `drone_7`.
The policy observation is a flat 40D vector ordered as:

```text
self_obs[19], neighbor_obs[12], obstacle_obs[9]
```

The action space is `Box(-1, 1, (4,))`. Actions are mapped once through the
reusable cpsquare-lab action helper to per-rotor thrust ratios and then to
physical thrust targets.

## Model Notes

The encoder follows the released main simulation architecture:

```text
self MLP, neighbor MLP, obstacle MLP
-> two attention tokens [neighbor_embed, obstacle_embed]
-> attention
-> concat self embedding and attended tokens
-> feed-forward latent
```

The attention implementation uses `torch.nn.functional.scaled_dot_product_attention`
for the core attention operation, but keeps the released projection geometry:
`n_head=4` and `d_k=d_v=d_model`. This is not the same shape as
`torch.nn.MultiheadAttention`, which splits `embed_dim` across heads.

The stock skrl YAML runner does not instantiate arbitrary custom PyTorch model
classes by class path. To keep the project on Isaac Lab's default `train.py` and
`play.py`, the task registration installs a narrow `skrl.utils.runner.torch.Runner`
model-factory hook. The hook only activates when the agent config has:

```yaml
models:
  factory: quad_swarm_paper_attention
```

All other skrl tasks continue through the original stock `Runner` path.

## Training And Playback

Run training with Isaac Lab's default skrl script and the task's IPPO config:

```bash
uv run python scripts/skrl/train.py --task Isaac-Quad-Swarm-Paper-Crazyflie-v0 --algorithm IPPO --num_envs 128 --max_iterations 10 --headless
```

Run playback with the default play script:

```bash
uv run python scripts/skrl/play.py --task Isaac-Quad-Swarm-Paper-Crazyflie-v0 --algorithm IPPO --num_envs 8 --checkpoint <checkpoint.pt> --headless
```

The default scripts import `environments.tasks`, which registers this task.

## Parity Report

| Item | Status |
| --- | --- |
| Simulation backend | Isaac Lab 3.0/skrl instead of Omnidrones/Sample Factory. |
| Action convention | Uses `[-1, 1]^4` at the policy boundary, mapped to `[0, 1]` thrust ratios in cpsquare-lab. |
| Attention implementation | Uses PyTorch SDPA while preserving released projection, residual, and LayerNorm structure. |
| Reward coefficients | Centralized in `paper_spec.py`; first-pass values need a source-level coefficient parity pass before score claims. |
| Replay | Lagged collision replay manager implemented as reusable cpsquare-lab utility and wired into reset path. |
| Shared policy | Current stock-skrl path uses independent per-drone policy/value modules. Reusing the same module instances with stock skrl IPPO creates multiple optimizers over the same parameters and has produced non-finite checkpoints. |
| Downwash | Not implemented in this first workable pass. |
| Obstacle actors | Logical obstacle field is implemented for every environment; env-0 visual cylinder obstacles are spawned for Kit/video inspection. |

## Validation Status

Passing pure Python checks:

```text
uv run pytest test/test_quad_swarm_model_forward.py test/test_quad_swarm_task_config.py
uv run pytest ../cpsquare-lab/test/test_swarm_knn.py ../cpsquare-lab/test/test_swarm_sdf_observations.py ../cpsquare-lab/test/test_swarm_events_rewards_replay.py
```

Simulator smoke tests are still pending. The current host prompts for the
NVIDIA Omniverse EULA when importing simulator-facing scripts, so task launch
validation should be run after that EULA is handled by the user.
