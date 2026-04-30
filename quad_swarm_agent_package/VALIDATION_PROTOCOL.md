# Validation protocol

Use this in order. Do not skip the early checks.

## A. Static checks

### A1. Compute hover ratio and hover action from the vehicle YAML

Run:

```bash
python quad_swarm_agent_package/scripts/compute_hover_action.py \
  ../cpsquare-lab/cpsquare_lab/embodiments/multirotor/cf2x/sim/cf2x.yaml
```

Expected for the uploaded translated Crazyflie:

- `hover_ratio` should be about `0.625`
- `hover_action` should be about `+0.25`
- total thrust-to-weight should be about `1.6`

### A2. Confirm model initialization

Add a one-time assertion or debug print after policy construction:

- `mean_head.bias` equals the hover-action vector within tolerance.
- `log_std_parameter` equals the configured initial value.

### A3. Confirm direct rotor mapping is still unchanged

Smoke check:

- action `-1` maps to `0` thrust
- action `+1` maps to `max_thrust`
- action `hover_action` maps close to `hover_thrust`

## B. No-training rollout smoke test

Goal: verify that a freshly initialized deterministic policy does not instantly crash all drones.

### Procedure

- Run one environment or a very small batch.
- Force deterministic inference if possible.
- Record the first 2 to 3 seconds of altitude, floor fraction, action mean, and thrust ratio mean.

### Pass condition

At minimum, the swarm should no longer show the old pattern of starting in the air and immediately falling to the ground without trying to fly.

A stronger pass condition is:

- median altitude at 1 second stays above `0.6 m`
- not all drones are on the floor by 1 second
- thrust ratio mean starts near `hover_ratio` rather than `0.5`

## C. Replay gate smoke test

Goal: verify that replay now becomes reachable.

### Suggested unit-style synthetic check

For `dt = 0.01`, `num_drones = 8`, and one drone crashed for 100 steps:

- old counter: `100`
- new crash signal: `100 * 0.01 * (1 / 8) = 0.125`

### Runtime pass condition

In early smoke training, `replay_active` eventually becomes `1` and `replay_reset` is nonzero.

## D. Short training smoke test

Use the current training command, but do not treat 100k steps as a convergence test.

```bash
uv run python scripts/skrl/train.py \
  --task Isaac-Quad-Swarm-Paper-Crazyflie-v0 \
  --algorithm IPPO
```

### What to look for

- `avg_altitude` stops collapsing immediately after reset.
- `floor_fraction` trends down relative to the old baseline.
- `replay_active` turns on.
- reward no longer looks dominated by long grounded segments.

## E. Longer training test

Once A through D pass, run a longer training budget. Treat `>= 1M` environment timesteps as the first meaningful comparison point, not `100k`.

### Pass condition

- reward trend improves materially over the baseline
- final-distance metric improves
- inference rollouts sustain flight and make visible progress toward goals

