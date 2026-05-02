# Instrumentation spec

These are the smallest metrics that will make debugging much faster.

## Startup print or logger line

Emit once at environment creation or policy construction:

- `vehicle_mass`
- `num_rotors`
- `hover_thrust`
- `max_thrust_per_rotor`
- `total_thrust_to_weight`
- `hover_ratio`
- `hover_action`
- `initial_policy_log_std`

## Per-episode metrics

Add these to the environment's metric set:

- `avg_altitude`
- `floor_fraction`
- `action_mean`
- `thrust_ratio_mean`
- `crash_signal`
- `replay_active`
- `replay_reset`
- `robot_robot_collisions`
- `robot_obstacle_collisions`
- `final_distance`

## Optional debug rollout dump

When a debug flag is enabled, dump the first 200 to 300 steps after reset for one environment:

- timestep
- altitude mean
- floor fraction
- sampled action mean
- applied thrust ratio mean
- goal distance mean

This can be written to CSV or JSONL. The goal is simply to prove whether the initial control signal is below hover, near hover, or wildly saturating.

## Minimal acceptance graph set

If you only plot four things in TensorBoard, plot these:

1. `avg_altitude`
2. `floor_fraction`
3. `thrust_ratio_mean`
4. `replay_active`
