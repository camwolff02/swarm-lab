# Environment specification

## Target environment form

Implement the task as a **`DirectMARLEnv`** with one homogeneous agent definition reused across all drones.

### Agent naming
Use deterministic agent IDs such as:
- `drone_0`
- `drone_1`
- ...
- `drone_{N-1}`

### State space
First pass: set `state_space = 0` (no centralized state).

### Observation space
Each agent gets a flat observation vector.

Recommended flat ordering:
1. `self_obs`
2. `neighbor_obs`
3. `obstacle_obs`

Keep this ordering identical in:
- the environment output
- model slicing code
- tests
- docs

## Base paper setting

Use this as the default parity target:
- number of robots: `8`
- sensed nearest neighbors: `K = 2`
- obstacle density: `0.2`
- obstacle size: `0.6 m`
- default obstacle count if matching the release integer cast: `int(0.2 * 8 * 8) = 12`
- room size: `10 x 10 x 10 m`
- obstacle sampling area: centered `8 x 8 m` region, discretized into `64` cells of `1 m^2`
- episode duration: `15.0 s`
- downwash: enabled for the parity baseline once the task is stable

## Observation definition

### Self observation
Use the paper self-state layout:
- relative position to goal: `3`
- linear velocity: `3`
- rotation matrix body-to-world: `9`
- angular velocity: `3`
- altitude: `1`

Total self observation size: `19`

### Neighbor observation
For each of the `K` nearest neighbors, include:
- relative position: `3`
- relative velocity: `3`

Per-neighbor size: `6`

For `K = 2`, total neighbor observation size: `12`

### Obstacle observation
Use the paper/repo local obstacle observation:
- `9` values
- a `3 x 3` local grid in the XY plane
- grid resolution `0.1 m`
- value at each cell is the local obstacle signed-distance-like value to the nearest obstacle center minus obstacle radius

Total obstacle observation size: `9`

### Total observation size
For `K = 2`:
- `19 + 12 + 9 = 40`

## K-nearest-neighbor selection

Implement reusable KNN logic in `cpsquare-lab`.

Requirements:
- select neighbors per agent each step
- exclude self
- sort by Euclidean distance in position space
- support fixed `K`
- return relative position and relative velocity blocks in deterministic order
- clip or pad only if needed by configuration; for the paper base setting, fixed `K=2` should be available directly

## Local obstacle SDF

Implement the obstacle observation in reusable cpsquare-lab code.

Behavior to match:
- for each drone, place a 3 x 3 grid centered on the drone in XY
- offsets are `[-resolution, 0, +resolution]` in both X and Y
- resolution default is `0.1 m`
- each cell stores the minimum distance from that XY sample point to any obstacle center, minus obstacle radius
- initialize missing/no-obstacle cells with `100.0` to match the release behavior

## Obstacle-room generation

Keep the exact room preset and obstacle layout code in the main workspace if it stays paper-specific.

Behavior to reproduce:
- sample obstacle occupancy over the centered `8 x 8` cell map
- obstacle count should match the configured density times cell count
- obstacle size should match the configured side length / radius convention used by the original code
- keep obstacle placement deterministic under seeds

## Actions

Recommended environment-facing action:
- `Box(-1, 1, (4,))`

Recommended physical control path:
- convert once in reusable cpsquare-lab control code into the thrust command expected by the multirotor dynamics

Document the chosen mapping clearly because the paper text and released code use slightly different conventions.

## Rewards

Compose the reward from reusable primitives, but keep the paper-specific coefficients in `paper_spec.py`.

Required terms:
- distance-to-goal reward
- robot-robot collision penalty
- robot-obstacle collision penalty
- close-proximity penalty to other robots
- floor/crash penalty
- angular-velocity penalty
- control-effort penalty
- orientation/tilt penalty

Critical behavior:
- collisions should be penalized **once per collision event**, not every step forever after contact is established

## Terminations / truncations

At minimum support:
- episode timeout at `15.0 s`
- success metric when a goal is reached, if the original task exposes one
- crash/floor termination if the original task uses it
- keep truncation vs termination semantics explicit and test them

## Reset behavior

Support two reset paths:
1. fresh reset from newly sampled initial conditions
2. replay reset from the collision replay manager

The task should not need to know replay-buffer internals beyond:
- requesting a replay start state when available and selected
- restoring that state
- reporting new collision states back to the replay manager

## Metrics to log

Track at least:
- success rate
- collision rate
- final distance to goal
- robot-robot collisions per episode
- robot-obstacle collisions per episode
- replay reset rate
- timeout rate
- crash rate

## Recommended tests

Add tests for:
- observation size and ordering
- KNN selection correctness
- local SDF values for a small known obstacle layout
- reward term activation for synthetic collision/proximity cases
- replay reset correctness
