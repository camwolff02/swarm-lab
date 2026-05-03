r"""Constants for the quadrotor swarm obstacle-navigation paper task.

The constants define the observation contract, room geometry, policy cadence,
collision radii, and reward scales used by the task implementation and model
builders. Distances are meters and times are seconds unless noted otherwise.

The observation dimension is computed from the self-state vector, visible-neighbor
features, and local obstacle SDF grid:

\[
D_\text{obs} = D_\text{self} + kD_\text{neighbor} + D_\text{sdf}.
\]

The dense reward scales in this module parameterize the environment equation

\[
r_\text{dense} = -\Delta t\left(
  w_d d + w_a\lVert a\rVert_2 + w_f f + w_q q + w_\omega\lVert\omega\rVert_2
\right).
\]
"""

from __future__ import annotations

NUM_DRONES = 8
VISIBLE_NEIGHBORS = 2
SELF_OBS_SIZE = 19
NEIGHBOR_OBS_SIZE = 6
OBSTACLE_OBS_SIZE = 9
# Observations are concatenated as ``[self, k nearest neighbors, local SDF]``.
# The self vector contains goal-relative pose and vehicle state, each neighbor
# contributes relative position and velocity, and the obstacle vector is a 3x3
# local signed-distance grid.
OBS_SIZE = SELF_OBS_SIZE + VISIBLE_NEIGHBORS * NEIGHBOR_OBS_SIZE + OBSTACLE_OBS_SIZE
ACTION_SIZE = 4

ROOM_SIZE = 10.0
ROOM_HEIGHT = 10.0
OBSTACLE_GRID_SHAPE = (8, 8)
OBSTACLE_CELL_SIZE = 1.0
OBSTACLE_DENSITY = 0.2
OBSTACLE_SIZE = 0.6
OBSTACLE_RADIUS = OBSTACLE_SIZE * 0.5
OBSTACLE_COUNT = int(OBSTACLE_DENSITY * OBSTACLE_GRID_SHAPE[0] * OBSTACLE_GRID_SHAPE[1])
LOCAL_SDF_RESOLUTION = 0.1

EPISODE_LENGTH_S = 15.0
SIM_DT = 0.01
# Decimation is the ratio between physics frequency and policy frequency. With
# ``SIM_DT = 0.01`` and ``DECIMATION = 2``, physics runs at 100 Hz and the policy
# updates at 50 Hz.
DECIMATION = 2
HIDDEN_SIZE = 256
ATTENTION_HEADS = 4
LEARNING_RATE = 1.0e-4
ROLLOUT_LENGTH = 128
BATCH_SIZE = 1024
REPLAY_PROBABILITY = 0.75
# Collision replay waits this many simulated seconds before storing a state that
# later collided. Resetting from the lagged state exposes the policy to the
# lead-up, not just the terminal collision frame.
REPLAY_LAG_S = 1.5

GOAL_REACHED_RADIUS = 0.35
FLOOR_CRASH_HEIGHT = 0.08
# The release computes robot-robot collision and proximity radii as multiples
# of the quad arm length: hitbox=2*arm and falloff=4*arm, with arm ~= 0.05m.
ROBOT_COLLISION_RADIUS = 0.10
ROBOT_PROXIMITY_RADIUS = 0.20
OBSTACLE_COLLISION_ROBOT_RADIUS = 0.05
COLLISION_GRACE_PERIOD_S = 1.5
REPLAY_ACTIVATION_EPISODES = 10
REPLAY_ACTIVATION_CRASH_SIGNAL_THRESHOLD = 1.0

OBSERVATION_CLIP = 10.0
REWARD_CLIP = 10.0
COLLISION_PENALTY_ANNEAL_STEPS = 0
DEBUG_ROLLOUT_DUMP = False

# These coefficients are centralized here so paper-vs-release drift is visible.
# These match the released obstacle baseline unless noted otherwise.
GOAL_REWARD_SCALE = 1.0
GOAL_REWARD_DISTANCE_SCALE = 1.0
ROBOT_COLLISION_PENALTY = 5.0
OBSTACLE_COLLISION_PENALTY = 5.0
PROXIMITY_PENALTY = 4.0
FLOOR_CRASH_PENALTY = 1.0
ANGULAR_VELOCITY_PENALTY_SCALE = 0.1
CONTROL_EFFORT_PENALTY_SCALE = 0.05
TILT_PENALTY_SCALE = 1.0
