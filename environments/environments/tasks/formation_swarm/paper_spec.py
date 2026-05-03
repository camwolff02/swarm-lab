r"""Constants for the Xie et al. multi-UAV formation-control task.

The values in this module define the task contract used by the environment,
models, and agent YAML files. Distances are meters, velocities are meters per
second, angular rates are radians per second unless a downstream controller
explicitly documents degrees, and time-step constants are simulator seconds.

The four MORL weights scalarize smoothness, obstacle, forward-progress, and formation
objectives:

\[
r = w_s r_\text{smooth} + w_o r_\text{obstacle}
  + w_f r_\text{forward} + w_\ell r_\text{formation}.
\]

The observation dimension follows the task observation layout:

\[
D_\text{obs} =
D_\text{self} + (N-1)D_\text{other} + B D_\text{dynamic} + D_\text{static}.
\]
"""

from __future__ import annotations

NUM_DRONES = 3
NUM_BALLS = 2
STATIC_OBSTACLES = 10

FORMATION = (
    (1.0, 0.0, 0.0),
    (-0.5, 0.866, 0.0),
    (-0.5, -0.866, 0.0),
)
# Equilateral triangle vertices. ``0.866`` approximates sqrt(3) / 2, so scaling
# by ``FORMATION_SIZE`` preserves the source paper's triangular shape.
FORMATION_SIZE = 1.0
TARGET_POS = (0.0, 0.0, 1.5)
TARGET_VEL = (0.0, 2.0, 0.0)
TARGET_HEADING = (1.0, 0.0, 0.0)

SIM_DT = 0.01
# The control policy runs every ``DECIMATION`` physics steps. With ``SIM_DT`` at
# 0.01 s and decimation 2, the physics loop runs at 100 Hz and actions update at
# 50 Hz.
DECIMATION = 2
EPISODE_LENGTH_S = 9.0

GRID_SIZE = 0.5
GRID_BORDER = 2.0
STATIC_MARGIN = 2.0
STATIC_HEIGHT = 5.0
COLUMN_RADIUS = 0.15
BALL_RADIUS = 0.15
OBS_RANGE = 10.0

MIN_BALL_SPEED = 2.0
MAX_BALL_SPEED = 5.0
THROW_THRESHOLD_STEPS = 150
THROW_TIME_RANGE_STEPS = 450

HARD_SAFE_DISTANCE = 0.15
SAFE_DISTANCE = 0.4
OBS_SAFE_DISTANCE = 0.4
SOFT_OBS_SAFE_DISTANCE = 0.6
CRASH_MIN_HEIGHT = 0.2
CRASH_MAX_HEIGHT = 2.8

SELF_OBS_DIM = 29
OTHER_OBS_DIM = 7
DYNAMIC_OBS_DIM = 10
STATIC_SDF_DIM = 9
OBS_DIM = SELF_OBS_DIM + (NUM_DRONES - 1) * OTHER_OBS_DIM + NUM_BALLS * DYNAMIC_OBS_DIM + STATIC_SDF_DIM
ACTION_DIM = 4

ATTENTION_DIM = 32
ATTENTION_HEADS = 1
MLP_HIDDEN = (256, 256, 256)
INITIAL_LOG_STD = 0.0

MORL_SMOOTH_WEIGHT = 0.5123452752249692
MORL_FORMATION_WEIGHT = 0.14240923087088264
MORL_OBSTACLE_WEIGHT = 0.2187620445033934
MORL_FORWARD_WEIGHT = 0.1264834494007547
# MORL weights are the scalarization coefficients used to combine smoothness,
# formation, obstacle-avoidance, and forward-progress objectives into one PPO
# reward. They intentionally sum to 1.0 within floating-point tolerance.

FORMATION_COEFF = 5.0
FORMATION_SIZE_COEFF = 5.0
SEPARATION_COEFF = 1.0
TOO_CLOSE_PENALTY = -10.0

BALL_REWARD_COEFF = 10.0
BALL_HARD_REWARD_COEFF = 100.0
STATIC_HARD_COEFF = 1.0
HIT_PENALTY = -20.0

VELOCITY_COEFF = 10.0
HEADING_COEFF = 1.0
HEIGHT_COEFF = 5.0
POSITION_REWARD_COEFF = 50.0
TRUNCATED_REWARD = 10.0
ACCEPTABLE_V_DIFF = 1.0

EFFORT_WEIGHT = 0.5
ACTION_SMOOTHNESS_WEIGHT = 1.0
SPIN_REWARD_COEFF = 1.0
THROTTLE_SMOOTHNESS_WEIGHT = 2.0
