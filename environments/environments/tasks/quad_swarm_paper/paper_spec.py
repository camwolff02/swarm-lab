"""Paper constants for the quadrotor swarm obstacle-navigation task."""

from __future__ import annotations

NUM_DRONES = 8
VISIBLE_NEIGHBORS = 2
SELF_OBS_SIZE = 19
NEIGHBOR_OBS_SIZE = 6
OBSTACLE_OBS_SIZE = 9
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
DECIMATION = (
    2  # Possibly increase from 2 to 4 to reduce RL frequency relative to the physics frequency, improve stuttering
    # Decimation controls how many environment steps we wait before acting.
    # The ration between the physics frequency and the control frequency
    # If physics runs at 100Hz, and Decimation = 2, agent acts at 50Hz. If Decimation = 5, agent acts at 20Hz
)
HIDDEN_SIZE = 25
ATTENTION_HEADS = 4
LEARNING_RATE = 1.0e-4
ROLLOUT_LENGTH = 128
BATCH_SIZE = 1024
REPLAY_PROBABILITY = 0.75
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
