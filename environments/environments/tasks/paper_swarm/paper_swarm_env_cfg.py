# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paper swarm waypoint navigation with static obstacle avoidance.

Task:
    A team of CTBR-controlled Crazyflie drones navigates to unique
    target pose commands while avoiding static column obstacles and
    inter-agent collisions.

Curriculum:
    1. Single-agent pretraining with passive hovering drones.  Basic
       quadrotor goal navigation and collision avoidance around
       drone-shaped objects.  Neighbor-attention encoder is initialised.
    2. MARL interaction learning.  Learning-drone count ramps from 2
       to 8 while sparse-to-medium static obstacles are introduced.
    3. Target fine-tuning with dense obstacles and strongest simple
       lab-realistic domain randomisation.

Training modes:
    - MAPPO: noisy actor observations plus centralized critic observations.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG, CRAZYFLIE_CTBR_CONTROLLER_CFG
from cpsquare_lab.embodiments.multirotor.common.actions import ActionType, CtbrActionCfg, HandleOutOfRangeAction
from cpsquare_lab.embodiments.multirotor.common.ctbr import hover_collective_thrust_from_multirotor_cfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.drone_arl.mdp.commands import DroneUniformPoseCommandCfg
from isaaclab_tasks.utils import PresetCfg

from environments.envs.manager_based_ma_env_cfg import AgentGroupCfg
from environments.envs.manager_based_marl_env_cfg import (
    AgentRlCfg,
    ManagerBasedMarlEnvCfg,
)

from . import mdp
from .paper_swarm_recorders import PaperSwarmRecorderManagerCfg

# -----------------------------------------------------------------------------
# Global task defaults
# -----------------------------------------------------------------------------

NUM_DRONES = 8
VISIBLE_NEIGHBORS = 2
INITIAL_ACTIVE_DRONES = 1
ACTIVE_AGENT_RAMP_STEPS = 200_000
OBSTACLE_CURRICULUM_START_STEP = 50_000
OBSTACLE_CURRICULUM_RAMP_STEPS = 250_000
SPAWN_TARGET_RANDOMIZATION_START_STEP = OBSTACLE_CURRICULUM_START_STEP + OBSTACLE_CURRICULUM_RAMP_STEPS
SPAWN_TARGET_RANDOMIZATION_RAMP_STEPS = 200_000
STAGE1_TIMESTEPS = 200_000
STAGE2_TIMESTEPS = 300_000
STAGE3_TIMESTEPS = 400_000
PASSIVE_DRONE_RAMP_STEPS = 150_000

NUM_ENVS = 32
ENV_SPACING = 8.0

DRONE_AGENT_IDS = [f"drone_{i}" for i in range(NUM_DRONES)]
ACTIVE_AGENT_MASK_KEY = "active_drones"

WORKSPACE_XY = (-4.0, 4.0)
WORKSPACE_Z = (1.0, 3.0)
START_Z = (1.0, 1.5)
SAFE_WAYPOINT_SEPARATION = 2.0
ROBOT_PROXIMITY_DISTANCE = 0.5
ROBOT_PROXIMITY_MAX_PENALTY = 10.0
COLLISION_DISTANCE = 0.12
OBSTACLE_COLLISION_DISTANCE = 0.2
TARGET_REACHED_DISTANCE = 0.35
TARGET_REACHED_YAW = 0.35
COMMAND_RESAMPLE_TIME = (3.0, 6.0)

STATIC_COLUMNS = 10
COLUMN_RADIUS = 0.15
COLUMN_HEIGHT = 3.0
COLUMN_GRID_SIZE = 0.5
COLUMN_GRID_BORDER = 2.0
COLUMN_MARGIN = 2.0
COLUMN_SAFE_DISTANCE = 0.6
COLUMN_POSITIONS_KEY = "column_positions"

DEFAULT_DRONE_CFG = CRAZYFLIE_CFG
SWARM_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 4 * 1024 * 1024


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class PaperSwarmSceneCfg(InteractiveSceneCfg):
    """Workspace with terrain, lighting, and one drone articulation per agent."""

    num_envs: int = NUM_ENVS
    env_spacing: float = ENV_SPACING
    replicate_physics: bool = True

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        debug_vis=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    def __post_init__(self):
        if DEFAULT_DRONE_CFG is MISSING:
            raise ValueError("DEFAULT_DRONE_CFG is MISSING. Set it to the actual drone ArticulationCfg.")
        for i in range(NUM_DRONES):
            agent_id = f"drone_{i}"
            setattr(self, agent_id, DEFAULT_DRONE_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/{agent_id}"))


# -----------------------------------------------------------------------------
# Commands and actions
# -----------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Per-agent target pose command."""

    target_pose = DroneUniformPoseCommandCfg(
        class_type="environments.tasks.paper_swarm.mdp.commands:PaperSwarmPoseCommand",
        asset_name="{entity_name}",
        body_name="base_link",
        resampling_time_range=COMMAND_RESAMPLE_TIME,
        debug_vis=False,
        ranges=DroneUniformPoseCommandCfg.Ranges(
            pos_x=WORKSPACE_XY,
            pos_y=WORKSPACE_XY,
            pos_z=WORKSPACE_Z,
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Per-agent CTBR action config."""

    ctbr = CtbrActionCfg(
        asset_name="{entity_name}",
        controller_cfg=CRAZYFLIE_CTBR_CONTROLLER_CFG,
        max_roll_pitch_rate=3.0,
        max_yaw_rate=2.0,
        action_type=ActionType.NORM_NEG_1_TO_1,
        handle_out_of_range=HandleOutOfRangeAction.TANH,
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Actor policy observations and critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- self block (26 dims) --
        root_lin_vel_b = ObsTerm(
            func=mdp.root_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("{entity_name}")},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        root_ang_vel_b = ObsTerm(
            func=mdp.root_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("{entity_name}")},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        projected_gravity_b = ObsTerm(
            func=mdp.projected_gravity_b,
            params={"asset_cfg": SceneEntityCfg("{entity_name}")},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        root_pos = ObsTerm(
            func=mdp.root_pos,
            params={"asset_cfg": SceneEntityCfg("{entity_name}")},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        root_rotation_matrix = ObsTerm(
            func=mdp.root_rotation_matrix,
            params={"asset_cfg": SceneEntityCfg("{entity_name}")},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        active_flag = ObsTerm(
            func=mdp.agent_active_flag,
            params={"agent_ids": DRONE_AGENT_IDS, "agent_id": "{agent_id}", "mask_key": ACTIVE_AGENT_MASK_KEY},
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "ctbr"})

        # -- neighbor block (max_neighbors * 6 dims) --
        neighbor_state = ObsTerm(
            func=mdp.neighbor_state_b,
            params={
                "asset_cfg": SceneEntityCfg("{entity_name}"),
                "agent_ids": DRONE_AGENT_IDS,
                "max_neighbors": VISIBLE_NEIGHBORS,
                "radius": 6.0,
                "mask_key": ACTIVE_AGENT_MASK_KEY,
            },
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

        # -- SDF block (9 dims) --
        static_sdf = ObsTerm(
            func=mdp.static_sdf,
            params={
                "asset_cfg": SceneEntityCfg("{entity_name}"),
                "column_positions_key": COLUMN_POSITIONS_KEY,
                "grid_size": 3,
                "grid_resolution": 0.1,
                "column_radius": COLUMN_RADIUS,
            },
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

        # -- goal block (6 dims) --
        target_pos_b = ObsTerm(
            func=mdp.relative_target_position_b,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        target_yaw_error = ObsTerm(
            func=mdp.target_yaw_error,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        goal_reached = ObsTerm(
            func=mdp.goal_reached_flag,
            params={
                "asset_cfg": SceneEntityCfg("{entity_name}"),
                "command_name": "target_pose",
                "distance_threshold": TARGET_REACHED_DISTANCE,
                "yaw_threshold": TARGET_REACHED_YAW,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DecentralizedCriticCfg(PolicyCfg):
        """Local per-agent critic observation without actor observation corruption."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CentralizedCriticCfg(ObsGroup):
        """Global critic observation for CTDE algorithms such as MAPPO."""

        swarm_state = ObsTerm(
            func=mdp.paper_swarm_global_state,
            params={
                "agent_ids": DRONE_AGENT_IDS,
                "command_name": "target_pose",
                "mask_key": ACTIVE_AGENT_MASK_KEY,
                "column_positions_key": COLUMN_POSITIONS_KEY,
                "max_static_columns": STATIC_COLUMNS,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: DecentralizedCriticCfg = DecentralizedCriticCfg()


@configclass
class MappoObservationsCfg(ObservationsCfg):
    """MAPPO observations with a centralized critic group."""

    critic: ObservationsCfg.CentralizedCriticCfg = ObservationsCfg.CentralizedCriticCfg()


# -----------------------------------------------------------------------------
# Rewards, terminations, events, curriculum
# -----------------------------------------------------------------------------


@configclass
class RewardsCfg:
    goal_distance = RewTerm(
        func=mdp.goal_distance_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    waypoint_tracking = RewTerm(
        func=mdp.waypoint_tracking_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "std": 0.8,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    heading_tracking = RewTerm(
        func=mdp.heading_tracking_reward,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "std": 0.7,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    reached_target_bonus = RewTerm(
        func=mdp.reached_target_pose,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "distance_threshold": TARGET_REACHED_DISTANCE,
            "yaw_threshold": TARGET_REACHED_YAW,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    robot_collision = RewTerm(
        func=mdp.robot_collision_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "agent_ids": DRONE_AGENT_IDS,
            "collision_distance": COLLISION_DISTANCE,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    robot_proximity = RewTerm(
        func=mdp.robot_proximity_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "agent_ids": DRONE_AGENT_IDS,
            "falloff_distance": ROBOT_PROXIMITY_DISTANCE,
            "max_penalty": ROBOT_PROXIMITY_MAX_PENALTY,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    obstacle_avoidance = RewTerm(
        func=mdp.obstacle_avoidance_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "column_positions_key": COLUMN_POSITIONS_KEY,
            "column_radius": COLUMN_RADIUS,
            "safe_distance": COLUMN_SAFE_DISTANCE,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    body_rate_l2 = RewTerm(
        func=mdp.body_rate_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    upright = RewTerm(
        func=mdp.upright_reward,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    obstacle_collision = RewTerm(
        func=mdp.obstacle_collision_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "column_positions_key": COLUMN_POSITIONS_KEY,
            "collision_distance": OBSTACLE_COLLISION_DISTANCE,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.drone_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "xy_bounds": (-6.0, 6.0),
            "z_bounds": (0.2, 5.0),
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    drone_collision = DoneTerm(
        func=mdp.drone_pairwise_collision,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "agent_ids": DRONE_AGENT_IDS,
            "collision_distance": COLLISION_DISTANCE,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    column_collision = DoneTerm(
        func=mdp.drone_column_collision,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "column_positions_key": COLUMN_POSITIONS_KEY,
            "column_radius": OBSTACLE_COLLISION_DISTANCE,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


@configclass
class ReplayTrainingTerminationsCfg:
    """Training terminations that leave collisions as replay events."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.drone_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "xy_bounds": (-6.0, 6.0),
            "z_bounds": (0.2, 5.0),
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


@configclass
class PaperSwarmEventsCfg:
    sample_static_columns = EventTerm(
        func=mdp.sample_static_columns,
        mode="reset",
        params={
            "num_columns": STATIC_COLUMNS,
            "grid_size": COLUMN_GRID_SIZE,
            "grid_border": COLUMN_GRID_BORDER,
            "margin": COLUMN_MARGIN,
            "height": COLUMN_HEIGHT,
            "column_radius": COLUMN_RADIUS,
            "column_positions_key": COLUMN_POSITIONS_KEY,
        },
    )
    reset_drone_root_state = EventTerm(
        func=mdp.reset_drone_root_state_uniform,
        mode="reset",
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "xy_bounds": WORKSPACE_XY,
            "z_bounds": START_Z,
            "min_separation": SAFE_WAYPOINT_SEPARATION,
            "lin_vel_range": (-0.05, 0.05),
            "ang_vel_range": (-0.05, 0.05),
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    reset_hover_thrust = EventTerm(
        func=mdp.reset_drone_hover_thrust,
        mode="reset",
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "collective_hover_thrust": hover_collective_thrust_from_multirotor_cfg(CRAZYFLIE_CFG),
        },
    )


@configclass
class Stage1EventsCfg(PaperSwarmEventsCfg):
    """Stage 1: active drone at random position, passive drones at sampled hover positions.

    Hover thrust is applied to passive drones enabled by the passive-drone
    curriculum. Inactive drones are parked near the ground.
    """

    reset_drone_root_state = EventTerm(
        func=mdp.reset_drone_root_state_uniform,
        mode="reset",
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "xy_bounds": WORKSPACE_XY,
            "z_bounds": START_Z,
            "min_separation": SAFE_WAYPOINT_SEPARATION,
            "lin_vel_range": (-0.05, 0.05),
            "ang_vel_range": (-0.05, 0.05),
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


@configclass
class CurriculumCfg:
    active_agent_count = CurrTerm(
        func=mdp.active_agent_count_curriculum,
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "min_agents": INITIAL_ACTIVE_DRONES,
            "max_agents": NUM_DRONES,
            "ramp_steps": ACTIVE_AGENT_RAMP_STEPS,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
            "selection": "prefix",
        },
    )
    paper_swarm_task = CurrTerm(
        func=mdp.paper_swarm_task_curriculum,
        params={
            "workspace_xy": WORKSPACE_XY,
            "workspace_z": WORKSPACE_Z,
            "max_static_columns": STATIC_COLUMNS,
            "obstacle_start_step": OBSTACLE_CURRICULUM_START_STEP,
            "obstacle_ramp_steps": OBSTACLE_CURRICULUM_RAMP_STEPS,
            "randomization_start_step": SPAWN_TARGET_RANDOMIZATION_START_STEP,
            "randomization_ramp_steps": SPAWN_TARGET_RANDOMIZATION_RAMP_STEPS,
            "start_safe_sampling_prob": 1.0,
            "end_safe_sampling_prob": 0.0,
            "start_min_separation": SAFE_WAYPOINT_SEPARATION,
            "end_min_separation": 0.0,
            "column_radius": COLUMN_RADIUS,
            "column_safe_distance": COLUMN_SAFE_DISTANCE,
        },
    )


@configclass
class Stage1CurriculumCfg:
    """Single-drone waypoint control with passive hovering drones.

    The active learning agent remains drone_0 while the passive drone count
    ramps from 1 up to 7.  Target range expands from zero outward.
    """

    active_agent_count = CurrTerm(
        func=mdp.active_agent_count_curriculum,
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "min_agents": INITIAL_ACTIVE_DRONES,
            "max_agents": INITIAL_ACTIVE_DRONES,
            "ramp_steps": 10_000_000,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
            "selection": "prefix",
        },
    )
    passive_drone_count = CurrTerm(
        func=mdp.passive_drone_count_curriculum,
        params={
            "min_passive": 1,
            "max_passive": 7,
            "ramp_steps": PASSIVE_DRONE_RAMP_STEPS,
        },
    )
    paper_swarm_task = CurrTerm(
        func=mdp.paper_swarm_task_curriculum,
        params={
            "workspace_xy": WORKSPACE_XY,
            "workspace_z": WORKSPACE_Z,
            "max_static_columns": 0,
            "obstacle_start_step": 1_000_000_000,
            "obstacle_ramp_steps": 1,
            "randomization_start_step": 1_000_000_000,
            "randomization_ramp_steps": 1,
            "start_safe_sampling_prob": 1.0,
            "end_safe_sampling_prob": 1.0,
            "start_min_separation": SAFE_WAYPOINT_SEPARATION,
            "end_min_separation": SAFE_WAYPOINT_SEPARATION,
            "column_radius": COLUMN_RADIUS,
            "column_safe_distance": COLUMN_SAFE_DISTANCE,
        },
    )
    expand_target_range = CurrTerm(
        func=mdp.expand_target_range_curriculum,
        params={
            "start_step": 0,
            "end_step": 50_000,
            "start_xy": 0.0,
            "end_xy": 1.5,
            "start_z_delta": 0.0,
            "end_z_delta": 0.5,
        },
    )
    observation_noise = CurrTerm(
        func=mdp.update_observation_noise_curriculum,
        params={"start_step": 30_000, "end_step": 100_000, "final_noise": 0.01},
    )


@configclass
class Stage2CurriculumCfg:
    """MARL interaction learning: agent count ramps 2→8, sparse→medium obstacles.

    Obstacles are introduced halfway through after agents have learned
    basic drone-drone interaction.
    """

    active_agent_count = CurrTerm(
        func=mdp.active_agent_count_curriculum,
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "min_agents": 2,
            "max_agents": NUM_DRONES,
            "ramp_steps": STAGE2_TIMESTEPS,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
            "selection": "prefix",
        },
    )
    paper_swarm_task = CurrTerm(
        func=mdp.paper_swarm_task_curriculum,
        params={
            "workspace_xy": WORKSPACE_XY,
            "workspace_z": WORKSPACE_Z,
            "max_static_columns": 6,
            "obstacle_start_step": OBSTACLE_CURRICULUM_START_STEP,
            "obstacle_ramp_steps": OBSTACLE_CURRICULUM_RAMP_STEPS,
            "randomization_start_step": 0,
            "randomization_ramp_steps": 100_000,
            "start_safe_sampling_prob": 0.5,
            "end_safe_sampling_prob": 0.0,
            "start_min_separation": 1.0,
            "end_min_separation": 0.0,
            "column_radius": COLUMN_RADIUS,
            "column_safe_distance": COLUMN_SAFE_DISTANCE,
        },
    )
    observation_noise = CurrTerm(
        func=mdp.update_observation_noise_curriculum,
        params={"start_step": 0, "end_step": 100_000, "final_noise": 0.03},
    )


@configclass
class Stage3CurriculumCfg:
    """Target fine-tuning: mostly 8-drone episodes, medium→dense obstacles, strong DR.

    Obstacles ramp from medium (4) to target dense (10).  Target/spawn
    sampling is fully randomised from the start.
    """

    active_agent_count = CurrTerm(
        func=mdp.active_agent_count_curriculum,
        params={
            "agent_ids": DRONE_AGENT_IDS,
            "min_agents": NUM_DRONES,
            "max_agents": NUM_DRONES,
            "ramp_steps": 1,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
            "selection": "prefix",
        },
    )
    paper_swarm_task = CurrTerm(
        func=mdp.paper_swarm_task_curriculum,
        params={
            "workspace_xy": WORKSPACE_XY,
            "workspace_z": WORKSPACE_Z,
            "max_static_columns": STATIC_COLUMNS,
            "obstacle_start_step": 0,
            "obstacle_ramp_steps": STAGE3_TIMESTEPS,
            "randomization_start_step": 0,
            "randomization_ramp_steps": 1,
            "start_safe_sampling_prob": 0.0,
            "end_safe_sampling_prob": 0.0,
            "start_min_separation": 0.0,
            "end_min_separation": 0.0,
            "column_radius": COLUMN_RADIUS,
            "column_safe_distance": COLUMN_SAFE_DISTANCE,
        },
    )
    observation_noise = CurrTerm(
        func=mdp.update_observation_noise_curriculum,
        params={"start_step": 0, "end_step": 150_000, "final_noise": 0.05},
    )


# -----------------------------------------------------------------------------
# Environment configs
# -----------------------------------------------------------------------------


@configclass
class PaperSwarmPhysicsCfg(PresetCfg):
    """Physics backend preset for PhysX and Newton."""

    default: PhysxCfg = PhysxCfg(
        gpu_total_aggregate_pairs_capacity=SWARM_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY,
    )
    physx: PhysxCfg = PhysxCfg(
        gpu_total_aggregate_pairs_capacity=SWARM_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY,
    )
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=200,
            nconmax=200,
            ls_iterations=20,
            cone="pyramidal",
            ls_parallel=True,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )


@configclass
class PaperSwarmBaseMarlEnvCfg(ManagerBasedMarlEnvCfg):
    initial_passive_drone_count: int = NUM_DRONES - INITIAL_ACTIVE_DRONES
    initial_static_column_count: int = STATIC_COLUMNS
    initial_workspace_xy: tuple[float, float] = WORKSPACE_XY
    initial_workspace_z: tuple[float, float] = WORKSPACE_Z
    initial_safe_sampling_prob: float = 1.0
    initial_spawn_min_separation: float = SAFE_WAYPOINT_SEPARATION
    initial_target_min_separation: float = SAFE_WAYPOINT_SEPARATION
    scene = PaperSwarmSceneCfg(num_envs=NUM_ENVS, env_spacing=ENV_SPACING)
    sim = SimulationCfg(
        dt=0.01,
        render_interval=2,
        physics=PaperSwarmPhysicsCfg(),
        render=sim_utils.RenderCfg(
            carb_settings={
                "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    viewer = ViewerCfg(eye=(7.5, 7.5, 5.0), lookat=(0.0, 0.0, 1.5))
    events = PaperSwarmEventsCfg()

    possible_agents = DRONE_AGENT_IDS
    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=ObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=ReplayTrainingTerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=CurriculumCfg(),
            ),
        )
    ]
    observation_group = "policy"
    active_agent_mask_key = ACTIVE_AGENT_MASK_KEY
    reset_on = "any"

    episode_length_s = 20.0
    is_finite_horizon = False
    decimation = 2
    replay_enabled = True
    replay_probability = 0.75
    replay_lag_s = 1.5
    replay_capacity = 1024
    replay_max_uses = 4
    collision_grace_period_s = 1.5
    collision_distance = COLLISION_DISTANCE
    obstacle_collision_distance = OBSTACLE_COLLISION_DISTANCE

    recorders = None


@configclass
class PaperSwarmMappoEnvCfg(PaperSwarmBaseMarlEnvCfg):
    """MAPPO variant with centralized critic observations."""

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=ReplayTrainingTerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=CurriculumCfg(),
            ),
        )
    ]


@configclass
class PaperSwarmIppoEnvCfg(PaperSwarmBaseMarlEnvCfg):
    """IPPO variant with decentralized critic observations."""


@configclass
class Stage1CommandsCfg:
    """Target starts at drone position, curriculum expands range outward."""

    target_pose = DroneUniformPoseCommandCfg(
        asset_name="{entity_name}",
        body_name="base_link",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        ranges=DroneUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(1.0, 1.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class Stage1EvalCommandsCfg:
    """Target pose ranges matching end-of-curriculum (Stage 1 eval)."""

    target_pose = DroneUniformPoseCommandCfg(
        asset_name="{entity_name}",
        body_name="base_link",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        ranges=DroneUniformPoseCommandCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-1.5, 1.5),
            pos_z=(0.5, 1.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class Stage1ActionsCfg:
    """Stage 1 uses TANH to match later stages (avoid transfer shock)."""

    ctbr = CtbrActionCfg(
        asset_name="{entity_name}",
        controller_cfg=CRAZYFLIE_CTBR_CONTROLLER_CFG,
        max_roll_pitch_rate=3.0,
        max_yaw_rate=2.0,
        action_type=ActionType.NORM_NEG_1_TO_1,
        handle_out_of_range=HandleOutOfRangeAction.TANH,
    )


@configclass
class Stage1RewardsCfg:
    waypoint_tracking = RewTerm(
        func=mdp.waypoint_tracking_reward,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "std": 0.8,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    heading_tracking = RewTerm(
        func=mdp.heading_tracking_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "std": 0.7,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    reached_target_bonus = RewTerm(
        func=mdp.reached_target_pose,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "distance_threshold": TARGET_REACHED_DISTANCE,
            "yaw_threshold": TARGET_REACHED_YAW,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    robot_collision = RewTerm(
        func=mdp.robot_collision_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "agent_ids": DRONE_AGENT_IDS,
            "collision_distance": COLLISION_DISTANCE,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    robot_proximity = RewTerm(
        func=mdp.robot_proximity_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "agent_ids": DRONE_AGENT_IDS,
            "falloff_distance": ROBOT_PROXIMITY_DISTANCE,
            "max_penalty": ROBOT_PROXIMITY_MAX_PENALTY,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    crash_penalty = RewTerm(
        func=mdp.crash_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "minimum_height": 0.2,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    body_rate_l2 = RewTerm(
        func=mdp.body_rate_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


@configclass
class Stage1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(
        func=mdp.drone_crash,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "minimum_height": 0.2,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    too_far_from_command = DoneTerm(
        func=mdp.pose_command_error_above_masked,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "command_name": "target_pose",
            "max_position_error": 4.0,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )
    out_of_bounds = DoneTerm(
        func=mdp.drone_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "xy_bounds": (-6.0, 6.0),
            "z_bounds": (0.2, 5.0),
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


@configclass
class PaperSwarmMappoStage1EnvCfg(PaperSwarmBaseMarlEnvCfg):
    """MAPPO stage 1: single-drone waypoint control with passive hovering drones.

    Only ``drone_0`` is managed by the RL pipeline.  The other 7 drones are
    physically simulated and actively hover in place via Lee position
    controllers — they appear in the neighbor-attention stream so the policy
    learns basic collision avoidance around drone-shaped objects.

    This stage uses 8192 environments for massive parallelisation.
    """

    scene = PaperSwarmSceneCfg(num_envs=8192, env_spacing=ENV_SPACING)
    decimation = 2
    events = Stage1EventsCfg()
    possible_agents = ["drone_0"]
    initial_passive_drone_count = 1
    initial_static_column_count = 0
    replay_enabled = False

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=1,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=Stage1ActionsCfg(),
                rewards=Stage1RewardsCfg(),
                terminations=Stage1TerminationsCfg(),
                commands=Stage1CommandsCfg(),
                curriculum=Stage1CurriculumCfg(),
            ),
        )
    ]


@configclass
class PaperSwarmMappoStage2EnvCfg(PaperSwarmBaseMarlEnvCfg):
    """MAPPO stage 2: MARL interaction learning with variable agent count.

    Learning-drones ramp from 2 to 8 over the stage.  Sparse-to-medium
    static obstacles are introduced halfway through.  Actor is initialised
    from the Stage 1 checkpoint.
    """

    initial_static_column_count = 0

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=ReplayTrainingTerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=Stage2CurriculumCfg(),
            ),
        )
    ]


@configclass
class PaperSwarmMappoStage3EnvCfg(PaperSwarmBaseMarlEnvCfg):
    """MAPPO stage 3: target fine-tuning with dense obstacles and strong DR.

    Mostly 8-drone episodes with medium-to-dense static obstacles.
    Hard-case collision/deadlock replay and strongest simple lab DR.
    """

    initial_static_column_count = 0

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=ReplayTrainingTerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=Stage3CurriculumCfg(),
            ),
        )
    ]


@configclass
class PaperSwarmEvalEnvCfg(PaperSwarmBaseMarlEnvCfg):
    """Evaluation variant for play.py -- few envs, short episodes, recorder enabled.

    Uses 4 parallel environments with 8 drones each. Episodes are shortened
    to help the RecorderManager flush HDF5 data more frequently.

    Usage with play.py::

        uv run scripts/skrl/play.py --task Isaac-Paper-Swarm-Waypoint-Eval-v0 \\
            --algorithm IPPO --checkpoint <path>

    After the run, recorder data is in ``/tmp/isaaclab/logs/paper_swarm_dataset.hdf5``.
    """

    scene = PaperSwarmSceneCfg(num_envs=4, env_spacing=ENV_SPACING)
    episode_length_s = 10.0  # shorter than train (20s)
    recorders = PaperSwarmRecorderManagerCfg()

    # Workspace bounds used by InitialStateCheckRecorder
    eval_xy_bound: float = abs(WORKSPACE_XY[1])
    eval_z_min: float = START_Z[0]
    eval_z_max: float = START_Z[1]
    eval_min_separation: float = SAFE_WAYPOINT_SEPARATION

    # All drones active during eval (no curriculum)
    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=ObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=TerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=None,
            ),
        )
    ]


@configclass
class PaperSwarmMappoEvalEnvCfg(PaperSwarmEvalEnvCfg):
    """Evaluation variant for MAPPO checkpoints with centralized critic observations."""

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=TerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=None,
            ),
        )
    ]


@configclass
class PaperSwarmMappoStage1EvalCfg(PaperSwarmEvalEnvCfg):
    """MAPPO Stage 1 eval: single-drone waypoint control with passive drones.

    Inherits recorder, shorter episodes, and eval workspace bounds.
    Uses Stage 1 actions/rewards/terminations and fixed target range
    matching end-of-curriculum XY and Z.
    """

    possible_agents = ["drone_0"]
    events = Stage1EventsCfg()
    initial_passive_drone_count = NUM_DRONES - 1
    initial_static_column_count = 0
    eval_z_max: float = WORKSPACE_Z[1]
    eval_min_separation: float = 1.0
    replay_enabled = False

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=1,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=Stage1ActionsCfg(),
                rewards=Stage1RewardsCfg(),
                terminations=Stage1TerminationsCfg(),
                commands=Stage1EvalCommandsCfg(),
                curriculum=None,
            ),
        )
    ]


@configclass
class PaperSwarmMappoStage3EvalCfg(PaperSwarmEvalEnvCfg):
    """MAPPO Stage 3 eval: 8-drone waypoint navigation with obstacles.

    Inherits recorder, shorter episodes, and eval workspace bounds.
    Uses Stage 3 actions/rewards/terminations/observations.  Obstacles are
    active (``PaperSwarmEventsCfg.sample_static_columns`` runs with
    ``num_columns=10`` and no curriculum to suppress them).
    """

    initial_static_column_count = STATIC_COLUMNS

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=TerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=None,
            ),
        )
    ]


@configclass
class PaperSwarmMappoStage2EvalCfg(PaperSwarmEvalEnvCfg):
    """MAPPO Stage 2 eval: multi-drone waypoint navigation with sparse obstacles.

    Inherits recorder, shorter episodes, and eval workspace bounds.
    Matches Stage 2 observations/actions/rewards/terminations.
    """

    initial_static_column_count = 0

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=NUM_DRONES,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=MappoObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=TerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=None,
            ),
        )
    ]


PaperSwarmMAPPORunnerCfg = PaperSwarmMappoEnvCfg
PaperSwarmIPPORunnerCfg = PaperSwarmIppoEnvCfg
