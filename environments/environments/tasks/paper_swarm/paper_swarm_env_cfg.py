# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paper swarm waypoint navigation with static obstacle avoidance.

Task:
    A team of CTBR-controlled Crazyflie drones navigates to unique
    target pose commands while avoiding static column obstacles and
    inter-agent collisions. Implements a hybrid of the collision-swarm
    and formation-swarm papers.

Curriculum:
    1. Start with one active drone while possible_agents remains fixed.
    2. Ramp active drone count to NUM_DRONES.
    3. Ramp waypoint sampling from collision-safe assignments to stochastic
       independent sampling.

Training modes:
    - IPPO: noisy actor observations plus decentralized, uncorrupted critic observations.
    - MAPPO: noisy actor observations plus centralized critic observations.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG, CRAZYFLIE_CTBR_CONTROLLER_CFG
from cpsquare_lab.embodiments.multirotor.common.actions import ActionType, CtbrActionCfg, HandleOutOfRangeAction

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg
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

from isaaclab_tasks.manager_based.drone_arl.mdp.commands import DroneUniformPoseCommandCfg

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
INITIAL_ACTIVE_DRONES = 1
ACTIVE_AGENT_RAMP_STEPS = 200_000
OBSTACLE_CURRICULUM_START_STEP = ACTIVE_AGENT_RAMP_STEPS
OBSTACLE_CURRICULUM_RAMP_STEPS = 100_000
SPAWN_TARGET_RANDOMIZATION_START_STEP = OBSTACLE_CURRICULUM_START_STEP + OBSTACLE_CURRICULUM_RAMP_STEPS
SPAWN_TARGET_RANDOMIZATION_RAMP_STEPS = 200_000

NUM_ENVS = 32
ENV_SPACING = 8.0

DRONE_AGENT_IDS = [f"drone_{i}" for i in range(NUM_DRONES)]
ACTIVE_AGENT_MASK_KEY = "active_drones"

WORKSPACE_XY = (-4.0, 4.0)
WORKSPACE_Z = (1.0, 3.0)
SAFE_WAYPOINT_SEPARATION = 2.0
COLLISION_DISTANCE = 0.45
TARGET_REACHED_DISTANCE = 0.25
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
        # -- self block (34 dims) --
        root_lin_vel_b = ObsTerm(func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        root_ang_vel_b = ObsTerm(func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        projected_gravity_b = ObsTerm(
            func=mdp.projected_gravity_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")}
        )
        root_pos = ObsTerm(func=mdp.root_pos, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        root_rotation_matrix = ObsTerm(
            func=mdp.root_rotation_matrix, params={"asset_cfg": SceneEntityCfg("{entity_name}")}
        )
        active_flag = ObsTerm(
            func=mdp.agent_active_flag,
            params={"agent_ids": DRONE_AGENT_IDS, "agent_id": "{agent_id}", "mask_key": ACTIVE_AGENT_MASK_KEY},
        )
        drone_identity = ObsTerm(
            func=mdp.drone_identity,
            params={"agent_ids": DRONE_AGENT_IDS, "agent_id": "{agent_id}"},
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "ctbr"})

        # -- neighbor block (max_neighbors * 6 dims) --
        neighbor_state = ObsTerm(
            func=mdp.neighbor_state_b,
            params={
                "asset_cfg": SceneEntityCfg("{entity_name}"),
                "agent_ids": DRONE_AGENT_IDS,
                "max_neighbors": NUM_DRONES - 1,
                "radius": 6.0,
                "mask_key": ACTIVE_AGENT_MASK_KEY,
            },
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
        )

        # -- goal block (6 dims) --
        target_pos_b = ObsTerm(
            func=mdp.relative_target_position_b,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
        )
        target_yaw_error = ObsTerm(
            func=mdp.target_yaw_error,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
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
    collision_avoidance = RewTerm(
        func=mdp.collision_avoidance_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("{entity_name}"),
            "agent_id": "{agent_id}",
            "agent_ids": DRONE_AGENT_IDS,
            "safe_distance": SAFE_WAYPOINT_SEPARATION,
            "collision_distance": COLLISION_DISTANCE,
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
            "column_radius": COLUMN_RADIUS,
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
            "xy_bounds": (-1.5, 1.5),
            "z_bounds": (1.0, 1.5),
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


# -----------------------------------------------------------------------------
# Environment configs
# -----------------------------------------------------------------------------


@configclass
class PaperSwarmBaseMarlEnvCfg(ManagerBasedMarlEnvCfg):
    scene = PaperSwarmSceneCfg(num_envs=NUM_ENVS, env_spacing=ENV_SPACING)
    sim = SimulationCfg(
        dt=0.01,
        render_interval=2,
        physics=PhysxCfg(
            gpu_total_aggregate_pairs_capacity=SWARM_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY,
        ),
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
                terminations=TerminationsCfg(),
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
                terminations=TerminationsCfg(),
                commands=CommandsCfg(),
                curriculum=CurriculumCfg(),
            ),
        )
    ]


@configclass
class PaperSwarmIppoEnvCfg(PaperSwarmBaseMarlEnvCfg):
    """IPPO variant with decentralized critic observations."""


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
    eval_xy_bound: float = 1.5
    eval_z_min: float = 1.0
    eval_z_max: float = 1.5
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


PaperSwarmMAPPORunnerCfg = PaperSwarmMappoEnvCfg
PaperSwarmIPPORunnerCfg = PaperSwarmIppoEnvCfg
