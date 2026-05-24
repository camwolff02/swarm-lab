# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cooperative drone waypoint MARL config for the V2 ManagerBasedMarlEnv.

Task:
    A team of CTBR-controlled drones navigates to target pose commands in an
    empty environment while avoiding inter-agent collisions.

Curriculum:
    1. Start with one active drone while possible_agents remains fixed.
    2. Ramp active drone count to NUM_DRONES.
    3. Ramp waypoint sampling from collision-safe assignments to stochastic
       independent sampling.

Training modes:
    - IPPO: decentralized policy observations only, state.mode="none".
    - MAPPO: policy observations plus centralized critic group state.
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG, CRAZYFLIE_CTBR_CONTROLLER_CFG
from cpsquare_lab.embodiments.multirotor.common.actions import ActionType, CtbrActionCfg, HandleOutOfRangeAction

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

from environments.envs.manager_based_ma_env_cfg import (
    AgentSetCfg,
    MultiAgentOptionsCfg,
)
from environments.envs.manager_based_marl_env_cfg import (
    AgentRlProfileCfg,
    ManagerBasedMarlEnvCfg,
    MultiAgentStateCfg,
)

from . import mdp

# -----------------------------------------------------------------------------
# Global task defaults
# -----------------------------------------------------------------------------

NUM_DRONES = 8
INITIAL_ACTIVE_DRONES = 1
ACTIVE_AGENT_RAMP_STEPS = 500_000
WAYPOINT_RANDOMIZATION_START_STEP = ACTIVE_AGENT_RAMP_STEPS
WAYPOINT_RANDOMIZATION_RAMP_STEPS = 300_000

NUM_ENVS = 4096
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

DEFAULT_DRONE_CFG = CRAZYFLIE_CFG


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class DroneWaypointSceneCfg(InteractiveSceneCfg):
    """Empty workspace containing one named drone articulation per agent."""

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
            # TODO replace with robot_i
            agent_id = f"drone_{i}"
            setattr(self, agent_id, DEFAULT_DRONE_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/{agent_id}"))


# -----------------------------------------------------------------------------
# Commands and actions
# -----------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Per-agent target pose command."""

    target_pose = DroneUniformPoseCommandCfg(
        asset_name="{entity_name}",
        body_name="base_link",
        resampling_time_range=COMMAND_RESAMPLE_TIME,
        debug_vis=True,
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
    """Actor policy observations and MAPPO centralized critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        root_lin_vel_b = ObsTerm(func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        root_ang_vel_b = ObsTerm(func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        projected_gravity_b = ObsTerm(
            func=mdp.projected_gravity_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")}
        )

        target_pos_b = ObsTerm(
            func=mdp.relative_target_position_b,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
        )
        target_yaw_error = ObsTerm(
            func=mdp.target_yaw_error,
            params={"asset_cfg": SceneEntityCfg("{entity_name}"), "command_name": "target_pose"},
        )

        neighbor_pos_b = ObsTerm(
            func=mdp.relative_neighbor_positions_b,
            params={
                "asset_cfg": SceneEntityCfg("{entity_name}"),
                "agent_ids": DRONE_AGENT_IDS,
                "max_neighbors": NUM_DRONES - 1,
                "radius": 6.0,
                "mask_key": ACTIVE_AGENT_MASK_KEY,
            },
        )
        neighbor_vel_b = ObsTerm(
            func=mdp.relative_neighbor_velocities_b,
            params={
                "asset_cfg": SceneEntityCfg("{entity_name}"),
                "agent_ids": DRONE_AGENT_IDS,
                "max_neighbors": NUM_DRONES - 1,
                "radius": 6.0,
                "mask_key": ACTIVE_AGENT_MASK_KEY,
            },
        )
        active_flag = ObsTerm(
            func=mdp.agent_active_flag,
            params={"agent_ids": DRONE_AGENT_IDS, "agent_id": "{agent_id}", "mask_key": ACTIVE_AGENT_MASK_KEY},
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "ctbr"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        swarm_state = ObsTerm(
            func=mdp.swarm_global_state,
            params={
                "agent_ids": DRONE_AGENT_IDS,
                "command_name": "target_pose",
                "include_root_state": True,
                "include_target_pose": True,
                "include_pairwise_distances": True,
                "mask_key": ACTIVE_AGENT_MASK_KEY,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05, params={"action_name": "ctbr"})
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


@configclass
class DroneEventsCfg:
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
    waypoint_randomization = CurrTerm(
        func=mdp.waypoint_randomization_curriculum,
        params={
            "command_name": "target_pose",
            "agent_ids": DRONE_AGENT_IDS,
            "workspace_xy": WORKSPACE_XY,
            "workspace_z": WORKSPACE_Z,
            "start_step": WAYPOINT_RANDOMIZATION_START_STEP,
            "ramp_steps": WAYPOINT_RANDOMIZATION_RAMP_STEPS,
            "start_safe_sampling_prob": 1.0,
            "end_safe_sampling_prob": 0.0,
            "start_min_separation": SAFE_WAYPOINT_SEPARATION,
            "end_min_separation": 0.0,
            "mask_key": ACTIVE_AGENT_MASK_KEY,
        },
    )


# -----------------------------------------------------------------------------
# Environment configs
# -----------------------------------------------------------------------------


@configclass
class DroneWaypointBaseMarlEnvCfg(ManagerBasedMarlEnvCfg):
    scene = DroneWaypointSceneCfg(num_envs=NUM_ENVS, env_spacing=ENV_SPACING)
    sim = SimulationCfg(dt=0.005, render_interval=4)
    viewer = ViewerCfg(eye=(7.5, 7.5, 5.0), lookat=(0.0, 0.0, 1.5))
    events = DroneEventsCfg()

    possible_agents = DRONE_AGENT_IDS
    profiles = {
        "drone": AgentRlProfileCfg(
            observations=ObservationsCfg(),
            actions=ActionsCfg(),
            rewards=RewardsCfg(),
            terminations=TerminationsCfg(),
            commands=CommandsCfg(),
            curriculum=CurriculumCfg(),
            entity_name="{agent_id}",
            metadata={
                "embodiment": "ctbr_drone",
                "task_role": "cooperative_waypoint_follower",
                "active_agent_mask_key": ACTIVE_AGENT_MASK_KEY,
            },
        )
    }
    sets = [
        AgentSetCfg(
            name="drones",
            count=NUM_DRONES,
            id_template="drone_{i}",
            profile="drone",
            policy="drone_policy",
            team="swarm",
            trainable=True,
            metadata={"cooperative": True, "shared_policy": True},
        )
    ]

    ma_options = MultiAgentOptionsCfg(
        observation_group="policy",
        # This uploaded task uses standard single-agent Isaac Lab manager terms.
        # V2 supports set-pooled managers, but this config selects the compatibility
        # grouping so the existing CtbrActionCfg and DroneUniformPoseCommandCfg
        # terms can run unchanged per drone.
        manager_grouping="agent",
        reset_on="any_agent",
        dynamic_agents=False,
        validate_spaces=True,
        validate_policy_space_sharing=True,
        expose_agent_metadata_maps=True,
        active_agent_mask_key=ACTIVE_AGENT_MASK_KEY,
    )

    state = MultiAgentStateCfg(mode="none", state_space=0)
    episode_length_s = 20.0
    is_finite_horizon = False
    decimation = 4


@configclass
class DroneWaypointMappoEnvCfg(DroneWaypointBaseMarlEnvCfg):
    """MAPPO variant with centralized critic state."""

    state = MultiAgentStateCfg(mode="observation_group", group_name="critic", state_space=-1, expose_as_dict=False)


@configclass
class DroneWaypointIppoEnvCfg(DroneWaypointBaseMarlEnvCfg):
    """IPPO variant with decentralized observations only."""

    state = MultiAgentStateCfg(mode="none", state_space=0, expose_as_dict=False)


DroneWaypointMAPPORunnerCfg = DroneWaypointMappoEnvCfg
DroneWaypointIPPORunnerCfg = DroneWaypointIppoEnvCfg
