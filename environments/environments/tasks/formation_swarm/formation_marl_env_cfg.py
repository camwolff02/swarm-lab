# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based MARL config for Xie et al. formation swarm control."""

from __future__ import annotations

from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG, CRAZYFLIE_CTBR_CONTROLLER_CFG
from cpsquare_lab.embodiments.multirotor.common.actions import (
    ActionType,
    CtbrActionCfg,
    HandleOutOfRangeAction,
)
import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.managers import (
    EventTermCfg as EventTerm,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.utils.configclass import configclass

from environments.envs.manager_based_ma_env_cfg import AgentGroupCfg
from environments.envs.manager_based_marl_env_cfg import (
    AgentRlCfg,
    ManagerBasedMarlEnvCfg,
)

from . import mdp
from .paper_spec import PaperSpecEnvCfg

# -----------------------------------------------------------------------------
# Task defaults from paper spec
# -----------------------------------------------------------------------------

_DEFAULTS = PaperSpecEnvCfg()
NUM_DRONES = _DEFAULTS.num_drones
NUM_BALLS = _DEFAULTS.num_balls
NUM_ENVS = 1024
ENV_SPACING = 12.0

AGENT_IDS = [f"robot_{i}" for i in range(NUM_DRONES)]

OBS_SELF_DIM = _DEFAULTS.self_obs_dim
OBS_OTHER_DIM = _DEFAULTS.other_obs_dim
OBS_DYNAMIC_DIM = _DEFAULTS.dynamic_obs_dim
OBS_STATIC_DIM = _DEFAULTS.static_sdf_dim
ACTION_DIM = _DEFAULTS.action_dim

# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class FormationSwarmSceneCfg(InteractiveSceneCfg):
    """Workspace with ground plane, lighting, and drone articulations."""

    num_envs: int = NUM_ENVS
    env_spacing: float = ENV_SPACING
    replicate_physics: bool = True

    def __post_init__(self):
        for i in range(NUM_DRONES):
            agent_id = f"robot_{i}"
            setattr(
                self,
                agent_id,
                CRAZYFLIE_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/{agent_id}"),
            )


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """No command manager — formation task has fixed target."""

    pass


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Per-agent CTBR action config (collective thrust + body rates)."""

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
    """Policy observations matching the paper's flat concatenation layout.

    Order: ego state (29) + other drones (14) + balls (20) + static SDF (9) = 72.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        ego_pos = ObsTerm(func=mdp.formation_ego_pos, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_quat = ObsTerm(func=mdp.formation_ego_quat, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_lin_vel = ObsTerm(func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_ang_vel = ObsTerm(func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_heading = ObsTerm(func=mdp.formation_ego_heading, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_up = ObsTerm(func=mdp.formation_ego_up, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        last_action = ObsTerm(func=mdp.formation_last_action, params={"action_name": "ctbr"})
        target_vel_rel = ObsTerm(func=mdp.formation_target_vel_rel, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        identity = ObsTerm(func=mdp.drone_identity, params={"agent_id": "{agent_id}"})
        other_drones = ObsTerm(func=mdp.formation_other_drone_obs, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ball_obs = ObsTerm(func=mdp.formation_ball_obs, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        static_sdf = ObsTerm(func=mdp.formation_static_sdf, params={"asset_cfg": SceneEntityCfg("{entity_name}")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observation: same as policy, without corruption."""

        ego_pos = ObsTerm(func=mdp.formation_ego_pos, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_quat = ObsTerm(func=mdp.formation_ego_quat, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_lin_vel = ObsTerm(func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_ang_vel = ObsTerm(func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_heading = ObsTerm(func=mdp.formation_ego_heading, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ego_up = ObsTerm(func=mdp.formation_ego_up, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        last_action = ObsTerm(func=mdp.formation_last_action, params={"action_name": "ctbr"})
        target_vel_rel = ObsTerm(func=mdp.formation_target_vel_rel, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        identity = ObsTerm(func=mdp.drone_identity, params={"agent_id": "{agent_id}"})
        other_drones = ObsTerm(func=mdp.formation_other_drone_obs, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        ball_obs = ObsTerm(func=mdp.formation_ball_obs, params={"asset_cfg": SceneEntityCfg("{entity_name}")})
        static_sdf = ObsTerm(func=mdp.formation_static_sdf, params={"asset_cfg": SceneEntityCfg("{entity_name}")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------


@configclass
class RewardsCfg:
    formation_smooth = RewTerm(
        func=mdp.formation_smooth_reward,
        weight=_DEFAULTS.morl_smooth_weight,
        params={"asset_cfg": SceneEntityCfg("{entity_name}")},
    )
    formation_obstacle = RewTerm(
        func=mdp.formation_obstacle_reward,
        weight=_DEFAULTS.morl_obstacle_weight,
        params={"asset_cfg": SceneEntityCfg("{entity_name}")},
    )
    formation_formation = RewTerm(
        func=mdp.formation_formation_reward,
        weight=_DEFAULTS.morl_formation_weight,
        params={"asset_cfg": SceneEntityCfg("{entity_name}")},
    )
    formation_forward = RewTerm(
        func=mdp.formation_forward_reward,
        weight=_DEFAULTS.morl_forward_weight,
        params={"asset_cfg": SceneEntityCfg("{entity_name}")},
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(
        func=mdp.drone_crash,
        params={"asset_cfg": SceneEntityCfg("{entity_name}"), "agent_id": "{agent_id}"},
    )
    too_close = DoneTerm(
        func=mdp.drone_too_close,
        params={"asset_cfg": SceneEntityCfg("{entity_name}"), "agent_id": "{agent_id}"},
    )
    hit_ball = DoneTerm(
        func=mdp.drone_hit_ball,
        params={"asset_cfg": SceneEntityCfg("{entity_name}"), "agent_id": "{agent_id}"},
    )
    hit_column = DoneTerm(
        func=mdp.drone_hit_column,
        params={"asset_cfg": SceneEntityCfg("{entity_name}"), "agent_id": "{agent_id}"},
    )


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------


@configclass
class EventsCfg:
    reset_swarm_state = EventTerm(func=mdp.reset_swarm_root_state, mode="reset")
    sample_columns = EventTerm(func=mdp.sample_static_columns, mode="reset")
    reset_balls = EventTerm(func=mdp.reset_balls, mode="reset")


# -----------------------------------------------------------------------------
# Curriculum
# -----------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    formation_obstacles = CurrTerm(func=mdp.formation_curriculum_obstacles)


# -----------------------------------------------------------------------------
# Environment config
# -----------------------------------------------------------------------------


@configclass
class FormationSwarmMarlEnvCfg(ManagerBasedMarlEnvCfg):
    """Manager-based MARL config for Xie et al. formation swarm control."""

    scene = FormationSwarmSceneCfg(num_envs=NUM_ENVS, env_spacing=ENV_SPACING)

    sim = SimulationCfg(
        dt=_DEFAULTS.sim_dt,
        render_interval=_DEFAULTS.decimation,
        physics=PhysxCfg(enable_external_forces_every_iteration=True, gpu_total_aggregate_pairs_capacity=4 * 1024 * 1024),
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

    viewer = ViewerCfg(eye=(5.0, -8.0, 5.0), lookat=(0.0, 5.0, 1.2))

    possible_agents = AGENT_IDS
    agent_groups = [
        AgentGroupCfg(
            name="robot",
            count=NUM_DRONES,
            id_template="robot_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=ObservationsCfg(),
                actions=ActionsCfg(),
                rewards=RewardsCfg(),
                terminations=TerminationsCfg(),
                commands=None,
                curriculum=CurriculumCfg(),
            ),
        )
    ]
    observation_group = "policy"
    reset_on = "any"

    events = EventsCfg()

    episode_length_s = _DEFAULTS.episode_length_s
    is_finite_horizon = False
    decimation = _DEFAULTS.decimation

    # Paper-spec parameters propagated through to the env
    num_drones = NUM_DRONES
    num_balls = NUM_BALLS
    static_obstacles = _DEFAULTS.static_obstacles

    formation = _DEFAULTS.formation
    formation_size = _DEFAULTS.formation_size
    target_pos = _DEFAULTS.target_pos
    target_vel = _DEFAULTS.target_vel
    target_heading = _DEFAULTS.target_heading

    curriculum_stage = 3

    # Reward coefficients
    morl_smooth_weight = _DEFAULTS.morl_smooth_weight
    morl_formation_weight = _DEFAULTS.morl_formation_weight
    morl_obstacle_weight = _DEFAULTS.morl_obstacle_weight
    morl_forward_weight = _DEFAULTS.morl_forward_weight

    formation_coeff = _DEFAULTS.formation_coeff
    formation_size_coeff = _DEFAULTS.formation_size_coeff
    separation_coeff = _DEFAULTS.separation_coeff
    too_close_penalty = _DEFAULTS.too_close_penalty

    ball_reward_coeff = _DEFAULTS.ball_reward_coeff
    ball_hard_reward_coeff = _DEFAULTS.ball_hard_reward_coeff
    static_hard_coeff = _DEFAULTS.static_hard_coeff
    hit_penalty = _DEFAULTS.hit_penalty

    velocity_coeff = _DEFAULTS.velocity_coeff
    heading_coeff = _DEFAULTS.heading_coeff
    height_coeff = _DEFAULTS.height_coeff
    position_reward_coeff = _DEFAULTS.position_reward_coeff
    truncated_reward = _DEFAULTS.truncated_reward
    has_obstacle_coeff = _DEFAULTS.has_obstacle_coeff
    no_obstacle_coeff = _DEFAULTS.no_obstacle_coeff
    after_throw_coeff = _DEFAULTS.after_throw_coeff

    effort_weight = _DEFAULTS.effort_weight
    action_smoothness_weight = _DEFAULTS.action_smoothness_weight
    spin_reward_coeff = _DEFAULTS.spin_reward_coeff
    throttle_smoothness_weight = _DEFAULTS.throttle_smoothness_weight

    action_dim = ACTION_DIM
    obs_dim = OBS_SELF_DIM + (NUM_DRONES - 1) * OBS_OTHER_DIM + NUM_BALLS * OBS_DYNAMIC_DIM + OBS_STATIC_DIM

    self_obs_dim = OBS_SELF_DIM
    other_obs_dim = OBS_OTHER_DIM
    dynamic_obs_dim = OBS_DYNAMIC_DIM
    static_sdf_dim = OBS_STATIC_DIM

    grid_size = _DEFAULTS.grid_size
    grid_border = _DEFAULTS.grid_border
    static_margin = _DEFAULTS.static_margin
    static_height = _DEFAULTS.static_height
    column_radius = _DEFAULTS.column_radius
    ball_radius = _DEFAULTS.ball_radius

    min_ball_speed = _DEFAULTS.min_ball_speed
    max_ball_speed = _DEFAULTS.max_ball_speed
    ball_speed = _DEFAULTS.ball_speed
    ball_hard_reward_coeff = _DEFAULTS.ball_hard_reward_coeff

    safe_distance = _DEFAULTS.safe_distance
    hard_safe_distance = _DEFAULTS.hard_safe_distance
    obs_safe_distance = _DEFAULTS.obs_safe_distance
    soft_obs_safe_distance = _DEFAULTS.soft_obs_safe_distance
    crash_min_height = _DEFAULTS.crash_min_height
    crash_max_height = _DEFAULTS.crash_max_height

    throw_threshold_steps = _DEFAULTS.throw_threshold_steps
    throw_time_range_steps = _DEFAULTS.throw_time_range_steps
    curriculum_delayed_throw_threshold_steps = _DEFAULTS.curriculum_delayed_throw_threshold_steps
    curriculum_delayed_throw_time_range_steps = _DEFAULTS.curriculum_delayed_throw_time_range_steps

    random_ball_speed = True
    use_cube_reward_mask = False
    spawn_obstacle_visuals = True
    use_ctbr_tanh = True


# -----------------------------------------------------------------------------
# Stage-specific configs
# -----------------------------------------------------------------------------


@configclass
class FormationSwarmStage1EnvCfg(FormationSwarmMarlEnvCfg):
    """Stage 1: obstacle-free formation flight.

    No static columns, no dynamic balls.  Drones learn basic formation
    maintenance with smooth forward flight.
    """

    curriculum_stage = 1
    spawn_obstacle_visuals = False


@configclass
class FormationSwarmStage2EnvCfg(FormationSwarmMarlEnvCfg):
    """Stage 2: formation flight with static obstacles.

    Static column obstacles are active.  No dynamic balls yet.
    """

    curriculum_stage = 2


@configclass
class FormationSwarmStage3EnvCfg(FormationSwarmMarlEnvCfg):
    """Stage 3: full formation flight with static columns and dynamic balls."""

    curriculum_stage = 3
