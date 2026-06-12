"""DirectMARLEnv configuration for the Xie et al. formation swarm task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG, CRAZYFLIE_CTBR_CONTROLLER_CFG
from cpsquare_lab.embodiments.multirotor.common.actions import (
    ActionType,
    CtbrActionCfg,
    HandleOutOfRangeAction,
)
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .paper_spec import PaperSpecEnvCfg

SWARM_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 4 * 1024 * 1024


def physx_swarm_cfg() -> PhysxCfg:
    """Return the PhysX preset used for large multirotor swarms."""
    return PhysxCfg(
        enable_external_forces_every_iteration=True,
        gpu_total_aggregate_pairs_capacity=SWARM_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY,
    )


class ActionTermDict(dict[str, CtbrActionCfg]):
    """Dictionary of action terms that also accepts manager runtime attributes."""


@configclass
class FormationSwarmEnvCfg(PaperSpecEnvCfg):
    """Configuration for Xie et al. formation maintenance with obstacle avoidance."""

    state_space = -1
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=12.0, replicate_physics=True)
    actions: ActionTermDict = ActionTermDict()

    # TODO make this modular to swap out Drone
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/robot_0")

    curriculum_stage: int = 3
    active_balls: int | None = None
    active_static_obstacles: int | None = None
    use_cube_reward_mask: bool = False

    spawn_obstacle_visuals: bool = True
    terminate_on_collision: bool = True
    terminate_on_crash: bool = True
    random_ball_speed: bool = True
    use_ctbr_tanh: bool = True

    def __post_init__(self) -> None:
        """Finalize derived Isaac Lab configuration fields."""
        super().__post_init__()

        # Scene Configuration
        self.viewer.eye = (5.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 5.0, 1.2)
        self.viewer.cam_prim_path = "/World/FormationSwarmCamera"

        # Scene Construction
        self.sim: SimulationCfg = SimulationCfg(
            dt=self.sim_dt,
            render_interval=self.decimation,
            physics=physx_swarm_cfg(),
            render=sim_utils.RenderCfg(
                carb_settings={"/rtx/hydra/readTransformsFromFabricInRenderDelegate": False},
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        self.possible_agents = [f"robot_{index}" for index in range(self.num_drones)]
        out_of_range_action = HandleOutOfRangeAction.TANH if self.use_ctbr_tanh else HandleOutOfRangeAction.CLIP
        self.actions = ActionTermDict({
            agent: CtbrActionCfg(
                asset_name=agent,
                controller_cfg=CRAZYFLIE_CTBR_CONTROLLER_CFG,
                max_roll_pitch_rate=3,
                max_yaw_rate=2,
                action_type=ActionType.NORM_NEG_1_TO_1,
                handle_out_of_range=out_of_range_action,
            )
            for agent in self.possible_agents
        })

        other_dim = (self.num_drones - 1) * self.other_obs_dim
        self.obs_dim = self.self_obs_dim + other_dim + self.num_balls * self.dynamic_obs_dim + self.static_sdf_dim

        self.observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
