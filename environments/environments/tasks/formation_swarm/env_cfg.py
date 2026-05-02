"""DirectMARLEnv configuration for the Xie et al. formation swarm task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG
from cpsquare_lab.tasks.common.physics import physx_swarm_cfg

from . import paper_spec as spec


@configclass
class FormationSwarmEnvCfg(DirectMARLEnvCfg):
    """Configuration for Xie et al. formation maintenance with obstacle avoidance."""

    decimation = spec.DECIMATION
    episode_length_s = spec.EPISODE_LENGTH_S
    possible_agents = [f"drone_{index}" for index in range(spec.NUM_DRONES)]
    observation_spaces = {
        agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(spec.OBS_DIM,), dtype=np.float32)
        for agent in possible_agents
    }
    action_spaces = {
        agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(spec.ACTION_DIM,), dtype=np.float32)
        for agent in possible_agents
    }
    state_space = -1

    sim: SimulationCfg = SimulationCfg(
        dt=spec.SIM_DT,
        render_interval=decimation,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=12.0, replicate_physics=True)
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/drone_0")

    num_drones: int = spec.NUM_DRONES
    num_balls: int = spec.NUM_BALLS
    static_obstacles: int = spec.STATIC_OBSTACLES
    formation_size: float = spec.FORMATION_SIZE
    target_pos: tuple[float, float, float] = spec.TARGET_POS
    target_vel: tuple[float, float, float] = spec.TARGET_VEL
    target_heading: tuple[float, float, float] = spec.TARGET_HEADING

    spawn_obstacle_visuals: bool = True
    terminate_on_collision: bool = True
    terminate_on_crash: bool = True
    random_ball_speed: bool = True
    use_ctbr_tanh: bool = True

    def __post_init__(self) -> None:
        self.possible_agents = [f"drone_{index}" for index in range(self.num_drones)]
        other_dim = (self.num_drones - 1) * spec.OTHER_OBS_DIM
        obs_dim = spec.SELF_OBS_DIM + other_dim + self.num_balls * spec.DYNAMIC_OBS_DIM + spec.STATIC_SDF_DIM
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(spec.ACTION_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.state_space = -1
        self.viewer.eye = (5.0, -8.0, 5.0)
        self.viewer.lookat = (0.0, 5.0, 1.2)
        self.viewer.cam_prim_path = "/World/FormationSwarmCamera"

