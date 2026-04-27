"""DirectMARLEnv configuration for the quad swarm paper task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_CFG
from cpsquare_lab.tasks.common.physics import physx_swarm_cfg

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import paper_spec as spec


@configclass
class QuadSwarmPaperEnvCfg(DirectMARLEnvCfg):
    """Configuration for the paper-style homogeneous quadrotor swarm task."""

    decimation = spec.DECIMATION
    episode_length_s = spec.EPISODE_LENGTH_S
    possible_agents = [f"drone_{index}" for index in range(spec.NUM_DRONES)]
    observation_spaces = {
        agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(spec.OBS_SIZE,), dtype=np.float32)
        for agent in possible_agents
    }
    action_spaces = {
        agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(spec.ACTION_SIZE,), dtype=np.float32)
        for agent in possible_agents
    }
    state_space = -1

    sim: SimulationCfg = SimulationCfg(
        dt=spec.SIM_DT,
        render_interval=decimation,
        physics=physx_swarm_cfg(),
        render=sim_utils.RenderCfg(
            carb_settings={
                # Isaac Lab's rendering experience enables this alongside the Fabric scene delegate.
                # RTX warns that this conflicts with geometry streaming and can prevent dynamic objects
                # from streaming correctly, which is especially visible for moving drones.
                "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
            },
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
    visible_neighbors: int = spec.VISIBLE_NEIGHBORS
    room_size: float = spec.ROOM_SIZE
    obstacle_density: float = spec.OBSTACLE_DENSITY
    obstacle_radius: float = spec.OBSTACLE_RADIUS
    local_sdf_resolution: float = spec.LOCAL_SDF_RESOLUTION
    enable_obstacles: bool = True
    spawn_obstacle_actors: bool = True
    enable_replay: bool = True
    replay_probability: float = spec.REPLAY_PROBABILITY
    replay_lag_s: float = spec.REPLAY_LAG_S

    goal_reached_radius: float = spec.GOAL_REACHED_RADIUS
    floor_crash_height: float = spec.FLOOR_CRASH_HEIGHT
    robot_collision_radius: float = spec.ROBOT_COLLISION_RADIUS
    robot_proximity_radius: float = spec.ROBOT_PROXIMITY_RADIUS

    goal_reward_scale: float = spec.GOAL_REWARD_SCALE
    goal_reward_distance_scale: float = spec.GOAL_REWARD_DISTANCE_SCALE
    robot_collision_penalty: float = spec.ROBOT_COLLISION_PENALTY
    obstacle_collision_penalty: float = spec.OBSTACLE_COLLISION_PENALTY
    proximity_penalty: float = spec.PROXIMITY_PENALTY
    floor_crash_penalty: float = spec.FLOOR_CRASH_PENALTY
    angular_velocity_penalty_scale: float = spec.ANGULAR_VELOCITY_PENALTY_SCALE
    control_effort_penalty_scale: float = spec.CONTROL_EFFORT_PENALTY_SCALE
    tilt_penalty_scale: float = spec.TILT_PENALTY_SCALE

    def __post_init__(self) -> None:
        self.possible_agents = [f"drone_{index}" for index in range(self.num_drones)]
        obs_size = spec.SELF_OBS_SIZE + self.visible_neighbors * spec.NEIGHBOR_OBS_SIZE + spec.OBSTACLE_OBS_SIZE
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(spec.ACTION_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.state_space = -1
        self.sim.render_interval = self.decimation
        self.viewer.eye = (8.0, 8.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.viewer.cam_prim_path = "/World/QuadSwarmCamera"
