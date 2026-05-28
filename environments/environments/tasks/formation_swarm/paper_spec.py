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

from collections.abc import Iterable

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.configclass import configclass

type Position = tuple[float, float, float]


@configclass
class PaperSpecEnvCfg(DirectMARLEnvCfg):
    """Configuration matching the paper's specification."""

    num_drones: int = 3
    num_balls: int = 2
    static_obstacles: int = 10

    formation: Iterable[Position] = (
        (1.0, 0.0, 0.0),
        (-0.5, 0.866, 0.0),
        (-0.5, -0.866, 0.0),
    )

    # Equilateral triangle vertices. ``0.866`` approximates sqrt(3) / 2, so scaling
    # by ``FORMATION_SIZE`` preserves the source paper's triangular shape.
    formation_size: float = 1.0
    target_pos: Position = (0.0, 0.0, 1.5)
    target_vel: Position = (0.0, 2.0, 0.0)
    target_heading: Position = (1.0, 0.0, 0.0)

    sim_dt: float = 0.01
    # The control policy runs every ``DECIMATION`` physics steps. With ``SIM_DT`` at
    # 0.01 s and decimation 2, the physics loop runs at 100 Hz and actions update at
    # 50 Hz.
    decimation: int = 2
    episode_length_s: float = 9.0

    grid_size: float = 0.5
    grid_border: float = 2.0
    static_margin: float = 2.0
    static_height: float = 5.0
    column_radius: float = 0.15
    ball_radius: float = 0.15
    obs_range: float = 10.0

    min_ball_speed: float = 2.0
    max_ball_speed: float = 5.0
    ball_speed: float = 3.0
    throw_threshold_steps: int = 150
    throw_time_range_steps: int = 450
    curriculum_delayed_throw_threshold_steps: int = 1000
    curriculum_delayed_throw_time_range_steps: int = 800

    hard_safe_distance: float = 0.15
    safe_distance: float = 0.4
    obs_safe_distance: float = 0.4
    soft_obs_safe_distance: float = 0.6
    crash_min_height: float = 0.2
    crash_max_height: float = 2.8

    self_obs_dim: int = 29
    other_obs_dim: int = 7
    dynamic_obs_dim: int = 10
    static_sdf_dim: int = 9

    action_dim: int = 4

    attention_dim: int = 32
    attention_heads: int = 1
    mlp_hidden: Iterable[int] = (256, 256, 256)
    initial_log_std: float = 0.0

    morl_smooth_weight: float = 0.5123452752249692
    morl_formation_weight: float = 0.14240923087088264
    morl_obstacle_weight: float = 0.2187620445033934
    morl_forward_weight: float = 0.1264834494007547
    # MORL weights are the scalarization coefficients used to combine smoothness,
    # formation, obstacle-avoidance, and forward-progress objectives into one PPO
    # reward. They intentionally sum to 1.0 within floating-point tolerance.

    formation_coeff: float = 5.0
    formation_size_coeff: float = 5.0
    separation_coeff: float = 1.0
    too_close_penalty: float = -10.0

    ball_reward_coeff: float = 10.0
    ball_hard_reward_coeff: float = 100.0
    static_hard_coeff: float = 1.0
    hit_penalty: float = -20.0

    velocity_coeff: float = 10.0
    heading_coeff: float = 1.0
    height_coeff: float = 5.0
    position_reward_coeff: float = 50.0
    truncated_reward: float = 10.0
    acceptable_v_diff: float = 1.0
    has_obstacle_coeff: float = 0.2
    no_obstacle_coeff: float = 1.0
    after_throw_coeff: float = 0.2

    effort_weight: float = 0.5
    action_smoothness_weight: float = 1.0
    spin_reward_coeff: float = 1.0
    throttle_smoothness_weight: float = 2.0

    def __post_init__(self):
        self.obs_dim = (
            self.self_obs_dim
            + (self.num_drones - 1) * self.other_obs_dim
            + self.num_balls * self.dynamic_obs_dim
            + self.static_sdf_dim
        )
