"""Isaac Lab DirectMARLEnv for Xie et al. formation swarm control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from cpsquare_lab.controllers import PIDRateController
from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_PARAMS
from cpsquare_lab.embodiments.multirotor.common.actions import CtbrAction, CtbrActionCfg
from cpsquare_lab.tasks.common.metrics import flush_episode_metrics, initialize_episode_metrics, record_metric, reset_episode_metrics

from . import paper_spec as spec
from .env_cfg import FormationSwarmEnvCfg

METRIC_NAMES = (
    "reward",
    "morl_smooth",
    "morl_formation",
    "morl_obstacle",
    "morl_forward",
    "formation_success",
    "action_success",
    "collision",
    "crash",
    "too_close",
    "hit_ball",
    "hit_column",
)


def _laplacian(points: torch.Tensor, normalize: bool) -> torch.Tensor:
    r"""Compute the pairwise-distance graph Laplacian for formation points.

    For pairwise distance matrix \(D_{ij}=\lVert p_i-p_j\rVert_2\) and degree
    \(d_i=\sum_j D_{ij}\), the unnormalized Laplacian is

    \[
    L = \operatorname{diag}(d) - D.
    \]

    With ``normalize=True``, the function returns

    \[
    L_\text{norm} = I - \operatorname{diag}(d)^{-1/2}D\operatorname{diag}(d)^{-1/2}.
    \]
    """
    distances = torch.cdist(points, points)
    degree = distances.sum(dim=-1)
    if normalize:
        scale = degree.clamp_min(1.0e-6).pow(-0.5)
        adj = scale.unsqueeze(-1) * distances * scale.unsqueeze(-2)
        eye = torch.eye(points.shape[-2], device=points.device, dtype=points.dtype)
        return eye.expand(points.shape[:-2] + eye.shape) - adj
    return degree.unsqueeze(-1) - distances


def _formation_cost(points: torch.Tensor, desired_laplacian: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    r"""Return the matrix-norm distance from the desired formation Laplacian.

    \[
    c_L(P) = \lVert L^\star - L(P)\rVert_F.
    \]
    """
    return torch.linalg.matrix_norm(desired_laplacian - _laplacian(points, normalize), dim=(-2, -1)).unsqueeze(-1)


def _pairwise_without_self(values: torch.Tensor) -> torch.Tensor:
    """Remove diagonal self-pairs from a batched pairwise tensor."""
    count = values.shape[-2]
    eye = torch.eye(count, device=values.device, dtype=torch.bool)
    return values[:, ~eye].reshape(values.shape[0], count, count - 1, values.shape[-1])


def _other_drone_observation(positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
    r"""Build relative position, distance, and velocity features for other drones.

    For ego drone \(i\) and neighbor \(j\), the feature block is

    \[
    \left[\frac{p_i-p_j}{2},\; \frac{\lVert p_i-p_j\rVert_2}{2},\; v_i-v_j\right].
    \]
    """
    relative_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
    relative_vel = velocities.unsqueeze(2) - velocities.unsqueeze(1)
    other_pos = _pairwise_without_self(relative_pos)
    other_vel = _pairwise_without_self(relative_vel)
    other_dist = torch.linalg.norm(other_pos, dim=-1, keepdim=True)
    return torch.cat((other_pos / 2.0, other_dist / 2.0, other_vel), dim=-1).reshape(positions.shape[0], positions.shape[1], -1)


class FormationSwarmEnv(DirectMARLEnv):
    r"""Directed formation flight with static columns and dynamic balls.

    The environment tracks a four-objective reward decomposition:

    \[
    r = w_s r_\text{smooth} + w_o r_\text{obstacle}
      + w_f r_\text{forward} + w_\ell r_\text{formation}.
    \]

    The formation objective compares the current pairwise-distance Laplacian against
    the desired formation Laplacian. For normalized formation cost \(c_L\), drone count
    \(N\), current formation diameter \(s\), and desired diameter \(s^\star\):

    \[
    r_L = \frac{1}{2}\left(
      \frac{1}{1 + \left(10(c_L-0.04)/N\right)^2} - 0.04N
    \right),
    \]

    \[
    r_\text{size} = 3\left(
      \frac{\frac{1}{1+(s-s^\star)^2} + \frac{1}{1+c_L^\text{raw}} - 2}{N}
      - 0.04N
    \right) + 2.36.
    \]

    These scalar terms are combined with separation penalties, obstacle penalties,
    forward/height/heading rewards, and smooth-action terms inside ``_get_rewards``.
    """

    cfg: FormationSwarmEnvCfg

    def __init__(self, cfg: FormationSwarmEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize the FormationSwarmEnv instance."""
        self._agent_ids = list(cfg.possible_agents)
        self._last_actions = torch.zeros(cfg.scene.num_envs, cfg.num_drones, spec.ACTION_DIM, device=cfg.sim.device)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._formation = torch.tensor(spec.FORMATION, device=self.device, dtype=torch.float32) * cfg.formation_size
        self._target_pos = torch.tensor(cfg.target_pos, device=self.device, dtype=torch.float32)
        self._target_vel = torch.tensor(cfg.target_vel, device=self.device, dtype=torch.float32)
        self._target_heading = torch.tensor(cfg.target_heading, device=self.device, dtype=torch.float32)
        self._final_pos = self._target_pos + self._target_vel * (self.max_episode_length * self.step_dt * 0.9)
        self._formation_l = _laplacian(self._formation, normalize=True)
        self._formation_l_unnormalized = _laplacian(self._formation, normalize=False)
        self._standard_formation_size = torch.cdist(self._formation, self._formation).max()
        self._drone_ids = torch.eye(3, device=self.device, dtype=torch.float32)[: self.cfg.num_drones]
        self._grid_offsets = torch.tensor(
            [[-0.1, -0.1], [-0.1, 0.0], [-0.1, 0.1], [0.0, -0.1], [0.0, 0.0], [0.0, 0.1], [0.1, -0.1], [0.1, 0.0], [0.1, 0.1]],
            device=self.device,
            dtype=torch.float32,
        )

        self._ctbr_actions = {
            agent: CtbrAction(
                CtbrActionCfg(
                    asset_name=agent,
                    controller_class=PIDRateController,
                    controller_params=CRAZYFLIE_PARAMS.__dict__,
                    thrust_ratio_range=(0.0, 0.9),
                    use_tanh=self.cfg.use_ctbr_tanh,
                ),
                self,
            )
            for agent in self._agent_ids
        }
        self._previous_rotor_actions = torch.zeros(
            self.num_envs, self.cfg.num_drones, CRAZYFLIE_PARAMS.num_rotors, device=self.device
        )
        self._current_rotor_actions = self._previous_rotor_actions.clone()

        self._columns = torch.zeros(self.num_envs, self.cfg.static_obstacles, 3, device=self.device)
        self._balls_pos = torch.zeros(self.num_envs, self.cfg.num_balls, 3, device=self.device)
        self._balls_vel = torch.zeros_like(self._balls_pos)
        self._ball_start_pos = torch.zeros_like(self._balls_pos)
        self._ball_start_vel = torch.zeros_like(self._balls_pos)
        self._ball_launch_step = torch.zeros(self.num_envs, self.cfg.num_balls, device=self.device, dtype=torch.long)
        self._ball_active = torch.zeros(self.num_envs, self.cfg.num_balls, device=self.device, dtype=torch.bool)
        self._last_hit_ball = torch.zeros(self.num_envs, self.cfg.num_drones, device=self.device, dtype=torch.bool)
        self._last_hit_column = torch.zeros_like(self._last_hit_ball)
        self._last_too_close = torch.zeros_like(self._last_hit_ball)
        self._last_crash = torch.zeros_like(self._last_hit_ball)
        initialize_episode_metrics(self, METRIC_NAMES)

    def _setup_scene(self) -> None:
        """Create drone assets, obstacles, ground, and cloned environments."""
        self.drones = {}
        for agent in self._agent_ids:
            robot_cfg = self.cfg.robot_cfg.replace(prim_path=f"/World/envs/env_.*/{agent}")
            self.drones[agent] = robot_cfg.class_type(robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(80.0, 80.0)))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for agent, drone in self.drones.items():
            self.scene.articulations[agent] = drone

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)
        self._spawn_view_camera()
        self._spawn_obstacle_visuals()

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        r"""Map formation policy actions into CTBR command order.

        Formation policies emit \([c,\omega_x,\omega_y,\omega_z]\). The shared
        ``CtbrAction`` term expects \([\omega_x,\omega_y,\omega_z,c]\), so the method
        applies that permutation before the CTBR action equations documented in
        :class:`cpsquare_lab.embodiments.multirotor.common.actions.CtbrAction`.
        """
        self._update_balls()
        for index, agent in enumerate(self._agent_ids):
            action = torch.nan_to_num(actions[agent].to(self.device), nan=0.0, posinf=1.0, neginf=-1.0)
            self._last_actions[:, index] = action
            ctbr_action = torch.cat((action[:, 1:4], action[:, 0:1]), dim=-1)
            self._ctbr_actions[agent].process_actions(ctbr_action)
            self._current_rotor_actions[:, index] = self._ctbr_actions[agent].processed_actions

    def _apply_action(self) -> None:
        """Apply action."""
        for action in self._ctbr_actions.values():
            action.apply_actions()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get observations."""
        state = self._collect_state()
        pos = state["pos"]
        vel = state["vel"]
        quat = state["quat"]
        omega = state["omega"]
        heading = math_utils.quat_apply(quat.reshape(-1, 4), torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(pos.numel() // 3, 3)).reshape(self.num_envs, self.cfg.num_drones, 3)
        up = math_utils.quat_apply(quat.reshape(-1, 4), torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(pos.numel() // 3, 3)).reshape(self.num_envs, self.cfg.num_drones, 3)
        rel_vel = self._target_vel.view(1, 1, 3) - vel
        ids = self._drone_ids.view(1, self.cfg.num_drones, 3).expand(self.num_envs, -1, -1)
        obs_self = torch.cat((pos, quat, vel, omega, heading, up, self._current_rotor_actions, rel_vel, ids), dim=-1)

        obs_others = _other_drone_observation(pos, vel)

        rel_ball_pos = self._balls_pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_ball_vel = self._balls_vel.unsqueeze(1) - vel.unsqueeze(2)
        ball_dist = torch.linalg.norm(rel_ball_pos, dim=-1, keepdim=True)
        obs_ball = torch.cat(
            (ball_dist, rel_ball_pos, rel_ball_vel, self._balls_vel.unsqueeze(1).expand(-1, self.cfg.num_drones, -1, -1)),
            dim=-1,
        )
        inactive = ~self._ball_active.unsqueeze(1).unsqueeze(-1)
        obs_ball = obs_ball.masked_fill(inactive, 0.0).reshape(self.num_envs, self.cfg.num_drones, -1)

        grid = pos[..., :2].unsqueeze(2) + self._grid_offsets.view(1, 1, -1, 2)
        rel_columns = self._columns[:, None, :, None, :2] - grid[:, :, None, :, :]
        obs_static = torch.linalg.norm(rel_columns, dim=-1).amin(dim=2)

        observations = torch.cat((obs_self, obs_others, obs_ball, obs_static), dim=-1).clamp(-20.0, 20.0)
        return {agent: observations[:, index] for index, agent in enumerate(self._agent_ids)}

    def _get_states(self) -> torch.Tensor:
        """Get states."""
        return torch.cat([self.obs_dict[agent].reshape(self.num_envs, -1) for agent in self._agent_ids], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        r"""Compute the multi-objective formation reward.

        The smoothness objective rewards low collective effort and small action
        changes:

        \[
        r_\text{effort} = \max\left(2.5-\sum_i \rho_i, 0\right),\qquad
        r_\Delta = \max\left(0.5-\lVert m_t-m_{t-1}\rVert_2, 0\right),
        \]

        with \(\rho_i=\operatorname{clip}((m_i+1)/2,0,1)\). Static-column obstacle
        cost uses XY distance \(d_c\):

        \[
        r_\text{column} =
        \operatorname{mean}\left(\operatorname{clip}(d_c, r_\text{col}, r_\text{safe}) - r_\text{safe}\right).
        \]

        The final per-drone reward is the weighted sum shown in the class docstring.
        """
        state = self._collect_state()
        pos = state["pos"]
        vel = state["vel"]
        omega = state["omega"]

        cost_l = _formation_cost(pos, self._formation_l, normalize=True)
        cost_l_unnormalized = _formation_cost(pos, self._formation_l_unnormalized, normalize=False)
        pairwise_dist = torch.cdist(pos, pos)
        pairwise_dist = pairwise_dist.masked_fill(torch.eye(self.cfg.num_drones, device=self.device, dtype=torch.bool), torch.inf)
        size = torch.cdist(pos, pos).amax(dim=(1, 2)).unsqueeze(-1)
        separation = pairwise_dist.amin(dim=-1)

        reward_formation = (1.0 / (1.0 + torch.square((cost_l - 0.04) / self.cfg.num_drones * 10.0)) - (0.04 * self.cfg.num_drones)) / 2.0
        reward_size = 1.0 / (1.0 + torch.square(size - self._standard_formation_size))
        reward_size = reward_size + 1.0 / (1.0 + cost_l_unnormalized)
        reward_size = ((reward_size - 2.0) / self.cfg.num_drones - (0.04 * self.cfg.num_drones)) * 3.0 + 2.36
        separation_reward = -(separation < spec.SAFE_DISTANCE).float()
        too_close_reward = (separation < spec.HARD_SAFE_DISTANCE).float()

        heading_error = torch.linalg.norm(self._target_heading.view(1, 1, 3) - state["heading"], dim=-1)
        reward_heading = torch.clamp(1.0 - heading_error, min=0.0)
        pos_error = torch.linalg.norm(pos - (self._final_pos.view(1, 1, 3) + self._formation.view(1, self.cfg.num_drones, 3)), dim=-1)
        pos_reward = 1.0 / (1.0 + pos_error)
        vel_error = vel - self._target_vel.view(1, 1, 3)
        vel_reward = torch.clamp(torch.clamp(torch.linalg.norm(self._target_vel), min=1.0) - torch.linalg.norm(vel_error, dim=-1), min=0.0)
        height_reward = torch.clamp(1.0 - (pos[..., 2] - self._target_pos[2]).abs(), min=0.0)

        rel_ball_pos = self._balls_pos.unsqueeze(1) - pos.unsqueeze(2)
        ball_dist = torch.linalg.norm(rel_ball_pos, dim=-1)
        active_ball = self._ball_active.unsqueeze(1)
        ball_hard = torch.zeros_like(ball_dist)
        ball_hard[ball_dist < spec.OBS_SAFE_DISTANCE] = -(spec.BALL_HARD_REWARD_COEFF / spec.BALL_REWARD_COEFF)
        k = 0.5 * (spec.BALL_HARD_REWARD_COEFF / spec.BALL_REWARD_COEFF) / (spec.SOFT_OBS_SAFE_DISTANCE - spec.OBS_SAFE_DISTANCE)
        ball_soft = (ball_dist.clamp(spec.OBS_SAFE_DISTANCE, spec.SOFT_OBS_SAFE_DISTANCE) - spec.SOFT_OBS_SAFE_DISTANCE) * k
        ball_soft = ball_soft + (ball_dist - spec.SOFT_OBS_SAFE_DISTANCE).clamp_min(0.0)
        ball_reward = ((ball_hard + ball_soft) * active_ball).amin(dim=-1)

        rel_col = self._columns.unsqueeze(1) - pos.unsqueeze(2)
        col_dist = torch.linalg.norm(rel_col[..., :2], dim=-1)
        cube_reward = (col_dist.clamp(spec.COLUMN_RADIUS, spec.OBS_SAFE_DISTANCE) - spec.OBS_SAFE_DISTANCE).mean(dim=-1)

        hit_ball = ((ball_dist < spec.BALL_RADIUS) & active_ball).any(dim=-1)
        hit_column = (col_dist < spec.COLUMN_RADIUS).any(dim=-1)
        hit_reward = (hit_ball | hit_column).float()
        crash = (pos[..., 2] < spec.CRASH_MIN_HEIGHT) | (pos[..., 2] > spec.CRASH_MAX_HEIGHT)
        too_close = separation < spec.HARD_SAFE_DISTANCE
        bad_terminate = crash | too_close | hit_ball | hit_column
        bad_env = bad_terminate.any(dim=-1, keepdim=True)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        truncated = time_out.view(-1, 1)

        morl_obstacle = (
            ball_reward * spec.BALL_REWARD_COEFF
            + cube_reward * spec.STATIC_HARD_COEFF
            + hit_reward * spec.HIT_PENALTY
            + truncated * spec.TRUNCATED_REWARD
            - bad_env * spec.TRUNCATED_REWARD
        )

        rotor_ratio = ((self._current_rotor_actions + 1.0) * 0.5).clamp(0.0, 1.0)
        effort = torch.clamp(2.5 - rotor_ratio.sum(dim=-1), min=0.0)
        throttle_diff = torch.linalg.norm(self._current_rotor_actions - self._previous_rotor_actions, dim=-1)
        throttle_smooth = torch.clamp(0.5 - throttle_diff, min=0.0)
        action_diff = torch.linalg.norm(self._last_actions - getattr(self, "_previous_actions", self._last_actions), dim=-1)
        action_smooth = torch.clamp(2.5 - action_diff, min=0.0)
        spin = torch.clamp(1.5 - omega[..., 2].abs(), min=0.0)
        morl_smooth = (
            effort * spec.EFFORT_WEIGHT
            + action_smooth * spec.ACTION_SMOOTHNESS_WEIGHT
            + spin * spec.SPIN_REWARD_COEFF
            + throttle_smooth * spec.THROTTLE_SMOOTHNESS_WEIGHT
            + truncated * spec.TRUNCATED_REWARD
            - bad_env * spec.TRUNCATED_REWARD
        )

        column_near = (col_dist < (spec.SOFT_OBS_SAFE_DISTANCE + 1.0)).any(dim=(1, 2)).unsqueeze(-1)
        obstacle_present = active_ball.any(dim=-1) | column_near
        coeff = torch.where(obstacle_present, torch.full_like(obstacle_present, 0.2, dtype=torch.float32), torch.ones_like(obstacle_present, dtype=torch.float32))
        morl_formation = (
            reward_size * coeff * spec.FORMATION_SIZE_COEFF
            + reward_formation * spec.FORMATION_COEFF * coeff
            + separation_reward * spec.SEPARATION_COEFF
            + too_close_reward * spec.TOO_CLOSE_PENALTY
            + truncated * spec.TRUNCATED_REWARD
            - bad_env * spec.TRUNCATED_REWARD
        )
        morl_forward = (
            height_reward * spec.HEIGHT_COEFF * coeff
            + pos_reward * spec.POSITION_REWARD_COEFF * truncated
            + vel_reward * spec.VELOCITY_COEFF * coeff
            + reward_heading * spec.HEADING_COEFF
            + truncated * spec.TRUNCATED_REWARD
            - bad_env * spec.TRUNCATED_REWARD
        )
        reward = (
            morl_smooth * spec.MORL_SMOOTH_WEIGHT
            + morl_obstacle * spec.MORL_OBSTACLE_WEIGHT
            + morl_forward * spec.MORL_FORWARD_WEIGHT
            + morl_formation * spec.MORL_FORMATION_WEIGHT
        )

        formation_success = cost_l_unnormalized < 5.0
        action_success = throttle_diff.mean(dim=-1, keepdim=True) < 0.005
        record_metric(self, "reward", reward.mean(dim=-1))
        record_metric(self, "morl_smooth", morl_smooth.mean(dim=-1))
        record_metric(self, "morl_formation", morl_formation.mean(dim=-1))
        record_metric(self, "morl_obstacle", morl_obstacle.mean(dim=-1))
        record_metric(self, "morl_forward", morl_forward.mean(dim=-1))
        record_metric(self, "formation_success", formation_success.float().squeeze(-1))
        record_metric(self, "action_success", action_success.float().squeeze(-1))
        record_metric(self, "collision", (hit_ball | hit_column | too_close).any(dim=-1).float())
        record_metric(self, "crash", crash.any(dim=-1).float())
        record_metric(self, "too_close", too_close.any(dim=-1).float())
        record_metric(self, "hit_ball", hit_ball.any(dim=-1).float())
        record_metric(self, "hit_column", hit_column.any(dim=-1).float())

        self._last_hit_ball = hit_ball
        self._last_hit_column = hit_column
        self._last_too_close = too_close
        self._last_crash = crash
        self._previous_rotor_actions.copy_(self._current_rotor_actions)
        self._previous_actions = self._last_actions.clone()
        return {agent: reward[:, index] for index, agent in enumerate(self._agent_ids)}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Get dones."""
        state = self._collect_state()
        pos = state["pos"]
        pairwise_dist = torch.cdist(pos, pos)
        pairwise_dist = pairwise_dist.masked_fill(torch.eye(self.cfg.num_drones, device=self.device, dtype=torch.bool), torch.inf)
        separation = pairwise_dist.amin(dim=-1)
        ball_dist = torch.linalg.norm(self._balls_pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1)
        active_ball = self._ball_active.unsqueeze(1)
        col_dist = torch.linalg.norm((self._columns.unsqueeze(1) - pos.unsqueeze(2))[..., :2], dim=-1)
        self._last_hit_ball = ((ball_dist < spec.BALL_RADIUS) & active_ball).any(dim=-1)
        self._last_hit_column = (col_dist < spec.COLUMN_RADIUS).any(dim=-1)
        self._last_too_close = separation < spec.HARD_SAFE_DISTANCE
        self._last_crash = (pos[..., 2] < spec.CRASH_MIN_HEIGHT) | (pos[..., 2] > spec.CRASH_MAX_HEIGHT)
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.terminate_on_collision:
            terminated |= (self._last_hit_ball | self._last_hit_column | self._last_too_close).any(dim=-1)
        if self.cfg.terminate_on_crash:
            terminated |= self._last_crash.any(dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return {agent: terminated for agent in self._agent_ids}, {agent: time_out for agent in self._agent_ids}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Reset idx."""
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        flush_episode_metrics(self, env_ids_tensor)
        super()._reset_idx(env_ids_tensor)
        positions = self._target_pos.view(1, 1, 3) + self._formation.view(1, self.cfg.num_drones, 3)
        positions = positions.expand(len(env_ids_tensor), -1, -1)
        self._write_swarm_state(env_ids_tensor, positions, None, torch.zeros_like(positions), torch.zeros_like(positions))
        self._sample_columns(env_ids_tensor)
        self._reset_balls(env_ids_tensor)
        self._last_actions[env_ids_tensor] = 0.0
        self._previous_actions = self._last_actions.clone()
        for index, action in enumerate(self._ctbr_actions.values()):
            action.reset(env_ids_tensor)
            self._current_rotor_actions[env_ids_tensor, index] = action.processed_actions[env_ids_tensor]
            self._previous_rotor_actions[env_ids_tensor, index] = action.processed_actions[env_ids_tensor]
        self._sync_obstacle_visuals(env_ids_tensor)
        reset_episode_metrics(self, env_ids_tensor)

    def _collect_state(self) -> dict[str, torch.Tensor]:
        """Collect state."""
        pos, quat, vel, omega = [], [], [], []
        for drone in self.drones.values():
            pos.append(wp.to_torch(drone.data.root_pos_w) - self.scene.env_origins)
            quat.append(wp.to_torch(drone.data.root_quat_w))
            vel.append(wp.to_torch(drone.data.root_lin_vel_w))
            omega.append(wp.to_torch(drone.data.root_ang_vel_w))
        pos_t = torch.stack(pos, dim=1)
        quat_t = torch.stack(quat, dim=1)
        vel_t = torch.stack(vel, dim=1)
        omega_t = torch.stack(omega, dim=1)
        heading = math_utils.quat_apply(quat_t.reshape(-1, 4), torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(pos_t.numel() // 3, 3)).reshape(self.num_envs, self.cfg.num_drones, 3)
        return {"pos": pos_t, "quat": quat_t, "vel": vel_t, "omega": omega_t, "heading": heading}

    def _write_swarm_state(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        orientations: torch.Tensor | None,
        linear_velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
    ) -> None:
        """Write swarm state."""
        for drone_index, drone in enumerate(self.drones.values()):
            root_pose = wp.to_torch(drone.data.default_root_pose)[env_ids].clone()
            root_vel = wp.to_torch(drone.data.default_root_vel)[env_ids].clone()
            joint_pos = wp.to_torch(drone.data.default_joint_pos)[env_ids].clone()
            joint_vel = wp.to_torch(drone.data.default_joint_vel)[env_ids].clone()
            root_pose[:, :3] = positions[:, drone_index] + self.scene.env_origins[env_ids]
            if orientations is not None:
                root_pose[:, 3:7] = orientations[:, drone_index]
            root_vel[:, :3] = linear_velocities[:, drone_index]
            root_vel[:, 3:6] = angular_velocities[:, drone_index]
            drone.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            drone.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
            drone.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            drone.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _sample_columns(self, env_ids: torch.Tensor) -> None:
        """Sample columns."""
        target_speed = torch.linalg.norm(self._target_vel[:2]).clamp_min(1.0)
        length = float(target_speed * self.cfg.episode_length_s) - 2.0 * spec.STATIC_MARGIN
        cols = int((2.0 * spec.GRID_BORDER) // spec.GRID_SIZE)
        rows = max(int(length // spec.GRID_SIZE), 1)
        total = rows * cols
        random_values = torch.randint(total, (len(env_ids), total), device=self.device)
        indices = torch.argsort(random_values, dim=-1)[:, : self.cfg.static_obstacles]
        grid_a = indices // cols
        grid_b = indices % cols
        x0 = grid_a.float() * spec.GRID_SIZE + spec.STATIC_MARGIN
        y0 = grid_b.float() * spec.GRID_SIZE - spec.GRID_BORDER
        y0 = torch.where((grid_a % 2) == 0, y0 + spec.GRID_SIZE / 2.0, y0)
        sin_theta = self._target_vel[1] / target_speed
        cos_theta = self._target_vel[0] / target_speed
        x = x0 * cos_theta - y0 * sin_theta
        y = x0 * sin_theta + y0 * cos_theta
        z = torch.zeros_like(x)
        self._columns[env_ids] = torch.stack((x, y, z), dim=-1)

    def _reset_balls(self, env_ids: torch.Tensor) -> None:
        """Reset balls."""
        self._ball_active[env_ids] = False
        self._balls_pos[env_ids] = torch.tensor((0.0, 0.0, -10.0), device=self.device)
        self._balls_vel[env_ids] = 0.0
        launch_offsets = torch.rand(len(env_ids), self.cfg.num_balls, device=self.device) * spec.THROW_TIME_RANGE_STEPS
        self._ball_launch_step[env_ids] = (spec.THROW_THRESHOLD_STEPS + launch_offsets).long()

    def _update_balls(self) -> None:
        """Update balls."""
        if self.cfg.num_balls == 0:
            return
        should_launch = (self.episode_length_buf.unsqueeze(-1) >= self._ball_launch_step) & ~self._ball_active
        if should_launch.any():
            state = self._collect_state()
            center = state["pos"].mean(dim=1)
            env_ids, ball_ids = should_launch.nonzero(as_tuple=True)
            speed = torch.rand(len(env_ids), device=self.device) * (spec.MAX_BALL_SPEED - spec.MIN_BALL_SPEED) + spec.MIN_BALL_SPEED
            direction_xy = torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0
            direction_xy = direction_xy / torch.linalg.norm(direction_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
            t_hit = torch.rand(len(env_ids), device=self.device) * 0.8 + 0.8
            start_z = torch.rand(len(env_ids), device=self.device) * center[env_ids, 2] + 0.5 * center[env_ids, 2]
            vel = torch.zeros(len(env_ids), 3, device=self.device)
            vel[:, :2] = direction_xy * speed.unsqueeze(-1)
            vel[:, 2] = (center[env_ids, 2] - start_z) / t_hit + 0.5 * 9.81 * t_hit
            start = center[env_ids].clone()
            start[:, :2] = center[env_ids, :2] - vel[:, :2] * t_hit.unsqueeze(-1)
            start[:, 2] = start_z
            self._ball_start_pos[env_ids, ball_ids] = start
            self._ball_start_vel[env_ids, ball_ids] = vel
            self._balls_pos[env_ids, ball_ids] = start
            self._balls_vel[env_ids, ball_ids] = vel
            self._ball_launch_step[env_ids, ball_ids] = self.episode_length_buf[env_ids]
            self._ball_active[env_ids, ball_ids] = True

        active_envs, active_balls = self._ball_active.nonzero(as_tuple=True)
        if len(active_envs) > 0:
            t = (self.episode_length_buf[active_envs] - self._ball_launch_step[active_envs, active_balls]).float() * self.step_dt
            start = self._ball_start_pos[active_envs, active_balls]
            vel = self._ball_start_vel[active_envs, active_balls]
            pos = start + vel * t.unsqueeze(-1)
            pos[:, 2] = start[:, 2] + vel[:, 2] * t - 0.5 * 9.81 * t.square()
            cur_vel = vel.clone()
            cur_vel[:, 2] = vel[:, 2] - 9.81 * t
            self._balls_pos[active_envs, active_balls] = pos
            self._balls_vel[active_envs, active_balls] = cur_vel
            landed = pos[:, 2] < 0.2
            self._ball_active[active_envs[landed], active_balls[landed]] = False
        self._sync_obstacle_visuals(torch.arange(self.num_envs, device=self.device))

    def _spawn_view_camera(self) -> None:
        """Spawn view camera."""
        camera_path = self.cfg.viewer.cam_prim_path
        if not camera_path or camera_path == "/OmniverseKit_Persp" or self.sim.stage.GetPrimAtPath(camera_path).IsValid():
            return
        eye = torch.tensor([self.cfg.viewer.eye], dtype=torch.float32)
        target = torch.tensor([self.cfg.viewer.lookat], dtype=torch.float32)
        orientation = math_utils.quat_from_matrix(math_utils.create_rotation_matrix_from_view(eye, target, up_axis="Z"))[0]
        camera_cfg = sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=20.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0))
        camera_cfg.func(camera_path, camera_cfg, translation=tuple(float(v) for v in eye[0]), orientation=tuple(float(v) for v in orientation))

    def _spawn_obstacle_visuals(self) -> None:
        """Spawn obstacle visuals."""
        self._column_prims = []
        self._ball_prims = []
        if not self.cfg.spawn_obstacle_visuals:
            return
        env_index = 0
        column_cfg = sim_utils.CylinderCfg(radius=spec.COLUMN_RADIUS, height=spec.STATIC_HEIGHT, axis="Z", visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.8)))
        ball_cfg = sim_utils.SphereCfg(radius=spec.BALL_RADIUS, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))
        for index in range(self.cfg.static_obstacles):
            prim = column_cfg.func(f"/World/envs/env_{env_index}/columns/column_{index:02d}", column_cfg, translation=(0.0, 0.0, -10.0))
            self._column_prims.append(prim)
        for index in range(self.cfg.num_balls):
            prim = ball_cfg.func(f"/World/envs/env_{env_index}/balls/ball_{index:02d}", ball_cfg, translation=(0.0, 0.0, -10.0))
            self._ball_prims.append(prim)

    def _sync_obstacle_visuals(self, env_ids: torch.Tensor) -> None:
        """Sync obstacle visuals."""
        if not getattr(self, "_column_prims", None) and not getattr(self, "_ball_prims", None):
            return
        if not (env_ids == 0).any():
            return
        origin = self.scene.env_origins[0].detach().cpu()
        for index, prim in enumerate(self._column_prims):
            position = self._columns[0, index].detach().cpu() + origin
            position[2] = spec.STATIC_HEIGHT / 2.0
            sim_utils.standardize_xform_ops(prim, translation=tuple(float(v) for v in position))
        for index, prim in enumerate(self._ball_prims):
            position = self._balls_pos[0, index].detach().cpu() + origin
            sim_utils.set_prim_visibility(prim, bool(self._ball_active[0, index]))
            sim_utils.standardize_xform_ops(prim, translation=tuple(float(v) for v in position))
