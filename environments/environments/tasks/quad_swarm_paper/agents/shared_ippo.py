"""Shared-policy/shared-optimizer IPPO for the quad swarm paper task.

Stock skrl multi-agent IPPO keeps per-agent optimizer/update streams. This
module owns the paper-faithful path instead: one homogeneous decentralized
policy, one local value function, pooled rollouts over all drones/environments,
and one optimizer per role.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import tqdm
from skrl.models.torch import Model
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from environments.tasks.quad_swarm_paper.models.quad_swarm_encoder import QuadSwarmEncoderCfg
from environments.tasks.quad_swarm_paper.models.quad_swarm_skrl_models import (
    QuadSwarmDeterministicValue,
    QuadSwarmGaussianPolicy,
)

SHARED_HOMOGENEOUS_IPPO_KEY = "shared_homogeneous_ippo"


@dataclass(frozen=True)
class SharedIPPOComponents:
    """Shared model/optimizer ownership for the homogeneous decentralized policy."""

    agent_ids: tuple[str, ...]
    policy: QuadSwarmGaussianPolicy
    value: QuadSwarmDeterministicValue
    policy_optimizer: torch.optim.Optimizer
    value_optimizer: torch.optim.Optimizer


@dataclass
class SharedRolloutStorage:
    """Contiguous pooled rollout storage with explicit ``[T, E, N, ...]`` axes."""

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    index: int = 0

    @classmethod
    def create(
        cls,
        *,
        rollouts: int,
        num_envs: int,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> SharedRolloutStorage:
        tensor_shape = (int(rollouts), int(num_envs), int(num_agents))
        return cls(
            observations=torch.zeros((*tensor_shape, observation_dim), device=device, dtype=torch.float32),
            actions=torch.zeros((*tensor_shape, action_dim), device=device, dtype=torch.float32),
            log_probs=torch.zeros((*tensor_shape, 1), device=device, dtype=torch.float32),
            values=torch.zeros((*tensor_shape, 1), device=device, dtype=torch.float32),
            rewards=torch.zeros((*tensor_shape, 1), device=device, dtype=torch.float32),
            terminated=torch.zeros((*tensor_shape, 1), device=device, dtype=torch.bool),
            truncated=torch.zeros((*tensor_shape, 1), device=device, dtype=torch.bool),
        )

    @property
    def full(self) -> bool:
        return self.index >= self.observations.shape[0]

    def add(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        if self.full:
            raise RuntimeError("Shared rollout storage is full. Call reset() after PPO update.")
        self.observations[self.index].copy_(observations.detach())
        self.actions[self.index].copy_(actions.detach())
        self.log_probs[self.index].copy_(log_probs.detach())
        self.values[self.index].copy_(values.detach())
        self.rewards[self.index].copy_(rewards.detach())
        self.terminated[self.index].copy_(terminated.detach())
        self.truncated[self.index].copy_(truncated.detach())
        self.index += 1

    def reset(self) -> None:
        self.index = 0


class SharedIPPOAgent:
    """Paper-faithful homogeneous decentralized IPPO with one shared update stream."""

    def __init__(self, env: Any, cfg: Mapping[str, Any]) -> None:
        self.env = env
        self.cfg = cfg
        self.components = build_shared_ippo_components(env, cfg)
        self.agent_ids = self.components.agent_ids
        self.policy = self.components.policy
        self.value = self.components.value
        self.policy_optimizer = self.components.policy_optimizer
        self.value_optimizer = self.components.value_optimizer
        self.device = torch.device(env.device)
        self.num_envs = int(env.num_envs)
        self.num_agents = len(self.agent_ids)

        agent_cfg = cfg.get("agent", {})
        if not isinstance(agent_cfg, Mapping):
            agent_cfg = {}
        trainer_cfg = cfg.get("trainer", {})
        if not isinstance(trainer_cfg, Mapping):
            trainer_cfg = {}
        experiment_cfg = agent_cfg.get("experiment", {})
        if not isinstance(experiment_cfg, Mapping):
            experiment_cfg = {}

        self.rollouts = int(agent_cfg.get("rollouts", 128))
        self.learning_epochs = int(agent_cfg.get("learning_epochs", 8))
        self.mini_batches = int(agent_cfg.get("mini_batches", 16))
        self.shared_mini_batch_size = _resolve_shared_mini_batch_size(
            agent_cfg=agent_cfg,
            rollouts=self.rollouts,
            num_envs=self.num_envs,
            mini_batches=self.mini_batches,
        )
        self.discount_factor = float(agent_cfg.get("discount_factor", 0.99))
        self.lambda_coefficient = float(agent_cfg.get("lambda", 0.95))
        self.grad_norm_clip = float(agent_cfg.get("grad_norm_clip", 0.0))
        self.ratio_clip = float(agent_cfg.get("ratio_clip", 0.2))
        self.value_clip = float(agent_cfg.get("value_clip", 0.2))
        self.clip_predicted_values = bool(agent_cfg.get("clip_predicted_values", True))
        self.entropy_loss_scale = float(agent_cfg.get("entropy_loss_scale", 0.0))
        self.value_loss_scale = float(agent_cfg.get("value_loss_scale", 1.0))
        self.kl_threshold = float(agent_cfg.get("kl_threshold", 0.0))
        self.mixed_precision = bool(agent_cfg.get("mixed_precision", False))
        self._device_type = self.device.type
        self._scaler_enabled = self.mixed_precision and self._device_type == "cuda"
        self.policy_scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._scaler_enabled)
        self.value_scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._scaler_enabled)

        reference_agent = self.agent_ids[0]
        observation_dim = _flat_space_size(env.observation_spaces[reference_agent])
        action_dim = _flat_space_size(env.action_spaces[reference_agent])
        self.storage = SharedRolloutStorage.create(
            rollouts=self.rollouts,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            observation_dim=observation_dim,
            action_dim=action_dim,
            device=self.device,
        )

        directory = str(experiment_cfg.get("directory") or os.path.join(os.getcwd(), "runs"))
        experiment_name = str(experiment_cfg.get("experiment_name") or "shared_homogeneous_ippo")
        self.experiment_dir = os.path.join(directory, experiment_name)
        self.write_interval = _resolve_interval(experiment_cfg.get("write_interval", "auto"), trainer_cfg, divisor=100)
        self.checkpoint_interval = _resolve_interval(
            experiment_cfg.get("checkpoint_interval", "auto"), trainer_cfg, divisor=10
        )
        self.writer = SummaryWriter(log_dir=self.experiment_dir) if self.write_interval > 0 else None
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

        self.tracking_data: dict[str, list[float | torch.Tensor]] = {}
        self._instant_reward_sum = torch.zeros((), device=self.device)
        self._instant_reward_count = 0
        self._last_next_observations: torch.Tensor | None = None
        self._running_mode = "train"

    def set_running_mode(self, mode: str) -> None:
        self._running_mode = mode

    def set_mode(self, mode: str) -> None:
        self.policy.set_mode(mode)
        self.value.set_mode(mode)

    def act(
        self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        del timestep, timesteps
        observations = stack_agent_observations(states, self.agent_ids)
        flat_observations = flatten_agent_batch(observations)

        with torch.autocast(device_type=self._device_type, enabled=self._scaler_enabled):
            actions, log_probs, _ = self.policy.act({"states": flat_observations}, role="policy")
            values, _, _ = self.value.act({"states": flat_observations}, role="value")

        action_tensor = unflatten_agent_batch(actions, num_envs=self.num_envs, agent_ids=self.agent_ids)
        log_prob_tensor = unflatten_agent_batch(log_probs, num_envs=self.num_envs, agent_ids=self.agent_ids)
        value_tensor = unflatten_agent_batch(values, num_envs=self.num_envs, agent_ids=self.agent_ids)
        return unstack_agent_actions(action_tensor, self.agent_ids), log_prob_tensor, value_tensor

    def record_transition(
        self,
        *,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        observations = stack_agent_observations(states, self.agent_ids)
        action_tensor = stack_agent_tensors(actions, self.agent_ids, label="actions")
        reward_tensor = stack_agent_tensors(rewards, self.agent_ids, label="rewards")
        terminated_tensor = stack_agent_tensors(terminated, self.agent_ids, label="terminated").to(dtype=torch.bool)
        truncated_tensor = stack_agent_tensors(truncated, self.agent_ids, label="truncated").to(dtype=torch.bool)

        self.storage.add(
            observations=observations,
            actions=action_tensor,
            log_probs=log_probs,
            values=values,
            rewards=reward_tensor,
            terminated=terminated_tensor,
            truncated=truncated_tensor,
        )
        self._last_next_observations = stack_agent_observations(next_states, self.agent_ids).detach()

        # Keep TensorBoard enabled but avoid stock skrl's per-step .item() GPU syncs.
        self._instant_reward_sum += reward_tensor.sum(dim=1).mean().detach()
        self._instant_reward_count += 1

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        if self.storage.full:
            self.set_mode("train")
            self._update(timestep=timestep, timesteps=timesteps)
            self.set_mode("eval")
            self.storage.reset()

        step = timestep + 1
        if self.writer is not None and self.write_interval > 0 and step > 1 and not step % self.write_interval:
            self.write_tracking_data(step)
        if self.checkpoint_interval > 0 and step > 1 and not step % self.checkpoint_interval:
            self.write_checkpoint(timestep=step, timesteps=timesteps)

    def track_data(self, tag: str, value: float | torch.Tensor) -> None:
        self.tracking_data.setdefault(tag, []).append(value)

    def track_environment_info(self, infos: Mapping[str, Any], key: str) -> None:
        if key not in infos:
            return
        for name, value in infos[key].items():
            if isinstance(value, torch.Tensor):
                self.track_data(f"Info / {name}", value.detach().mean())
            elif isinstance(value, int | float):
                self.track_data(f"Info / {name}", float(value))

    def write_tracking_data(self, timestep: int) -> None:
        if self.writer is None:
            return
        if self._instant_reward_count:
            self.track_data(
                "Reward / Instantaneous reward (mean)",
                self._instant_reward_sum / self._instant_reward_count,
            )
            self._instant_reward_sum.zero_()
            self._instant_reward_count = 0

        for tag, values in self.tracking_data.items():
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                scalar = torch.stack([value.detach().to(self.device).float() for value in values]).mean().item()
            else:
                scalar = float(sum(float(value) for value in values) / len(values))
            self.writer.add_scalar(tag, scalar, timestep)
        self.writer.flush()
        self.tracking_data.clear()

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        del timesteps
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        tag = str(timestep)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "timestep": timestep,
                "agent_ids": self.agent_ids,
                "shared_homogeneous_ippo": True,
            },
            os.path.join(checkpoint_dir, f"agent_{tag}.pt"),
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "policy" in checkpoint and "value" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"])
            self.value.load_state_dict(checkpoint["value"])
            if "policy_optimizer" in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            if "value_optimizer" in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
            return

        # Compatibility with stock skrl per-agent checkpoints for evaluation warm starts.
        first_agent = next(iter(checkpoint))
        self.policy.load_state_dict(checkpoint[first_agent]["policy"])
        self.value.load_state_dict(checkpoint[first_agent]["value"])

    def _update(self, *, timestep: int, timesteps: int) -> None:
        del timestep, timesteps
        if self._last_next_observations is None:
            raise RuntimeError("Cannot update shared IPPO before at least one transition has been recorded.")

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._scaler_enabled):
            next_values, _, _ = self.value.act(
                {"states": flatten_agent_batch(self._last_next_observations.float())}, role="value"
            )
            next_values = unflatten_agent_batch(next_values, num_envs=self.num_envs, agent_ids=self.agent_ids)
            returns, advantages = self._compute_gae(next_values=next_values)

        flat_observations = flatten_rollout_tensor(self.storage.observations)
        flat_actions = flatten_rollout_tensor(self.storage.actions)
        flat_log_probs = flatten_rollout_tensor(self.storage.log_probs)
        flat_values = flatten_rollout_tensor(self.storage.values)
        flat_returns = flatten_rollout_tensor(returns)
        flat_advantages = flatten_rollout_tensor(advantages)
        num_samples = flat_observations.shape[0]
        # Stock skrl IPPO minibatches each agent independently. This shared path
        # pools all drones for a single optimizer, so keep the backward batch at
        # the old per-agent scale instead of multiplying it by num_agents.
        minibatch_size = min(max(self.shared_mini_batch_size, 1), num_samples)
        minibatches_per_epoch = (num_samples + minibatch_size - 1) // minibatch_size

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        kl_total = 0.0
        update_count = 0

        for _epoch in range(self.learning_epochs):
            indexes = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                batch_indexes = indexes[start : start + minibatch_size]
                mb_observations = flat_observations[batch_indexes]
                mb_actions = flat_actions[batch_indexes]
                mb_old_log_probs = flat_log_probs[batch_indexes]
                mb_old_values = flat_values[batch_indexes]
                mb_returns = flat_returns[batch_indexes]
                mb_advantages = flat_advantages[batch_indexes]

                with torch.autocast(device_type=self._device_type, enabled=self._scaler_enabled):
                    _, next_log_probs, _ = self.policy.act(
                        {"states": mb_observations, "taken_actions": mb_actions}, role="policy"
                    )
                    ratio_log = next_log_probs - mb_old_log_probs
                    ratio = torch.exp(ratio_log)
                    surrogate = mb_advantages * ratio
                    surrogate_clipped = mb_advantages * torch.clip(
                        ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                    entropy_loss = -self.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()

                self.policy_optimizer.zero_grad(set_to_none=True)
                self.policy_scaler.scale(policy_loss + entropy_loss).backward()
                if self.grad_norm_clip > 0:
                    self.policy_scaler.unscale_(self.policy_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clip)
                self.policy_scaler.step(self.policy_optimizer)
                self.policy_scaler.update()

                with torch.autocast(device_type=self._device_type, enabled=self._scaler_enabled):
                    predicted_values, _, _ = self.value.act({"states": mb_observations}, role="value")
                    if self.clip_predicted_values:
                        predicted_values = mb_old_values + torch.clip(
                            predicted_values - mb_old_values,
                            min=-self.value_clip,
                            max=self.value_clip,
                        )
                    value_loss = self.value_loss_scale * F.mse_loss(mb_returns, predicted_values)

                self.value_optimizer.zero_grad(set_to_none=True)
                self.value_scaler.scale(value_loss).backward()
                if self.grad_norm_clip > 0:
                    self.value_scaler.unscale_(self.value_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_norm_clip)
                self.value_scaler.step(self.value_optimizer)
                self.value_scaler.update()

                with torch.no_grad():
                    kl = ((torch.exp(ratio_log) - 1.0) - ratio_log).mean()
                policy_loss_total += float(policy_loss.detach().item())
                value_loss_total += float(value_loss.detach().item())
                entropy_total += float((-entropy_loss).detach().item())
                kl_total += float(kl.detach().item())
                update_count += 1
                if self.kl_threshold > 0 and kl > self.kl_threshold:
                    break

        denominator = max(update_count, 1)
        self.track_data("Loss / Shared policy loss", policy_loss_total / denominator)
        self.track_data("Loss / Shared value loss", value_loss_total / denominator)
        self.track_data("Policy / Shared entropy", entropy_total / denominator)
        self.track_data("Policy / Shared KL", kl_total / denominator)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().detach())
        self.track_data("Learning / Pooled advantage mean", flat_advantages.mean().detach())
        self.track_data("Learning / Pooled advantage std", flat_advantages.std().detach())
        self.track_data("Learning / Shared minibatch size", float(minibatch_size))
        self.track_data("Learning / Shared minibatches per epoch", float(minibatches_per_epoch))

    def _compute_gae(self, *, next_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = self.storage.rewards
        dones = self.storage.terminated | self.storage.truncated
        values = self.storage.values
        advantages = torch.zeros_like(rewards)
        advantage = torch.zeros_like(next_values)
        for step in reversed(range(self.rollouts)):
            next_step_values = values[step + 1] if step < self.rollouts - 1 else next_values
            not_done = dones[step].logical_not()
            delta = rewards[step] + self.discount_factor * not_done * next_step_values - values[step]
            advantage = delta + self.discount_factor * self.lambda_coefficient * not_done * advantage
            advantages[step] = advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        return returns, advantages


class SharedIPPOTrainer:
    """Minimal sequential trainer for the shared homogeneous IPPO agent."""

    def __init__(self, env: Any, agent: SharedIPPOAgent, cfg: Mapping[str, Any]) -> None:
        self.env = env
        self.agent = agent
        self.timesteps = int(cfg.get("timesteps", 0))
        self.initial_timestep = int(cfg.get("initial_timestep", 0))
        self.environment_info = cfg.get("environment_info", "episode")
        self.close_environment_at_exit = bool(cfg.get("close_environment_at_exit", True))
        self.disable_progressbar = bool(cfg.get("disable_progressbar", False))

    def train(self) -> None:
        self.agent.set_running_mode("train")
        self.agent.set_mode("eval")
        states, infos = self.env.reset()
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            with torch.no_grad():
                actions, log_probs, values = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                self.agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_probs=log_probs,
                    values=values,
                )
                self.agent.track_environment_info(infos, self.environment_info)

            self.agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
            if _any_done(terminated, truncated):
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

        if self.close_environment_at_exit:
            self.env.close()

    def eval(self) -> None:
        self.agent.set_running_mode("eval")
        self.agent.set_mode("eval")
        states, _ = self.env.reset()
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            with torch.no_grad():
                actions, _, _ = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)
                states, _, terminated, truncated, _ = self.env.step(actions)
                if _any_done(terminated, truncated):
                    states, _ = self.env.reset()


def shared_homogeneous_ippo_enabled(cfg: Mapping[str, Any]) -> bool:
    """Return whether the explicit shared-training mode is requested."""

    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, Mapping):
        return False
    return bool(training_cfg.get(SHARED_HOMOGENEOUS_IPPO_KEY, False))


def encoder_cfg_from_model_config(model_cfg: Mapping[str, Any]) -> QuadSwarmEncoderCfg:
    """Build the paper encoder config from the skrl YAML model section."""

    return QuadSwarmEncoderCfg(
        self_obs_dim=int(model_cfg.get("self_obs_dim", 19)),
        neighbor_obs_dim=int(model_cfg.get("neighbor_obs_dim", 12)),
        obstacle_obs_dim=int(model_cfg.get("obstacle_obs_dim", 9)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        initial_log_std=float(model_cfg.get("initial_log_std", -1.0)),
        init_policy_to_hover=bool(model_cfg.get("init_policy_to_hover", True)),
    )


def build_shared_ippo_components(env: Any, cfg: Mapping[str, Any]) -> SharedIPPOComponents:
    """Create exactly one policy, one value function, and one optimizer per role."""

    agent_ids = tuple(env.possible_agents)
    if not agent_ids:
        raise ValueError("Shared IPPO requires at least one agent in env.possible_agents.")

    _validate_homogeneous_spaces(env, agent_ids)
    model_cfg = cfg.get("models", {})
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    agent_cfg = cfg.get("agent", {})
    if not isinstance(agent_cfg, Mapping):
        agent_cfg = {}

    encoder_cfg = encoder_cfg_from_model_config(model_cfg)
    reference_agent = agent_ids[0]
    policy = QuadSwarmGaussianPolicy(
        env.observation_spaces[reference_agent],
        env.action_spaces[reference_agent],
        env.device,
        encoder_cfg=encoder_cfg,
    )
    value = QuadSwarmDeterministicValue(
        env.observation_spaces[reference_agent],
        env.action_spaces[reference_agent],
        env.device,
        encoder_cfg=encoder_cfg,
    )
    policy.init_state_dict(role="policy")
    value.init_state_dict(role="value")

    learning_rate = float(agent_cfg.get("learning_rate", 1.0e-4))
    return SharedIPPOComponents(
        agent_ids=agent_ids,
        policy=policy,
        value=value,
        # Unlike stock skrl IPPO, these optimizers are owned once globally,
        # not duplicated for each named drone.
        policy_optimizer=torch.optim.Adam(policy.parameters(), lr=learning_rate),
        value_optimizer=torch.optim.Adam(value.parameters(), lr=learning_rate),
    )


def stack_agent_observations(obs_dict: Mapping[str, torch.Tensor], agent_ids: Sequence[str]) -> torch.Tensor:
    """Stack per-agent observation dicts into ``[num_envs, num_agents, obs_dim]``.

    The ordering is always the explicit ``agent_ids`` order, expected to be
    ``env.possible_agents`` for the quad swarm task.
    """

    tensors = _ordered_agent_tensors(obs_dict, agent_ids, label="observations")
    return torch.stack(tensors, dim=1)


def stack_agent_tensors(
    tensor_dict: Mapping[str, torch.Tensor], agent_ids: Sequence[str], *, label: str
) -> torch.Tensor:
    """Stack a homogeneous per-agent tensor dict into ``[num_envs, num_agents, ...]``."""

    tensors = _ordered_agent_tensors(tensor_dict, agent_ids, label=label)
    return torch.stack(tensors, dim=1)


def flatten_agent_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten ``[num_envs, num_agents, ...]`` to ``[num_envs * num_agents, ...]``."""

    if tensor.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions [E, N, ...], got shape {tuple(tensor.shape)}.")
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def unflatten_agent_batch(tensor: torch.Tensor, *, num_envs: int, agent_ids: Sequence[str]) -> torch.Tensor:
    """Unflatten ``[num_envs * num_agents, ...]`` to ``[num_envs, num_agents, ...]``."""

    num_agents = len(agent_ids)
    if num_envs <= 0 or num_agents <= 0:
        raise ValueError("num_envs and agent_ids must describe a non-empty [E, N] batch.")
    expected = num_envs * num_agents
    if tensor.shape[0] != expected:
        raise ValueError(f"Expected leading dimension {expected}, got {tensor.shape[0]}.")
    return tensor.reshape(num_envs, num_agents, *tensor.shape[1:])


def unstack_agent_actions(action_tensor: torch.Tensor, agent_ids: Sequence[str]) -> dict[str, torch.Tensor]:
    """Convert ``[num_envs, num_agents, act_dim]`` actions back to the env dict API."""

    if action_tensor.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions [E, N, ...], got shape {tuple(action_tensor.shape)}.")
    if action_tensor.shape[1] != len(agent_ids):
        raise ValueError(
            f"Action tensor agent dimension {action_tensor.shape[1]} does not match {len(agent_ids)} agent ids."
        )
    return {agent_id: action_tensor[:, index].contiguous() for index, agent_id in enumerate(agent_ids)}


def flatten_rollout_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten pooled rollout tensors ``[T, E, N, ...]`` to ``[T * E * N, ...]``."""

    if tensor.ndim < 4:
        raise ValueError(f"Expected at least 4 dimensions [T, E, N, ...], got shape {tuple(tensor.shape)}.")
    return tensor.reshape(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], *tensor.shape[3:])


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    """Return optimizer-owned parameter object ids in param-group order."""

    return tuple(id(param) for group in optimizer.param_groups for param in group["params"])


def assert_optimizer_owns_model_once(model: Model, optimizer: torch.optim.Optimizer) -> None:
    """Validate that an optimizer owns each model parameter exactly once."""

    model_param_ids = tuple(id(param) for param in model.parameters())
    optimizer_param_ids = optimizer_parameter_ids(optimizer)
    if len(optimizer_param_ids) != len(set(optimizer_param_ids)):
        raise ValueError("Optimizer contains duplicate parameter references.")
    if set(optimizer_param_ids) != set(model_param_ids):
        raise ValueError("Optimizer parameter ownership does not match the model parameters.")


def _ordered_agent_tensors(
    tensor_dict: Mapping[str, torch.Tensor], agent_ids: Sequence[str], *, label: str
) -> list[torch.Tensor]:
    if not agent_ids:
        raise ValueError(f"Cannot collate {label} without agent ids.")

    missing = [agent_id for agent_id in agent_ids if agent_id not in tensor_dict]
    if missing:
        raise KeyError(f"Missing {label} for agents: {missing}.")

    tensors = [tensor_dict[agent_id] for agent_id in agent_ids]
    reference_shape = tuple(tensors[0].shape)
    for agent_id, tensor in zip(agent_ids[1:], tensors[1:], strict=True):
        if tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"Expected all {label} tensors to have shape {reference_shape}; "
                f"{agent_id} has shape {tuple(tensor.shape)}."
            )
    return tensors


def _validate_homogeneous_spaces(env: Any, agent_ids: Sequence[str]) -> None:
    reference_agent = agent_ids[0]
    reference_obs_shape = tuple(env.observation_spaces[reference_agent].shape)
    reference_action_shape = tuple(env.action_spaces[reference_agent].shape)
    for agent_id in agent_ids[1:]:
        obs_shape = tuple(env.observation_spaces[agent_id].shape)
        action_shape = tuple(env.action_spaces[agent_id].shape)
        if obs_shape != reference_obs_shape or action_shape != reference_action_shape:
            raise ValueError(
                "Shared homogeneous IPPO requires identical per-agent observation/action spaces; "
                f"{agent_id} has obs {obs_shape}, action {action_shape}, expected "
                f"obs {reference_obs_shape}, action {reference_action_shape}."
            )


def _flat_space_size(space: Any) -> int:
    if not hasattr(space, "shape") or space.shape is None:
        raise ValueError(f"Expected a Box-like space with a shape, got {space}.")
    size = 1
    for dim in space.shape:
        size *= int(dim)
    return size


def _resolve_interval(value: Any, trainer_cfg: Mapping[str, Any], *, divisor: int) -> int:
    if value == "auto":
        return int(int(trainer_cfg.get("timesteps", 0)) / divisor)
    return int(value)


def _resolve_shared_mini_batch_size(
    *, agent_cfg: Mapping[str, Any], rollouts: int, num_envs: int, mini_batches: int
) -> int:
    configured = agent_cfg.get("shared_mini_batch_size")
    if configured is not None:
        return int(configured)

    # ``mini_batches`` in stock skrl is per-agent. A homogeneous shared policy
    # still trains from pooled agent samples, but each optimizer step should not
    # silently grow by the number of drones.
    return max((int(rollouts) * int(num_envs)) // max(int(mini_batches), 1), 1)


def _any_done(terminated: Mapping[str, torch.Tensor], truncated: Mapping[str, torch.Tensor]) -> bool:
    first_terminated = next(iter(terminated.values()))
    first_truncated = next(iter(truncated.values()))
    return bool((first_terminated | first_truncated).any().item())
