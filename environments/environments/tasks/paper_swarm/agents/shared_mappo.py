"""Paper-aligned MAPPO variant for the paper_swarm task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import torch
from skrl import config
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch.base import MultiAgent
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.resources.schedulers.torch import KLAdaptiveLR
from torch import nn
from torch.nn import functional as F


class ValueNorm1(nn.Module):
    """Exponential value normalizer used by the paper's released MAPPO code."""

    def __init__(self, input_shape: int | Sequence[int] = 1, *, beta: float = 0.995, epsilon: float = 1.0e-5) -> None:
        """Initialize the ValueNorm1 instance."""
        super().__init__()
        shape = torch.Size((input_shape,)) if isinstance(input_shape, int) else torch.Size(input_shape)
        self.input_shape = shape
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_mean_sq", torch.zeros(shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    def running_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the command-line entry point."""
        debiasing_term = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debiasing_term
        mean_sq = self.running_mean_sq / debiasing_term
        return mean, (mean_sq - mean.square()).clamp(min=1.0e-2)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Update."""
        dims = tuple(range(values.dim() - len(self.input_shape)))
        weight = self.beta
        self.running_mean.mul_(weight).add_(values.mean(dim=dims) * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(values.square().mean(dim=dims) * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 - weight)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize."""
        mean, var = self.running_mean_var()
        return (values - mean) / torch.sqrt(var)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize."""
        mean, var = self.running_mean_var()
        return values * torch.sqrt(var) + mean


class PaperSharedMAPPO(MAPPO):
    """MAPPO with shared actor/critic and observation critic input."""

    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Mapping[str, Model]],
        memories: Mapping[str, Memory] | None = None,
        observation_spaces: Mapping[str, int | Sequence[int] | gym.Space] | None = None,
        action_spaces: Mapping[str, int | Sequence[int] | gym.Space] | None = None,
        device: str | torch.device | None = None,
        cfg: dict[str, Any] | None = None,
        shared_observation_spaces: Mapping[str, int | Sequence[int] | gym.Space] | None = None,
    ) -> None:
        """Initialize the PaperSharedMAPPO instance."""
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=cfg,
            shared_observation_spaces=shared_observation_spaces,
        )
        self._shared_uid = self.possible_agents[0]
        self.policy = self.policies[self._shared_uid]
        self.value = self.values[self._shared_uid]
        self._use_value_norm = bool(self.cfg.get("use_value_norm", True))
        self._value_normalizer = ValueNorm1(1, beta=float(self.cfg.get("value_norm_beta", 0.995))).to(self.device)
        self._huber_delta = float(self.cfg.get("huber_delta", 10.0))
        self._use_huber_loss = bool(self.cfg.get("use_huber_loss", True))
        self._scale_actor_loss_by_action_dim = bool(self.cfg.get("scale_actor_loss_by_action_dim", True))

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate[self._shared_uid])
        self.critic_optimizer = torch.optim.Adam(self.value.parameters(), lr=self._learning_rate[self._shared_uid])
        self.actor_scheduler = self._make_scheduler(self.actor_optimizer)
        self.critic_scheduler = self._make_scheduler(self.critic_optimizer)
        self.optimizers = {uid: self.actor_optimizer for uid in self.possible_agents}
        self.schedulers = {uid: self.actor_scheduler for uid in self.possible_agents if self.actor_scheduler is not None}

        for uid in self.possible_agents:
            self.checkpoint_modules[uid].pop("optimizer", None)
            self.checkpoint_modules[uid]["policy"] = self.policy
            self.checkpoint_modules[uid]["value"] = self.value
            self.checkpoint_modules[uid]["actor_optimizer"] = self.actor_optimizer
            self.checkpoint_modules[uid]["critic_optimizer"] = self.critic_optimizer
            if self._use_value_norm:
                self.checkpoint_modules[uid]["value_normalizer"] = self._value_normalizer

    def reset_optimizer_state(self) -> None:
        """Recreate optimizers and schedulers from config after loading model weights."""
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate[self._shared_uid])
        self.critic_optimizer = torch.optim.Adam(self.value.parameters(), lr=self._learning_rate[self._shared_uid])
        self.actor_scheduler = self._make_scheduler(self.actor_optimizer)
        self.critic_scheduler = self._make_scheduler(self.critic_optimizer)
        self.optimizers = {uid: self.actor_optimizer for uid in self.possible_agents}
        self.schedulers = {uid: self.actor_scheduler for uid in self.possible_agents if self.actor_scheduler is not None}
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["actor_optimizer"] = self.actor_optimizer
            self.checkpoint_modules[uid]["critic_optimizer"] = self.critic_optimizer

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Make scheduler."""
        scheduler_cls = self._learning_rate_scheduler[self._shared_uid]
        if scheduler_cls is None:
            return None
        return scheduler_cls(optimizer, **self._learning_rate_scheduler_kwargs[self._shared_uid])

    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record transition."""
        MultiAgent.record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if not self.memories:
            return

        self._current_shared_next_states = dict(next_states)
        for uid in self.possible_agents:
            if self._rewards_shaper is not None:
                rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act(
                    {"states": self._shared_state_preprocessor[uid](states[uid])},
                    role="value",
                )
                bootstrap_values = self._denormalize_values(values)

            if self._time_limit_bootstrap[uid]:
                rewards[uid] += self._discount_factor[uid] * bootstrap_values * truncated[uid]

            self.memories[uid].add_samples(
                states=states[uid],
                actions=actions[uid],
                rewards=rewards[uid],
                next_states=next_states[uid],
                terminated=terminated[uid],
                truncated=truncated[uid],
                log_prob=self._current_log_prob[uid],
                values=values,
                shared_states=states[uid],
            )

    def _denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values."""
        if not self._use_value_norm:
            return values
        return self._value_normalizer.denormalize(values)

    def _normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns."""
        if not self._use_value_norm:
            return returns
        return self._value_normalizer.normalize(returns)

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        *,
        discount_factor: float,
        lambda_coefficient: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gae."""
        advantage = torch.zeros_like(last_values)
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        for index in reversed(range(rewards.shape[0])):
            next_values = values[index + 1] if index < rewards.shape[0] - 1 else last_values
            advantage = rewards[index] - values[index] + discount_factor * not_dones[index] * (
                next_values + lambda_coefficient * advantage
            )
            advantages[index] = advantage
        return advantages + values, advantages

    def _flatten_memory_tensor(self, name: str) -> torch.Tensor:
        """Flatten memory tensor."""
        return torch.cat(
            [self.memories[uid].get_tensor_by_name(name).reshape(-1, self.memories[uid].get_tensor_by_name(name).shape[-1]) for uid in self.possible_agents],
            dim=0,
        )

    def _make_minibatches(self, batch_size: int, mini_batches: int) -> torch.Tensor:
        """Make minibatches."""
        usable = (batch_size // mini_batches) * mini_batches
        return torch.randperm(usable, device=self.device).reshape(mini_batches, -1)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Update."""
        returns_by_uid: dict[str, torch.Tensor] = {}
        advantages_by_uid: dict[str, torch.Tensor] = {}

        for uid in self.possible_agents:
            memory = self.memories[uid]
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                self.value.train(False)
                last_values, _, _ = self.value.act(
                    {"states": self._shared_state_preprocessor[uid](self._current_shared_next_states[uid].float())},
                    role="value",
                )
                self.value.train(True)

            values = self._denormalize_values(memory.get_tensor_by_name("values"))
            returns, advantages = self._compute_gae(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                values=values,
                last_values=self._denormalize_values(last_values),
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )
            returns_by_uid[uid] = returns
            advantages_by_uid[uid] = advantages

        all_returns = torch.cat([returns.reshape(-1, returns.shape[-1]) for returns in returns_by_uid.values()], dim=0)
        all_advantages = torch.cat([adv.reshape(-1, adv.shape[-1]) for adv in advantages_by_uid.values()], dim=0)
        advantages_mean = all_advantages.mean()
        advantages_std = all_advantages.std()
        if self._use_value_norm:
            self._value_normalizer.update(all_returns)

        for uid in self.possible_agents:
            self.memories[uid].set_tensor_by_name("returns", self._normalize_returns(returns_by_uid[uid]))
            self.memories[uid].set_tensor_by_name(
                "advantages",
                (advantages_by_uid[uid] - advantages_mean) / (advantages_std + 1.0e-8),
            )

        states = self._flatten_memory_tensor("states")
        shared_states = self._flatten_memory_tensor("shared_states")
        actions = self._flatten_memory_tensor("actions")
        old_log_prob = self._flatten_memory_tensor("log_prob")
        old_values = self._flatten_memory_tensor("values")
        returns = self._flatten_memory_tensor("returns")
        advantages = self._flatten_memory_tensor("advantages")

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_explained_var = 0.0
        updates = 0

        for epoch in range(self._learning_epochs[self._shared_uid]):
            kl_divergences = []
            for indices in self._make_minibatches(states.shape[0], self._mini_batches[self._shared_uid]):
                sampled_states = states[indices]
                sampled_shared_states = shared_states[indices]
                sampled_actions = actions[indices]
                sampled_old_log_prob = old_log_prob[indices]
                sampled_old_values = old_values[indices]
                sampled_returns = returns[indices]
                sampled_advantages = advantages[indices]

                _, next_log_prob, _ = self.policy.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="policy",
                )
                ratio_log = next_log_prob - sampled_old_log_prob
                with torch.no_grad():
                    kl_divergence = ((torch.exp(ratio_log) - 1.0) - ratio_log).mean()
                    kl_divergences.append(kl_divergence)
                if self._kl_threshold[self._shared_uid] and kl_divergence > self._kl_threshold[self._shared_uid]:
                    break

                ratio = torch.exp(ratio_log)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clamp(
                    ratio,
                    1.0 - self._ratio_clip[self._shared_uid],
                    1.0 + self._ratio_clip[self._shared_uid],
                )
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                if self._scale_actor_loss_by_action_dim:
                    policy_loss = policy_loss * sampled_actions.shape[-1]
                entropy_loss = -self._entropy_loss_scale[self._shared_uid] * self.policy.get_entropy(role="policy").mean()

                self.actor_optimizer.zero_grad()
                (policy_loss + entropy_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                if self._grad_norm_clip[self._shared_uid] > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip[self._shared_uid])
                self.actor_optimizer.step()

                predicted_values, _, _ = self.value.act({"states": sampled_shared_states}, role="value")
                value_pred_clipped = sampled_old_values + torch.clamp(
                    predicted_values - sampled_old_values,
                    min=-self._value_clip[self._shared_uid],
                    max=self._value_clip[self._shared_uid],
                )
                value_loss_original = self._critic_loss(sampled_returns, predicted_values)
                value_loss_clipped = self._critic_loss(sampled_returns, value_pred_clipped)
                value_loss = torch.max(value_loss_original, value_loss_clipped).mean() * self._value_loss_scale[self._shared_uid]

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                if config.torch.is_distributed:
                    self.value.reduce_parameters()
                if self._grad_norm_clip[self._shared_uid] > 0:
                    nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip[self._shared_uid])
                self.critic_optimizer.step()

                with torch.no_grad():
                    explained_var = 1.0 - F.mse_loss(predicted_values, sampled_returns) / sampled_returns.var().clamp_min(1.0e-8)
                cumulative_policy_loss += float(policy_loss.detach())
                cumulative_entropy_loss += float(entropy_loss.detach())
                cumulative_value_loss += float(value_loss.detach())
                cumulative_explained_var += float(explained_var.detach())
                updates += 1

            if self._kl_threshold[self._shared_uid] and kl_divergences and kl_divergences[-1] > self._kl_threshold[self._shared_uid]:
                break

        self._step_schedulers(kl_divergences)
        divisor = max(updates, 1)
        self.track_data("Loss / Policy loss (shared)", cumulative_policy_loss / divisor)
        self.track_data("Loss / Value loss (shared)", cumulative_value_loss / divisor)
        self.track_data("Loss / Entropy loss (shared)", cumulative_entropy_loss / divisor)
        self.track_data("Learning / Explained variance (shared)", cumulative_explained_var / divisor)
        self.track_data("Learning / Advantages mean (shared)", float(advantages_mean.detach()))
        self.track_data("Learning / Advantages std (shared)", float(advantages_std.detach()))
        self.track_data("Policy / Standard deviation (shared)", self.policy.distribution(role="policy").stddev.mean().item())
        if self.actor_scheduler is not None:
            self.track_data("Learning / Actor learning rate (shared)", self.actor_scheduler.get_last_lr()[0])
        if self.critic_scheduler is not None:
            self.track_data("Learning / Critic learning rate (shared)", self.critic_scheduler.get_last_lr()[0])

    def _critic_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Critic loss."""
        if self._use_huber_loss:
            return F.huber_loss(prediction, target, delta=self._huber_delta, reduction="none")
        return F.mse_loss(prediction, target, reduction="none")

    def _step_schedulers(self, kl_divergences: list[torch.Tensor]) -> None:
        """Step schedulers."""
        for scheduler in (self.actor_scheduler, self.critic_scheduler):
            if scheduler is None:
                continue
            if isinstance(scheduler, KLAdaptiveLR):
                kl = torch.stack(kl_divergences).mean() if kl_divergences else torch.zeros((), device=self.device)
                scheduler.step(float(kl.detach()))
            else:
                scheduler.step()
