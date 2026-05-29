"""Minimal MAPPO wrapper that feeds policy observations (not centralized state) to the policy.

SKRL's stock MAPPO passes ``states`` (from ``env.state()``) to both the policy and the
value network.  In our task the policy input (86 dims) differs from the critic input
(232 dims).  This thin subclass overrides ``act`` to inject ``observations`` into the
policy's input dict so the attention encoder sees the correct 86-dim observation.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch
from skrl.multi_agents.torch.mappo import MAPPO


class PaperMAPPO(MAPPO):
    """MAPPO that passes policy observations to the policy instead of states."""

    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        timestep: int = 0,
        timesteps: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Act with policy observations, not centralized states."""
        self._current_observations = observations
        self._current_states = states
        actions: dict[str, torch.Tensor] = {}
        log_prob: dict[str, torch.Tensor] = {}
        outputs: dict[str, Any] = {}
        current_values: dict[str, torch.Tensor] = {}
        for uid in self.possible_agents:
            inputs = {
                "states": self._state_preprocessor[uid](states[uid]),
                "observations": observations[uid],
            }
            actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
            log_prob[uid] = outputs[uid].get("log_prob")

            if self.training:
                values, _ = self.values[uid].act(inputs, role="value")
                current_values[uid] = self._value_preprocessor[uid](values, inverse=True)

        self._current_log_prob = log_prob
        self._current_values = current_values
        return actions, outputs

    def reset_optimizer_state(self) -> None:
        """Reset optimiser / scheduler state for all agents to facilitate curriculum transfer.

        Recreates the Adam optimiser and any configured learning-rate scheduler
        from scratch while leaving model weights and preprocessors untouched.
        """
        for uid in self.possible_agents:
            self.optimizers[uid] = torch.optim.Adam(
                itertools.chain(self.policies[uid].parameters(), self.values[uid].parameters()),
                lr=self.cfg.learning_rate[uid][0],
            )
            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

            if self.schedulers[uid] is not None:
                sched_cls = self.cfg.learning_rate_scheduler[uid][0]
                sched_kwargs = self.cfg.learning_rate_scheduler_kwargs[uid][0]
                self.schedulers[uid] = sched_cls(self.optimizers[uid], **sched_kwargs)
