"""Minimal MAPPO wrapper that feeds policy observations (not centralized state) to the policy.

SKRL's stock MAPPO passes ``states`` (from ``env.state()``) to both the policy and the
value network.  In our task the policy input (86 dims) differs from the critic input
(232 dims).  This thin subclass overrides ``act`` to inject ``observations`` into the
policy's input dict so the attention encoder sees the correct 86-dim observation.
"""

from __future__ import annotations

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
        outputs: dict[str, Any] = {}
        for uid in self.possible_agents:
            inputs = {
                "states": self._state_preprocessor[uid](states[uid]),
                "observations": observations[uid],
            }
            actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
        return actions, outputs
