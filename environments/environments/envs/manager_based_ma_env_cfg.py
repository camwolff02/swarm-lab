# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for manager-based multi-agent environments.

This module defines the classical multi-agent layer: observations, actions,
commands, and the fixed PettingZoo/SKRL agent universe. It intentionally does
not define rewards, terminations, or curriculum; those belong to
:mod:`manager_based_marl_env_cfg`.

The public authoring model is deliberately small:

* :class:`AgentCfg` describes one concrete agent's manager terms.
* :class:`AgentGroupCfg` declares multiple agents that share one
  :class:`AgentCfg` and substitutes templates such as ``"{agent_id}"`` and
  ``"{i}"``.
* :class:`ManagerBasedMaEnvCfg` combines explicit agents and groups, then
  :func:`compile_multi_agent_spec` expands the declaration into concrete runtime
  metadata before managers are constructed.

Groups expand into concrete agent configs before managers are built. Runtime
APIs still expose the fixed ``possible_agents`` list expected by
PettingZoo-compatible multi-agent trainers.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import MISSING, dataclass, field
from typing import Any

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.configclass import configclass

AgentID = str
AgentGroupName = str


class _SafeFormatDict(dict):
    """Dictionary that preserves unknown format placeholders."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@configclass
class AgentCfg:
    """Manager configuration for one concrete PettingZoo agent.

    The contained manager configs should be ordinary IsaacLab manager
    configuration objects. String fields inside the config are materialized with
    the current agent context when the config is compiled. The most common
    placeholder is ``"{agent_id}"`` for selecting the scene asset owned by that
    agent.

    Attributes:
        asset_name: Scene asset name owned by the agent.
        observations: Observation manager configuration.
        actions: Action manager configuration.
        commands: Optional command manager configuration.
    """

    asset_name: str = "{agent_id}"

    observations: Any = MISSING
    actions: Any = MISSING
    commands: Any | None = None


@configclass
class AgentGroupCfg:
    """Convenience declaration for agents that share the same :class:`AgentCfg`.

    Groups are a compact way to declare multiple agents with the same manager
    config. Each group expands into concrete ``dict[agent_id, AgentCfg]``
    entries before managers are built. Use a group when many agents only differ
    by templated names such as asset ids.

    Attributes:
        name: Group name used for metadata and template substitution.
        agent_cfg: Agent config template to copy for each concrete id.
        agent_ids: Explicit concrete agent ids. Mutually exclusive with
            :attr:`count`.
        count: Number of agents to generate from :attr:`id_template`. Mutually
            exclusive with :attr:`agent_ids`.
        id_template: Template used when :attr:`count` is supplied.
    """

    name: AgentGroupName = MISSING
    agent_cfg: AgentCfg = MISSING

    agent_ids: list[AgentID] | None = None
    count: int | None = None
    id_template: str = "{name}_{i}"

    def expand_agent_ids(self) -> list[AgentID]:
        """Return the concrete agent ids described by the group."""

        has_agent_ids = self.agent_ids is not None
        has_count = self.count is not None
        if has_agent_ids == has_count:
            raise ValueError(
                f"AgentGroupCfg(name={self.name!r}) must specify exactly one of agent_ids or count."
            )
        if has_agent_ids:
            return list(self.agent_ids or [])
        if self.count is None or self.count < 0:
            raise ValueError(f"AgentGroupCfg(name={self.name!r}) count must be >= 0.")
        return [
            self.id_template.format_map(_SafeFormatDict({"name": self.name, "group_name": self.name, "i": i}))
            for i in range(self.count)
        ]


@configclass
class ManagerBasedMaEnvCfg(DirectMARLEnvCfg):
    """Configuration for a manager-based classical multi-agent environment.

    This config bridges IsaacLab manager terms to the public ABI inherited from
    :class:`isaaclab.envs.DirectMARLEnv`. It keeps ``possible_agents`` fixed for
    PettingZoo/SKRL compatibility and derives observation/action spaces from the
    managers at runtime.

    Attributes:
        possible_agents: Fixed public agent universe. If omitted, it is derived
            from explicit agents and groups in declaration order.
        agents: Explicit per-agent configs.
        agent_groups: Group declarations that expand to concrete agents.
        observation_group: Observation manager group exposed as the policy
            observation for each agent.
        active_agent_mask_key: Optional environment attribute name containing a
            ``[num_envs, num_agents]`` mask used to zero inactive-agent actions.
        recorders: Optional recorder manager configuration.
    """

    possible_agents: list[AgentID] | None = None

    # DirectMARLEnvCfg required fields. These are derived from managers at runtime,
    # but must be non-MISSING before DirectMARLEnvCfg validation runs.
    observation_spaces: dict[AgentID, Any] = field(default_factory=dict)
    action_spaces: dict[AgentID, Any] = field(default_factory=dict)
    state_space: Any = 0
    episode_length_s: float = math.inf

    agents: dict[AgentID, AgentCfg] = field(default_factory=dict)
    agent_groups: list[AgentGroupCfg] = field(default_factory=list)

    observation_group: str = "policy"
    active_agent_mask_key: str | None = None

    # Optional ManagerBasedEnv-style recorder config. DirectMARLEnvCfg does not
    # define this field, but the manager-based runtime can use it when supplied.
    recorders: Any | None = None


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """Materialized metadata for one concrete PettingZoo agent."""

    agent_id: AgentID
    agent_index: int
    group_name: AgentGroupName | None
    asset_name: str
    cfg: AgentCfg


@dataclass(frozen=True)
class AgentGroupRuntimeSpec:
    """Manager execution unit produced from the compiled agent declaration.

    The runtime builds one manager bundle per concrete agent. The group metadata
    is still preserved so tasks and loggers can recover how an agent was
    declared.
    """

    name: AgentGroupName
    agent_ids: tuple[AgentID, ...]
    agent_indices: tuple[int, ...]
    asset_names: tuple[str, ...]
    cfg: AgentCfg

    @property
    def num_agents(self) -> int:
        """Number of concrete agents in the execution group."""

        return len(self.agent_ids)

    @property
    def entity_names(self) -> tuple[str, ...]:
        """Backward-compatible alias for asset names."""

        return self.asset_names


@dataclass(frozen=True)
class MultiAgentRuntimeSpec:
    """Compiled multi-agent declaration used by ManagerBasedMa/MarlEnv."""

    possible_agents: tuple[AgentID, ...]
    agents: dict[AgentID, AgentRuntimeSpec]
    execution_groups: dict[AgentGroupName, AgentGroupRuntimeSpec]
    groups: dict[AgentGroupName, list[AgentID]]

    @property
    def agent_ids(self) -> tuple[AgentID, ...]:
        """Alias for the fixed PettingZoo-style possible-agent list."""

        return self.possible_agents

    @property
    def num_agents(self) -> int:
        """Number of fixed possible agents."""

        return len(self.possible_agents)


def compile_multi_agent_spec(cfg: ManagerBasedMaEnvCfg) -> MultiAgentRuntimeSpec:
    """Resolve explicit agents and agent groups into runtime metadata.

    Args:
        cfg: Multi-agent environment config to compile.

    Returns:
        Concrete runtime metadata with one :class:`AgentRuntimeSpec` per public
        agent and one :class:`AgentGroupRuntimeSpec` execution group per agent.

    Raises:
        ValueError: If the declaration is empty, has duplicate ids, or the fixed
            ``possible_agents`` universe does not match the configured agents.
    """

    resolved_agents: dict[str, AgentCfg] = {}
    agent_to_group: dict[str, str | None] = {}
    derived_agents: list[str] = []

    for agent_id, agent_cfg in cfg.agents.items():
        resolved_agents[agent_id] = _materialize_agent_cfg(agent_cfg, agent_id=agent_id, group_name=None, index=0)
        agent_to_group[agent_id] = None
        derived_agents.append(agent_id)

    groups: dict[str, list[str]] = {}
    for group in cfg.agent_groups:
        agent_ids = group.expand_agent_ids()
        groups[group.name] = agent_ids
        for index, agent_id in enumerate(agent_ids):
            if agent_id in resolved_agents:
                raise ValueError(f"Agent {agent_id!r} is configured more than once.")
            resolved_agents[agent_id] = _materialize_agent_cfg(
                group.agent_cfg, agent_id=agent_id, group_name=group.name, index=index
            )
            agent_to_group[agent_id] = group.name
            derived_agents.append(agent_id)

    possible_agents = list(cfg.possible_agents or _unique_preserving_order(derived_agents))
    if not possible_agents:
        raise ValueError("ManagerBasedMaEnvCfg requires possible_agents or at least one configured agent/group.")
    if len(possible_agents) != len(set(possible_agents)):
        raise ValueError(f"cfg.possible_agents contains duplicates: {possible_agents!r}")

    missing = [agent_id for agent_id in possible_agents if agent_id not in resolved_agents]
    if missing:
        raise ValueError(f"The following possible_agents are missing AgentCfg entries: {missing!r}")
    extras = sorted(set(resolved_agents) - set(possible_agents))
    if extras:
        raise ValueError(f"AgentCfg entries are not present in possible_agents: {extras!r}")

    runtime_agents = {
        agent_id: AgentRuntimeSpec(
            agent_id=agent_id,
            agent_index=index,
            group_name=agent_to_group[agent_id],
            asset_name=resolved_agents[agent_id].asset_name,
            cfg=resolved_agents[agent_id],
        )
        for index, agent_id in enumerate(possible_agents)
    }

    execution_groups = {
        agent_id: AgentGroupRuntimeSpec(
            name=agent_id,
            agent_ids=(agent_id,),
            agent_indices=(runtime_agents[agent_id].agent_index,),
            asset_names=(runtime_agents[agent_id].asset_name,),
            cfg=runtime_agents[agent_id].cfg,
        )
        for agent_id in possible_agents
    }

    return MultiAgentRuntimeSpec(
        possible_agents=tuple(possible_agents),
        agents=runtime_agents,
        execution_groups=execution_groups,
        groups=groups,
    )


def _materialize_agent_cfg(agent_cfg: AgentCfg, *, agent_id: str, group_name: str | None, index: int) -> AgentCfg:
    context = {
        "agent_id": agent_id,
        "entity_name": agent_id,
        "group_name": group_name or "",
        "name": group_name or agent_id,
        "i": index,
    }
    return _materialize_value(copy.deepcopy(agent_cfg), context)


def _unique_preserving_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _format_template(value: str, context: dict[str, Any]) -> str:
    return value.format_map(_SafeFormatDict(context))


def _materialize_value(value: Any, context: dict[str, Any]) -> Any:
    """Recursively substitute string templates in common config containers."""

    if isinstance(value, str):
        return _format_template(value, context)
    if isinstance(value, list):
        return [_materialize_value(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_value(item, context) for item in value)
    if isinstance(value, dict):
        return {_materialize_value(key, context): _materialize_value(item, context) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        for field_info in dataclasses.fields(value):
            if not hasattr(value, field_info.name):
                continue
            current = getattr(value, field_info.name)
            try:
                setattr(value, field_info.name, _materialize_value(current, context))
            except Exception:
                pass
        return value
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
        ("isaaclab", "isaaclab_tasks", "environments", "cpsquare_lab")
    ):
        for key, current in list(vars(value).items()):
            try:
                setattr(value, key, _materialize_value(current, context))
            except Exception:
                pass
        return value
    return value
