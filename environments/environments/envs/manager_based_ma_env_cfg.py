# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for manager-based multi-agent environments.

``ManagerBasedMaEnv`` is the classical/control layer. It is intentionally
multi-agent and manager-backed, but it does not define RL-specific reward,
termination, curriculum, or centralized-state configuration. The MARL extension
adds those in ``manager_based_marl_env_cfg.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from collections import defaultdict
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.configclass import configclass

AgentID = str
AgentSetName = str
AgentProfileName = str
PolicyName = str
TeamName = str

AgentSelectionMode = Literal["agents", "generated", "regex"]
ManagerGrouping = Literal["set", "agent"]
ResetMode = Literal["global", "any_agent", "all_agents"]


class _SafeFormatDict(dict):
    """Dictionary that preserves unknown format placeholders."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@configclass
class AgentProfileCfg:
    """Reusable classical manager bundle for one embodiment or role.

    A profile describes the observation, action, and optional command manager
    configs used by an agent set. MARL-only terms are added by
    :class:`AgentRlProfileCfg` in ``manager_based_marl_env_cfg.py``.
    """

    observations: Any = MISSING
    actions: Any = MISSING
    commands: Any | None = None

    observation_noise_model: Any | None = None
    action_noise_model: Any | None = None

    # Backward-compatible single-agent-style template. In grouped execution it
    # is expanded once per concrete agent to produce AgentSetRuntimeSpec.entity_names.
    entity_name: str = "{agent_id}"

    metadata: dict[str, Any] = field(default_factory=dict)


@configclass
class AgentSetCfg:
    """A homogeneous or role-homogeneous set of concrete PettingZoo agents."""

    name: AgentSetName = MISSING

    # Exactly one membership mode must be used.
    agents: list[AgentID] | None = None
    count: int | None = None
    id_template: str | None = None
    regex: str | None = None

    profile: AgentProfileName = MISSING

    # Lightweight trainer/logger metadata. These do not affect classical
    # stepping, but are useful for wrappers and MARL configs that inherit this.
    policy: PolicyName | None = None
    team: TeamName | None = None
    trainable: bool = True

    decision_period: int = 1
    hold_action_between_decisions: bool = True

    # Dotted-path overrides applied after deep-copying the profile for this set.
    overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def selection_mode(self) -> AgentSelectionMode:
        """Return the declared membership mode.

        Raises:
            ValueError: If zero or more than one membership mode is specified.
        """

        has_agents = self.agents is not None
        has_generated = self.count is not None or self.id_template is not None
        has_regex = self.regex is not None
        selected = int(has_agents) + int(has_generated) + int(has_regex)
        if selected != 1:
            raise ValueError(
                f"AgentSetCfg(name={self.name!r}) must declare exactly one of agents, count/id_template, or regex."
            )
        if has_agents:
            return "agents"
        if has_generated:
            return "generated"
        return "regex"


@configclass
class MultiAgentOptionsCfg:
    """Runtime compatibility and pooling options."""

    observation_group: str = "policy"

    # ``set`` is the performant V2 default. ``agent`` is a compatibility escape
    # hatch for manager terms that only support one entity at a time.
    manager_grouping: ManagerGrouping = "set"

    # Used by MARL reset aggregation; harmless for the classical base env whose
    # done hooks are neutral placeholders.
    reset_on: ResetMode = "global"

    dynamic_agents: bool = False
    validate_unique_agents: bool = True
    validate_profile_references: bool = True
    validate_spaces: bool = True
    validate_policy_space_sharing: bool = True
    expose_agent_metadata_maps: bool = True

    # Optional bool tensor on the root env with shape
    # ``(num_envs, len(possible_agents))``. Inactive agents remain part of the
    # public ABI but their actions/rewards/terminations may be masked out.
    active_agent_mask_key: str | None = None


@configclass
class ManagerBasedMaEnvCfg(DirectMARLEnvCfg):
    """Configuration for the classical manager-based multi-agent env.

    This class inherits ``DirectMARLEnvCfg`` so the runtime can reuse
    DirectMARLEnv's PettingZoo-like multi-agent API. Observation/action spaces
    are derived from pooled managers at runtime, so inherited DirectMARLEnvCfg
    fields are neutralized here instead of left as ``MISSING``.
    """

    possible_agents: list[AgentID] | None = None

    # DirectMARLEnvCfg required fields. These are filled/overridden by the
    # runtime after managers are created, but must be non-MISSING before
    # cfg.validate() runs in DirectMARLEnv.__init__.
    observation_spaces: dict[AgentID, Any] = field(default_factory=dict)
    action_spaces: dict[AgentID, Any] = field(default_factory=dict)
    state_space: Any = 0
    episode_length_s: float = math.inf

    # Optional ManagerBasedEnv-style recorder config. DirectMARLEnvCfg does not
    # define this field, but the manager-based runtime can use it when supplied.
    recorders: Any | None = None

    profiles: dict[AgentProfileName, AgentProfileCfg] = MISSING
    sets: list[AgentSetCfg] = MISSING
    ma_options: MultiAgentOptionsCfg = MultiAgentOptionsCfg()


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """Materialized metadata for one concrete PettingZoo agent."""

    agent_id: AgentID
    profile_name: AgentProfileName
    set_name: AgentSetName
    policy: PolicyName | None
    team: TeamName | None
    trainable: bool
    decision_period: int
    hold_action_between_decisions: bool
    entity_name: str
    profile: AgentProfileCfg
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSetRuntimeSpec:
    """Pooled execution unit used by V2 manager creation."""

    name: AgentSetName
    profile_name: AgentProfileName
    agent_ids: tuple[AgentID, ...]
    agent_indices: tuple[int, ...]
    entity_names: tuple[str, ...]
    policy: PolicyName | None
    team: TeamName | None
    trainable: bool
    decision_period: int
    hold_action_between_decisions: bool
    profile: AgentProfileCfg
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        """Number of concrete agents in the execution set."""

        return len(self.agent_ids)


@dataclass(frozen=True)
class MultiAgentRuntimeSpec:
    """Compiled multi-agent declaration used by ManagerBasedMa/MarlEnv."""

    possible_agents: tuple[AgentID, ...]
    agents: dict[AgentID, AgentRuntimeSpec]
    execution_sets: dict[AgentSetName, AgentSetRuntimeSpec]
    sets: dict[AgentSetName, list[AgentID]]
    policies: dict[PolicyName, list[AgentID]]
    teams: dict[TeamName, list[AgentID]]

    @property
    def agent_ids(self) -> tuple[AgentID, ...]:
        """Alias for the fixed PettingZoo-style possible-agent list."""

        return self.possible_agents

    @property
    def num_agents(self) -> int:
        """Number of fixed possible agents."""

        return len(self.possible_agents)


def compile_multi_agent_spec(cfg: ManagerBasedMaEnvCfg) -> MultiAgentRuntimeSpec:
    """Compile the flattened public config into agent and set runtime specs.

    Args:
        cfg: Manager-based multi-agent environment config.

    Returns:
        A normalized runtime spec with both per-agent metadata and pooled
        execution-set metadata.

    Raises:
        ValueError: If profiles, sets, generated IDs, regex assignment, or
        uniqueness constraints are invalid.
    """

    if not cfg.profiles:
        raise ValueError("ManagerBasedMaEnvCfg.profiles must not be empty.")
    if not cfg.sets:
        raise ValueError("ManagerBasedMaEnvCfg.sets must not be empty.")

    canonical_agents = list(cfg.possible_agents or [])
    derived_agents: list[str] = []

    for set_cfg in cfg.sets:
        mode = set_cfg.selection_mode()
        if mode == "regex":
            if not canonical_agents:
                raise ValueError(
                    f"AgentSetCfg(name={set_cfg.name!r}) uses regex but cfg.possible_agents is not provided."
                )
            continue
        derived_agents.extend(_expand_non_regex_set(set_cfg))

    possible_agents = canonical_agents if canonical_agents else _unique_preserving_order(derived_agents)
    if len(possible_agents) != len(set(possible_agents)):
        raise ValueError(f"cfg.possible_agents contains duplicates: {possible_agents!r}")

    possible_agent_set = set(possible_agents)
    agent_to_set: dict[str, AgentSetCfg] = {}
    set_to_agents: dict[str, list[str]] = {}

    for set_cfg in cfg.sets:
        if cfg.ma_options.validate_profile_references and set_cfg.profile not in cfg.profiles:
            raise ValueError(f"AgentSetCfg(name={set_cfg.name!r}) references unknown profile {set_cfg.profile!r}.")

        if set_cfg.selection_mode() == "regex":
            regex = re.compile(set_cfg.regex or "")
            agent_ids = [agent_id for agent_id in possible_agents if regex.fullmatch(agent_id)]
        else:
            agent_ids = _expand_non_regex_set(set_cfg)

        if not agent_ids:
            raise ValueError(f"AgentSetCfg(name={set_cfg.name!r}) did not match/generate any agents.")

        unknown = sorted(set(agent_ids) - possible_agent_set)
        if unknown:
            raise ValueError(f"AgentSetCfg(name={set_cfg.name!r}) assigns agents not in possible_agents: {unknown!r}.")

        set_to_agents[set_cfg.name] = list(agent_ids)
        for agent_id in agent_ids:
            if agent_id in agent_to_set and cfg.ma_options.validate_unique_agents:
                old_set = agent_to_set[agent_id].name
                raise ValueError(f"Agent {agent_id!r} is assigned by multiple sets: {old_set!r} and {set_cfg.name!r}.")
            agent_to_set[agent_id] = set_cfg

    missing = [agent_id for agent_id in possible_agents if agent_id not in agent_to_set]
    if missing:
        raise ValueError(f"The following possible_agents are not assigned to any set: {missing!r}")

    runtime_agents: dict[str, AgentRuntimeSpec] = {}
    execution_sets: dict[str, AgentSetRuntimeSpec] = {}
    policies: dict[str, list[str]] = defaultdict(list)
    teams: dict[str, list[str]] = defaultdict(list)

    for set_cfg in cfg.sets:
        if set_cfg.name not in set_to_agents:
            continue
        profile_name = set_cfg.profile
        profile_template = cfg.profiles[profile_name]
        agent_ids = tuple(set_to_agents[set_cfg.name])
        agent_indices = tuple(possible_agents.index(agent_id) for agent_id in agent_ids)

        entity_names = tuple(
            _format_template(
                profile_template.entity_name,
                _context(
                    agent_id=agent_id,
                    set_cfg=set_cfg,
                    profile_name=profile_name,
                    entity_name="",
                    entity_names=(),
                ),
            )
            for agent_id in agent_ids
        )

        set_context = _context(
            agent_id=None,
            set_cfg=set_cfg,
            profile_name=profile_name,
            entity_name="",
            entity_names=entity_names,
        )
        set_profile = copy.deepcopy(profile_template)
        set_profile = _materialize_value(set_profile, set_context)
        _apply_overrides(set_profile, set_cfg.overrides, set_context)

        set_metadata = {}
        set_metadata.update(copy.deepcopy(getattr(profile_template, "metadata", {}) or {}))
        set_metadata.update(copy.deepcopy(getattr(set_cfg, "metadata", {}) or {}))
        set_metadata = _materialize_value(set_metadata, set_context)

        execution_sets[set_cfg.name] = AgentSetRuntimeSpec(
            name=set_cfg.name,
            profile_name=profile_name,
            agent_ids=agent_ids,
            agent_indices=agent_indices,
            entity_names=entity_names,
            policy=set_cfg.policy,
            team=set_cfg.team,
            trainable=set_cfg.trainable,
            decision_period=set_cfg.decision_period,
            hold_action_between_decisions=set_cfg.hold_action_between_decisions,
            profile=set_profile,
            metadata=set_metadata,
        )

        for agent_id, entity_name in zip(agent_ids, entity_names):
            agent_context = _context(
                agent_id=agent_id,
                set_cfg=set_cfg,
                profile_name=profile_name,
                entity_name=entity_name,
                entity_names=entity_names,
            )
            agent_profile = copy.deepcopy(profile_template)
            agent_profile = _materialize_value(agent_profile, agent_context)
            _apply_overrides(agent_profile, set_cfg.overrides, agent_context)

            agent_metadata = {}
            agent_metadata.update(copy.deepcopy(getattr(profile_template, "metadata", {}) or {}))
            agent_metadata.update(copy.deepcopy(getattr(set_cfg, "metadata", {}) or {}))
            agent_metadata = _materialize_value(agent_metadata, agent_context)

            runtime_agents[agent_id] = AgentRuntimeSpec(
                agent_id=agent_id,
                profile_name=profile_name,
                set_name=set_cfg.name,
                policy=set_cfg.policy,
                team=set_cfg.team,
                trainable=set_cfg.trainable,
                decision_period=set_cfg.decision_period,
                hold_action_between_decisions=set_cfg.hold_action_between_decisions,
                entity_name=entity_name,
                profile=agent_profile,
                metadata=agent_metadata,
            )
            if set_cfg.policy:
                policies[set_cfg.policy].append(agent_id)
            if set_cfg.team:
                teams[set_cfg.team].append(agent_id)

    return MultiAgentRuntimeSpec(
        possible_agents=tuple(possible_agents),
        agents=runtime_agents,
        execution_sets=execution_sets,
        sets={name: list(agent_ids) for name, agent_ids in set_to_agents.items()},
        policies={name: list(agent_ids) for name, agent_ids in policies.items()},
        teams={name: list(agent_ids) for name, agent_ids in teams.items()},
    )


def _context(
    *,
    agent_id: str | None,
    set_cfg: AgentSetCfg,
    profile_name: str,
    entity_name: str,
    entity_names: tuple[str, ...],
) -> dict[str, Any]:
    context = {
        "set_name": set_cfg.name,
        "profile": profile_name,
        "policy": set_cfg.policy or "",
        "team": set_cfg.team or "",
        "entity_name": entity_name,
        "entity_names": entity_names,
        "num_agents": len(entity_names),
    }
    if agent_id is not None:
        context["agent_id"] = agent_id
    return context


def _expand_non_regex_set(set_cfg: AgentSetCfg) -> list[str]:
    mode = set_cfg.selection_mode()
    if mode == "agents":
        return list(set_cfg.agents or [])
    if mode == "generated":
        if set_cfg.count is None or set_cfg.id_template is None:
            raise ValueError(f"AgentSetCfg(name={set_cfg.name!r}) generated mode requires count and id_template.")
        if set_cfg.count < 0:
            raise ValueError(f"AgentSetCfg(name={set_cfg.name!r}) count must be >= 0.")
        return [set_cfg.id_template.format(i=i) for i in range(set_cfg.count)]
    raise ValueError(f"_expand_non_regex_set called for regex set {set_cfg.name!r}.")


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
                # Some config fields may be read-only descriptors. Leave them unchanged.
                pass
        return value
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
        ("isaaclab", "isaaclab_tasks", "environments")
    ):
        for key, current in list(vars(value).items()):
            try:
                setattr(value, key, _materialize_value(current, context))
            except Exception:
                pass
        return value
    return value


def _apply_overrides(target: Any, overrides: dict[str, Any], context: dict[str, Any]) -> None:
    """Apply dotted-path overrides into a profile object or dictionary."""

    for path, raw_value in (overrides or {}).items():
        value = _materialize_value(copy.deepcopy(raw_value), context)
        parts = path.split(".")
        obj = target
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)
