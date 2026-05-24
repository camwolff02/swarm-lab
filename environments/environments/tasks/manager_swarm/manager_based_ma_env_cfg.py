# Copyright (c) 2026, The IsaacLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for manager-based multi-agent environments.

This module defines the v1 public configuration surface for a proposed
``ManagerBasedMaEnv`` / ``ManagerBasedMarlEnv`` stack.

Design goals
------------
* Keep the public API close to IsaacLab's existing manager-based configs.
* Expose PettingZoo-compatible agent identities via ``possible_agents``.
* Support homogeneous and heterogeneous agents through reusable profiles and
  agent sets.
* Keep trainer-specific concepts such as policy sharing and teams lightweight.
* Leave graph observations, communication, opponent metadata, and advanced
  research-specific features inside normal manager terms or ``metadata`` until
  they need first-class runtime support.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections import defaultdict
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass

# Type aliases used for readability. Concrete manager config types are kept as
# ``Any`` because IsaacLab manager configs are task-specific @configclass objects.
AgentID = str
AgentSetName = str
AgentProfileName = str
PolicyName = str
TeamName = str

ResetMode = Literal["global", "any_agent", "all_agents", "per_agent"]
AgentSelectionMode = Literal["agents", "generated", "regex"]


@configclass
class AgentProfileCfg:
    """Reusable manager configuration bundle for one agent type.

    A profile describes *what an agent is*: its observation manager config,
    action manager config, optional reward/termination configs, and optional
    per-agent metadata. Many concrete agents may reuse the same profile.

    The profile is intentionally manager-shaped. For example,
    ``observations`` should usually be a normal IsaacLab observations config
    containing groups such as ``policy`` and optionally ``critic``.
    """

    # Required manager groups for externally controlled agents.
    observations: Any = MISSING
    actions: Any = MISSING

    # Optional manager groups. These are required by ManagerBasedMarlEnv for
    # trainable RL agents, but remain optional here so the same profile concept
    # works for classical ManagerBasedMaEnv tasks.
    rewards: Any | None = None
    terminations: Any | None = None
    commands: Any | None = None
    curriculum: Any | None = None

    # Optional per-agent noise models. These mirror the conceptual role of
    # DirectMARLEnvCfg's per-agent observation/action noise support.
    observation_noise_model: Any | None = None
    action_noise_model: Any | None = None

    # String used by the compiler when materializing manager term params.
    # Common values:
    #   "{agent_id}"                 -> one scene entity per agent id
    #   "robot"                      -> all agents act on one articulation
    #   "{team}_{agent_id}"          -> expanded using set/team context
    entity_name: str = "{agent_id}"

    # Free-form metadata available to custom term functions or wrappers.
    # Examples: {"role": "pursuer"}, {"opponents": ["red"]},
    # {"limb": "left_arm"}, {"capabilities": ["fly", "carry"]}
    metadata: dict[str, Any] = field(default_factory=dict)


# TODO rename to agent group instead of agent set
@configclass
class AgentSetCfg:
    """A homogeneous or role-homogeneous set of concrete agents.

    A set describes *which agents exist* and which profile/training metadata
    they use. This is the ergonomic layer for homogeneous swarms, role groups,
    heterogeneous teams, and regex-based assignment.

    Agent membership can be declared in one of three ways:

    * ``agents``: explicit PettingZoo agent ids.
    * ``count`` + ``id_template``: generated ids, e.g. ``drone_{i}``.
    * ``regex``: match ids from ``ManagerBasedMaEnvCfg.possible_agents``.

    Regex sets require the top-level ``possible_agents`` list to be provided.
    Generated and explicit sets can be used to derive ``possible_agents``.
    """

    name: AgentSetName = MISSING

    # Membership declaration. Exactly one mode should be used in normal configs.
    agents: list[AgentID] | None = None
    count: int | None = None
    id_template: str | None = None
    regex: str | None = None

    # Reusable profile key.
    profile: AgentProfileName = MISSING

    # Lightweight trainer-facing metadata. These are not required by PettingZoo,
    # but are useful for wrappers, logging, and policy mapping.
    policy: PolicyName | None = None
    team: TeamName | None = None

    # Agent sets with the same policy are expected to share parameters. If this
    # is False, wrappers may still expose the policy name but should not optimize
    # or train it.
    trainable: bool = True

    # Optional decision cadence for macro-actions or lower-rate controllers.
    # Runtime code can process actions every ``decision_period`` environment
    # steps and hold actions in between.
    decision_period: int = 1
    hold_action_between_decisions: bool = True

    # Conflict resolution when multiple regex/explicit sets match the same
    # agent. Higher priority wins if the runtime compiler allows overlap.
    priority: int = 0

    # Per-set overrides applied after deep-copying the referenced profile.
    # This lets a role share most of a profile while changing one term param.
    overrides: dict[str, Any] = field(default_factory=dict)

    # Free-form set metadata used by reward/termination/observation terms,
    # wrappers, curriculum, or experiment logging.
    metadata: dict[str, Any] = field(default_factory=dict)

    def selection_mode(self) -> AgentSelectionMode:
        """Return the membership declaration mode for this set."""
        if self.agents is not None:
            return "agents"
        if self.count is not None or self.id_template is not None:
            return "generated"
        if self.regex is not None:
            return "regex"
        raise ValueError(f"AgentSetCfg(name={self.name!r}) does not declare any agents.")


@configclass
class MultiAgentOptionsCfg:
    """Runtime compatibility and validation options."""

    # Observation group returned as the PettingZoo/SKRL per-agent observation.
    observation_group: str = "policy"

    # How vectorized sub-environments are reset from per-agent done buffers.
    reset_on: ResetMode = "global"

    # PettingZoo allows ``agents`` to shrink during an episode. IsaacLab
    # vectorized training is usually easier and faster with fixed agents, so the
    # v1 default is static agents.
    dynamic_agents: bool = False

    # Validation knobs used by the runtime compiler/wrappers.
    validate_unique_agents: bool = True
    validate_profile_references: bool = True
    validate_spaces: bool = True
    validate_policy_space_sharing: bool = True

    # If True, expose helper maps such as agent_to_policy, agent_to_team,
    # policy_to_agents, and team_to_agents for wrappers/loggers.
    expose_agent_metadata_maps: bool = True

    # Optional name of a bool tensor stored on the env with shape
    # (num_envs, len(possible_agents)). Inactive agents remain present in the
    # public PettingZoo/SKRL ABI but are masked out by task terms.
    active_agent_mask_key: str | None = None


@configclass
class ManagerBasedMaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a manager-based multi-agent classical environment.

    This class intentionally flattens the multi-agent v1 public API onto the
    environment config instead of nesting it under a separate ``MultiAgentCfg``.
    That keeps the shape close to existing IsaacLab manager-based configs while
    still separating reusable agent definitions into ``profiles`` and ``sets``.
    """

    # Optional PettingZoo universe of agent ids. If omitted, it can be derived
    # from explicit/generated AgentSetCfg entries. It is required when using
    # regex-only sets.
    observations: Any | None = None
    actions: Any | None = None
    rewards: Any | None = None
    terminations: Any | None = None

    possible_agents: list[AgentID] | None = None

    # Reusable manager bundles and membership declarations.
    profiles: dict[AgentProfileName, AgentProfileCfg] = MISSING
    sets: list[AgentSetCfg] = MISSING

    # Optional centralized/global state config. The structured CTDE config is
    # declared in manager_based_marl_env_cfg.py to keep MARL-specific state
    # semantics out of classical multi-agent tasks.
    state: Any | None = None

    # Runtime options and validation controls.
    ma_options: MultiAgentOptionsCfg = MultiAgentOptionsCfg()


@dataclass(frozen=True)
class AgentRuntimeSpec:
    """Materialized runtime declaration for one concrete PettingZoo agent."""

    agent_id: str
    profile_name: str
    set_name: str
    policy: str | None
    team: str | None
    trainable: bool
    decision_period: int
    hold_action_between_decisions: bool
    entity_name: str
    profile: AgentProfileCfg
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiAgentRuntimeSpec:
    """Compiled multi-agent declaration used by ManagerBasedMa/MarlEnv."""

    possible_agents: tuple[str, ...]
    agents: dict[str, AgentRuntimeSpec]
    sets: dict[str, list[str]]
    policies: dict[str, list[str]]
    teams: dict[str, list[str]]

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self.possible_agents

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)


def compile_multi_agent_spec(cfg: ManagerBasedMaEnvCfg) -> MultiAgentRuntimeSpec:
    """Compile a flattened multi-agent config into a normalized runtime spec."""
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
    assigned: dict[str, tuple[AgentSetCfg, str]] = {}
    set_to_agents: dict[str, list[str]] = {}

    for set_cfg in sorted(cfg.sets, key=lambda item: getattr(item, "priority", 0), reverse=True):
        if cfg.ma_options.validate_profile_references and set_cfg.profile not in cfg.profiles:
            raise ValueError(f"AgentSetCfg(name={set_cfg.name!r}) references unknown profile {set_cfg.profile!r}.")

        mode = set_cfg.selection_mode()
        if mode == "regex":
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
            if agent_id in assigned and cfg.ma_options.validate_unique_agents:
                old_set = assigned[agent_id][0].name
                raise ValueError(f"Agent {agent_id!r} is assigned by multiple sets: {old_set!r} and {set_cfg.name!r}.")
            assigned[agent_id] = (set_cfg, set_cfg.profile)

    missing = [agent_id for agent_id in possible_agents if agent_id not in assigned]
    if missing:
        raise ValueError(f"The following possible_agents are not assigned to any set: {missing!r}")

    runtime_agents: dict[str, AgentRuntimeSpec] = {}
    policies: dict[str, list[str]] = defaultdict(list)
    teams: dict[str, list[str]] = defaultdict(list)

    for agent_id in possible_agents:
        set_cfg, profile_name = assigned[agent_id]
        profile_template = cfg.profiles[profile_name]
        context = {
            "agent_id": agent_id,
            "set_name": set_cfg.name,
            "profile": profile_name,
            "policy": set_cfg.policy or "",
            "team": set_cfg.team or "",
        }
        entity_name = _format_value(profile_template.entity_name, context)
        context["entity_name"] = entity_name

        profile = copy.deepcopy(profile_template)
        profile = _materialize_value(profile, context)
        _apply_overrides(profile, set_cfg.overrides, context)

        metadata = {}
        metadata.update(copy.deepcopy(getattr(profile_template, "metadata", {}) or {}))
        metadata.update(copy.deepcopy(getattr(set_cfg, "metadata", {}) or {}))
        metadata = _materialize_value(metadata, context)

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
            profile=profile,
            metadata=metadata,
        )
        if set_cfg.policy:
            policies[set_cfg.policy].append(agent_id)
        if set_cfg.team:
            teams[set_cfg.team].append(agent_id)

    return MultiAgentRuntimeSpec(
        possible_agents=tuple(possible_agents),
        agents=runtime_agents,
        sets={name: list(agent_ids) for name, agent_ids in set_to_agents.items()},
        policies={name: list(agent_ids) for name, agent_ids in policies.items()},
        teams={name: list(agent_ids) for name, agent_ids in teams.items()},
    )


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


def _format_value(value: str, context: dict[str, Any]) -> str:
    return value.format(**context)


def _materialize_value(value: Any, context: dict[str, Any]) -> Any:
    """Recursively substitute string templates in common config containers."""
    if isinstance(value, str):
        return value.format(**context)
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
