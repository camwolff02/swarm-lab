"""Environment-owned feature cache for the quad swarm paper task."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import warp as wp

import isaaclab.utils.math as math_utils

CachePhase = Literal["reward", "observation"]


@dataclass(frozen=True)
class KinematicsState:
    """Low-level root state for one simulated asset."""

    root_pos_env: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    root_quat_w: torch.Tensor
    rot_mat_w: torch.Tensor


@dataclass(frozen=True)
class SwarmKinematicsState:
    """Stacked kinematics for a stable agent ordering."""

    asset_names: tuple[str, ...]
    root_pos_env: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    root_quat_w: torch.Tensor
    rot_mat_w: torch.Tensor


@dataclass(frozen=True)
class TrackingState:
    """Goal-tracking features derived from cached kinematics."""

    action_term: Any | None
    kinematics: KinematicsState | SwarmKinematicsState
    target_pos: torch.Tensor
    position_error: torch.Tensor
    distance: torch.Tensor


class EnvCache:
    """Cache persistent handles and phase-local derived state for an Isaac Lab env."""

    _VALID_PHASES = frozenset(("reward", "observation"))

    def __init__(self, env: Any) -> None:
        self._env = env
        self._assets: dict[str, Any] = {}
        self._action_terms: dict[str, Any] = {}
        self._phase_caches: dict[CachePhase, dict[tuple[Any, ...], Any]] = {
            "reward": {},
            "observation": {},
        }
        self.stats: defaultdict[str, int] = defaultdict(int)

    def new_step(self) -> None:
        """Invalidate per-step feature caches before simulator state advances."""

        self._clear_phase_caches()

    def on_reset(self) -> None:
        """Invalidate derived features after reset mutates simulator state."""

        self._clear_phase_caches()

    def invalidate_all(self) -> None:
        """Clear handles and all derived caches after unusual out-of-band mutation."""

        self._assets.clear()
        self._action_terms.clear()
        self._clear_phase_caches()

    def asset(self, asset_name: str) -> Any:
        """Return a persistent asset handle by name."""

        if asset_name in self._assets:
            self.stats["asset_hits"] += 1
            return self._assets[asset_name]

        self.stats["asset_misses"] += 1
        asset = self._resolve_asset(asset_name)
        self._assets[asset_name] = asset
        return asset

    def action_term(self, term_name: str) -> Any:
        """Return a persistent action-term handle by name."""

        if term_name in self._action_terms:
            self.stats["action_term_hits"] += 1
            return self._action_terms[term_name]

        self.stats["action_term_misses"] += 1
        action_manager = getattr(self._env, "action_manager", None)
        if action_manager is None or not hasattr(action_manager, "get_term"):
            raise AttributeError("The environment does not expose an action_manager.get_term(...) API.")
        term = action_manager.get_term(term_name)
        self._action_terms[term_name] = term
        return term

    def kinematics(self, asset_name: str, phase: CachePhase) -> KinematicsState:
        """Return cached root kinematics for one asset in the requested phase."""

        cache = self._phase_cache(phase)
        key = ("kinematics", asset_name)
        if key in cache:
            self.stats[f"{phase}_kinematics_hits"] += 1
            return cache[key]

        self.stats[f"{phase}_kinematics_misses"] += 1
        asset = self.asset(asset_name)
        data = asset.data
        root_quat = _as_torch(data.root_quat_w)
        angular_velocity = data.root_ang_vel_b if hasattr(data, "root_ang_vel_b") else data.root_ang_vel_w
        state = KinematicsState(
            root_pos_env=_as_torch(data.root_pos_w) - self._env.scene.env_origins,
            root_lin_vel_w=_as_torch(data.root_lin_vel_w),
            root_ang_vel_b=_as_torch(angular_velocity),
            root_quat_w=root_quat,
            rot_mat_w=math_utils.matrix_from_quat(root_quat),
        )
        cache[key] = state
        return state

    def swarm_kinematics(
        self,
        asset_names: Iterable[str],
        phase: CachePhase,
    ) -> SwarmKinematicsState:
        """Return cached stacked kinematics for the provided asset order."""

        names = tuple(asset_names)
        cache = self._phase_cache(phase)
        key = ("swarm_kinematics", names)
        if key in cache:
            self.stats[f"{phase}_swarm_kinematics_hits"] += 1
            return cache[key]

        self.stats[f"{phase}_swarm_kinematics_misses"] += 1
        states = [self.kinematics(name, phase) for name in names]
        swarm_state = SwarmKinematicsState(
            asset_names=names,
            root_pos_env=torch.stack([state.root_pos_env for state in states], dim=1),
            root_lin_vel_w=torch.stack([state.root_lin_vel_w for state in states], dim=1),
            root_ang_vel_b=torch.stack([state.root_ang_vel_b for state in states], dim=1),
            root_quat_w=torch.stack([state.root_quat_w for state in states], dim=1),
            rot_mat_w=torch.stack([state.rot_mat_w for state in states], dim=1),
        )
        cache[key] = swarm_state
        return swarm_state

    def tracking(self, asset_name: str, action_term_name: str | None, phase: CachePhase) -> TrackingState:
        """Return cached target-tracking features for one asset."""

        cache = self._phase_cache(phase)
        key = ("tracking", asset_name, action_term_name)
        if key in cache:
            self.stats[f"{phase}_tracking_hits"] += 1
            return cache[key]

        self.stats[f"{phase}_tracking_misses"] += 1
        action_term = self.action_term(action_term_name) if action_term_name is not None else None
        kinematics = self.kinematics(asset_name, phase)
        target_pos = self._target_position(asset_name, action_term)
        position_error = target_pos - kinematics.root_pos_env
        tracking = TrackingState(
            action_term=action_term,
            kinematics=kinematics,
            target_pos=target_pos,
            position_error=position_error,
            distance=torch.linalg.norm(position_error, dim=-1),
        )
        cache[key] = tracking
        return tracking

    def swarm_tracking(
        self,
        asset_names: Iterable[str],
        phase: CachePhase,
        action_term_name: str | None = None,
    ) -> TrackingState:
        """Return cached stacked target-tracking features for a stable asset order."""

        names = tuple(asset_names)
        cache = self._phase_cache(phase)
        key = ("swarm_tracking", names, action_term_name)
        if key in cache:
            self.stats[f"{phase}_swarm_tracking_hits"] += 1
            return cache[key]

        self.stats[f"{phase}_swarm_tracking_misses"] += 1
        action_term = self.action_term(action_term_name) if action_term_name is not None else None
        kinematics = self.swarm_kinematics(names, phase)
        target_pos = self._swarm_target_position(names, action_term)
        position_error = target_pos - kinematics.root_pos_env
        tracking = TrackingState(
            action_term=action_term,
            kinematics=kinematics,
            target_pos=target_pos,
            position_error=position_error,
            distance=torch.linalg.norm(position_error, dim=-1),
        )
        cache[key] = tracking
        return tracking

    def _phase_cache(self, phase: CachePhase) -> dict[tuple[Any, ...], Any]:
        if phase not in self._VALID_PHASES:
            raise ValueError(f"Unknown cache phase {phase!r}. Expected one of {sorted(self._VALID_PHASES)}.")
        return self._phase_caches[phase]

    def _clear_phase_caches(self) -> None:
        for cache in self._phase_caches.values():
            cache.clear()

    def _resolve_asset(self, asset_name: str) -> Any:
        scene = getattr(self._env, "scene", None)
        if scene is not None:
            try:
                return scene[asset_name]
            except (KeyError, TypeError, AttributeError):
                pass
            articulations = getattr(scene, "articulations", None)
            if articulations is not None and asset_name in articulations:
                return articulations[asset_name]

        drones = getattr(self._env, "drones", None)
        if drones is not None and asset_name in drones:
            return drones[asset_name]

        raise KeyError(f"Unable to resolve asset {asset_name!r} from the environment.")

    def _target_position(self, asset_name: str, action_term: Any | None) -> torch.Tensor:
        target = _read_target_from_action_term(action_term)
        if target is not None:
            return target

        goals = getattr(self._env, "_goals", None)
        agent_ids = getattr(self._env, "_agent_ids", ())
        if goals is None or asset_name not in agent_ids:
            raise AttributeError(f"No target source is available for asset {asset_name!r}.")
        return goals[:, agent_ids.index(asset_name)]

    def _swarm_target_position(self, asset_names: tuple[str, ...], action_term: Any | None) -> torch.Tensor:
        target = _read_target_from_action_term(action_term)
        if target is not None:
            return target

        goals = getattr(self._env, "_goals", None)
        agent_ids = getattr(self._env, "_agent_ids", ())
        if goals is None:
            raise AttributeError("No target source is available for swarm tracking.")
        indices = [agent_ids.index(asset_name) for asset_name in asset_names]
        return goals[:, indices]


def _as_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


def _read_target_from_action_term(action_term: Any | None) -> torch.Tensor | None:
    if action_term is None:
        return None

    for name in ("target_pos_w", "target_position_w", "target_pos", "target_position", "command"):
        if not hasattr(action_term, name):
            continue
        value = getattr(action_term, name)
        if callable(value):
            value = value()
        if value is not None:
            return _as_torch(value)
    return None
