# Patch recipe

This section is intentionally concrete enough for an implementation agent to apply in code, but still leaves room for exact naming and style choices.

## 1) Hover-biased policy initialization

**Primary file**

- `swarm-rl/environments/tasks/quad_swarm_paper/models/quad_swarm_skrl_models.py`

**Rationale**

The current policy head is a plain linear mean head. If it starts centered at zero, then the direct rotor mapping sends zero action to 50 percent of max thrust. For the translated Crazyflie in `cf2x.yaml`, hover is closer to 62.5 percent of max thrust, so the default operating point is below hover.

**Implementation sketch**

```python
from cpsquare_lab.embodiments.multirotor.cf2x.sim.robot import CRAZYFLIE_PARAMS


def _compute_hover_action() -> torch.Tensor:
    max_thrusts = torch.tensor(CRAZYFLIE_PARAMS.max_thrusts, dtype=torch.float32)
    hover_ratio = float(CRAZYFLIE_PARAMS.hover_thrust) / max_thrusts
    return hover_ratio * 2.0 - 1.0
```

Then inside `QuadSwarmGaussianPolicy.__init__`:

```python
self.mean_head = nn.Linear(encoder_cfg.output_dim, _space_size(action_space))
self.log_std_parameter = nn.Parameter(torch.full((_space_size(action_space),), -1.0))

hover_action = _compute_hover_action().to(device=self.device)
with torch.no_grad():
    nn.init.zeros_(self.mean_head.weight)
    self.mean_head.bias.copy_(hover_action)
```

**Notes**

- Zeroing or very-small-initializing `mean_head.weight` matters. Bias-only initialization is not enough if the final layer weights are still random.
- Keep the value head unchanged.
- If you want a config knob, add something like `init_policy_to_hover: true` and `initial_policy_log_std: -1.0`.

## 2) Replay activation fix

**Primary file**

- `swarm-rl/environments/tasks/quad_swarm_paper/env.py`

**Replace**

- `_episode_floor_crash_counts`

**With**

- `_episode_crash_signal`

**Recommended accumulation**

```python
self._episode_crash_signal += self.step_dt * floor_events.float().mean(dim=1)
```

This keeps units in seconds of average crashed-drone occupancy, rather than raw contact-step counts.

**Recommended activation update**

```python
signals = self._episode_crash_signal[env_ids][completed].detach().cpu().tolist()
self._replay_activation_history.extend(float(signal) for signal in signals)
required = int(self.cfg.replay_activation_episodes)
if len(self._replay_activation_history) >= required:
    average_signal = sum(self._replay_activation_history) / len(self._replay_activation_history)
    self._replay_active = average_signal < float(self.cfg.replay_activation_crash_signal_threshold)
```

**Config additions**

Add a threshold field in `paper_spec.py` and `env_cfg.py`:

```python
REPLAY_ACTIVATION_CRASH_SIGNAL_THRESHOLD = 1.0
```

## 3) Observation and reward clipping

**Primary files**

- `swarm-rl/environments/tasks/quad_swarm_paper/env.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/paper_spec.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/env_cfg.py`

**Observation clipping**

At the end of `_get_observations`:

```python
observations = observations.clamp(-float(self.cfg.observation_clip), float(self.cfg.observation_clip))
```

**Reward clipping**

At the end of `_get_rewards`, just before splitting per-agent rewards:

```python
reward = reward.clamp(-float(self.cfg.reward_clip), float(self.cfg.reward_clip))
```

**Suggested defaults**

```python
OBSERVATION_CLIP = 10.0
REWARD_CLIP = 10.0
```

## 4) Collision-penalty annealing

This is optional after the hover and replay fixes, but worthwhile.

**Minimal implementation**

Add a helper that linearly scales collision penalties from 0 to the configured target over `collision_penalty_anneal_steps` environment steps.

```python
def _anneal_scale(self) -> float:
    steps = max(int(self.cfg.collision_penalty_anneal_steps), 1)
    return min(float(self.common_step_counter) / float(steps), 1.0)
```

Then multiply:

```python
collision_scale = self._anneal_scale() if self.cfg.collision_penalty_anneal_steps > 0 else 1.0
```

and apply `collision_scale` to robot and obstacle collision penalties and the proximity falloff penalty.

## 5) Diagnostics

Extend `METRIC_NAMES` and `record_metric(...)` calls to include:

- `avg_altitude`
- `floor_fraction`
- `action_mean`
- `thrust_ratio_mean`
- `crash_signal`

**Implementation hint**

```python
avg_altitude = state["positions"][..., 2].mean(dim=1)
floor_fraction = floor_events.float().mean(dim=1)
action_mean = self._last_actions.mean(dim=(1, 2))
stacked_thrust_ratio = torch.stack(
    [self._thrust_targets[agent] / self._max_thrusts for agent in self._agent_ids],
    dim=1,
)
thrust_ratio_mean = stacked_thrust_ratio.mean(dim=(1, 2))
```

## 6) Leave the runner safety guard alone for now

Do not flip `share_parameters: true` under the current stock skrl IPPO runner patch. If you want paper-fidelity parameter sharing later, implement a true shared-optimizer path rather than reusing one module instance across multiple per-agent optimizers.
