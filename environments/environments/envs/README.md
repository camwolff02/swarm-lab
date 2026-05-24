<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Manager-Based Multi-Agent Environments

This package adapts IsaacLab manager terms to the multi-agent ABI provided by
`DirectMARLEnv`. The runtime exposes PettingZoo/SKRL-style dictionaries for
observations, actions, rewards, terminations, truncations, infos, and optional
critic observations while still letting task authors write ordinary IsaacLab
manager configs.

## Authoring Model

The public configuration surface is intentionally small.

- `possible_agents` is the fixed public agent universe. Keep it stable for
  PettingZoo and SKRL compatibility, even when a curriculum masks inactive
  agents.
- `AgentCfg` describes one concrete agent's observation, action, and optional
  command managers.
- `AgentGroupCfg` declares multiple agents with the same `AgentCfg`. It expands
  into concrete agent configs before managers are built.
- `ManagerBasedMaEnvCfg` owns the classical control surface: agents, groups,
  observations, actions, commands, masks, and optional recorders.
- `AgentRlCfg` extends `AgentCfg` with rewards, terminations, and curriculum.
- `ManagerBasedMarlEnvCfg` owns the RL surface: `AgentRlCfg`, reset aggregation,
  and episode horizon.
- A `critic` observation group is the training state channel. It can be local
  and decentralized for IPPO, or centralized/global for MAPPO.

## Compile Flow

At environment construction time, `compile_multi_agent_spec()` expands explicit
agents and `AgentGroupCfg` declarations into concrete runtime metadata.

1. Explicit `agents` are copied by id.
2. Each `AgentGroupCfg` expands either `agent_ids` or `count` plus
   `id_template`.
3. String templates inside each copied config are materialized with
   `agent_id`, `group_name`, `name`, `entity_name`, and `i`.
4. The compiler validates that configured agents exactly match
   `possible_agents`.
5. The runtime builds one manager bundle per concrete agent and derives
   Gymnasium observation/action spaces from the managers.

`AgentGroupCfg` does not change the public runtime API. After compilation, every
public method still uses concrete agent ids:

- `possible_agents`, `agents`
- `observation_spaces`, `action_spaces`, `state_space`, `state_spaces`
- `observation_space(agent)`, `action_space(agent)`
- `reset() -> (obs_dict, info_dict)`
- `step(action_dict) -> (obs, rewards, terminated, truncated, infos)`
- `state()` returns the per-agent `critic` observation group when configured.

## Manager Bundles

The runtime builds one manager bundle per concrete agent. This is the most
predictable shape for IsaacLab manager terms because action, observation,
reward, termination, command, and curriculum terms can be written exactly like
single-agent manager terms. `AgentGroupCfg` only affects declaration and
metadata; it does not require group-aware tensor handling at runtime.

## Minimal RL Config

```python
from environments.envs import AgentGroupCfg, AgentRlCfg, ManagerBasedMarlEnvCfg


class ObservationsCfg:
    policy = PolicyObsCfg()
    critic = DecentralizedCriticObsCfg()


class MappoObservationsCfg(ObservationsCfg):
    critic = CentralizedCriticObsCfg()


class SwarmEnvCfg(ManagerBasedMarlEnvCfg):
    possible_agents = [f"drone_{i}" for i in range(4)]

    agent_groups = [
        AgentGroupCfg(
            name="drone",
            count=4,
            id_template="drone_{i}",
            agent_cfg=AgentRlCfg(
                asset_name="{agent_id}",
                observations=ObservationsCfg(),
                actions=ActionsCfg(),
                commands=CommandsCfg(),
                rewards=RewardsCfg(),
                terminations=TerminationsCfg(),
                curriculum=CurriculumCfg(),
            ),
        )
    ]

    observation_group = "policy"
    reset_on = "any"
```

For active-agent curricula, keep `possible_agents` fixed and expose an
environment mask through `active_agent_mask_key`. The runtime zeros inactive
agent actions while preserving the dictionary keys expected by multi-agent
trainers.
