---
name: env-info
description: Show paper_swarm environment configuration, registered tasks, and stage details. Use when user asks about env config, tasks, stages, registered Gym IDs, or wants to verify a config loads correctly.
---

# Env Info

Inspect paper_swarm environment configurations, registered tasks, and stage details.

## List all registered tasks

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 10 uv run python -c "
from environments import tasks as local_tasks
import gymnasium as gym
for task_id in sorted(gym.registry.keys()):
    if 'Paper-Swarm' in task_id:
        print(f'  {task_id}')
" 2>&1
```

## Show stage config details

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 30 uv run python -c "
from environments import tasks as local_tasks
from isaaclab_tasks.utils import resolve_task_config

for stage, entry in [('Stage1','skrl_mappo_cfg_entry_point'),('Stage2','skrl_mappo_cfg_entry_point'),('Stage3','skrl_mappo_cfg_entry_point')]:
    local_tasks.register_tasks_for(f'Isaac-Paper-Swarm-Waypoint-MAPPO-{stage}-v0')
    ec, tc = resolve_task_config(f'Isaac-Paper-Swarm-Waypoint-MAPPO-{stage}-v0', entry)
    g = ec.agent_groups[0]
    print(f'\n=== {stage} ===')
    print(f'  possible_agents: {ec.possible_agents}')
    print(f'  num_envs: {ec.scene.num_envs}')
    print(f'  episode_length_s: {ec.episode_length_s}')
    print(f'  agent count: {g.count}')
    t = g.agent_cfg.terminations
    print(f'  terminations: {[k for k in vars(t) if not k.startswith(\"_\")]}')
    r = g.agent_cfg.rewards
    print(f'  rewards: {[k for k in vars(r) if not k.startswith(\"_\")]}')
    c = g.agent_cfg.curriculum
    if c and hasattr(c, 'active_agent_count'):
        print(f'  active_agents: min={c.active_agent_count.params.get(\"min_agents\")}, max={c.active_agent_count.params.get(\"max_agents\")}')
    if c and hasattr(c, 'paper_swarm_task'):
        p = c.paper_swarm_task.params
        print(f'  columns: {p.get(\"max_static_columns\")}, obs_start: {p.get(\"obstacle_start_step\")}')
    print(f'  train: rollouts={tc[\"agent\"][\"rollouts\"]}, mini_batches={tc[\"agent\"][\"mini_batches\"]}, lr={tc[\"agent\"][\"learning_rate\"]}, timesteps={tc[\"trainer\"][\"timesteps\"]}')
" 2>&1
```

## Verify a specific config loads

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 30 uv run python -c "
from environments import tasks as local_tasks
from isaaclab_tasks.utils import resolve_task_config
local_tasks.register_tasks_for('TASK_ID_HERE')
ec, tc = resolve_task_config('TASK_ID_HERE', 'skrl_mappo_cfg_entry_point')
print('Config loaded OK')
print(f'  num_envs: {ec.scene.num_envs}')
print(f'  possible_agents: {ec.possible_agents}')
" 2>&1
```

## Show observation dimensions

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 10 uv run python -c "
from environments.tasks.paper_swarm.models import PaperAttentionEncoderCfg
c = PaperAttentionEncoderCfg()
print(f'self_obs_dim: {c.self_obs_dim}')
print(f'other_obs_dim: {c.other_obs_dim}')
print(f'max_neighbors: {c.max_neighbors}')
print(f'static_sdf_dim: {c.static_sdf_dim}')
print(f'goal_obs_dim: {c.goal_obs_dim}')
print(f'observation_dim: {c.observation_dim}')
print(f'attention_dim: {c.attention_dim}')
print(f'hidden_units: {c.hidden_units}')
" 2>&1
```

## Current training process info

```bash
ps aux | grep train.py | grep -v grep | head -5
```
