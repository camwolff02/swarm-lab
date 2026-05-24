# Cameron Swarm

Cameron cooperative drone waypoint task for Isaac Lab 3.0.0 beta 2.

This package provides:

- the concrete Cameron drone waypoint task configuration and registration.
- fixed `possible_agents` with an active-agent mask curriculum.
- CTBR multirotor actions and target-pose commands.
- IPPO and MAPPO task registrations:
  - `Isaac-Cameron-Drone-Waypoint-IPPO-v0`
  - `Isaac-Cameron-Drone-Waypoint-MAPPO-v0`

The reusable manager-based multi-agent runtime lives in
`environments.envs`.

## Layout

```text
cameron_swarm/
  drone_waypoint_marl_env_cfg.py # IPPO/MAPPO task configs
  mdp/                           # task observations, rewards, resets, curricula
  config/                        # SKRL smoke-training configs
  register.py                    # Gymnasium task registration
  tests/                         # compiler and simulator smoke tests
```
