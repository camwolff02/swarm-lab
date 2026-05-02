# Likely repo-relative file map

This map reconstructs the likely tree from the uploaded flat files and import statements.

## swarm-rl

- `swarm-rl/environments/tasks/quad_swarm_paper/__init__.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/env.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/env_cfg.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/paper_spec.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/obstacle_room.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/models/attention.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/models/quad_swarm_encoder.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/models/quad_swarm_skrl_models.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/agents/runner.py`
- `swarm-rl/environments/tasks/quad_swarm_paper/agents/skrl_ippo_cfg.yaml`

## cpsquare-lab

- `cpsquare-lab/cpsquare_lab/embodiments/common/events.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/common/action_mapping.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/common/actions.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/common/events.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/common/multirotor.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/common/params.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/cf2x/sim/robot.py`
- `cpsquare-lab/cpsquare_lab/embodiments/multirotor/cf2x/sim/cf2x.yaml`
- `cpsquare-lab/cpsquare_lab/controllers/base.py`
- `cpsquare-lab/cpsquare_lab/controllers/lee_position.py`
- `cpsquare-lab/cpsquare_lab/controllers/pid_rate.py`
- `cpsquare-lab/cpsquare_lab/tasks/swarm/collision_replay.py`
- `cpsquare-lab/cpsquare_lab/tasks/swarm/events.py`
- `cpsquare-lab/cpsquare_lab/tasks/swarm/grid_sdf.py`
- `cpsquare-lab/cpsquare_lab/tasks/swarm/knn.py`
- `cpsquare-lab/cpsquare_lab/tasks/swarm/observations.py`
- `cpsquare-lab/cpsquare_lab/tasks/swarm/rewards.py`

If your actual paths differ, keep the task ownership the same: environment/model/runner changes in `swarm-rl`, reusable helpers and vehicle parameters in `cpsquare-lab`.
