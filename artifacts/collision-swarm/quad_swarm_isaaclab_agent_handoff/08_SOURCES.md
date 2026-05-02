# Sources and reference links

These are the primary sources the implementation agent should use as the source of truth.

## Paper and project pages
- Paper project page: https://sites.google.com/view/obst-avoid-swarm-rl
- Paper on arXiv: https://arxiv.org/abs/2309.13285
- arXiv PDF: https://arxiv.org/pdf/2309.13285
- Released source code: https://github.com/Zhehui-Huang/quad-swarm-rl

## Isaac Lab docs and code
- Isaac Lab docs index (develop): https://isaac-sim.github.io/IsaacLab/develop/index.html
- Environment overview: https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html
- `DirectMARLEnv` API: https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.html
- skrl wrapper API: https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_rl/isaaclab_rl.html
- Isaac Lab skrl training script (develop branch): https://github.com/isaac-sim/IsaacLab/blob/main/scripts/reinforcement_learning/skrl/train.py

## skrl docs
- skrl getting started: https://skrl.readthedocs.io/en/latest/intro/getting_started.html
- IPPO docs: https://skrl.readthedocs.io/en/latest/api/multi_agents/ippo.html
- Gaussian model mixin: https://skrl.readthedocs.io/en/latest/api/models/gaussian.html
- Deterministic model mixin: https://skrl.readthedocs.io/en/latest/api/models/deterministic.html

## Original repo files to mirror closely

### Model files
- Attention block: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/models/attention_layer.py
- Main multi-agent model: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/models/quad_multi_model.py

### Training/config files
- Obstacle baseline config: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/runs/obstacles/quad_obstacle_baseline.py
- Obstacle environment config entrypoint: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/runs/obstacles/quads_multi_obstacles.py
- Training entrypoint: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/train.py

### Environment and utility files
- Quadrotor params: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/env_wrappers/quadrotor_params.py
- Reward shaping wrapper: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/env_wrappers/reward_shaping.py
- Raw-control wrapper and replay hooks: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/env_wrappers/quad_utils.py
- Core quadrotor multi-agent env: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/gym_art/quadrotor_multi/quadrotor_multi.py
- Obstacle observation helper: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/gym_art/quadrotor_multi/obstacles/obstacles.py
- Obstacle utility functions: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/gym_art/quadrotor_multi/obstacles/utils.py
- Drone-drone collision helper: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/gym_art/quadrotor_multi/collisions/quadrotors.py

## Community context worth checking before finalizing the trainer path

These are not primary sources, but they are useful context when deciding whether stock Isaac Lab/skrl scripts cleanly support shared homogeneous policies across named agents:
- Isaac Lab issue about custom critic observations with skrl runner: https://github.com/isaac-sim/IsaacLab/issues/4454
- skrl discussion about sharing one decentralized model in Isaac Lab multi-agent setups: https://github.com/Toni-SM/skrl/discussions/1036

Use the paper, the release, Isaac Lab docs, and skrl docs as the primary truth. Use community threads only for implementation context.
