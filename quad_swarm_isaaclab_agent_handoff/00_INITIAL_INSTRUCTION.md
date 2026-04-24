# Initial instruction for the implementation agent

You are implementing a paper-faithful Isaac Lab 3.0 port of the ICRA 2024 paper **"Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning"** on top of **cpsquare-lab**, which is the Isaac Lab port of Omnidrones.

## Mission

Recreate the environment, observation pipeline, reward structure, collision replay logic, and model architecture from the paper and released code, while using Isaac Lab 3.0 beta on the `develop` branch for simulation and skrl for RL training.

## Hard constraints

1. **Push reusable and generic functionality into `cpsquare-lab`.**
   - Generic drone assets/configs
   - Generic actuation / action-to-thrust mapping
   - Generic multirotor state observations
   - Generic swarm neighbor observations
   - Generic local obstacle SDF observations
   - Generic collision/event tracking
   - Generic reward primitives
   - Generic replay-buffer / reset-manager utilities

2. **Keep the paper-specific package in the main workspace thin.**
   The main workspace should mostly contain:
   - task registration
   - the thin `DirectMARLEnv` composition layer
   - paper constants/configs
   - the paper-specific attention encoder and skrl model wrappers
   - train/eval scripts
   - tests and documentation for this paper task

3. **Do not rewrite PPO.**
   Use Isaac Lab + skrl IPPO unless there is a proven blocker. Treat any need for a custom trainer as a narrow extension around skrl usage, not a full RL reimplementation.

4. **Do not modify Isaac Lab core unless there is no extension point.**
   Exhaust environment wrappers, custom models, and custom training scripts first.

5. **Start from the main simulation architecture, not the compressed sim-to-real model.**
   First target:
   - hidden size 256
   - multi-head attention block
   - separate actor and critic networks
   - no RNN

6. **When paper and repo disagree, prefer the released code path, but document the discrepancy.**
   Important example: the paper text describes thrusts in `[0, 1]^4`, while the release uses a raw-control path with `zero_action_middle=True`.

7. **Make shared-policy behavior an explicit checkpoint.**
   The paper goal is a homogeneous decentralized controller reused across all drones. Do not silently accept per-agent parameter duplication without evaluating whether skrl can share module instances cleanly.

## Deliverables

Produce the following as working code and docs:
- a paper-faithful Isaac Lab task that runs under `DirectMARLEnv`
- a custom skrl IPPO training script
- custom actor and critic model classes with the paper encoder
- cpsquare-lab reusable utilities extracted from the task
- smoke tests, unit tests, and a short parity report documenting any remaining deviations from the released implementation

## Success criteria

The port is successful when:
- the task runs end-to-end in Isaac Lab 3.0 on the `develop` branch
- the environment observations, rewards, and replay behavior match the paper/repo design
- the custom attention encoder is used in both actor and critic
- the main workspace environment stays thin because reusable pieces were moved into `cpsquare-lab`
- all unresolved divergences are listed explicitly

Read the rest of this handoff pack before writing code.
