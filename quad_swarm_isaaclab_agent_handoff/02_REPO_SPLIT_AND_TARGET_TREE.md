# Repository split and target tree

This project must be split so that **generic reusable functionality goes to `cpsquare-lab`** and the **paper-specific task stays thin in the main workspace**.

## Split rule

### Put this in `cpsquare-lab`
Anything that is reusable across more than one multirotor/swarm task:
- drone assets and configs
- generic actuator and thrust helpers
- self-state observation helpers
- goal-relative observation helpers
- K-nearest-neighbor observation helpers
- local obstacle SDF utilities
- robot-robot and robot-obstacle collision bookkeeping
- reward primitives
- collision replay state storage and replay sampling utilities
- generic tests for the above

### Keep this in the main workspace
Anything that is specific to this paper/task:
- task registration
- exact paper constants and environment presets
- the task composition layer that wires together reusable cpsquare-lab pieces
- paper-specific obstacle-room sampler if it is not broadly reusable
- paper-specific attention encoder and skrl wrappers
- train/eval scripts for this task
- parity report and task-local tests

## Recommended target tree

Use this as the default target tree. Rename the paper package if the workspace already has a preferred package name.

```text
cpsquare-lab/
  source/cpsquare_lab/
    assets/
      multirotors/
        crazyflie.py
        multirotor_common.py
    control/
      action_mapping.py
      motor_commands.py
    mdp/
      observations/
        multirotor_self.py
        goal_relative.py
        swarm_neighbors.py
        local_sdf.py
      rewards/
        navigation.py
        swarm_collisions.py
        control_penalties.py
      events/
        collision_events.py
        contact_events.py
      terminations/
        multirotor_terminations.py
    managers/
      collision_replay.py
    utils/
      knn.py
      obstacle_grid.py
      tensor_ops.py
  tests/
    test_swarm_neighbors.py
    test_local_sdf.py
    test_collision_replay.py
    test_action_mapping.py

main-workspace/
  source/quad_swarm_paper/
    tasks/
      quad_swarm_obstacles/
        __init__.py
        registration.py
        paper_spec.py
        env_cfg.py
        env.py
        obstacle_room.py
        metrics.py
    models/
      attention.py
      quad_swarm_encoder.py
      quad_swarm_skrl_models.py
    utils/
      parity_report.py
  scripts/
    reinforcement_learning/
      skrl/
        train_quad_swarm_ippo.py
        eval_quad_swarm.py
  tests/
    test_quad_swarm_env_shapes.py
    test_quad_swarm_model_forward.py
    test_quad_swarm_reward_terms.py
    test_quad_swarm_smoke.py
  docs/
    quad_swarm_port_notes.md
```

## File ownership guidance

### `cpsquare-lab/source/cpsquare_lab/...`
Should contain implementation details that are independent of this paper name.

Examples:
- `swarm_neighbors.py` should know how to compute nearest-neighbor relative position/velocity blocks for any multirotor swarm task.
- `local_sdf.py` should know how to generate a local 3 x 3 SDF-like obstacle observation around each robot for any task that uses a 2D obstacle field.
- `collision_replay.py` should know how to snapshot state, sample replay starts, and retire replay entries according to configurable thresholds.

### `main-workspace/source/quad_swarm_paper/...`
Should mostly be thin composition code.

Examples:
- `paper_spec.py` stores constants such as `K=2`, obstacle density defaults, episode duration, hidden size, and training defaults.
- `env.py` uses the reusable cpsquare-lab functions to build the final observation, reward, and reset pipeline.
- `quad_swarm_encoder.py` owns the paper-specific encoder and nothing generic.

## Anti-patterns to avoid

Do not leave these in the paper task if they can be generic:
- reusable KNN code
- reusable SDF generation code
- reusable collision bookkeeping
- reusable action normalization logic
- reusable reward primitives like control effort or tilt penalties

Do not move these into `cpsquare-lab` unless they clearly generalize:
- exact paper constants
- paper-specific room layout presets
- paper-specific training hyperparameter presets
- paper-specific attention encoder naming
