# 3-Stage Curriculum for Sim2Real Multi-Agent Quadrotor Navigation

## Objective

Train a shared attention-based quadrotor policy for multi-agent goal navigation with collision avoidance and dense obstacle avoidance.

Core research hypothesis:

> Single-agent pretraining with passive hovering drones can initialize both the quadrotor controller and the neighbor-attention encoder, improving downstream MAPPO training stability and sample efficiency.

The curriculum has three stages:

1. **Single-agent passive-drone pretraining**
2. **MARL interaction learning**
3. **Target fine-tuning with dense obstacles and simple sim2real DR**

---

# Global Assumptions

## Policy / Architecture

- Use the same actor architecture across all stages.
- Actor receives:
  - self-state input,
  - goal input,
  - neighbor-drone tokens,
  - obstacle tokens,
  - optional previous action.
- Use attention over neighbor-drone tokens.
- Use attention or pooling over obstacle tokens if supported.
- In Stage 1, true obstacle tokens should be empty/masked.
- In Stages 2–3, use shared policy parameters across all learning drones.
- In Stages 2–3, use MAPPO with centralized critic and decentralized actor execution.

## Sim2Real Setting

- Target deployment is a controlled indoor lab with MoCap.
- SysID for quadrotor dynamics is assumed to already exist.
- Domain randomization should stay simple and lab-realistic.
- Prioritize:
  - MoCap-like position noise,
  - velocity-estimation noise,
  - neighbor relative-state noise,
  - thrust-scale randomization,
  - small command noise,
  - small latency/control-rate jitter.

Avoid unnecessary complexity initially:

- no strong wind randomization,
- no aggressive sensor dropout,
- no extreme mass/inertia randomization,
- no complex perception-failure model unless needed.

---

# Stage 1: Single-Agent Passive-Drone Pretraining

## Purpose

Initialize:

- basic quadrotor goal navigation,
- collision avoidance around drone-shaped objects,
- neighbor-attention encoder,
- smooth control behavior.

This stage should be single-agent from the learning perspective, but not empty: the learning drone observes passive hovering drones through the neighbor-drone attention pathway.

## Environment

| Parameter            | Setting                                  |
| -------------------- | ---------------------------------------- |
| Learning drones      | 1                                        |
| Passive drones       | 1–8 hovering drones                      |
| True obstacles       | 0                                        |
| Algorithm            | PPO or single-agent MAPPO-compatible PPO |
| Actor initialization | scratch                                  |
| Critic               | single-agent critic                      |
| Domain randomization | light                                    |

## Curriculum Knobs

| Parameter                               |        Start |              End |
| --------------------------------------- | -----------: | ---------------: |
| Passive hovering drones                 |          1–2 |              6–8 |
| Start-goal distance                     | short/medium |  full task range |
| Goal placement near drones              |         rare |           common |
| Passive-drone blockage of shortest path |         rare |       occasional |
| Spawn clearance                         |     generous | target clearance |
| True obstacles                          |            0 |                0 |

## Suggested Sub-Schedule

| Stage progress | Passive drones | Goal difficulty           | Clearance difficulty | DR    |
| -------------- | -------------: | ------------------------- | -------------------- | ----- |
| 0–30%          |            1–2 | easy/medium               | wide clearance       | light |
| 30–70%         |            3–5 | full-range goals          | moderate clearance   | light |
| 70–100%        |            6–8 | goals requiring avoidance | target clearance     | light |

## Domain Randomization

| DR term                               | Setting                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| Own position noise                    | small Gaussian                                                |
| Own velocity noise                    | small Gaussian                                                |
| Passive-drone relative-position noise | small Gaussian                                                |
| Passive-drone relative-velocity noise | small Gaussian or zero if passive drones are perfectly static |
| Yaw noise                             | small Gaussian if yaw is observed                             |
| Action/command noise                  | optional, very small                                          |
| Thrust-scale randomization            | very narrow around SysID value                                |
| Latency                               | none, or fixed 1 timestep if trivial to implement             |
| Mass/inertia randomization            | off or very narrow                                            |
| Wind/disturbance                      | off                                                           |
| Sensor/token dropout                  | off                                                           |

## Promotion Criteria

Advance to Stage 2 when:

- goal success rate is high and stable,
- passive-drone collision rate is very low,
- timeout/deadlock rate is low,
- policy is not overly conservative,
- control outputs are smooth enough for transfer.

---

# Stage 2: MARL Interaction Learning

## Purpose

Transfer the Stage 1 actor into MAPPO and train reciprocal drone-drone avoidance among learning agents.

This stage introduces true multi-agent learning and sparse-to-medium true obstacles, but the main learning focus should be drone-drone interaction.

## Environment

| Parameter             | Setting                                                          |
| --------------------- | ---------------------------------------------------------------- |
| Learning drones       | variable, mostly 2 → target-N                                    |
| Passive drones        | optional 0–4                                                     |
| True obstacles        | sparse → medium                                                  |
| Algorithm             | MAPPO                                                            |
| Actor initialization  | Stage 1 actor                                                    |
| Critic initialization | fresh centralized critic, unless compatible reuse is implemented |
| Domain randomization  | moderate                                                         |

## Curriculum Knobs

| Parameter         |           Start |                                     End |
| ----------------- | --------------: | --------------------------------------: |
| Learning drones   |      mostly 2–4 |                        mixed 4–target-N |
| True obstacles    |     none/sparse |                           sparse/medium |
| Goal interactions | simple crossing | swaps, converging paths, denser traffic |
| Spawn clearance   |        generous |                        target clearance |
| Passive drones    |    optional 0–2 |                            optional 0–4 |
| DR                |    low-moderate |                                moderate |

## Suggested Sub-Schedule

| Stage progress | Learning-drone distribution | True obstacles | Goal/interactions               | DR           |
| -------------- | --------------------------- | -------------- | ------------------------------- | ------------ |
| 0–30%          | mostly 2, some 4            | none/sparse    | simple crossing and swaps       | low-moderate |
| 30–70%         | mostly 4–8                  | sparse         | more crossing/converging goals  | moderate     |
| 70–100%        | mixed 4–target-N            | sparse/medium  | target-like multi-agent traffic | moderate     |

## Domain Randomization

| DR term                          | Setting                                     |
| -------------------------------- | ------------------------------------------- |
| Own position noise               | small-to-moderate Gaussian                  |
| Own velocity noise               | small-to-moderate Gaussian                  |
| Neighbor relative-position noise | moderate Gaussian                           |
| Neighbor relative-velocity noise | moderate Gaussian                           |
| Obstacle position noise          | small Gaussian                              |
| Thrust-scale randomization       | modest around SysID value                   |
| Command/action noise             | small                                       |
| Latency                          | fixed 1 timestep or randomized 0–1 timestep |
| Control-rate jitter              | optional small jitter                       |
| Mass randomization               | narrow                                      |
| Wind/disturbance                 | off or tiny random force                    |
| Sensor/token dropout             | optional very low neighbor-token dropout    |

## Promotion Criteria

Advance to Stage 3 when:

- 2–4 drone success rate is high,
- larger-N success rate is improving and usable,
- drone-drone collision rate is low,
- sparse/medium obstacle collision rate is low,
- deadlock rate is low,
- MAPPO updates are stable,
- policy does not collapse after increasing agent count.

---

# Stage 3: Target Fine-Tuning

## Purpose

Train the final deployment task:

- target number of drones,
- dense true obstacles,
- reciprocal collision avoidance,
- simple lab-realistic sim2real robustness.

This is the main final training stage.

## Environment

| Parameter             | Setting                                              |
| --------------------- | ---------------------------------------------------- |
| Learning drones       | mostly target-N, with some smaller-N episodes        |
| Passive drones        | optional 0–2                                         |
| True obstacles        | medium → dense                                       |
| Algorithm             | MAPPO                                                |
| Actor initialization  | Stage 2 actor                                        |
| Critic initialization | Stage 2 critic                                       |
| Domain randomization  | strongest simple lab DR                              |
| Replay                | collision/near-collision/deadlock replay recommended |

## Curriculum Knobs

| Parameter             |                    Start |                                                  End |
| --------------------- | -----------------------: | ---------------------------------------------------: |
| Learning drones       | mixed medium-to-target N |                                      mostly target-N |
| True obstacle density |                   medium |                                         target dense |
| Goal placement        | random/moderate crossing | target distribution, bottlenecks, dense interactions |
| Replay usage          |                      low |                                    adaptive moderate |
| DR                    |                 moderate |                              strongest simple lab DR |

## Suggested Sub-Schedule

| Stage progress | Learning-drone distribution    | Obstacle density | Replay usage              | DR                  |
| -------------- | ------------------------------ | ---------------- | ------------------------- | ------------------- |
| 0–30%          | mixed 4–target-N               | medium           | low                       | moderate            |
| 30–70%         | mostly target-N, some medium-N | medium → dense   | moderate                  | moderate-strong     |
| 70–100%        | mostly target-N                | target dense     | adaptive hard-case replay | strongest simple DR |

## Domain Randomization

| DR term                          | Setting                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| Own position noise               | realistic MoCap noise range                                             |
| Own velocity noise               | realistic estimator noise range                                         |
| Neighbor relative-position noise | realistic MoCap/reconstruction noise range                              |
| Neighbor relative-velocity noise | realistic estimator noise range                                         |
| Obstacle position noise          | small map/calibration error                                             |
| Thrust-scale randomization       | moderate around SysID value                                             |
| Motor lag / response delay       | modest if implemented                                                   |
| Command/action noise             | small-to-moderate                                                       |
| Latency                          | randomized 0–1 timestep, or measured lab latency range                  |
| Control-rate jitter              | small if easy                                                           |
| Mass randomization               | narrow-to-moderate                                                      |
| Wind/disturbance                 | off or tiny                                                             |
| Sensor/token dropout             | very low neighbor/obstacle token dropout only if expected in deployment |

## Replay Buffer

Use a hard-case replay buffer for:

- drone-drone near collisions,
- drone-obstacle near collisions,
- actual collisions,
- deadlocks,
- timeouts,
- bottleneck failures.

Replay should not dominate training. Use it as a minority fraction of Stage 3 episodes or resets.

## Final Evaluation Criteria

Evaluate on held-out seeds with curriculum disabled.

| Metric                       | Desired outcome                                          |
| ---------------------------- | -------------------------------------------------------- |
| Goal success rate            | high on target-N and target obstacle density             |
| Drone-drone collision rate   | very low                                                 |
| Obstacle collision rate      | very low                                                 |
| Timeout/deadlock rate        | low                                                      |
| Minimum inter-agent distance | respects safety margin                                   |
| Path efficiency              | not overly conservative                                  |
| Control smoothness           | deployable on real quadrotors                            |
| Robustness                   | stable under held-out noise/latency/thrust randomization |
| Sim2Real readiness           | passes progressive real-world validation ladder          |

---

# Stage Transition Mechanics

## Stage 1 → Stage 2

- Initialize MAPPO actor from Stage 1 actor.
- Start a fresh centralized critic unless compatible critic transfer is already implemented.
- Consider resetting optimizer state.
- Temporarily reduce actor learning rate.
- Optionally increase entropy bonus briefly to avoid overly conservative behavior.
- Begin with mostly 2-agent episodes.

## Stage 2 → Stage 3

- Continue actor and critic from Stage 2.
- Lower actor learning rate at the transition if instability appears.
- Increase obstacle density gradually.
- Increase target-N sampling gradually.
- Enable hard-case replay.
- Increase DR to final lab-realistic settings.

---

# Minimal Implementation Checklist

## Required

- [ ] Stage 1 environment with 1 learning drone and passive hovering drones.
- [ ] Stage 1 has 0 true obstacles.
- [ ] Passive drone count increases from 1–2 to 6–8.
- [ ] Passive drones are encoded through the neighbor-drone attention stream.
- [ ] Same actor architecture is used across all stages.
- [ ] Stage 2 initializes actor from Stage 1.
- [ ] Stage 2 uses MAPPO with shared actor and centralized critic.
- [ ] Stage 2 increases learning-agent count over time.
- [ ] Stage 2 introduces sparse-to-medium true obstacles.
- [ ] Stage 3 increases obstacle density to target level.
- [ ] Stage 3 mostly trains target-N episodes.
- [ ] DR increases progressively: light → moderate → strongest simple lab DR.
- [ ] Final evaluation uses fixed held-out seeds with curriculum disabled.

## Recommended

- [ ] Reset optimizer state at Stage 1 → Stage 2.
- [ ] Use hard-case replay in Stage 3.
- [ ] Keep a small fraction of easier episodes during Stages 2–3.
- [ ] Track separate collision rates for drone-drone and drone-obstacle collisions.
- [ ] Track timeout/deadlock rate.
- [ ] Track path efficiency and control smoothness.
- [ ] Run ablations against MAPPO-from-scratch and no-passive-drone pretraining.

## Not Needed Initially

- [ ] Strong wind randomization.
- [ ] Complex perception dropout.
- [ ] Extreme mass/inertia randomization.
- [ ] Many named curriculum stages.
- [ ] Full outdoor sim2real robustness.

---

# Suggested Ablations

| Ablation                                          | Purpose                                            |
| ------------------------------------------------- | -------------------------------------------------- |
| MAPPO from scratch on Stage 3 task                | Tests whether curriculum helps at all              |
| Stage 1 without passive drones                    | Tests whether single-agent pretraining alone helps |
| Stage 1 with passive drones                       | Tests core proposed method                         |
| Stage 1 with passive drones but no actor transfer | Tests whether benefit comes from initialization    |
| Stage 3 without hard-case replay                  | Tests replay contribution                          |
| Stage 3 without DR                                | Tests sim2real robustness contribution             |

---

# Core Research Question

Does single-agent pretraining with passive hovering drones initialize a useful quadrotor controller and neighbor-attention encoder, improving MAPPO training stability, sample efficiency, and sim2real transfer for multi-agent quadrotor navigation?
