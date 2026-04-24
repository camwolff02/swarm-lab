# Validation and acceptance criteria

## Validation sequence

Run validation in this order so failures are easy to localize.

### 1) Static/unit tests
- observation helper tests in `cpsquare-lab`
- action mapping tests in `cpsquare-lab`
- local SDF tests in `cpsquare-lab`
- collision replay tests in `cpsquare-lab`
- environment shape tests in the main workspace
- model forward-pass tests in the main workspace

### 2) Environment smoke tests
- 2 drones, no obstacles, random actions
- 8 drones, no obstacles, random actions
- 8 drones, obstacles on, random actions
- 8 drones, obstacles on, replay enabled but training off

### 3) Short training sanity checks
- no obstacles, short run: verify policy improves over pure random baseline
- obstacles on, replay off: verify end-to-end train loop stability
- obstacles on, replay on: verify replay resets actually happen and do not crash the env

### 4) Paper-feature ablations
- no attention
- no replay
- no obstacle SDF

The goal is not exact score reproduction on day one, but confirming that each major paper component is implemented and has the expected directional effect.

## Acceptance criteria

The work is accepted only if **all** of the following are true:

1. **Thin task boundary**
   - the main workspace paper task mostly wires together reusable `cpsquare-lab` pieces
   - reusable observations/rewards/assets are not duplicated in the task package

2. **Environment correctness**
   - `DirectMARLEnv` task launches, resets, and steps correctly
   - agent observation and action spaces are deterministic and documented
   - observation ordering matches the model slicing logic exactly

3. **Model correctness**
   - the custom paper encoder is used in both actor and critic
   - the attention block matches the release semantics closely enough to justify parity claims
   - model unit tests pass

4. **Training correctness**
   - the custom skrl IPPO script runs end-to-end
   - checkpoints and logs are produced
   - replay and any annealing hooks can be enabled from config

5. **Parity reporting**
   - any divergence from the paper or release is documented in a short parity report
   - unresolved items are clearly labeled as confirmed / probable / unknown

6. **Shared-policy checkpoint**
   - the implementation either uses shared modules across drones or explicitly documents the blocker and the chosen fallback

## Required parity report contents

The parity report should include a table with at least these rows:
- simulation backend difference: Isaac Lab/skrl vs Omnidrones/Sample Factory
- action convention difference, if any
- attention implementation status
- reward coefficient source of truth
- replay implementation status
- shared-policy status
- downwash status
- any remaining unsupported repo feature

## Definition of done

The port is done when someone else can:
- install the workspace
- launch the task
- train with the provided skrl script
- read one short note explaining what matches the paper and what still differs
