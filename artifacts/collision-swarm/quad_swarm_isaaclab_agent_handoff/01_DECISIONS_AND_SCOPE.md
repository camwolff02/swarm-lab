# Decisions and scope

## Core architectural decisions

### 1) Environment base class
Use **`DirectMARLEnv`** for the task.

Reason:
- Isaac Lab's direct multi-agent workflow is the right fit for a paper-faithful swarm environment.
- skrl is the Isaac Lab backend with true multi-agent training support for IPPO/MAPPO.
- This task does not need the manager-based workflow as the first implementation target.

### 2) RL algorithm path
Use **skrl IPPO** as the first implementation path.

Reason:
- The paper uses an independent PPO style setup rather than a centralized MAPPO formulation.
- The custom part of the paper is in the **observation encoder and replay/reset logic**, not in PPO itself.
- Replacing the whole trainer should be avoided until there is concrete evidence that the stock IPPO data flow cannot support the required parameter sharing or logging.

### 3) Model integration approach
Use **custom skrl model classes in Python**, not the stock YAML/model-instantiator path.

Reason:
- The encoder is not a plain MLP. It slices observations into self, neighbor, and obstacle blocks and applies a custom attention block.
- The actor and critic should be explicit Python modules with clear forward paths and test coverage.

### 4) Model fidelity target
Port the released architecture **as implemented**, not a loose approximation.

Concretely:
- three 2-layer MLP encoders: self / neighbors / obstacles
- a custom attention block applied over two tokens: neighbor embedding and obstacle embedding
- concatenate self embedding with the attended outputs
- separate actor and critic heads on top
- separate actor and critic networks for the first faithful pass

Do **not** replace the repository attention module with a stock `torch.nn.MultiheadAttention` unless you first prove the projection dimensions and residual/LayerNorm behavior are identical.

### 5) Critic observation design
First pass: keep the critic local and set `state_space = 0`.

Reason:
- This is closer to the paper's independent PPO setup.
- A centralized critic is not required to reproduce the published architecture.
- If later experiments need MAPPO-style centralization, that can be added as a follow-up.

### 6) Action-space convention
Recommended convention:
- expose **`Box(-1, 1, (4,))`** to the RL policy
- map it once, inside reusable cpsquare-lab control code, to physical rotor thrust commands

Reason:
- The paper text describes thrust levels in `[0, 1]^4`, but the release uses `raw_control_zero_middle=True`.
- A single conversion point in reusable control code is cleaner than mixing conventions in the task and trainer.

Document the chosen convention in the task docs and parity report.

## Scope boundaries

### In scope
- direct Isaac Lab 3.0 environment port
- paper observation structure
- reward and termination logic
- replay-reset curriculum
- custom attention encoder
- skrl IPPO train/eval pipeline
- reusable library extraction into `cpsquare-lab`
- parity-oriented tests and ablations

### Out of scope for first pass
- exact Sample Factory async learner/collector parity
- on-board deployment constraints from the compressed sim-to-real model
- hardware-specific latency optimization
- full reproduction of every plotting or logging utility from the original repo

## Shared-weights checkpoint

Treat the following as a mandatory design checkpoint early in implementation:

**Goal:** one homogeneous decentralized policy (and one value function) reused across all drones.

Possible outcomes:
1. **Best case:** skrl accepts shared module instances for all agents.
2. **Acceptable fallback:** a thin custom training layer pools all agents' samples and updates one shared actor/value pair.
3. **Not acceptable without explicit sign-off:** silently training separate parameters per named drone and claiming paper parity.

If the stock multi-agent path forces separate per-agent modules, stop and document the exact blocker before proceeding.
