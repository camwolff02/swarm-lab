# Open questions and risks

## 1) Shared policy/value across named agents

### Why this matters
The paper intent is a homogeneous decentralized controller. Some multi-agent APIs are organized around per-agent model dictionaries, which can accidentally lead to duplicated parameters per named drone.

### Required action
Treat this as an early checkpoint:
- verify whether skrl accepts shared module instances for all agent IDs
- if not, build the smallest possible shared-parameter adapter
- do not claim paper parity until this is resolved or explicitly documented

## 2) Sample Factory async IPPO vs skrl execution model

### Why this matters
The original implementation is trained in the Sample Factory ecosystem. Isaac Lab + skrl may differ in rollout collection and learner scheduling.

### Mitigation
- first reproduce the environment/model/replay design in skrl IPPO
- document trainer-level differences in the parity report
- only escalate to a more custom trainer if the current path blocks shared-policy fidelity or training stability

## 3) Attention fidelity risk

### Why this matters
Replacing the repo attention module with a superficially similar stock attention layer can change the architecture.

### Mitigation
- port the local attention block closely
- write focused encoder tests
- keep the main simulation model separate from the compressed sim-to-real model

## 4) Action convention mismatch

### Why this matters
The paper describes thrust commands in `[0, 1]^4`, while the release uses a raw-control path centered at zero.

### Mitigation
- choose one explicit interface
- keep a single conversion point in reusable control code
- document the choice and rationale in the parity report

## 5) Physics parity risk

### Why this matters
The released environment sits on Omnidrones/Isaac Sim assumptions that may not transfer one-to-one to your Isaac Lab port.

### Mitigation
- isolate dynamics/asset/control code in `cpsquare-lab`
- first validate without obstacles and without replay
- then enable downwash and collision replay
- do not tune reward/model hyperparameters until basic dynamics sanity is confirmed

## 6) Obstacle-room parity risk

### Why this matters
Subtle differences in obstacle placement, cell-center definitions, radius interpretation, or spawn logic can affect results a lot.

### Mitigation
- preserve the paper/repo room defaults exactly where practical
- keep obstacle-room sampling code centralized in one paper-specific module
- add deterministic seed-based tests for obstacle map generation

## 7) Reward coefficient drift

### Why this matters
The reward structure has multiple terms. Changing coefficients or term semantics can invalidate comparisons.

### Mitigation
- copy coefficients from the released baseline into `paper_spec.py`
- do not invent values from memory
- reference the source file in comments or docs

## 8) Replay-state serialization complexity

### Why this matters
A replay reset requires enough simulator state to resume a pre-collision scenario correctly.

### Mitigation
- define a minimal but complete replay snapshot schema early
- include robot poses, velocities, angular rates, motor/actuator state if required, obstacle layout, goals, and any episode bookkeeping needed to resume safely
- unit-test replay restore before using it in training
