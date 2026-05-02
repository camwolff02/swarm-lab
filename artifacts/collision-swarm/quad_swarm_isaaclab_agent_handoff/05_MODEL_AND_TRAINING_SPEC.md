# Model and training specification

## Model fidelity target

Reproduce the released model structure, not a generic attention policy.

## Encoder structure

Create a reusable paper encoder with this exact high-level shape:

```text
flat observation
  -> slice into self / neighbor / obstacle blocks
  -> self MLP encoder (2 layers)
  -> neighbor MLP encoder (2 layers)
  -> obstacle MLP encoder (2 layers)
  -> create 2-token sequence: [neighbor_embed, obstacle_embed]
  -> custom multi-head attention block over those 2 tokens
  -> flatten attended tokens
  -> concatenate with self embedding
  -> latent vector
```

### Important fidelity note
The released code does **not** treat all neighbors as separate attention tokens. Instead, the fixed-width neighbor block is first encoded to one embedding, and the obstacle block is encoded to one embedding; attention is then applied over those **two embeddings**.

Do not silently change this into:
- attention over every neighbor as a token
- attention over every obstacle cell as a token
- a stock Transformer encoder over the entire observation

Those are different architectures.

## Attention implementation

Port the repository attention semantics carefully.

Requirements:
- preserve projection semantics
- preserve residual connection
- preserve LayerNorm behavior
- keep the number of heads consistent with the released main model

Preferred implementation:
- add a small local module in `models/attention.py` that mirrors the release behavior closely
- write a focused unit test comparing tensor shapes and residual behavior

## Actor and critic structure

### First-pass fidelity target
Use **separate** actor and critic networks with the same encoder structure.

Reason:
- the released baseline sets separate actor/critic weights
- this avoids ambiguity in the parity pass

### skrl wrapper classes
Implement explicit skrl classes:
- policy: `GaussianMixin + Model`
- value: `DeterministicMixin + Model`

Each class should:
- consume `inputs["observations"]`
- call the paper encoder
- output either action distribution parameters or value scalars

## Parameter-sharing plan

The physical drones are homogeneous. The intended control law is a shared decentralized policy.

### Preferred outcome
All agent IDs use the same policy module and the same value module.

### Implementation checkpoint
Before wiring the full trainer, verify one of these approaches:
1. skrl accepts a model dictionary where each agent ID references the same module instances
2. skrl can be wrapped so all agent experiences are pooled into a single shared update

If neither works cleanly, implement a thin custom adapter rather than accepting hidden per-agent duplication.

## Training script design

Create a dedicated Python training script, for example:
- `scripts/reinforcement_learning/skrl/train_quad_swarm_ippo.py`

The script should:
1. create the Isaac Lab environment
2. wrap it with `SkrlVecEnvWrapper`
3. build the custom actor and critic modules
4. instantiate IPPO directly in Python
5. configure memory, normalization, logging, checkpoints, and evaluation hooks explicitly

Do not rely on the stock generic runner config if it prevents explicit model wiring.

## Starting hyperparameters

Use the release baseline as the first parity target, then tune only if Isaac Lab dynamics require it.

Initial defaults to encode in `paper_spec.py` or a local config module:
- learning rate: `1e-4`
- rollout length: `128`
- batch size: `1024`
- hidden size: `256`
- actor/critic shared weights: `False`
- with v-trace: `False`
- recurrence: `off`
- replay sampling probability: `0.75`
- episode duration: `15.0 s`
- visible neighbors: `2`
- obstacle density: `0.2`
- obstacle size: `0.6`
- downwash: `True` for the parity baseline after stability is confirmed

## Replay and curriculum hooks

Keep replay logic out of the model.

The training script should only need to:
- enable or disable replay behavior through env/task config
- log replay metrics
- optionally enable reward-annealing schedules that match the release baseline

## Suggested implementation order

1. Implement the encoder and unit-test it with dummy observations.
2. Implement the policy/value wrappers and test sampling/value output.
3. Run a tiny no-obstacle smoke training job.
4. Enable the paper obstacle room and reward terms.
5. Enable replay and any annealing schedules.
6. Run ablations.

## Minimum model tests

Add tests that verify:
- correct observation slicing sizes
- encoder output size is stable
- actor outputs valid means/log-std or equivalent parameters
- critic outputs one scalar per agent observation
- forward pass works for batched inputs from multiple environments and agents
