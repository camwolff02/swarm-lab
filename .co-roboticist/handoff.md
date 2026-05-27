# Agent Handoff: Stage 1 NaN Fix

Date: 2026-05-26 | Branch: `new-paper` | Classification: `bugfix`

---

## What Was Fixed

**Problem**: Stage 1 MAPPO training always diverged to NaN at ~15K steps, or crashed at init. Zero successful training runs.

**Root cause**: `paper_swarm_global_state()` returned a 272-dim vector of zeros because Stage 1 has `possible_agents = ["drone_0"]` (only 1 managed agent) but the centralized critic tried to access all 8 drones through `_agent_to_bundle`. Unmanaged drones (1–7) weren't in the bundle dict → function fell through to `return torch.zeros(...)`.

The value network received constant-zero input → learned nothing → GAE advantages were garbage → policy gradient exploded → NaN.

**Fix** (1 file, 1 function):

`environments/environments/tasks/paper_swarm/mdp/observations.py` — `paper_swarm_global_state()`:

1. **Removed** the early-exit guard:
   ```python
   # BEFORE:
   if not all(agent_id in getattr(root, "_agent_to_bundle", {}) for agent_id in agent_ids):
       return torch.zeros(root.num_envs, state_dim, device=root.device)
   # AFTER: (removed)
   ```

2. **Added fallback** for unmanaged drone commands:
   ```python
   # BEFORE:
   bundle = root._manager_bundles[root._agent_to_bundle[agent_id]]
   commands.append(bundle.command_manager.get_command(command_name))
   
   # AFTER:
   if agent_id in agent_to_bundle:
       bundle = root._manager_bundles[agent_to_bundle[agent_id]]
       commands.append(bundle.command_manager.get_command(command_name))
   else:
       commands.append(torch.zeros(root.num_envs, 7, device=root.device))
   ```

**Why safe**: Value network architecture unchanged (always 272-dim input). Policy architecture unchanged (always 61-dim input). Checkpoints remain compatible across Stages 1→2→3.

---

## Smoke Test Results

- Started cleanly (no init crash)
- 96 training steps (3 rollouts), all metrics finite
- Policy weights updated across all layers
- Value weights updated (1015 total abs change)
- No NaN in any parameter
- Log std stable: -1.5 → -1.51 (healthy)
- Checkpoints saved: `agent_9.pt` through `agent_96.pt`

---

## Active Training Run

A Stage 1 training run was spawned in tmux:

```
tmux attach -t stage1-train
```

Command:
```bash
uv run scripts/skrl/train.py --algorithm MAPPO --task Isaac-Paper-Swarm-Waypoint-MAPPO-Stage1-v0 --headless
```

Log directory will be: `logs/skrl/paper_swarm/mappo_stage1/<timestamp>_mappo_torch/`

---

## How to Analyze the Training

### 1. Check if still running
```bash
tmux capture-pane -t stage1-train -p | tail -20
```

### 2. Extract metrics from latest run
```bash
cd /home/cam/Development/cpsquare/swarm-lab && uv run python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, math
stage_dir = 'logs/skrl/paper_swarm/mappo_stage1'
runs = sorted(os.listdir(stage_dir))
r = runs[-1]
ea = EventAccumulator(f'{stage_dir}/{r}')
ea.Reload()
tags = ea.Tags()['scalars']
print(f'Run: {r}')
for t in ['Episode / Total timesteps (mean)','Reward / Total reward (mean)',
          'Policy / Standard deviation (drone_0)','Loss / Value loss (drone_0)',
          'Loss / Policy loss (drone_0)']:
    if t in tags:
        events = ea.Scalars(t)
        has_nan = any(math.isnan(e.value) for e in events)
        last = events[-1]
        print(f'  {t.split(\"/\")[-1]:>25}: {last.value:.4f} @step{last.step}{\" *** NaN ***\" if has_nan else \"\"}')
"
```

### 3. Check latest checkpoint health
```bash
cd /home/cam/Development/cpsquare/swarm-lab && uv run python -c "
import torch, os, glob
stage_dir = 'logs/skrl/paper_swarm/mappo_stage1'
runs = sorted(os.listdir(stage_dir))
r = runs[-1]
ckpts = sorted(glob.glob(f'{stage_dir}/{r}/checkpoints/*.pt'))
if ckpts:
    ckpt = ckpts[-1]
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    steps = sd.get('timestep', '?')
    print(f'Latest: {os.path.basename(ckpt)} (step {steps})')
    for net in ['policy', 'value']:
        for k in sd['drone_0'][net]:
            if isinstance(sd['drone_0'][net][k], torch.Tensor):
                if sd['drone_0'][net][k].isnan().any():
                    print(f'  *** NaN in {net}.{k} ***')
    # Log std
    lp = sd['drone_0']['policy']
    for k in lp:
        if 'log_std' in k:
            print(f'  log_std: {lp[k].numpy()} -> std: {lp[k].exp().numpy()}')
"
```

### 4. Key metrics to watch

| Metric | Healthy | Warning |
|--------|---------|---------|
| Episode length (mean) | Increasing toward 1000 | Flat at low values (< 50) |
| Total reward (mean) | Positive, rising | Negative or flat |
| Policy std | Decreasing toward ~0.1 | Rising above 0.6 |
| Value loss | Decreasing toward ~0.5 | Stuck above 3.0 |
| Env stepping (ms) | < 100ms | > 500ms |

---

## Context Map

Full project knowledge at `.co-roboticist/context-map.md`
Detailed training analysis at `.co-roboticist/stage2-training-analysis.md`
Config ingest summary at `.co-roboticist/stage1-paper-swarm-ingest.md`

## Working Tree State

11 files modified (uncommitted). The Stage 1 config has been significantly reworked from the committed version:
- `possible_agents = ["drone_0"]` (was all 8)
- `num_envs = 512` (was 256)
- Fixed-origin resets + `expand_target_range_curriculum` (lab_5 hover pattern)
- CLIP actions (was TANH)
- `clip_log_std: true` with range [-4.0, -0.5]
- Rollouts 32, mini_batches 64 (was 64/32)
