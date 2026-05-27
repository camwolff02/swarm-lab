---
name: training-metrics
description: Read paper_swarm training logs and TensorBoard metrics. Use when user asks about training progress, reward, episode length, loss curves, training health, or wants to compare runs.
---

# Training Metrics

Read and analyze training metrics from paper_swarm TensorBoard event files.

## Quick summary (last known metrics)

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 15 uv run python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, glob
stage_dir = 'logs/skrl/paper_swarm/mappo_stage1'
runs = sorted(os.listdir(stage_dir))
for r in runs[-3:]:
    try:
        ea = EventAccumulator(f'{stage_dir}/{r}')
        ea.Reload()
        tags = ea.Tags()['scalars']
        for t in ['Episode / Total timesteps (mean)','Reward / Total reward (mean)',
                  'Stats / Env stepping time (ms)','Loss / Value loss (drone_0)',
                  'Policy / Standard deviation (drone_0)']:
            if t in tags:
                e = ea.Scalars(t)[-1]
                print(f'{r[-19:]} {t.split(\"/\")[-1]:>25}: {e.value:.1f} @step {e.step}')
    except: pass
" 2>&1
```

## Full metrics dump (all tags at last step)

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 15 uv run python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
stage_dir = 'logs/skrl/paper_swarm/mappo_stage1'
run = sorted(os.listdir(stage_dir))[-1]
ea = EventAccumulator(f'{stage_dir}/{run}')
ea.Reload()
print(f'=== {run} ===')
for tag in sorted(ea.Tags()['scalars']):
    events = ea.Scalars(tag)
    if events:
        last = events[-1]
        trend = ''
        if len(events) >= 2:
            f2 = events[-3] if len(events) >= 3 else events[0]
            trend = f' ({f2.value:.2f} -> {last.value:.2f})'
        print(f'  {tag:<50} = {last.value:.4f}{trend}'[:120])
" 2>&1
```

## Key metrics interpretation

| Metric | Healthy range | Warning sign |
|---|---|---|
| Episode length (mean) | 100+ | < 30 (drones crashing/terminating fast) |
| Total reward (mean) | Positive, rising | Negative or flat at low values |
| Policy std | Decreasing or stable | Rising above 0.8 |
| Value loss | Decreasing toward 0.5 | Stuck above 3.0 |
| Learning rate | Stable at target | Dropping to zero (KLAdaptiveLR runaway) |
| Env stepping (ms) | < 100ms | > 500ms (episodes resetting every step) |

## Compare agent checkpoint metrics across stages

To check what model weights look like after training:

```bash
cd /home/cam/Development/cpsquare/swarm-lab && timeout 15 uv run python -c "
import torch, os, glob
stage_dir = 'logs/skrl/paper_swarm/mappo_stage1'
runs = sorted(os.listdir(stage_dir))
for r in runs[-1:]:
    ckpts = glob.glob(f'{stage_dir}/{r}/checkpoints/*.pt')
    if ckpts:
        ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        sd = torch.load(ckpt, map_location='cpu')
        steps = sd.get('timestep', '?')
        print(f'Latest checkpoint: {os.path.basename(ckpt)} (step {steps})')
        policy_keys = [k for k in sd.get('policy',{}).keys() if 'encoder' in k or 'mean_head' in k]
        print(f'Policy keys: {policy_keys[:5]}...')
" 2>&1
```

## Training run logs directory structure

```
logs/skrl/paper_swarm/
  mappo_stage1/    # Stage 1 training runs
  mappo_stage2/    # Stage 2 training runs
  mappo_stage3/    # Stage 3 training runs
  mappo/           # Original MAPPO runs
```

Each run directory contains:
- `events.out.tfevents.*` — TensorBoard scalar metrics
- `checkpoints/agent_*.pt` — Model checkpoints
- `params/` — Saved config parameters
