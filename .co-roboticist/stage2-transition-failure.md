# Stage 2 Transition Failure — Analysis Correction

Generated: 2026-05-30

---

## Errata: Prior Analysis Was Wrong

The earlier diagnosis claimed `clip_log_std` was the root cause of Stage 2 collapse. **This was incorrect.**

### What I said before

> "Root cause: Stage 2/3 configs missing `clip_log_std`, `min_log_std`, `max_log_std` — PPO drove log_std to −2.62 unchecked"

### What the evidence shows

| Claim | Evidence | Verdict |
|---|---|---|
| "PPO drove log_std to −2.62" | log_std was −2.62 in the **Stage 1** checkpoint before Stage 2 began | Wrong — the value was inherited, not caused during Stage 2 |
| "Unbounded collapse" | Both runs (with and without `clip_log_std`) have **bit-identical** weights to Stage 1 | Wrong — zero gradient updates occurred in both runs |
| "clip_log_std fix resolves Stage 2" | Stage 2 v2 with the fix is identical to v1 | Wrong — fix doesn't affect behavior |

### Why I got it wrong

1. **Correlation ≠ causation**: I saw the same metrics (std=0.073, losses=0, LR=floor) in the initial Stage 2 run and assumed the collapse happened during Stage 2 training. The std of 0.073 looked like a collapsed value and I assumed unbounded log_std was responsible.

2. **Didn't compare weights directly**: I checked TensorBoard metrics and the YAML config but did not verify whether the Stage 2 checkpoint weights differed from Stage 1. A simple `torch.equal()` comparison would have immediately shown that training was frozen.

3. **Overlooked the first logged step**: The metrics at step 3000 (first logged event) already showed std=0.073, LR=5e-5, losses=0. This means the collapse was **pre-existing**, not emergent during training.

4. **During v2 analysis**: I correctly identified that the preprocessor + KLAdaptiveLR was preventing training, but missed that the weights were NEVER UPDATED (not even once). I should have checked the checkpoint weights before doing any deep-dive analysis.

5. **Confirmation bias**: Finding that the Stage 2 config was missing `clip_log_std` felt like the natural answer. I fitted the evidence to the hypothesis rather than the reverse.

---

## Current Understanding

### Stage 2 makes zero gradient updates

Both Stage 2 v1 (2026-05-28, without clip_log_std) and Stage 2 v2 (2026-05-29, with clip_log_std) produced **no training**:

```
Stage 1 best_agent weights == Stage 2 v1 best_agent weights → True
Stage 1 best_agent weights == Stage 2 v2 best_agent weights → True
Stage 1 best_agent weights == Stage 2 v2 agent_300000 weights → True
```

Every policy parameter (encoder attention, trunk MLP, mean head, log_std) and value function parameter is bit-identical to the Stage 1 checkpoint. All 8 drones have the same weights (shared via Python object identity in `runner.py:68-73`).

### What the TensorBoard data shows

| Metric | Step 3000 | Step 300000 | Meaning |
|---|---|---|---|
| Policy loss | 0.0 | 0.0 | Surrogate objective has zero gradient |
| Value loss | 0.0 | 0.0 | MSE(returns, predicted_values) = 0 |
| Entropy loss | 0.0 | 0.0 | Entropy bonus has zero gradient |
| Policy std | 0.073 | 0.073 | Frozen at Stage 1 checkpoint value |
| Learning rate | 5e-5 | 5e-5 | KLAdaptiveLR hit floor at first check |
| Mean total reward | -29.79 | -50.13 | Negative throughout |

### Observed physical behavior

- drone_0: flies toward waypoints (the Stage 1 policy was trained for this)
- drones 1-7: fly up and away (out-of-distribution observations for Stage 1 policy)

---

## Diagnostic Findings

### Stage 1 checkpoint is from step 20,000, not step 200,000

The `best_agent.pt` loaded into Stage 2 has identical weights to `agent_20000.pt` from Stage 1. The Stage 1 policy at step 20,000 had:
- log_std ≈ -2.6 (std ≈ 0.07) — **very low entropy**
- The Stage 1 policy CONTINUED training until step 200,000 and the std **recovered to 0.17**
- The reward also improved from ~33 to ~86 over training

**So the checkpoint loaded into Stage 2 had early, low-entropy policy weights, not the converged, higher-entropy final weights.**

### Value network produces near-constant output

The value network outputs ≈ -4.59 for both Stage 1-like and Stage 2-like states (difference = 0.0038). This is because:
1. The state preprocessor (RunningStandardScaler, clip=5.0) scales all states to [-5, 5]
2. LayerNorm per-sample normalization in the value network compresses variation further
3. The value function from the early Stage 1 checkpoint never learned to distinguish diverse multi-agent states

### Preprocessor update has negligible effect

Despite 126/272 state dims having near-zero std from Stage 1 (drones 1-7 were parked at fixed positions), the RunningStandardScaler's count of 163 million means a single 64-sample mini-batch of Stage 2 data barely changes the statistics:
```
Mean change: max = 0.0000
Var change: max = 0.0001
```
The observation and value preprocessors also barely change.

### Why losses are exactly zero (hypothesis)

The most likely mechanism:

1. **First mini-batch, first epoch**: KL ≈ 0 (same policy, same preprocessor), so the KL check passes. Losses are computed:
   - Policy loss: advantages are GAE-normalized, then normalized again to mean=0, std=1. With ratio=1 (same policy), `policy_loss = -mean(advantages * 1) = 0` (because advantages.mean() = 0 after normalization).
   - Value loss: predicted values are clipped relative to normalized values. MSE is small (≈ 0.01).
   - Entropy loss: non-zero from log_std ≈ -2.6.

2. **Gradient step**: The optimizer takes one step with these losses (LR = 5e-4).

3. **Mini-batch 2-N, first epoch**: The policy has changed slightly. KL > 0.01 → break fires. Only 1/32 mini-batches contributed.

4. **Scheduler update**: KLAdaptiveLR sees mean KL > 0.01 → drops LR to floor (5e-5).

5. **Subsequent iterations**: With LR at floor, gradient steps produce negligible weight changes. The KL threshold always fires immediately (because stored rollout log_probs differ from current policy log_probs). No meaningful learning ever occurs.

This explains zero policy loss (advantage normalization creates zero-mean advantages, ratio=1). The value and entropy losses would be small but non-zero. However, TensorBoard shows exactly 0.000000 — this may be due to the logged values being divided by (4 epochs × 32 mini-batches = 128), and the per-iteration losses being << 0.01, resulting in logged values below the TensorBoard display precision.

### Why Stage 1 succeeded

Stage 1 started with `initial_log_std = -1.5` (std ≈ 0.22), which is higher than -2.6. The first policy gradient step had larger entropy. The initial random policy + higher std provided diverse experience, allowing the value function to learn useful predictions. The LR (fixed 3e-4, no scheduler) stayed high, enabling continued learning.

---

## Summary of Issues for Fixing Stage 2

| Issue | Finding |
|---|---|
| **Low-entropy checkpoint** | Stage 1's `best_agent.pt` (from step 20k) has log_std ≈ -2.6. Need to either use final checkpoint (step 200k, std≈0.17) or reset log_std at transition |
| **Value function is near-constant** | The loaded value function outputs ≈ -4.59 for diverse states. Combined with advantage normalization → zero-mean advantages → zero policy loss |
| **KLAdaptiveLR too aggressive** | Scheduler drops LR to floor immediately after first gradient step, preventing recovery |
| **Preprocessor statistics from Stage 1** 126/272 state dims have near-zero std → extreme normalization → value network sees degenerate states |

## What I Should Have Done Differently

1. Before analyzing metrics, check: `torch.equal(stage1_weights, stage2_weights)` — would have immediately shown zero training
2. First verify the checkpoint loading mechanism (how does 1-agent → 8-agent transition work?)
3. Check which Stage 1 checkpoint was loaded (early vs final)
4. Run a simple forward-pass test on the value function before speculating about policy collapse
