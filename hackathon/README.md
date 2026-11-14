# Proposal: train a model to edit another model's weights

State on 2026-11-13.

For the [AI Research Hackathon](https://luma.com/sht6n8du), San Francisco, 14 to 15 November 2026. Track: Recursive Self-Improvement.

## Question

Can reinforcement learning make an open model better at editing another model's weights?

A frontier model can already improve a small model by reading its weights and activations and overwriting a few values, with no gradients ([MIT CSAIL post](https://rikiyat.github.io/blog/weight-space-programming/), [arXiv 2611.09382](https://arxiv.org/abs/2611.09382)). Nobody has trained a model to do this better.

## Baseline

One search per editor. Each tries to improve Qwen3.5-0.8B on math. Held-out MATH-500, 1,000 answers per arm:

| Editor | Original | Edited | Change |
|---|---|---|---|
| Claude Opus 5.5 | 7.6% | 34.8% | +27.2 |
| The published edit from 2611.09382, applied by us | 7.1% | 28.9% | +21.8 |
| Inkling 2 Small | 6.6% | 20.2% | +13.6 |
| Muse Glimmer-30B | 7.1% | 6.0% | -1.1 |
| Muse Glimmer-30B, second search | 6.1% | no edit | |

Lessons:

- Qwen fails in one way: it writes "Wait..." and rechecks until it runs out of tokens. Every gain comes from stopping that.
- Opus found the unit that pushes "Wait" by measuring each unit's share of that token's score. Glimmer compared averages over two texts and found nothing.
- Inkling, an open model, got half of Opus's gain. That gap is the room for training.
- The original model's score moves by about a point between runs, so small changes mean nothing.

## Experiment

Train Inkling 2 Small with RL as the editor. Compare the trained and untrained Inkling by average held-out gain over several searches.

```
                              TINKER
        ┌──────────────────────────────────────────────┐
        │  Inkling 2 Small, the editor being trained   │
        │    sample a reply  ·  train  ·  checkpoints  │
        └──────▲───────────────────────────┬───────────┘
               │ sample                    │ train on scored rollouts,
               │                           │ once per RL step
 ═════════════ │ ═════════ MODAL ═════════ │ ══════════════════════════
               │                           │
        ┌──────┴───────────────────────────▼───────────┐
        │  HARNESS (CPU)                               │
        │    runs many rollouts at once                │
        │    passes each tool call on, scores the edit │
        └──────┬───────────────────────────┬───────────┘
               │ python tool               │ forward tool
               ▼                           ▼
     ┌────────────────────┐     ┌──────────────────────────────┐
     │ SANDBOX (CPU)      │     │ TARGET SERVICE (L40S pool)   │
     │ one per rollout    │     │  Qwen3.5-0.8B                │
     │  read weights      │────▶│  original weights, shared    │
     │  analyse           │edit │  one edited copy per rollout │
     │  save an edit      │     │  generate · read text        │
     └────────────────────┘     └──────────────────────────────┘
```

- **Editor:** Inkling 2 Small on Tinker.
- **Target:** Qwen3.5-0.8B on Modal. Each rollout edits a fresh copy.
- **Tools:** a Python sandbox over the target's weights, and a forward tool that has the target answer problems or read text and return activations. 32 forward passes per rollout.
- **Rollout:** one editing search, about 160 turns, ending in a submitted edit.
- **Reward:** accuracy with the edit minus accuracy without, on 256 fixed answers the editor never sees.
- **Groups:** one per task (MATH, IFEval, SciBench), 8 rollouts each. Advantage is reward minus the group's average.
- **Training:** Tinker's RL loop, with our harness as its environment.

## Budget

Three people. Per-rollout costs are estimates.

| | Modal | Tinker |
|---|---|---|
| Credit | $750 | $300 |
| Cost per rollout | $1.40 | $0.63 |
| Rollouts, keeping a third in reserve | about 360 | about 320 |

| Rollouts per RL step | RL steps |
|---|---|
| 16 (2 tasks × 8) | about 19 |
| 24 (3 tasks × 8) | about 12 |
