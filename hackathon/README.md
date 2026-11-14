# Training a model to edit another model's weights

State on 2026-11-13.

For the [AI Research Hackathon](https://luma.com/sht6n8du), San Francisco, 14 to 15 November 2026. Track: Recursive Self-Improvement. Background is in [continual-learning-benchmarks/README.md](../continual-learning-benchmarks/README.md), sections 2.8 and 2.9.

## Where this comes from

Two recent results show a frontier model improving a small model by editing its weights by hand. No gradients: the editor reads weights and activations, reasons, and overwrites a few values.

- "Sparks of Weight-Space Programming" ([blog post](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL, 2 Nov 2026). GPT-6 Astra edits Qwen3.5-0.8B and MATH-500 goes from 7.9% to 48.1%.
- "Exploring Weight-Space Programming" ([2611.09382](https://arxiv.org/abs/2611.09382), Shanghai AI Laboratory, 6 Nov 2026). Same editor, sixteen searches. The best one changes a single column of one matrix and takes MATH-500 from 8.35% to 37.80%. Six of the sixteen searches find nothing. No code is released.

Both use a closed frontier model as the editor. We want to know how strong the editor has to be, whether an open model can do it, and whether a model can do it to itself.

## What we built

A harness where the editor and the target are both settings.

The target is Qwen3.5-0.8B. The editor gets two tools:

- A Python sandbox with the target's weights on disk. CPU only, no internet, no autograd. 300 calls.
- A forward tool, 50 calls. Each call either has the target answer practice problems (original weights, edited weights, or both side by side), or runs the target over text the editor supplies and returns log-probabilities and activations.

The editor can change any weight inside the transformer blocks, in place. The embedding matrix and final norm are off limits.

Practice problems come from the MATH and GSM8K training sets. The test is MATH-500, which the editor never sees. It is run once after the search, 2 samples per problem, original weights against edited.

The editor can stop whenever it wants. It is told to submit an edit only if it wins clearly on at least 300 fresh practice answers, and to submit the original weights otherwise.

Compared with 2611.09382, we cap answers at 8,192 tokens instead of 32,768 and use 2 test samples per problem instead of 4.

## What we found

Qwen3.5-0.8B fails at math in one way: it keeps writing "Wait..." and rechecking until it runs out of tokens. About 95% of its answers end that way.

Held-out MATH-500, 1,000 answers per arm:

| Editor | Original | Edited | Change |
|---|---|---|---|
| The paper's published edit, applied by us as a check | 7.1% | 28.9% | +21.8 |
| Muse Glimmer-30B, first search | 7.1% | 6.0% | -1.1 |
| Muse Glimmer-30B, second search | 6.1% | submitted no edit | |
| Claude Opus 5.5 | 7.6% | 34.8% | +27.2 |
| Inkling 2 Small, with Opus's methods in its prompt | 6.6% | 20.2% | +13.6 |

The published edit works in our harness, so the harness is sound.

Glimmer-30B found nothing in two searches. It saw the looping, but it picked which unit to edit by comparing average activations between two texts, only ever tried setting one column to zero, and tested on 50 answers at a time, which is too few to tell a gain from chance.

Opus 5.5 found an edit. It asked which unit contributes most to the score of the token "Wait" at the moment just before the model writes it, and the answer was one unit in layer 16. Flipping the sign of that unit's output, plus a second change that makes the model close its reasoning, took practice accuracy from about 6% to about 34% on 768 fresh answers, and held-out MATH-500 from 7.6% to 34.8%: 274 answers went from wrong to right and 2 from right to wrong. That is a larger gain than the published edit gets in our harness. The same unit shows up in two edits in 2611.09382.

Inkling 2 Small, given a description of Opus's methods, also found an edit: it strengthens the same end-of-reasoning signal in the last four layers' MLPs. It stopped on its own after 25 of its 50 forward passes. Held-out MATH-500 went from 6.6% to 20.2%, about half of Opus's gain. It did not find the "Wait" unit, and Opus's own tests showed that the two changes together do much better than either alone. Its practice numbers (+20 to +23) were higher than the held-out result.

The original model's own score came out at 7.1%, 6.1%, 7.6% and 6.6% in four evaluations, so a one-point change means nothing.

We also measured the two open models as possible targets for editing themselves, on the same questions:

| | MATH-500 | MMLU-Pro | SciBench | MedXpertQA |
|---|---|---|---|---|
| Muse Glimmer-30B | 92.7% | 83.2% | 64.0% | 38.8% |
| Inkling 2 Small | 96.7% | 86.0% | 79.6% | 59.2% |

Neither has a looping failure. Their mistakes are mostly completed wrong answers, which is where published edits gave nothing. The one clean behaviour found is Glimmer on SciBench: told to put only the number in the box, it does so in 33% of answers (Inkling: 90%). No model has edited itself yet.

## Goal for the weekend

Train Inkling 2 Small with RL to be a better weight editor of Qwen3.5-0.8B, and show it: trained against untrained, by average held-out gain. One untrained run so far gave +13.6 on MATH-500.

## The RL loop

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

- **RL step:** collect rollouts, score them, train Inkling on Tinker, save a checkpoint.
- **Rollout:** one editing search on a fresh copy of Qwen, about 160 turns, ending in a submitted edit.
- **Tool call:** one Python run, or one forward request to the target.

Choices:

- **Reward:** accuracy with the edit minus accuracy without, on 256 fixed answers the editor never sees.
- **Groups:** one per task (MATH, IFEval, SciBench), 8 rollouts each. Advantage is reward minus the group's average.
- **Training:** Tinker's RL loop, with our harness as its environment.
- **Per rollout:** 32 forward passes.
- **Throughput:** original answers generated once and reused; several rollouts per GPU.

## Budget

Three people: $750 Modal, $300 Tinker. About $2 per rollout (estimate), so about 300 rollouts: 12 to 19 RL steps.

## Risks

- Twelve to nineteen updates may be too few to see a change.
- The reward is noisy: 256 answers, and the original model alone moves by a point on 1,000.
- Inkling may learn the one fix for Qwen's looping and nothing more general.
