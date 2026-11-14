# Proposal: which models can edit weights directly, and can one edit its own

State on 2026-11-13.

For the [AI Research Hackathon](https://luma.com/sht6n8du) in San Francisco, 14 to 15 November 2026. Track: Recursive Self-Improvement.

Background reading is in [continual-learning-benchmarks/README.md](../continual-learning-benchmarks/README.md), sections 2.8 (models that learn to write their own weights) and 2.9 (editing weights from outside).

## The question

Can an open-weight model find and apply useful edits to a model's weights, with no gradients, and in the end to its own?

Two sources show that a frontier model can do this to a small one:

- **"Sparks of Weight-Space Programming"** (Takehi, Teoh, Zhang, Khattab, MIT CSAIL blog, 2 Nov 2026, [post](https://rikiyat.github.io/blog/weight-space-programming/)). GPT-6 Astra inspects and edits Qwen3.5-0.8B and raises MATH-500 from 7.9% to 48.1%.
- **"Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations"** ([2611.09382](https://arxiv.org/abs/2611.09382), Shanghai AI Laboratory, 6 Nov 2026). Sixteen searches on Qwen3.5-0.8B and Qwen3.5-9B with GPT-6 Astra as the editor. On 0.8B, a one-column edit of the layer-23 MLP down-projection raises held-out MATH-500 from 8.35 to 37.80. On 9B the largest kept gain is 3.03 points, and six of the sixteen searches keep the original weights. No code is released.

Both use a closed frontier model as the editor. Open questions: how strong the editor has to be, whether an open model can do it, and whether a weaker editor can be handed the methods of a stronger one.

## Setup

One harness, where the editor and the target are settings.

- **Target.** Qwen3.5-0.8B on one L40S. 24 layers, hidden size 1024, MLP down-projections of shape [1024, 3584].
- **Editor.** Any model that can call two tools. It never gets gradients.
- **Python tool.** A CPU sandbox with no internet and autograd off. The editor loads weight tensors, analyses them and saved activations, and saves an edited tensor. 300 calls.
- **Forward tool.** 50 calls, of two kinds, as in 2611.09382:
  - generation on practice problems with the original weights, the edited weights, or both paired, returning graded answers;
  - teacher-forced passes over supplied text, returning per-token log-probabilities, hidden states and MLP activations.
- **Edit space.** Any tensor inside the transformer blocks, changed in place. Off limits: the embedding matrix, the final norm, and multiplying a whole tensor by a constant.
- **Data.** Practice set of 3,000 MATH training problems and 1,500 GSM8K training problems. Held-out test is MATH-500, which the editor never sees.
- **Ending a search.** The editor may finish at any time. It is told to submit an edit only if a paired comparison on at least 300 fresh generations shows a gain above 2 standard errors, and otherwise to submit the original weights.
- **Evaluation.** Once, after the search: MATH-500, 2 samples per problem, original against edited, paired by problem.

Differences from 2611.09382: the generation cap is 8,192 tokens instead of 32,768, held-out uses 2 samples per problem instead of 4, and the grader, practice set and prompt wording are our own.

## Results so far

Target Qwen3.5-0.8B, held-out MATH-500, 1,000 graded answers per arm.

| Editor | Edit submitted | Original | Edited | Difference (SE) |
|---|---|---|---|---|
| None: the Math (Primary) edit of 2611.09382 applied as a control | layer-23 MLP down-projection, column 507, minus a doubt-versus-progress direction | 7.1% | 28.9% | +21.8 (1.57) |
| Muse Glimmer-30B, first prompt | layer-20 MLP down-projection, column 427 set to zero | 7.1% | 6.0% | -1.1 (0.79) |
| Muse Glimmer-30B, current prompt | none; it submitted the original weights | 6.1% | n/a | n/a |
| Claude Opus 5.5 | layer-16 MLP column 998 negated, plus a rank-one gain along the end-of-thinking token's direction on nine output matrices in layers 15 to 22 | running | running | practice: +29.3, +26.2, +28.1 (3.4 each) on 768 paired answers |
| Inkling 2 Small, given the methods Opus used | running | | | |

What these show:

- **The harness measures a working edit.** The published edit reproduces most of its gain under our shorter token cap.
- **Glimmer-30B did not find an edit in two searches.** It diagnosed the failure correctly (reasoning that loops until the token cap) but chose units by comparing mean activations between two texts, only ever zeroed single columns, and tested on sets too small to tell a gain from noise.
- **Opus 5.5 found one.** It split the score of the "Wait" token into per-unit contributions at the positions just before "Wait", across layers 12 to 23, and found the largest contributor. That unit, column 998 of layer 16, is also edited in two checkpoints of 2611.09382 for other tasks.
- **The original model's score moves by about a point between evaluations** (7.1% and 6.1% on the same problems and seeds), so small differences mean nothing.

## Glimmer-30B as a target

A diagnostic sweep of unedited Glimmer-30B: MATH-500 92.7%, MMLU-Pro 83.2%, SciBench 64.0%, MedXpertQA 38.8%. It hits the token cap in 0 to 1.6% of answers, so it has no looping failure to repair. The one repeatable, program-checkable failure found: on SciBench, told to put only the number in the box, it complies in 33% of answers.

Glimmer editing itself has not been run. Two things argue against it as the main line: Glimmer has not yet edited the small model, and it has no failure as clean as the one every published gain depends on.

## Path for the hackathon

1. **Compare editors on one fixed target.** Same target, tools, budget and held-out test for Glimmer-30B, Opus 5.5 and Inkling 2 Small. This gives how strong an editor has to be, with one open-weight point on each side.
2. **Hand a weaker editor the stronger editor's methods.** Inkling 2 Small's prompt carries the procedures Opus used: a large first diagnostic pass, fixed data splits, the per-unit share-of-token-score calculation, the two edit forms, dose sweeps and confirmation on fresh problems. It does not name the token, layer or unit Opus found. The comparison is the same editor with and without that section.
3. **Repeat searches.** Each cell above is a single search. In 2611.09382 three searches on this task gave +29.45, +5.00 and +27.30, so a rate needs several per editor.
4. **Then self-editing.** Take the strongest open editor that can also be hosted as a target and run it on its own weights, on a failure found by a diagnostic sweep.

A further direction if time allows: treat held-out lift as a reward and train the editor on it. The only related work found, "Reinforcement Learning for Neural Model Editing" ([2606.13461](https://arxiv.org/abs/2606.13461)), trains a small PPO policy to edit one layer of an MNIST CNN and of a text classifier; nobody has trained a language-model editor this way.

## What gets reported

- Held-out accuracy, original against edited, paired by problem, with the standard error.
- The edit itself: which tensors, which columns or directions, how many values changed.
- How each editor spent its budget: forward passes, sample sizes, and how it chose where to intervene.
- A baseline that needs no weight edit: a logit bias or repetition penalty aimed at the same failure. Neither source runs this.
- A general-capability probe on an untouched benchmark, to catch damage.

## Measured cost

Estimated from run times and list prices.

- Glimmer-30B search: 2 to 2.3 hours on one H100 plus one L40S, about $13.
- Opus 5.5 search: 2.8 hours, $7.49 of API spend plus about $6 for the L40S.
- Held-out evaluation, both arms: about 50 minutes on one L40S, about $2.
- Inkling 2 Small is called on Tinker at $0.30 input and $1.00 output per million tokens. Hosting it needs about 500 GB of GPU memory.

## Risks

- **One search per editor is not a rate.** Outcomes vary widely between searches of the same editor.
- **The gains come from one failure.** Edits remove an obstacle (looping); they do not add ability. Where the original model has no such failure, published edits land within noise.
- **Practice gains can fail to hold.** In 2611.09382, a +17.19 development gain on MedXpertQA became +0.24 held-out.
- **Self-editing needs a model that is both a capable editor and cheap to host.** Glimmer-30B is cheap to host but has not edited anything; Inkling 2 Small may edit but is expensive to host.
- **Long searches can be interrupted.** A search driver must run on non-preemptible capacity and must not restart on top of an earlier session's files.
