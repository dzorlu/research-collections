# Proposal: a model that edits its own weights

State on 2026-11-12. Nothing here has been built or run.

For the [AI Research Hackathon](https://luma.com/sht6n8du) in San Francisco, 14 to 15 November 2026. Track: Recursive Self-Improvement.

Background reading is in [continual-learning-benchmarks/README.md](../continual-learning-benchmarks/README.md), sections 2.8 (models that learn to write their own weights) and 2.9 (editing weights from outside).

## The question

Can an open-weight model find and apply useful edits to its own weights, with no gradients?

Two sources show that a stronger model can do this to a weaker one:

- **"Sparks of Weight-Space Programming"** (Takehi, Teoh, Zhang, Khattab, MIT CSAIL blog, 2 Nov 2026, [post](https://rikiyat.github.io/blog/weight-space-programming/)). GPT-6 Astra inspects and edits Qwen3.5-0.8B and raises MATH-500 from 7.9% to 48.1%.
- **"Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations"** ([2611.09382](https://arxiv.org/abs/2611.09382), Shanghai AI Laboratory, 6 Nov 2026, [code](https://github.com/WSP-Scope/WSP-Scope-Project)). Sixteen searches on Qwen3.5-0.8B and Qwen3.5-9B. On 0.8B, a one-column edit of the layer-23 MLP down-projection raises held-out MATH-500 from 8.35 to 37.80. On 9B the largest kept gain is 3.03 points, and six of the sixteen searches keep the original weights.

In both, the editor and the target are different models. Nobody has run the case where they are the same model.

## Model

Muse Glimmer-30B (Meta, Apache 2.0, open weights). It fits on one 80 GB GPU in BF16.

## Setup

- **One GPU on Modal** holds the base weights. It serves plain Glimmer as the editor, and Glimmer plus a candidate LoRA as the target.
- **Python sandbox.** CPU only, no internet. The editor can read its own weight files and saved activations. Autograd is off.
- **Forward tool.** Two request types, as in 2611.09382:
  - generation on practice problems, for base, edited, or both paired, returning scores and optional activations;
  - teacher-forced passes returning per-token log-probabilities, hidden states and MLP activations.
- **Edit space.** A rank-1 LoRA on a small set of layers. The base weights are never changed. A single-column edit and a rank-one update, which make up most of the kept edits in 2611.09382, are both expressible this way.
- **Budget per search.** 50 forward requests and 300 sandbox runs, matching 2611.09382.

## Loop

1. Glimmer runs itself on practice problems and reads the failures.
2. It inspects activations and proposes an edit.
3. The forward tool compares base against edited on the same problems.
4. The edit is kept only if it wins on fresh practice problems.
5. Recursive variant: the edited model becomes the editor for the next round.

## First step: find a failure worth fixing

Every large gain in the two sources comes from repairing one visible failure. On Qwen3.5-0.8B, 111 of 128 practice answers hit the 32,768-token cap, and the edit suppresses a "Wait" loop. Where the base model has no such failure, edits land within noise.

So the first job is a diagnostic sweep of Glimmer over a few benchmarks (MATH-500, IFEval, SciBench, GPQA Diamond, a tool-calling set) to find a repeatable failure with headroom. If none is found, the expected result is no gain.

## Arms

| Arm | Editor | Target |
|---|---|---|
| Self | Glimmer-30B | Glimmer-30B |
| External | GPT-6 Astra | Glimmer-30B |
| Sanity | Glimmer-30B | Qwen3.5-0.8B |
| Recursive | Glimmer-30B plus its kept edits | the same |

- **Self against External** separates the editor's ability from access to its own weights.
- **Sanity** checks that Glimmer can act as an editor at all, by trying to reproduce the published MATH-500 fix.
- **Recursive against Self** asks whether an edited model writes better edits than the original.

## What gets reported

- Held-out accuracy, base against edited, with paired intervals clustered by question.
- A baseline that needs no weight edit: a logit bias or repetition penalty aimed at the same failure. Neither source runs this.
- A general-capability probe on an untouched benchmark, to catch damage.
- The edit itself: which layer, which direction, how many values changed.

## Risks

- **The editor may be too weak.** All published results use GPT-6 Astra at maximum reasoning effort.
- **There may be nothing to fix.** A 30B model is unlikely to have a failure as clean as the 0.8B loop.
- **Development gains often do not hold.** In 2611.09382, a +17.19 development gain on MedXpertQA became +0.24 held-out.
- **Cost per search is unmeasured.** A rough guess is 5 to 10 H100-hours, $20 to $40.

## Before Saturday

- Serve Glimmer-30B on Modal with LoRA hot-swap and time one forward request.
- Run the Sanity arm once.
- Run the diagnostic sweep and pick the target failure.
- Confirm the hackathon allows a harness prepared in advance.
