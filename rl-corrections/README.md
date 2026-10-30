# RL corrections in miles

Every correction below changes one factor in the per-token policy gradient on the logits:

```
g_logits = w * A * (p - y_onehot) / N
```

`w` is a per-token weight, `A` the advantage, `p` the trainer's distribution, `y_onehot` the written token,
and `N` the token count. [01-multi-step-clipping/cispo_backward.ipynb](01-multi-step-clipping/cispo_backward.ipynb)
derives this line by hand and checks it against PyTorch.

Three probabilities appear in the formulas: `p_now` (trainer, current weights, with gradient), `p_old` (trainer,
recomputed before the step, no gradient) and `p_engine` (the SGLang engine that sampled the token).

Loss normalization (per token vs per response, GRPO vs Dr. GRPO vs DAPO): [normalization.ipynb](normalization.ipynb).
Score centering, step by step with numbers: [score_centering.md](score_centering.md).
New agent: read [HANDOFF.md](HANDOFF.md) first.
Start with [corrections.md](corrections.md): one table and the PyTorch code for each method.
To see it move, open [viz/clipping_network.html](viz/clipping_network.html): pick a method, press Step, and watch the network's connections and the kept, capped and dropped tokens.
All of them run on one toy batch in [corrections.ipynb](corrections.ipynb): PyTorch, shapes in the comments,
one section per problem (engine vs trainer, staleness across steps, score centering).

**Table 1. Each correction, what it changes, where miles computes it, and whether our runs use it.**

| Method | Changes | Rule | miles code | Our runs |
|---|---|---|---|---|
| PPO clip | `w` | `w = r = p_now / p_old`; `w = 0` if A > 0 and r > 1.28, or A < 0 and r < 0.8 | `math_utils.py:255-278` | all rows (`--eps-clip 0.2 --eps-clip-high 0.28`) |
| Dual-clip PPO | `w` | for A < 0, also `w = 0` when r > `eps_clip_c` | `math_utils.py:268-274` | no (`--eps-clip-c` unset) |
| CISPO | `w` | `w = clip(r, 0, 4)`, frozen; no zeroing | `tinker_losses.py:192-213` | no (Tinker API only) |
| TIS | `w` | PPO `w` times `clip(p_old / p_engine, 0, 2)`, frozen | `corrections.py:7-32` | GLM-5.3 torchtitan recipe (`--use-tis`) |
| IcePop / token MIS | `w` | PPO `w` times `p_old / p_engine` inside `[low, high]`, else 0 | `corrections.py:35-62` | no |
| Decoupled PPO | `w` | PPO clip on `p_now / p_old`, times the TIS weight on `p_old / p_engine` | `losses.py:137-140, 234-265` | GLM-5.3 torchtitan recipe (it is what `--use-tis` gives) |
| Bypass PPO | `w` | PPO clip on `p_now / p_engine`; no trainer recompute | `losses.py:137-138` | no |
| GSPO | `w` | one ratio per response, `exp(mean_t(log p_now - log p_old))`, then the PPO clip | `math_utils.py:225-251` | no |
| OPSM | `w` | `w = 0` for every token of a response with A < 0 and `mean_t(log p_old - log p_now) > 1e-4` | `math_utils.py:184-222` | no |
| Score centering | the score `(y_onehot - p)` | subtracts the score's average under `p_engine`; with weights 1 and a full candidate list it becomes `(y_onehot - p_engine)` | `score_centering/estimator.py:64-95` | no |

**Table 2. The four pages.**

| Page | Question it answers |
|---|---|
| [01-multi-step-clipping/](01-multi-step-clipping/) | What limits a token's update when one rollout gets several optimizer steps (PPO clip, CISPO)? |
| [02-train-inference-mismatch/](02-train-inference-mismatch/) | How is a token reweighted when the engine that sampled it computes a different probability than the trainer (TIS, IcePop, MIS, decoupled and bypass PPO)? |
| [03-sequence-level/](03-sequence-level/) | What changes when the ratio or the mask is computed per response instead of per token (GSPO, OPSM)? |
| [04-score-centering/](04-score-centering/) | How does the loss cancel the update that sampling from `p_engine` adds even when the reward carries no signal? |

Our runs take one optimizer step per rollout, so `p_now = p_old` at the step and every method that only uses
`p_now / p_old` (PPO clip, dual clip, GSPO, OPSM) does almost nothing: the PPO clip fires on 0.019% of
Qwen tokens and 0% of GLM tokens (page 01, section 4). What remains is the trainer-vs-engine gap, which
only TIS, IcePop/MIS and score centering act on.

Code: miles = radixark/miles main at 34914c948f (all files under
`miles/backends/training_utils/loss/hub/`).

## TODO

Two infra-level sources of train/inference mismatch, left for later (both need a closer look at attention and kernels):

- **Sparse attention indexer.** DSA layers pick the top-k earlier tokens to attend to; a near-cutoff token can be
  picked by the engine and not the trainer. miles replays the engine's picks with `--use-rollout-indexer-replay`
  (about 0.5 MB per token on DeepSeek-V3.2). GLM-5.3-Flash has 11 DSA layers; our recipes do not replay the
  indexer, and its share of GLM's 0.021 to 0.031 log-prob gap is unmeasured.
- **Kernels.** Batch-dependent kernels and reduction order make the engine's numbers depend on its batch, phase
  (decode vs prefill) and GPU layout. The engine's decode-vs-prefill self-gap on GLM-5.3 is 0.031 to 0.033 (mean
  absolute log-prob), a floor R3 cannot remove. The fix is batch-invariant kernels (see corrections.md section 7).

