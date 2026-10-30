# Score centering

The REINFORCE gradient on the logits for one sampled token is `-A * (y_onehot - p)`. When the token was sampled
from `p_engine` and not from the trainer's `p`, this gradient has a nonzero average even when the reward carries
no information. Score centering subtracts that average. Paper: arXiv 2609.20807, Appendix A.

## 1. The drift, on a three-token vocabulary

```python
p = [0.506, 0.307, 0.186]   # trainer, softmax of logits [0.5, 0.0, -0.5]
q = [0.20,  0.30,  0.50]    # engine distribution that sampled the tokens
A = 1.0                     # same reward for every sample: no signal

# average of -A * (y_onehot - p) over y ~ q  =  -A * (q - p)
plain    = [0.306, 0.007, -0.314]   # nonzero: training moves p toward q
# score centering replaces (y_onehot - p) by (y_onehot - q)
centered = [0.0, 0.0, 0.0]          # average of -A * (y_onehot - q) over y ~ q
```

**Table 1. Logit gradient for one sample (token 0 written, A = 1).**

| Loss | Token 0 | Token 1 | Token 2 |
|---|---|---|---|
| REINFORCE, `-A * (y_onehot - p)` | -0.494 | 0.307 | 0.186 |
| Score centering, `-A * (y_onehot - q)` | -0.800 | 0.300 | 0.500 |

The table's second row comes from running miles's loss formula (below) through PyTorch autograd with all three
tokens in the candidate list and weights of 1; it matches `-A * (y_onehot - q)` exactly. The
`y_onehot - q` form holds only in that case. With importance weights or a partial candidate list, the
correction is the general formula in section 2.

## 2. The loss in miles

```
loss = -A * ( f(r) * log p[token]  -  Σ_k in H  c_k * log p[k] )      all weights detached
c_k  = q[k] * f(p[k] / q[k])  -  alpha * p[k]
```

- `H` is the list of candidates the engine records per token: its top 128 (`--rollout-top-logprobs-num 128`)
  when sampling from the full vocabulary, or the whole top-k/top-p support when sampling is filtered.
- `f` is the importance weight: `none` gives `f = 1`, `tis` gives `min(r, 2)`, `mis` keeps r inside `[0.5, 5]`
  and sets it to 0 outside (`--score-centering-is`).
- `alpha` rescales the trainer's probability outside `H`: the engine's tail is modeled as
  `rho * p` with `rho = (1 - Σ q[H]) / (1 - Σ p[H])`, and `alpha = rho * f(1 / rho)`.

Code (miles main 34914c948f, `miles/backends/training_utils/loss/hub/score_centering/`):
- `estimator.py:64-95`: `score_centering_loss`; the loss line is 88.
- `importance_sampling.py:20-76`: the three `f` options.
- Enabled with `--loss-type score_centering` (`miles/backends/training_utils/loss/hub/losses.py:522-523`). It is its own REINFORCE-style loss: the PPO
  clip does not apply, and it replaces `--use-tis`. It also turns on an FP32 LM head in the engine
  (`miles/backends/sglang_utils/arguments.py:192-193`) so `q` is not rounded.

The full recipe and data contract are in miles `examples/infra_features/score_centering/README.md`.

## 3. In our runs

Not used. It needs the engine to return 128 candidate log-probs per generated token (about 1 KiB per position,
per the miles README) and, per that README, is documented for text-only GRPO recipes: multimodal token expansion is not supported.
