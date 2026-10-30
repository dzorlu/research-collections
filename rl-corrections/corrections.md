# Policy-gradient corrections in miles: what reaches the logits

Training changes the weights through one quantity: the gradient on the logits, `dz`. Every method here (PPO,
CISPO, TIS, GSPO, score centering, ...) is a different choice for one of its four terms.

```python
dz = w * A * (y_hat - y_onehot) / normalization          # (T, V) for one response: row t is token t's gradient
```

The methods fix three problems:

1. **Staleness.** The trainer took several optimizer steps on one rollout, or the samples are from older weights, so
   the model being trained is no longer the one that sampled the tokens.
2. **Mismatch.** The inference engine (SGLang) sampled the tokens. With the same weights, the engine and the
   trainer compute slightly different probabilities (other kernels, other precision).
3. **Drift.** Because the engine samples (from `q`) but the update subtracts the trainer's `y_hat`, each update also
   carries `A_avg * (y_hat - q)`: a pull toward the engine where a state's average advantage `A_avg` is positive, a
   push away where it is negative. It has nothing to do with which token earned the reward (section 4).

Runnable versions: [corrections.ipynb](corrections.ipynb) (every method on one toy batch, PyTorch) and
[viz/clipping_network.html](viz/clipping_network.html) (interactive: watch the weights change, see which tokens
are dropped).

## 1. How to calculate dz

Shapes: `B` batch (responses), `T` tokens per response, `V` vocabulary size.

```python
z        = trainer logits                                   # (B, T, V)
y        = sampled tokens (chosen by the inference engine)  # (B, T)

y_hat    = torch.softmax(z, -1)                             # (B, T, V)  trainer probabilities
y_onehot = torch.nn.functional.one_hot(y, V)                # (B, T, V)
w        = ...                                              # (B, T)     section 2: how strongly each token is trained
A        = ...                                              # (B, 1)     section 3: how good each response was
norm     = ...                                              # (B, T)     section 5: how the token losses are averaged

dz = (w * A / norm).reshape(B, T, 1) * (y_hat - y_onehot)   # (B, T, V)
```

Backprop then carries `dz` down through the network the same way for every method.

**Table 1. The four decisions.**

| Term | Decision | Options | Section |
|---|---|---|---|
| `w` | how strongly each token is trained | REINFORCE, PPO, CISPO, TIS, MIS, IcePop, decoupled PPO, OPSM, GSPO | 2 |
| `A` | how good each response (or token) was | REINFORCE, GRPO, Dr. GRPO, PPO with a value model, REINFORCE++ | 3 |
| `y_hat - y_onehot` | which direction to push | REINFORCE direction, or score centering's `q - y_onehot` | 4 |
| `norm` | how token losses are averaged | per token, per response (miles default), GRPO, Dr. GRPO | 5 |

Two ratios decide `w`. Both are per token, `(B, T)`:

```python
r   = p_now / p_old        # (B, T)  staleness: trainer now vs trainer when the token was sampled
rho = p_old / p_engine     # (B, T)  mismatch: trainer vs engine, same weights
```

The examples use the toy batch from the notebook: two responses of four tokens. Response 0 was good (`A = +1`),
response 1 was bad (`A = -1`).

## 2. `w`: how strongly each token is trained

### 2.1 REINFORCE

```python
w = torch.ones(B, T)                                        # (B, T)
```

No correction: every token at full strength.

Loss: notebook `reinforce`.

### 2.2 PPO (and GRPO)

Fixes staleness. Matters when one rollout gets several optimizer steps (mini-batches) or the samples are from older
weights. On the first step `r = 1`.

```python
w = r.clone()                    # (B, T)
w[(A > 0) & (r > 1.28)] = 0      # (B, T)  already moved far enough up: stop pushing
w[(A < 0) & (r < 0.8)]  = 0      # (B, T)  already moved far enough down: stop pushing
```

A token that became 1.5x more likely counts 1.5x. Once it has moved past the band in the direction `A` wants, it
gets no gradient. This keeps each update close to the model that generated the data (a trust region). On the other
side `r` is not limited.

Example: response 1 is bad (`A = -1`). Token 3 has `r = 0.21`, already 5x less likely: `w = 0`. Token 2 has
`r = 2.95`, it became more likely: `w = 2.95`, uncapped.

Loss: notebook `ppo`; miles `math_utils.py:255` (band `[1 - eps_clip, 1 + eps_clip_high]`, our recipes 0.8 and 1.28).

### 2.3 CISPO

Fixes staleness without dropping tokens.

```python
w = torch.clamp(r, 0, 4)         # (B, T)
```

Never 0; at most 4. The paper (MiniMax-M1, arXiv 2506.13585) "abandons the trust region constraint and instead
clips the importance sampling weights".

Example: response 1, token 3 (`r = 0.21`). PPO gives `w = 0`; CISPO gives `w = 0.21`.

miles's version divides by the engine instead: `w = clamp(p_now / p_engine, 0, 4)`, so it also reacts to mismatch
(same token: `w = 0.06`).

Loss: notebook `cispo_paper`, `cispo_miles`; miles `tinker_losses.py:192` (Tinker API only).

### 2.4 TIS (truncated importance sampling)

Fixes mismatch by capping.

```python
c = torch.clamp(rho, 0, 2)       # (B, T)  importance weight
w = c * w_ppo                    # (B, T)  w_ppo: PPO's w (2.2); 1 with one optimizer step
```

Each token is weighted by how much more likely the trainer finds it than the engine did, capped at 2.

Example: response 1, token 4 has `rho = 2.31` (the engine gave it less than half the trainer's probability):
`c = 2`.

Loss: notebook `tis`; miles `corrections.py:7` (`--use-tis`).

### 2.5 MIS (masked importance sampling)

Fixes mismatch by dropping.

```python
c = torch.where((rho >= 0.5) & (rho <= 2), rho, 0)    # (B, T)
w = c * w_ppo                                         # (B, T)
```

A token on which engine and trainer disagree by more than 2x either way gets `c = 0`: dropped.

Example: response 1, token 3 (`rho = 0.29`) and token 4 (`rho = 2.31`): both `c = 0`.

Loss: notebook `mis`; miles `examples/infra_features/train_infer_mismatch_helper/mis.py` (`tis_mode: mask`).

### 2.6 IcePop

Fixes mismatch by dropping, both sides. From the Ring-1T paper (arXiv 2510.18855, section 2.3.2).

```python
c = torch.where((rho >= 0.5) & (rho <= 5), rho, 0)    # (B, T)  paper: alpha = 0.5, beta = 5
w = c * w_ppo                                         # (B, T)
```

Why: in MoE models the engine and the trainer disagree more, because small numeric differences change which
experts a token is routed to, and the paper shows the disagreement compounds over training steps. A token on which
they disagree a lot gives an unreliable gradient. TIS still trains on it (capped at 2); IcePop drops it. The paper
reports IcePop beat TIS on AIME25 (Ring-mini-2.0) and kept gradient norms and the engine/trainer gap from growing
on Ring-1T.

IcePop and token-level MIS in mask mode give the same `dz`: IcePop sets the token's importance weight to 0, MIS
removes the token from the loss mask (miles keeps it in the denominator, so the result is the same). They differ in
the rest: MIS (miles `mis.py`) can also weight per sequence, rescale the weights to mean 1 across the batch, and
veto whole sequences, and its example config runs truncation, not masking. miles's `icepop_function` reads its band
from `--tis-clip-low` and `--tis-clip` (defaults 0 and 2, so by default it only drops above 2); the paper's band is
`--tis-clip-low 0.5 --tis-clip 5`.

Example: response 1, token 3 (`rho = 0.29`, below 0.5): `c = 0`. Token 4 (`rho = 2.31`, inside the band): `c = 2.31`.

Loss: notebook `icepop` (miles default band); miles `corrections.py:35`.

### 2.7 Decoupled PPO

Fixes staleness and mismatch, each with its own ratio.

```python
w = torch.clamp(rho, 0, 2) * w_ppo     # (B, T)  same as TIS
```

PPO's clip on `r` keeps the update near the trainer's own model; the importance weight on `rho` corrects for the
engine. miles uses it when `--use-tis` is on and `--use-rollout-logprobs` is off.

Loss: notebook `tis`.

### 2.8 OPSM (off-policy sequence masking)

Fixes staleness, one decision per response.

```python
moved = (log p_old - log p_now).mean over the response's tokens     # (B, 1)  how far the model moved from it
w = w_ppo.clone()                                                  # (B, T)
w[(A < 0) & (moved > 1e-4)] = 0                                    # (B, T)  the whole response
```

A bad response the model has already moved away from is dropped entirely.

Example: response 1 is bad and the model moved away from it by 0.27: all four tokens `w = 0`.

Loss: notebook `opsm`; miles `math_utils.py:184` (`--use-opsm`).

### 2.9 GSPO (group sequence policy optimization)

Fixes staleness, one ratio per response.

```python
s = exp((log p_now - log p_old).mean over the response's tokens)    # (B, 1), geometric mean of r
w = s.expand(B, T).clone()                                          # (B, T)  every token of the response gets s
w[((A > 0) & (s > 1.28)) | ((A < 0) & (s < 0.8))] = 0               # (B, T)  PPO's rule, applied to s
```

The whole response is kept or clipped together, so one noisy token cannot be clipped alone.

Example: response 1 (`A = -1`) has `s = 0.77 < 0.8`: all four tokens `w = 0`. Response 0 (`A = +1`) has
`s = 0.64`: every token `w = 0.64`.

Loss: notebook `gspo`; miles `math_utils.py:225` (`--advantage-estimator gspo`).

## 3. `A`: how good each response was

`R` = each response's reward, `(G,)` for the `G` rollouts of one prompt. miles picks the estimator with
`--advantage-estimator` (`advantages.py:16`).

```python
A = R                              # (G,)  REINFORCE: the raw reward
A = (R - R.mean()) / R.std()       # (G,)  GRPO: compared with the other rollouts of the same prompt
A = R - R.mean()                   # (G,)  Dr. GRPO (miles --disable-grpo-std-normalization): no / std
```

`A` is copied to every token of the response. Per-token estimators give each token its own `A`, `(B, T)`: PPO with
a learned value model (GAE), REINFORCE++ (discounted token rewards), REINFORCE++ baseline (group-centered reward
minus the token's KL to the reference model).

Example: rewards `[1, 0, 0, 0]`. GRPO: `A = [+1.5, -0.5, -0.5, -0.5]`. Dr. GRPO: `[+0.75, -0.25, -0.25, -0.25]`.

## 4. `y_hat - y_onehot`: the direction

Every method above keeps it. Score centering changes it. Full derivation with numbers:
[score_centering.md](score_centering.md).

At one state the engine picks token `k` with probability `q[k]`, so the score `onehot(k) - y_hat` averages to
`s_bar = q - y_hat`, not 0. Each update therefore carries drift, `A_avg * (y_hat - q)`, where `A_avg` is the
advantage averaged over the engine's picks: a pull toward the engine when `A_avg > 0`, a push away when `A_avg < 0`.
Score centering subtracts `s_bar` from every pick's score, which removes one term from `dz`:

```python
dz = A[k] * (y_hat - onehot(k))                       # (V,)  plain
dz = A[k] * (y_hat - onehot(k)) - A[k] * (y_hat - q)  # (V,)  score centering: subtract A[k] * (y_hat - q)
   = A[k] * (q - onehot(k))                           # (V,)
dz = (w * A / norm).reshape(B, T, 1) * (q - y_onehot) # (B, T, V)  full form, w = 1
```

The centered score `onehot(k) - q` averages to 0 under the engine's picks, so the drift is 0 and only the signal
(which token was better than the state's average) is left.

Example from [score_centering.md](score_centering.md): `y_hat = [0.5, 0.3, 0.2]`, `q = [0.6, 0.3, 0.1]`,
`A = [0.75, -0.25, -0.25]`, so `A_avg = 0.35`. Plain REINFORCE's average `dz` is `[-0.275, 0.18, 0.095]` = drift
`[-0.035, 0, 0.035]` + signal `[-0.24, 0.18, 0.06]`. With score centering it is `[-0.24, 0.18, 0.06]`.

miles has only the engine's top candidates per token (`--rollout-top-logprobs-num`, e.g. 128) and models the rest,
so `q - y_onehot` is exact only when those cover the vocabulary.

Loss: notebook `score_centering_check`; miles `score_centering/estimator.py:66` (`--loss-type score_centering`).

## 5. `norm`: how token losses are averaged

The loss averages the token losses; the backward pass gives each token the factor it had in that average.
`T_i` = length of the token's response, `G` = rollouts per prompt, `L_max` = maximum length.

```python
norm = total number of tokens    # ()      per token (DAPO, CISPO paper, miles --calculate-per-token-loss)
norm = T_i                       # (B, T)  per response, summed (miles default)
norm = G * T_i                   # (B, T)  per response, averaged (GRPO paper)
norm = G * L_max                 # ()      a constant (Dr. GRPO)
```

Example from [normalization.ipynb](normalization.ipynb): a 3-token response and a 1-token response, 4 tokens.

| Method | `norm` per token | 1-token response vs each token of the 3-token response |
|---|---|---|
| per token | 4, 4, 4, 4 | same |
| miles default | 3, 3, 3, 1 | 3x larger `dz` |
| GRPO | 6, 6, 6, 2 | 3x larger `dz` |
| Dr. GRPO | 8, 8, 8, 8 | same |

With `norm = T_i`, each token of a short response gets a larger `dz`: short correct responses are pushed up harder
per token, long wrong ones are pushed down less. With a constant `norm`, every token gets the same `dz`, and a long
response contributes more in total.

## 6. In our runs

- `w`: one optimizer step per rollout, so `r = 1` and PPO, CISPO (paper), OPSM and GSPO change nothing. Our
  GLM-5.3 recipe (torchtitan) uses TIS (`w = clamp(rho, 0, 2)`); our Qwen3.8 recipes use
  PPO with `r` about 1, so `w` is about 1.
- `A`: GRPO with `/ std`.
- Direction: `y_hat - y_onehot` (no score centering).
- `norm`: the miles default, `T_i`. The Muse recipe uses per token.

## 7. Removing the mismatch at the source

The methods above correct the mismatch after the fact. The other route is to make the trainer compute exactly the
engine's probabilities, so `rho = 1` and TIS, MIS and IcePop have nothing to do. The vLLM team does this by having
the training forward pass call the inference engine's own kernels, with training adding only the backward pass
([vLLM blog, 2026-10-14](https://vllm-project.github.io/2026/10/14/train-inference-consistency.html)).

Sources of mismatch they remove: summation order in split-K matrix multiplies, kernels chosen by batch size (a
token's numbers depend on its batch-mates), different attention kernels for prefill and decode, cross-GPU reduction
order, and rounding differences that discrete choices (Top-K ties, MoE routing) amplify.

Reported results: on DeepSeek-V4-Flash (32 GB200), the log-prob difference is 0 at every step at about 91% of the
baseline's rollout throughput; the uncorrected baseline's train-inference KL rose from 2e-3 to 29.9 while reward fell
back. On Nemotron-3.5 agentic RL (SWE-bench Pro), the consistent run's score went 0.56 to 0.59 at 87.5% throughput;
the baseline collapsed to 0.22. Their view: algorithmic corrections "reduce the effect of the mismatch but do not
remove it", and the two stack. They note that zero difference is "not a sufficient condition for training quality"
and that neither baseline was tuned or used TIS/MIS.

Code: miles = radixark/miles main at 7e7a446815.
