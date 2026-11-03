# Bias and variance of the policy gradient

## 1. The question

RL for language models works like this: sample responses, score them, and nudge the model toward the good ones. The
nudge is computed from a sampled batch, so it is only an estimate of the gradient we want. And the samples usually
come from a different copy of the model than the one being trained: a separate inference engine, fast but a few
updates behind and numerically slightly different [[1]](#ref-1)[[2]](#ref-2).

Methods like PPO, importance sampling, TIS and DPPO each change that estimate. This post asks: how far is the
estimate from the gradient we want, and what does each method do to the gap?

## 2. The update

The model turns logits into probabilities `p` with a softmax. When a token `y` is written and gets advantage `A`
(how much better than usual the response was), the policy gradient on the logits is [[3]](#ref-3):

```python
dlogits = w * A * (p - y_onehot)    # [V]  one written token; V = vocabulary size
```

`y_onehot` is 1 at the written token and 0 elsewhere. `p - y_onehot` is the gradient of `-log p[y]` for a softmax. `w` is a per-token weight: 1 in plain REINFORCE, and the knob
every correction method turns. This is the gradient of the loss, so training subtracts it: a negative entry pushes
that token's logit up.

The two inputs have separate jobs: `y_onehot` says which token was written, `A` says how good that was. The reward
enters only through `A`.

For a batch of `B` responses of `S` tokens each, average over all `B * S` tokens.

## 3. The true gradient

The goal is the model's average reward. A toy with one position and three tokens:

```python
p = [0.5, 0.3, 0.2]               # [V]  how often the model writes each token
# reward: 1 if the model writes token 2, else 0
J = 0.5*0 + 0.3*0 + 0.2*1 = 0.2   # average reward over everything the model could write
```

The true gradient says how `J` changes when each logit changes. Find it by nudging one logit at a time by 0.01 and
watching `J`:

```python
nudge logit 0:  p = [0.5025, 0.2985, 0.1990]  ->  J = 0.1990,  slope (0.1990 - 0.2) / 0.01 = -0.10
nudge logit 1:  p = [0.4985, 0.3021, 0.1994]  ->  J = 0.1994,  slope -0.06
nudge logit 2:  p = [0.4990, 0.2994, 0.2016]  ->  J = 0.2016,  slope +0.16
```

Raising logit 2 raises `J`, because token 2 is the rewarded one. Raising the others steals probability from token 2.
So the gradient of `J` is `[-0.10, -0.06, +0.16]`. The update in section 2 is written as a loss to minimize, which
flips the sign: the true `dlogits` is `[0.10, 0.06, -0.16]`.

The same numbers come from averaging section 2's update over what the model writes (with its sign flipped back to
`onehot - p`). Each token contributes how often it is written, times its `A`, times `onehot - p`. With no baseline,
`A` is just the reward:

```python
#           how often   A        onehot - p
token 0:    0.5       * 0      * [ 0.5, -0.3, -0.2]  =  [ 0.00,  0.00, 0.00]
token 1:    0.3       * 0      * [-0.5,  0.7, -0.2]  =  [ 0.00,  0.00, 0.00]
token 2:    0.2       * 1      * [-0.5, -0.3,  0.8]  =  [-0.10, -0.06, 0.16]
                                                 sum =  [-0.10, -0.06, 0.16]    # the gradient of J again
```

This second form is the policy gradient identity: the gradient of an average reward equals the average, over what
the model writes, of reward times the gradient of `log p[y]` [[3]](#ref-3)[[4]](#ref-4). For a softmax that gradient is
`onehot - p`. It is an average over every response the model could write. With three responses we can add them all
up. A real model has `V**S` possible responses, far more than atoms in the universe, so training samples a
batch of `B` and averages instead. The rest of this post follows one number, token 2's `-0.16`, and asks how the
batch's version of it behaves.

## 4. One sample, one guess

One position (`S = 1`), no baseline (`A` = reward), `w = 1`. Each sample adds one of two values to token 2's entry:

```python
y = 2:      A = 1  ->  1 * (0.2 - 1) = -0.8    # written 20% of the time
y = 0 or 1: A = 0  ->  0                       # written 80% of the time
average = 0.2 * -0.8 = -0.16                   # the true gradient
```

The batch's estimate is the average of its `B` samples.

Two numbers describe the estimate:

- **Bias**: its average over many batches, minus the true value. Bias does not shrink with more samples.
- **Variance**: how far one sample lands from the average, squared and averaged. A batch of `B` divides it by `B`.

```python
error = bias**2 + variance / B
```

This split of the mean squared error is standard [[5]](#ref-5). Averaging `B` independent samples divides the variance by
`B` and leaves the bias alone.

## 5. Baseline: less variance, no bias

Use `A = R - b`, where `R` is the reward (0 or 1), with `b = 0.2`, the average reward. The samples change, their average does not: over the
model's own samples, `p - y_onehot` averages to 0, so subtracting `b` times it subtracts 0. This is why baselines are free [[3]](#ref-3)[[6]](#ref-6); Greensmith et al. also derive the
baseline that minimizes variance [[6]](#ref-6).

![baseline](figs/bv_baseline.png)

*Figure 1: 20,000 batches of 16, model sampling itself. Both baselines center on the true value (no bias). With
`b = 0.2` the spread shrinks from 0.102 to 0.058 per sample.*

GRPO uses the group's mean reward as `b` [[7]](#ref-7). Its other two steps, dividing by the group's reward std and by
response length, do add bias [[8]](#ref-8).

## 6. When the engine writes the tokens

Now `y` comes from the engine, so tokens show up at the engine's rates, not the model's. We still want the model's
`-0.16`. Two engines, `b = 0`:

```python
p     = [0.5,  0.3,  0.2 ]    # [V]  model
close = [0.6,  0.3,  0.1 ]    # [V]  engine writes token 2 half as often as the model
far   = [0.69, 0.30, 0.01]    # [V]  engine almost never writes token 2
```

### 6.1 Two distances between engine and model

**Total variation (TV): how much probability moved.**

```python
p - close = [-0.1, 0.0, +0.1]    # take 0.1 from token 0, give it to token 2
```

`sum(|p - close|)` is 0.2, but it counts every move twice, once where it leaves and once where it lands. So the
amount moved is `TV = 0.5 * sum(|p - p_engine|)`: 0.10 for close, 0.19 for far. This is the total variation
distance [[9]](#ref-9).

**Chi-squared: how big the weights get.** The weight that undoes the mismatch is `w = p / p_engine`, the importance
weight [[10]](#ref-10)[[12]](#ref-12):

```python
w_close = [0.83, 1.0,  2.0]    # [V]
w_far   = [0.72, 1.0, 20.0]    # [V]
```

Under the engine the weight always averages to 1: each term `p_engine[k] * (p[k] / p_engine[k])` is just `p[k]`. So
the weight's variance is its average square minus 1:

```python
close: 0.6*0.83**2  + 0.3*1**2 + 0.1*2**2   - 1 = 0.12
far:   0.69*0.72**2 + 0.3*1**2 + 0.01*20**2 - 1 = 3.66
```

That is the chi-squared distance; the second moment of the importance weight is exactly `1 + chi-squared`, and it
controls the error of importance sampling [[11]](#ref-11). Each token adds `p**2 / p_engine`, so a token the engine rarely writes blows it up.

![distance](figs/bv_distance.png)

*Figure 2: engine `[0.7 - e, 0.3, e]` as `e`, the engine's probability of token 2, falls from 0.2 (same as the
model) to 0.005. TV levels off at 0.2. Chi-squared explodes.*

### 6.2 Three weights

Each sample now adds `-0.8 * w[2]` if it wrote token 2, and 0 otherwise.

- **REINFORCE**, `w = 1`: token 2 shows up at the engine's rate, so the average is `-0.8 * 0.01 = -0.008` with the
  far engine. Bias `+0.152`: it barely pushes token 2.
- **IS**, `w = p / p_engine = 20` on token 2: the average is `-0.8 * 20 * 0.01 = -0.16`, exact. But 85% of batches
  of 16 never see token 2 and estimate 0; the 14% that see it once estimate `-1.0`.
- **TIS**, `w` capped at 2: in between, mostly bias. Truncating weights to trade variance for bias goes back to
  Ionides [[13]](#ref-13); for LLM training it was proposed against engine mismatch [[1]](#ref-1).

![weights](figs/bv_weights_far.png)

*Figure 3: the far engine. Left: 300 single batches of 16. Every REINFORCE and TIS batch lands near 0; most IS batches
land at 0 and a few at -1 or -2. Right: average more and more batches. REINFORCE and TIS settle near 0, the wrong
place (bias). IS swings for hundreds of batches (variance) before settling on the true -0.16.*

### 6.3 Every method, both engines

Each method chooses `w`. Here are their errors at `B = 16`, split into bias² and variance:

![methods](figs/bv_methods.png)

*Figure 4: squared error of the batch estimate at `B = 16`. Close engine (left): IS, TIS and DPPO coincide (the cap
of 2 does not bind, and DPPO passes since only 0.10 moved). Far engine (right): IS is mostly variance, the rest are
mostly bias.*

Table 1: the numbers behind Figure 4 (`b = 0`, per-sample variance, error at `B = 16`).

| engine | method | `w` on token 2 | bias | variance | error |
|---|---|---|---|---|---|
| close | REINFORCE | 1 | +0.080 | 0.058 | 0.010 |
| close | IS, TIS, DPPO | 2 | 0 | 0.230 | 0.014 |
| close | PPO (ratio to engine) | 0 | +0.160 | 0 | 0.026 |
| far | REINFORCE | 1 | +0.152 | 0.006 | 0.024 |
| far | IS | 20 | 0 | 2.534 | 0.158 |
| far | TIS | 2 | +0.144 | 0.025 | 0.022 |
| far | PPO, DPPO | 0 | +0.160 | 0 | 0.026 |

Three things to take from this:

1. **Unbiased is not the same as better.** With the close engine, plain REINFORCE beats IS at `B = 16`.
2. **The trust-region methods look worst here, by design.** PPO [[14]](#ref-14) (with its ratio taken against the engine)
   and DPPO [[2]](#ref-2) set `w = 0` on token 2, so they estimate 0. They are not trying to
   estimate one gradient well. They stop the step on tokens that already moved too far. That benefit only shows up
   across steps (section 7).
3. **PPO and DPPO disagree with the close engine.** Token 2's ratio is 2, so PPO blocks it. Only 0.10 of
   probability moved, under DPPO's 0.15, so DPPO keeps it.

### 6.4 Where to put the cap

A cap trades variance for bias. Where the best trade sits depends on the batch size:

![cap](figs/bv_cap.png)

*Figure 5: far engine, error versus the cap on `w`. At `B = 16` the best cap is about 3. At `B = 256` the variance is
16 times smaller, so the best cap rises to about 14.*

Common TIS implementations cap at 2 [[1]](#ref-1). The same trade appears in classic RL as V-trace [[15]](#ref-15) and
Retrace [[16]](#ref-16).

## 7. Across steps: the blind spot and the trust region

So far, one estimate. Training takes steps, and a step can hide a second kind of bias.

`w` fixes which token is written at a prefix. It cannot fix how often each prefix is reached. Two tokens; reward 1 if
the second token is `b`. The engine writes the first token 50/50, and after `a` it writes `b` 10% of the time. A step
moves the first token toward `a` by `d`, and "after `a`" toward `b` by `d`.

Table 2: the gain the weighted batch predicts (surrogate) vs the true gain in average reward.

| `d` | true gain | surrogate | gap |
|---|---|---|---|
| 0.01 | 0.0011 | 0.0010 | 0.0001 |
| 0.1 | 0.0200 | 0.0100 | 0.0100 |
| 0.3 | 0.1200 | 0.0300 | 0.0900 |

The stepped model reaches "after `a`" more often, and it also improved there. The batch was written by the engine and
never saw that. This gap is classic: the performance difference lemma [[17]](#ref-17) writes the true gain exactly, and TRPO
[[18]](#ref-18)[[19]](#ref-19) bounds the gap by the TV between the two policies. The DPPO paper adapts the bound to LLM responses
(Theorem 3.2 in [[2]](#ref-2)): the gap is at most a constant times the TV summed over the response. So keep TV small at every token. That is a trust region,
and it is what the zeros in Figure 4 pay for.

PPO measures movement with the ratio, `|r - 1|`. DPPO measures `|p - p_engine|`, which equals `p_engine * |r - 1|`,
the token's actual share of TV.

Table 3: PPO and DPPO on a rare token with `A > 0` and a confident token with `A < 0`.

| token | `p_engine` → `p` | `r` | moved | PPO | DPPO (`delta` 0.15) |
|---|---|---|---|---|---|
| rare | 0.0001 → 0.01 | 100 | 0.0099 | `w = 0` | `w = 100` |
| confident | 0.99 → 0.80 | 0.81 | 0.19 | `w = 0.81` | `w = 0` |

PPO blocks a token that barely moved and passes one that moved a lot. In the paper, collapse came from the second
kind (Figure 5 in [[2]](#ref-2)), and the mean `|p - p_engine|` rose before reward fell (Figure 3 in [[2]](#ref-2)).

## 8. Summary

Table 4: what each choice of `w` does.

| method | `w` | bias | variance |
|---|---|---|---|
| REINFORCE | 1 | grows with TV | low |
| IS | `p / p_engine` | none | grows with chi-squared |
| TIS | IS, capped | in between; best cap grows with `B` | in between |
| PPO, DPPO | 0 on tokens that moved too far | adds bias on those tokens | low |

TV is how much probability moved; ignoring it costs bias. Chi-squared is how big the weights get; correcting for it
costs variance. A trust region adds a little bias now to keep the next step's bias (the blind spot) small.

Next: the same measurement on a real training run, per method, instead of a toy.

Figures are generated by `bias_variance_figs.py`; every number above was checked in PyTorch.

## References

1. <a id="ref-1"></a>F. Yao, L. Liu, D. Zhang, C. Dong, J. Shang, J. Gao. *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training.* Blog, 2025. https://fengyao.notion.site/off-policy-rl
2. <a id="ref-2"></a>P. Qi, X. Zhou, Z. Liu, T. Pang, C. Du, M. Lin, W. S. Lee. *Rethinking the Trust Region in LLM Reinforcement Learning* (DPPO). arXiv:2602.04879, 2026.
3. <a id="ref-3"></a>R. J. Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE). Machine Learning 8:229-256, 1992.
4. <a id="ref-4"></a>R. S. Sutton, D. McAllester, S. Singh, Y. Mansour. *Policy Gradient Methods for Reinforcement Learning with Function Approximation.* NeurIPS, 2000.
5. <a id="ref-5"></a>L. Wasserman. *All of Statistics.* Springer, 2004 (bias-variance decomposition of mean squared error).
6. <a id="ref-6"></a>E. Greensmith, P. L. Bartlett, J. Baxter. *Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.* JMLR 5:1471-1530, 2004.
7. <a id="ref-7"></a>Z. Shao et al. *DeepSeekMath* (GRPO). arXiv:2402.03300, 2024.
8. <a id="ref-8"></a>Z. Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO). arXiv:2503.20783, 2025.
9. <a id="ref-9"></a>D. A. Levin, Y. Peres, E. L. Wilmer. *Markov Chains and Mixing Times.* AMS, 2009 (total variation distance, chapter 4).
10. <a id="ref-10"></a>A. B. Owen. *Monte Carlo Theory, Methods and Examples*, chapter 9: Importance Sampling. 2013. https://artowen.su.domains/mc/
11. <a id="ref-11"></a>S. Agapiou, O. Papaspiliopoulos, D. Sanz-Alonso, A. M. Stuart. *Importance Sampling: Intrinsic Dimension and Computational Cost.* Statistical Science, 2017; arXiv:1511.06196 (eq. 2.4: second moment of the weight minus 1 is the chi-squared divergence).
12. <a id="ref-12"></a>T. Degris, M. White, R. S. Sutton. *Off-Policy Actor-Critic.* ICML, 2012; arXiv:1205.4839.
13. <a id="ref-13"></a>E. L. Ionides. *Truncated Importance Sampling.* Journal of Computational and Graphical Statistics 17(2):295-311, 2008.
14. <a id="ref-14"></a>J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
15. <a id="ref-15"></a>L. Espeholt et al. *IMPALA* (V-trace). arXiv:1802.01561, 2018.
16. <a id="ref-16"></a>R. Munos, T. Stepleton, A. Harutyunyan, M. G. Bellemare. *Safe and Efficient Off-Policy Reinforcement Learning* (Retrace). arXiv:1606.02647, 2016.
17. <a id="ref-17"></a>S. Kakade, J. Langford. *Approximately Optimal Approximate Reinforcement Learning.* ICML, 2002.
18. <a id="ref-18"></a>J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel. *Trust Region Policy Optimization.* arXiv:1502.05477, 2015.
19. <a id="ref-19"></a>J. Achiam, D. Held, A. Tamar, P. Abbeel. *Constrained Policy Optimization.* arXiv:1705.10528, 2017.
