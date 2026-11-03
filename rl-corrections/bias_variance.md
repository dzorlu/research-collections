# Bias and variance of the policy gradient

## 1. The question

RL for language models estimates the policy gradient from a sampled batch. The tokens are usually written by an
inference engine rather than the model being trained: a separate copy that is a few updates behind and numerically
slightly different [[1]](#ref-1)[[2]](#ref-2). Importance sampling (IS), truncated IS (TIS), PPO and DPPO each modify
the estimator to cope with that mismatch. This post measures, on a toy where everything is exact, how far each
modified estimator lands from the true gradient, split into bias and variance.

## 2. The update

Let $z \in \mathbb{R}^V$ be the logits at one position, $p = \operatorname{softmax}(z)$ the trained model's
distribution, and $p_{\text{engine}}$ the distribution of the engine that samples. For a sampled token $y$ with
one-hot $e_y$, advantage $A$ and per-token weight $w$, the policy-gradient update on the logits is [[3]](#ref-3)

$$\text{dlogits} = w\,A\,(p - e_y).$$

This is a loss gradient ($p - e_y = \nabla_z[-\log p_y]$), so training subtracts it. The reward enters only through
$A$. Plain REINFORCE has $w = 1$; every correction method below is a choice of $w$. A batch of $B$ responses of $S$
tokens averages this over all $BS$ tokens.

## 3. The true gradient

The true gradient is the update from section 2, averaged over which token gets written, each weighted by how often
the model writes it. By the policy gradient identity this equals $-\nabla_z J$, the gradient of the average reward
in loss sign [[3]](#ref-3)[[4]](#ref-4):

$$g = \sum_y p_y \cdot r_y\,(p - e_y) \qquad (w = 1,\ A = r_y).$$

Every method below is an estimate of $g$. The toy has one position, three tokens, and only token 2 is rewarded:

```python
p = [0.5, 0.3, 0.2]    # model
r = [0, 0, 1]          # reward if token k is written
```

| $y$ | $p_y$ | $A = r_y$ | $p - e_y$ | $p_y \cdot A \cdot (p - e_y)$ |
|---|---|---|---|---|
| 0 | 0.5 | 0 | (−0.5, 0.3, 0.2) | (0, 0, 0) |
| 1 | 0.3 | 0 | (0.5, −0.7, 0.2) | (0, 0, 0) |
| 2 | 0.2 | 1 | (0.5, 0.3, −0.8) | (0.10, 0.06, −0.16) |
| | | | **sum** | **$g$ = (0.10, 0.06, −0.16)** |

The weight $p_y$ is how often the model itself writes each token. That is what breaks when another model writes the
tokens (section 6). A real model can't sum over all $V^S$ responses, so it averages $B$ samples instead.

## 4. Bias and variance

Call the update of one sampled token a row, $x = w\,A\,(p - e_y)$, and the average of $B$ rows $\hat g_B$. In
section 3's table only the $y = 2$ row is nonzero. Call it $v$:

$$\begin{gathered}
v = p - e_2 = (0.5,\ 0.3,\ -0.8), \qquad g = 0.2\,v = (0.10,\ 0.06,\ -0.16), \\
\|v\|^2 = 0.25 + 0.09 + 0.64 = 0.98.
\end{gathered}$$

With $q$ the probability that a sampled token is 2, one row is $x = v$ with probability $q$ and $0$ otherwise:
a Bernoulli($q$) scaled by $v$. A batch average is $v$ times the fraction of 2s it drew.

The model writes token 2 at $q = p_2 = 0.2$. The close engine of section 6 writes it half as often,
$p_{\text{engine},2} = 0.1$. Draw many batches of 16 from each:

![model vs engine batches](figs/bv_bias_scatter.png)

*Figure 1: where batches of 16 land, 20,000 from each sampler. Token 2's coordinate of $\hat g_B$; dashed is $g$.*

The spread averages out. The engine's offset does not.

**Error.** The error of the estimator is the mean squared miss over batches, $\mathbb{E}\,\|\hat g_B - g\|^2$. It
decomposes into bias and variance [[5]](#ref-5):

$$\mathbb{E}\,\|\hat g_B - g\|^2 = \underbrace{\|\mathbb{E}[\hat g_B] - g\|^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\,\|\hat g_B - \mathbb{E}[\hat g_B]\|^2}_{\text{variance}}.$$

:::note bias–variance decomposition
For any estimator $\hat\theta$ of a target $\theta$, with $m = \mathbb{E}[\hat\theta]$:

$$\hat\theta - \theta = (\hat\theta - m) + (m - \theta).$$

Square both sides, $\|a + b\|^2 = \|a\|^2 + 2\,a\cdot b + \|b\|^2$, and take expectations. The constant
$m - \theta$ comes out of the cross term, and what is left of it is zero:

$$\mathbb{E}\big[2\,(\hat\theta - m)\cdot(m - \theta)\big] = 2\,(m - \theta)\cdot\big(\mathbb{E}[\hat\theta] - m\big) = 0.$$

So

$$\mathbb{E}\,\|\hat\theta - \theta\|^2 = \underbrace{\mathbb{E}\,\|\hat\theta - m\|^2}_{\text{variance}} + \underbrace{\|m - \theta\|^2}_{\text{bias}^2}.$$

Bias is systematic: the estimator's average is off target, and more samples do not move it. Variance is random
fluctuation around that average; for a mean of $B$ independent samples it is $\operatorname{Var}(\text{one sample})/B$.
The two trade off: a biased estimator with small variance can beat an unbiased one with large variance, and which
wins depends on $B$. Here $\hat\theta = \hat g_B$ and $\theta = g$ [[5]](#ref-5).
:::

A batch averages $B$ independent rows, so $\mathbb{E}[\hat g_B] = \mathbb{E}[x]$ and
$\operatorname{Var}(\hat g_B) = \operatorname{Var}(x)/B$:

$$\mathbb{E}\,\|\hat g_B - g\|^2 = \|\mathbb{E}[x] - g\|^2 + \frac{\operatorname{Var}(x)}{B}.$$

Only $\operatorname{Var}(x)/B$ shrinks with $B$. Our row is $x = v\,Z$ with $Z \sim \text{Bernoulli}(q)$ and $v$
the $y = 2$ row from above:

$$\mathbb{E}[x] = q\,v, \qquad \operatorname{Var}(x) = \|v\|^2\, q(1-q).$$

| | model, $q = 0.2$ | engine, $q = 0.1$ |
|---|---|---|
| $\mathbb{E}[x] = q\,v$ | $0.2\,v$ = (0.10, 0.06, −0.16) | $0.1\,v$ = (0.05, 0.03, −0.08) |
| bias $= \mathbb{E}[x] - g$ | (0, 0, 0) | (−0.05, −0.03, +0.08) $= -0.1\,v$ |
| $\|\text{bias}\|^2$ | 0 | $0.05^2 + 0.03^2 + 0.08^2 = 0.0098$ |
| $\operatorname{Var}(x) = \|v\|^2\, q(1-q)$ | $0.98 \cdot 0.2 \cdot 0.8 = 0.157$ | $0.98 \cdot 0.1 \cdot 0.9 = 0.088$ |
| error $= \|\text{bias}\|^2 + \operatorname{Var}(x)/B$ | $0.157 / B$ | $0.0098 + 0.088 / B$ |
| at $B = 16$ | 0.0098 | $0.0098 + 0.0055 = 0.0153$ |

The model's bias is 0 because it writes token 2 at exactly the rate $g$ weights the $y = 2$ row, $q = p_2$
(section 3). Bias appears when that match breaks: $y$ drawn from a different distribution than $p$, as with the
engine, or the row changed ($w \ne 1$, $A$ rescaled).

The model's error goes to 0 as $B$ grows; the engine's never falls below $\|\text{bias}\|^2$, 0.0098. Yet for $B < 7$
the engine's error is the smaller one: its lower variance outweighs its bias until the batch is large enough for the
variance term to fall below 0.0098.

![error vs q](figs/bv_q.png)

*Figure 2: $\|\text{bias}\|^2 = 0.98\,(q - 0.2)^2$ and $\operatorname{Var}(x)/B = 0.98\,q(1-q)/B$ as the sampler's
probability of token 2 varies. The engine at $q = 0.1$ beats the model at $B = 4$ and loses at $B = 16$.*

The engine's variance is lower because at $q = 0.1$ most rows are $0$, so they agree with each other more than the
model's do; $q(1-q)$ shrinks toward either end. But they agree on $0.1\,v$, not $g$. Here low variance and high bias
are the same fact seen from two sides. Section 5 lowers $\operatorname{Var}(x)$ without touching the bias; every method in section 6
trades one term for the other.

## 5. Baseline: less variance, no bias

A baseline subtracts a constant from the reward, $A = r_y - b$, which subtracts $b\,(p - e_y)$ from every row.
Over the model's own samples $e_y$ averages to $p$, so $p - e_y$ averages to zero and the average row is unchanged
[[3]](#ref-3)[[6]](#ref-6). With $b = 0$ and $b = J = 0.2$:

| $y$ | $p_y$ | $p - e_y$ | $p_y\,(p - e_y)$ | row, $b = 0$ | row, $b = 0.2$ |
|---|---|---|---|---|---|
| 0 | 0.5 | (−0.5, 0.3, 0.2) | (−0.25, 0.15, 0.10) | (0, 0, 0) | (0.10, −0.06, −0.04) |
| 1 | 0.3 | (0.5, −0.7, 0.2) | (0.15, −0.21, 0.06) | (0, 0, 0) | (−0.10, 0.14, −0.04) |
| 2 | 0.2 | (0.5, 0.3, −0.8) | (0.10, 0.06, −0.16) | (0.50, 0.30, −0.80) | (0.40, 0.24, −0.64) |
| | | **sum** | **(0, 0, 0)** | **average (0.10, 0.06, −0.16)** | **average (0.10, 0.06, −0.16)** |
| | | | | **$\operatorname{Var}(x)$ 0.157** | **$\operatorname{Var}(x)$ 0.103** |

The fourth column sums to zero, so the baseline moves each row but the moves cancel: token 2's row shrinks toward
$g$ and the other two move off zero, and all three sit closer to their common average. Greensmith et al. derive the
variance-minimizing $b$ [[6]](#ref-6). Under the engine $e_y$ averages to $p_{\text{engine}}$ instead, the column
sums to $p - p_{\text{engine}} \ne 0$, and the baseline does shift the average; section 6 therefore uses $b = 0$.

![baseline](figs/bv_baseline.png)

*Figure 3: 20,000 batches of 16, model sampling itself, showing token 2's coordinate of the batch estimate. Both
baselines center on the true value; $b = 0.2$ is narrower.*

GRPO uses the group's mean reward as $b$ [[7]](#ref-7). Its other two steps, dividing by the group's reward standard
deviation and by response length, rescale $A$ and so add bias [[8]](#ref-8).

## 6. When the engine writes the tokens

Now $y \sim p_{\text{engine}}$ while the target is still $g$. Two engines, $b = 0$:

```python
p     = [0.5,  0.3,  0.2 ]    # model
close = [0.6,  0.3,  0.1 ]    # writes token 2 half as often
far   = [0.69, 0.30, 0.01]    # almost never writes token 2
```

### 6.1 Two distances between engine and model

**Total variation** measures how much probability mass moved [[9]](#ref-9):

$$\mathrm{TV} = \tfrac{1}{2}\textstyle\sum_k |p_k - p_{\text{engine},k}|,$$

which is 0.10 for close (0.1 moved from token 0 to token 2) and 0.19 for far.

**Chi-squared** measures how large the correcting weights get. The importance weight is
$w_k = p_k / p_{\text{engine},k}$ [[10]](#ref-10)[[12]](#ref-12):

```python
w_close = [0.83, 1.0,  2.0]
w_far   = [0.72, 1.0, 20.0]
```

It has mean 1 under the engine, and its variance is the chi-squared divergence, which controls the error of
importance sampling [[11]](#ref-11):

$$\operatorname{Var}_{p_{\text{engine}}}(w) = \sum_k \frac{p_k^2}{p_{\text{engine},k}} - 1 = \chi^2(p \,\|\, p_{\text{engine}}).$$

This is 0.12 for close and 3.66 for far. A token the engine rarely writes contributes $p_k^2 / p_{\text{engine},k}$,
so it dominates.

![distance](figs/bv_distance.png)

*Figure 4: engine $(0.7 - \epsilon,\ 0.3,\ \epsilon)$ as $\epsilon$, the engine's probability of token 2, falls from
0.2 (same as the model) to 0.005. TV levels off at 0.2. Chi-squared explodes.*

### 6.2 Three weights

A row is now $x = w_2\,v$ if $y = 2$ and $0$ otherwise, with $y = 2$ at rate $p_{\text{engine},2}$. With the far
engine, $p_{\text{engine},2} = 0.01$:

- **REINFORCE**, $w = 1$: $\mathbb{E}[x] = 0.01\,v$, a twentieth of $g$. Bias $-0.19\,v$, $\|\text{bias}\|^2 = 0.035$.
- **IS**, $w = p / p_{\text{engine}}$, so $w_2 = 20$: $\mathbb{E}[x] = 0.01 \cdot 20\,v = g$, unbiased. But 85% of
  batches of 16 never see token 2 and estimate $0$; the 14% that see it once estimate $\tfrac{20}{16}\,v$, six
  times $g$.
- **TIS**, $w = \min(p / p_{\text{engine}},\ 2)$: in between, mostly bias. Truncating weights to trade variance for
  bias goes back to Ionides [[13]](#ref-13); for LLM training it was proposed against engine mismatch [[1]](#ref-1).

![weights](figs/bv_weights_far.png)

*Figure 5: the far engine, token 2's coordinate of the batch estimate. Left: 300 single batches of 16. Every
REINFORCE and TIS batch lands near 0; most IS batches land at 0 and a few at -1 or -2. Right: running average over
batches. REINFORCE and TIS settle near 0 (bias). IS swings for hundreds of batches (variance) before settling on
the true -0.16.*

### 6.3 Every method, both engines

Two more choices of $w$, both built on the ratio $\rho_y = p_y / p_{\text{engine},y}$:

- **PPO** [[14]](#ref-14), with the ratio taken against the engine, sets $w = 0$ when $\rho_y$ leaves
  $[1 - \epsilon, 1 + \epsilon]$ in the direction $A$ favors, and $w = \rho_y$ otherwise.
- **DPPO** [[2]](#ref-2) sets $w = 0$ when $|p_y - p_{\text{engine},y}| > \delta$ (here $\delta = 0.15$) in the
  direction $A$ favors, and $w = \rho_y$ otherwise.

![methods](figs/bv_methods.png)

*Figure 6: squared error of the batch estimate at $B = 16$. Close engine (left): IS, TIS and DPPO coincide (the cap
of 2 does not bind, and DPPO passes since only 0.10 moved). Far engine (right): IS is mostly variance, the rest are
mostly bias.*

Table 1: the numbers behind Figure 6 ($b = 0$, per-sample variance, error at $B = 16$). Every bias is a multiple
of $v$, so one coefficient gives it.

| engine | method | $w_2$ | bias | $\|\text{bias}\|^2$ | $\operatorname{Var}(x)$ | error |
|---|---|---|---|---|---|---|
| close | REINFORCE | 1 | $-0.10\,v$ | 0.0098 | 0.088 | 0.015 |
| close | IS, TIS, DPPO | 2 | 0 | 0 | 0.353 | 0.022 |
| close | PPO (ratio to engine) | 0 | $-g$ | 0.0392 | 0 | 0.039 |
| far | REINFORCE | 1 | $-0.19\,v$ | 0.0354 | 0.010 | 0.036 |
| far | IS | 20 | 0 | 0 | 3.881 | 0.243 |
| far | TIS | 2 | $-0.18\,v$ | 0.0318 | 0.039 | 0.034 |
| far | PPO, DPPO | 0 | $-g$ | 0.0392 | 0 | 0.039 |

1. **Unbiased is not better.** With the close engine, REINFORCE beats IS at $B = 16$.
2. **The trust-region methods look worst, by design.** PPO and DPPO set $w_2 = 0$, so they estimate the zero vector and their bias is $-g$. They do not
   aim to estimate one gradient well; they stop the step on tokens that already moved too far. That pays off only
   across steps (section 7).
3. **PPO and DPPO disagree on the close engine.** $\rho_2 = 2$, so PPO blocks token 2. Only 0.10 of probability
   moved, under $\delta = 0.15$, so DPPO keeps it.

### 6.4 Where to put the cap

A cap trades variance for bias, and the best trade depends on $B$:

![cap](figs/bv_cap.png)

*Figure 7: far engine, error versus the cap on $w$. At $B = 16$ the best cap is about 3. At $B = 256$ the variance
term is 16 times smaller, so the best cap rises to about 14.*

Common TIS implementations cap at 2 [[1]](#ref-1). The same trade appears in classic off-policy RL as V-trace
[[15]](#ref-15) and Retrace [[16]](#ref-16).

## 7. Across steps: the blind spot and the trust region

Section 6 concerned one estimate. Over a step there is a second bias that no per-token $w$ removes: $w$ corrects
which token is written at a prefix, not how often each prefix is reached.

Two-token toy: the first token is $a$ or $b$, and the reward is 1 if the second token is $b$. The engine writes the
first token 50/50; after $a$ it writes $b$ with probability 0.1, after $b$ with probability 0.5, so $J = 0.3$. A step
raises $P(\text{first} = a)$ by $d$ and $P(b \mid a)$ by $d$. The true gain is $J(d) - J(0) = 0.1\,d + d^2$. The
surrogate (the importance-weighted objective evaluated on engine samples) keeps the engine's prefix frequencies and
predicts $0.1\,d$. The gap, $d^2$, is the stepped model reaching "after $a$" more often while also improving there.

Table 2: the gain the weighted batch predicts (surrogate) vs the true gain in average reward.

| $d$ | true gain | surrogate | gap |
|---|---|---|---|
| 0.01 | 0.0011 | 0.0010 | 0.0001 |
| 0.1 | 0.0200 | 0.0100 | 0.0100 |
| 0.3 | 0.1200 | 0.0300 | 0.0900 |

The performance difference lemma [[17]](#ref-17) writes the true gain exactly, and TRPO [[18]](#ref-18)[[19]](#ref-19)
bounds the gap by the TV between the two policies. DPPO adapts the bound to LLM responses (Theorem 3.2 in
[[2]](#ref-2)): the gap is at most a constant times the TV summed over the response. Keeping per-token TV small is a
trust region, and it is what the zeros in Figure 6 pay for.

PPO measures movement by $|\rho - 1|$. DPPO measures $|p - p_{\text{engine}}| = p_{\text{engine}}\,|\rho - 1|$, the
token's actual share of TV.

Table 3: PPO and DPPO on a rare token with $A > 0$ and a confident token with $A < 0$.

| token | $p_{\text{engine}} \to p$ | $\rho$ | moved | PPO | DPPO ($\delta = 0.15$) |
|---|---|---|---|---|---|
| rare | 0.0001 → 0.01 | 100 | 0.0099 | $w = 0$ | $w = 100$ |
| confident | 0.99 → 0.80 | 0.81 | 0.19 | $w = 0.81$ | $w = 0$ |

PPO blocks a token that barely moved and passes one that moved a lot. In the DPPO paper, collapse came from the
second kind (Figure 5 in [[2]](#ref-2)), and the mean $|p - p_{\text{engine}}|$ rose before reward fell (Figure 3
in [[2]](#ref-2)).

## 8. Summary

Table 4: what each choice of $w$ does.

| method | $w$ | bias | variance |
|---|---|---|---|
| REINFORCE | 1 | grows with TV | low |
| IS | $p / p_{\text{engine}}$ | none | grows with $\chi^2$ |
| TIS | IS, capped | in between; best cap grows with $B$ | in between |
| PPO, DPPO | 0 on tokens that moved too far | added on those tokens | low |

Ignoring the mismatch costs bias that scales with TV. Correcting it with $p / p_{\text{engine}}$ costs variance that
scales with $\chi^2$. A cap sits between the two, and the right cap depends on $B$. A trust region accepts some
bias in the current step to bound the next step's blind-spot bias.

Next: the same measurement on a real training run, per method.

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
