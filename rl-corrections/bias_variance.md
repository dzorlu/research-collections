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

Praise every move and the player learns nothing about which moves were good. A baseline is that praise: it adds the
same $-b$ to every reward, so on average it cannot push the model toward any token. It still changes each
individual update, and that is where the variance goes.

Replace $A = r_y$ with $A = r_y - b$ and multiply out the row:

$$(r_y - b)\,(p - e_y) = \underbrace{r_y\,(p - e_y)}_{\text{row without baseline}} \;-\; \underbrace{b\,(p - e_y)}_{\text{what the baseline subtracts}}.$$

The subtracted piece is not a constant: it depends on which $y$ was drawn and changes sign, and in proportion to
how often each $y$ is drawn the pieces cancel.

:::note a constant reward moves nothing: every step
Three tokens, $p = (0.6, 0.3, 0.1)$, reward $2$ whatever is drawn, so $A = 2$ for every row.

| | $y = 0$ | $y = 1$ | $y = 2$ |
|---|---|---|---|
| how often, $p_y$ | 0.6 | 0.3 | 0.1 |
| $e_y$ | (1, 0, 0) | (0, 1, 0) | (0, 0, 1) |
| $p - e_y$ | (−0.4, 0.3, 0.1) | (0.6, −0.7, 0.1) | (0.6, 0.3, −0.9) |
| row $= 2\,(p - e_y)$ | (−0.8, 0.6, 0.2) | (1.2, −1.4, 0.2) | (1.2, 0.6, −1.8) |
| $p_y \times$ row | (−0.48, 0.36, 0.12) | (0.36, −0.42, 0.06) | (0.12, 0.06, −0.18) |

Sum of the last line, coordinate by coordinate: $-0.48 + 0.36 + 0.12 = 0$, $0.36 - 0.42 + 0.06 = 0$,
$0.12 + 0.06 - 0.18 = 0$. Each row pushes its own token up (training subtracts the row) and the others down; token 0
gets a push of $-0.8$ often, token 2 a push of $-1.8$ rarely, and each token's upward pushes equal what the other
rows push it down. A baseline is this table with $-b$ in place of $2$.
:::

The toy, $r = (0, 0, 1)$, without a baseline and with $b = 0.2$:

| $y$ | $p_y$ | $r_y$ | $p - e_y$ | $A = r_y$ | row $= A\,(p - e_y)$ | $A = r_y - 0.2$ | row $= A\,(p - e_y)$ | subtracted $= 0.2\,(p - e_y)$ | $p_y \times$ subtracted |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0 | (−0.5, 0.3, 0.2) | 0 | (0, 0, 0) | −0.2 | (0.10, −0.06, −0.04) | (−0.10, 0.06, 0.04) | (−0.05, 0.03, 0.02) |
| 1 | 0.3 | 0 | (0.5, −0.7, 0.2) | 0 | (0, 0, 0) | −0.2 | (−0.10, 0.14, −0.04) | (0.10, −0.14, 0.04) | (0.03, −0.04, 0.01) |
| 2 | 0.2 | 1 | (0.5, 0.3, −0.8) | 1 | (0.50, 0.30, −0.80) | 0.8 | (0.40, 0.24, −0.64) | (0.10, 0.06, −0.16) | (0.02, 0.01, −0.03) |
| | | | | | **avg (0.10, 0.06, −0.16) $= g$** | | **avg (0.10, 0.06, −0.16) $= g$** | | **sum (0, 0, 0): bias the baseline adds** |
| | | | | | **$\operatorname{Var}(x)$ 0.157** | | **$\operatorname{Var}(x)$ 0.103** | | |

The average row is $g$ in both columns. The bias the baseline adds is the weighted sum of what it subtracts (last
column), and that is zero because $e_y$ lands on each token as often as $p$ says: $\mathbb{E}[e_y] = p$
[[3]](#ref-3)[[6]](#ref-6). The rows sit closer to the unchanged average, so lower variance.

![baseline](figs/bv_baseline.png)

*Figure 3: 20,000 batches of 16, model sampling itself, token 2's coordinate of the batch estimate. Both center on
the true value; $b = 0.2$ is narrower.*

## 6. When the engine writes the tokens

Now $y \sim p_{\text{engine}}$ while the target is still $g$. Two engines, $b = 0$:

```python
p     = [0.5,  0.3,  0.2 ]    # model
close = [0.6,  0.3,  0.1 ]    # writes token 2 half as often
far   = [0.69, 0.30, 0.01]    # almost never writes token 2
```

### 6.1 Two distances between engine and model

The engine's error on each token can be read as a difference or as a ratio. The far engine:

| token $k$ | $p_k$ | $p_{\text{engine},k}$ | difference $p_k - p_{\text{engine},k}$ | ratio $p_k / p_{\text{engine},k}$ |
|---|---|---|---|---|
| 0 | 0.50 | 0.69 | −0.19 | 0.72 |
| 1 | 0.30 | 0.30 | 0 | 1.0 |
| 2 | 0.20 | 0.01 | +0.19 | 20 |

**Total variation** adds up the differences [[9]](#ref-9):

$$\mathrm{TV} = \tfrac{1}{2}\textstyle\sum_k |p_k - p_{\text{engine},k}|,$$

half because each move is counted where it leaves and where it lands. It is the mass that moved, so it never
exceeds 1.

**Chi-squared** is the variance of the ratio. The ratio is the importance weight, $w_k = p_k / p_{\text{engine},k}$
[[10]](#ref-10)[[12]](#ref-12). Under the engine it is random, averages to 1 (since
$\sum_k p_{\text{engine},k} \cdot p_k / p_{\text{engine},k} = 1$), and its variance is [[11]](#ref-11)

$$\operatorname{Var}_{p_{\text{engine}}}(w) = \mathbb{E}[w^2] - 1 = \sum_k \frac{p_k^2}{p_{\text{engine},k}} - 1 = \chi^2(p \,\|\, p_{\text{engine}}).$$

| | close engine, (0.6, 0.3, 0.1) | far engine, (0.69, 0.30, 0.01) |
|---|---|---|
| TV $= \tfrac12 \sum_k \lvert p_k - p_{\text{engine},k} \rvert$ | $\tfrac12 (0.1 + 0 + 0.1) = 0.10$ | $\tfrac12 (0.19 + 0 + 0.19) = 0.19$ |
| weights $w$ | (0.83, 1.0, 2.0) | (0.72, 1.0, 20) |
| $\chi^2 = \sum_k p_{\text{engine},k}\, w_k^2 - 1$ | $0.6 \cdot 0.83^2 + 0.3 \cdot 1 + 0.1 \cdot 2^2 - 1 = 0.12$ | $0.69 \cdot 0.72^2 + 0.30 \cdot 1 + 0.01 \cdot 20^2 - 1 = 3.66$ |

Token 2 moved 0.19 either way. TV sees the difference, which is bounded by the mass that exists. $\chi^2$ sees the
ratio squared, which is unbounded as $p_{\text{engine},2} \to 0$: token 2 alone contributes $0.01 \cdot 20^2 = 4.0$.

TV sets the bias of ignoring the mismatch; $\chi^2$ sets the variance of correcting it. From close to far, TV
doubles and $\chi^2$ grows thirty-fold, so the same move costs REINFORCE a little more bias and IS far more variance.
That gap is why caps and trust regions exist (6.2 to 6.4).

### 6.2 Three weights

Far engine, $b = 0$. Only token 2's row is nonzero: $x = w_2\,${{$v$ @@ v = p − e_2 = (0.5, 0.3, −0.8), the y = 2 row of section 3's table; defined in section 4}} with
probability {{$q = p_{\text{engine},2} = 0.01$ @@ the far engine's probability of token 2, from the table in 6.1}}, else $0$. Section 4's formulas with the weight in place:

$$\begin{gathered}
\mathbb{E}[x] = q\,w_2\,v, \qquad \text{bias} = \mathbb{E}[x] - g = (q\,w_2 - p_2)\,v, \\
\operatorname{Var}(x) = w_2^2\,\|v\|^2\,q(1-q), \qquad \text{error} = \|\text{bias}\|^2 + \frac{\operatorname{Var}(x)}{B},
\end{gathered}$$

using {{$g = p_2\,v = 0.2\,v$ @@ section 3's table: only the y = 2 row is nonzero, weighted by p_2 = 0.2}} and {{$\|v\|^2 = 0.98$ @@ 0.5² + 0.3² + 0.8² = 0.98}}.

The bias is set by one number, {{$q\,w_2$ @@ how often token 2's row appears (q) times how much it counts when it does (w_2): its average weight}}. The target is the model's own frequency, $p_2 = 0.2$; the engine
supplies $q = 0.01$; $w_2$ is each method's way of making up the difference:

| | $w_2$ | $q\,w_2$ | bias $= (q\,w_2 - 0.2)\,v$ | $\|\text{bias}\|^2 = 0.98\,(q\,w_2 - 0.2)^2$ | $\operatorname{Var}(x) = w_2^2 \cdot 0.98 \cdot 0.01 \cdot 0.99$ | error, $B = 16$ |
|---|---|---|---|---|---|---|
| REINFORCE, $w = 1$ | 1 | 0.01 | $-0.19\,v$ | 0.035 | 0.010 | $0.035 + 0.010/16 = 0.036$ |
| IS, $w = p / p_{\text{engine}}$ | $0.2 / 0.01 = 20$ | 0.20 | 0 | 0 | 3.88 | $0 + 3.88/16 = 0.243$ |
| TIS, $w = \min(p / p_{\text{engine}},\ 2)$ [[13]](#ref-13)[[1]](#ref-1) | 2 | 0.02 | $-0.18\,v$ | 0.032 | 0.039 | $0.032 + 0.039/16 = 0.034$ |

- **REINFORCE** counts each token-2 row once. Token 2 shows up in 1% of rows instead of 20%, so the push is a
  twentieth of what it should be, every batch. Bias.
- **IS** counts each token-2 row 20 times, so the average push is right. But a batch either has no token-2 row
  (estimate $0$) or one, counted $\times 20$ (estimate six times the truth). Variance.
- **TIS** counts each one twice. The push is a tenth of what it should be, and a batch is $0$ or $0.6\times$ the
  truth: nearly REINFORCE's bias with a little of IS's variance.

In this toy the two errors are the two distances of 6.1: REINFORCE's $\|\text{bias}\|^2 = 0.98\,(p_2 - q)^2$ is the
moved mass squared, TV; IS's $\operatorname{Var}(x) = 0.98\,(p_2^2/q - p_2^2)$ is token 2's term of $\chi^2$.

![weights](figs/bv_weights_far.png)

*Figure 4: left, one dot per batch of 16, placed at its estimate; the dashed line is the truth. Right, the average
of the first $n$ batches as $n$ grows. REINFORCE and TIS settle short of the truth (bias); IS scatters widely and
takes hundreds of batches to settle on it (variance).*

### 6.3 Every method, both engines

Two more choices of $w$, both built on the ratio $\rho_y = p_y / p_{\text{engine},y}$:

- **PPO** [[14]](#ref-14), with the ratio taken against the engine, sets $w = 0$ when $\rho_y$ leaves
  $[1 - \epsilon, 1 + \epsilon]$ in the direction $A$ favors, and $w = \rho_y$ otherwise.
- **DPPO** [[2]](#ref-2) sets $w = 0$ when $|p_y - p_{\text{engine},y}| > \delta$ (here $\delta = 0.15$) in the
  direction $A$ favors, and $w = \rho_y$ otherwise.

![methods](figs/bv_methods.png)

*Figure 5: squared error of the batch estimate at $B = 16$. Close engine (left): IS, TIS and DPPO coincide (the cap
of 2 does not bind, and DPPO passes since only 0.10 moved). Far engine (right): IS is mostly variance, the rest are
mostly bias.*

Table 1: the numbers behind Figure 5 ($b = 0$, per-sample variance, error at $B = 16$). Every bias is a multiple
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

*Figure 6: far engine, error versus the cap on $w$. At $B = 16$ the best cap is about 3. At $B = 256$ the variance
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
trust region, and it is what the zeros in Figure 5 pay for.

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

## 8. What each algorithm trades

Every method fixes the two factors of the row $x = w\,A\,(p - e_y)$, and each choice moves one term of
$\|\text{bias}\|^2 + \operatorname{Var}(x)/B$ against the other.

### 8.1 The advantage

**Group-mean baseline** (GRPO [[7]](#ref-7)): $A = r_y - b$ with $b$ the mean reward of the $G$ responses to the
same prompt. A shift, so the subtracted piece averages to zero (section 5) and only the variance falls. One catch:
the mean includes the response's own reward, which shrinks the average row by $(G - 1)/G$. RLOO [[20]](#ref-20)
averages the other $G - 1$ responses, keeping $b$ independent of the response.

**Std and length division** (GRPO [[7]](#ref-7)): $A = (r_y - b)/\operatorname{std}$, and the token sum divided by
response length. These are rescales, not shifts. A shift subtracts a piece with mean zero; a rescale multiplies the
row by a factor computed from the same group, which does not average out: the average row becomes $g$ reweighted
across prompts and lengths. Variance falls, since every prompt's rows arrive at one scale. Dr. GRPO [[8]](#ref-8)
drops both. The toy has one prompt, so its std factor is a constant and shows no bias.

### 8.2 The weight

$w$ is a function of $\rho = p_y / p_{\text{engine},y}$, taken against the engine as in section 6.3. In practice
CISPO, PPO and GSPO take it against the recompute, the trainer's own log-probs before the step.

- **REINFORCE**, $w = 1$: the lowest variance; bias grows with TV.
- **IS**, $w = \rho$: no bias; variance grows with $\chi^2$, dominated by the rare row's $p_2^2 / p_{\text{engine},2}$.
- **TIS**, $w = \min(\rho, 2)$: the part of $\chi^2$ above the cap becomes bias; the best cap grows with $B$ (Figure 6).
- **CISPO** [[21]](#ref-21), $w = \min(\rho, 4)$ with the ratio held fixed under differentiation: TIS with a higher
  cap, and no row is zeroed. On the far engine, less bias and more variance than TIS; the same error at $B = 16$
  and the better of the two at $B = 256$ (Table 4).
- **PPO** [[14]](#ref-14), $w = \rho$ in the band, $0$ outside it in the direction $A$ favors: a zeroed row has bias
  $-g$ on that token and no variance. DAPO's clip-higher [[23]](#ref-23) widens the upper edge, so fewer rare tokens
  with positive $A$ are zeroed: less bias, more variance.
- **GSPO** [[22]](#ref-22): one ratio per response, the geometric mean of the per-token ratios, zeroed outside a
  tight band for every token of the response. Averaging log-ratios cuts the ratio's variance; whole-response zeros
  under a tight band add bias. On the one-position toy the sequence ratio is $\rho_2$ and GSPO coincides with PPO.
- **DPPO** [[2]](#ref-2), $w = \rho$, zeroed when $|p - p_{\text{engine}}| > \delta$ in the direction $A$ favors:
  the zero is placed by moved mass, not by $\rho$. A rare token passes uncapped, so DPPO carries IS's variance there;
  a confident token that moved far is zeroed (Table 3). Close engine: passes, equals IS. Far engine: zeroed, equals
  PPO. The bias on zeroed tokens buys the across-step bound of section 7.

Table 4: Table 1 with CISPO (cap 4) added and the error at a larger batch ($b = 0$, per-sample variance).

| engine | method | $w_2$ | bias | $\|\text{bias}\|^2$ | $\operatorname{Var}(x)$ | error, $B = 16$ | error, $B = 256$ |
|---|---|---|---|---|---|---|---|
| close | REINFORCE | 1 | $-0.10\,v$ | 0.0098 | 0.088 | 0.015 | 0.010 |
| close | IS, TIS, CISPO, DPPO | 2 | 0 | 0 | 0.353 | 0.022 | 0.0014 |
| close | PPO (ratio to engine) | 0 | $-g$ | 0.0392 | 0 | 0.039 | 0.039 |
| far | REINFORCE | 1 | $-0.19\,v$ | 0.0354 | 0.010 | 0.036 | 0.035 |
| far | IS | 20 | 0 | 0 | 3.881 | 0.243 | 0.015 |
| far | TIS (cap 2) | 2 | $-0.18\,v$ | 0.0318 | 0.039 | 0.034 | 0.032 |
| far | CISPO (cap 4) | 4 | $-0.16\,v$ | 0.0251 | 0.155 | 0.035 | 0.026 |
| far | PPO, DPPO | 0 | $-g$ | 0.0392 | 0 | 0.039 | 0.039 |

Table 5: each choice, the term it lowers and the term it raises.

| choice | $A$ | $w$ | lowers | raises |
|---|---|---|---|---|
| no baseline | $r_y$ | | nothing: the reference point | nothing |
| group mean (GRPO) | $r_y - b$ | | variance | bias: a shrink along $g$ by $(G-1)/G$ |
| leave-one-out mean (RLOO) | $r_y - b$, own reward excluded | | variance | nothing |
| std, length division (GRPO) | $(r_y - b)/\operatorname{std}$, $/\text{length}$ | | variance: one scale across prompts | bias: $g$ reweighted by group spread and length |
| REINFORCE | | $1$ | variance | bias, with TV |
| IS | | $\rho$ | bias, to zero | variance, with $\chi^2$ |
| TIS | | $\min(\rho, 2)$ | variance above the cap | bias where the cap binds |
| CISPO | | $\min(\rho, 4)$, ratio fixed under the gradient | bias, relative to TIS; no zeroed rows | variance, relative to TIS |
| PPO | | $\rho$; $0$ outside the band in the favored direction | variance on zeroed tokens; next-step bias (section 7) | bias $-g$ on zeroed tokens |
| GSPO | | sequence ratio; $0$ outside a tight band, per response | variance from per-token ratios | bias: whole responses zeroed |
| DPPO | | $\rho$; $0$ when moved mass $> \delta$ in the favored direction | next-step bias; fewer zeros on rare tokens than PPO | bias $-g$ on zeroed tokens; IS's variance on rare tokens that pass |

## 9. Summary

Table 6: what each choice of $w$ does.

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
20. <a id="ref-20"></a>A. Ahmadian, C. Cremer, M. Gallé, M. Fadaee, J. Kreutzer, O. Pietquin, A. Üstün, S. Hooker. *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (RLOO). arXiv:2402.14740, 2024.
21. <a id="ref-21"></a>MiniMax (A. Chen et al.). *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention* (CISPO). arXiv:2506.13585, 2025.
22. <a id="ref-22"></a>C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, J. Zhou, J. Lin. *Group Sequence Policy Optimization* (GSPO). arXiv:2507.18071, 2025.
23. <a id="ref-23"></a>Q. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476, 2025.
