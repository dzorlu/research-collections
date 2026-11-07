# Score centering

## Summary

In RL for language models an inference engine draws the tokens and a trainer computes the gradient, and the two
assign slightly different probabilities to the same token: quantization, a stale copy of the weights, different
kernels. Every correction in use reweights each token by a ratio of the two probabilities. That fixes the mismatch
exactly only when the ratio is left uncapped, and every practical method caps it. Score centering [[1]](#ref-1)
changes the direction of the gradient instead of its weight, and the whole method is one line. The argument fits in
a few lines.

**The update is advantage times score.** The score of token $v$ is the gradient of its log-probability with respect to
all $d$ parameters. At the logits it is one-hot minus the trainer's probabilities.

$$
s_v = \nabla_\theta \log p_\theta(v) \in \mathbb{R}^d, \qquad \text{at the logits: } s_v = e_v - p \in \mathbb{R}^V
$$

**The engine draws, so the expectation is under $q$.** One draw of token $y$ contributes $A_y s_y$. The engine draws
$v$ with probability $q_v$:

$$
\mathbb{E}_q[A\,s] = \sum_v q_v\, A_v\, s_v
$$

**The mean score under $q$ is $q - p$.** The one-hots collect $q$; $p$ is the same vector in every term and comes out
with the probabilities summing to one.

$$
\bar s = \sum_v q_v\, s_v = \sum_v q_v\,(e_v - p) = \sum_v q_v\, e_v - \Big(\sum_v q_v\Big)\, p = q - p
$$

If the trainer drew, this would be $p - p = 0$. The engine draws, so it is not.

**So the expected update carries drift.** A product of two random variables averages to mean times mean plus a
covariance. For advantage times score under $q$:

$$
\mathbb{E}_q[A\,s] = \underbrace{\bar A\,(q - p)}_{\text{drift}} + \underbrace{\sum_v q_v\,(A_v - \bar A)(s_v - \bar s)}_{\text{signal: } \mathrm{Cov}_q(A, s)}
$$

The drift pushes the trainer toward the engine, scaled by the mean advantage, blind to which token earned the reward.
The signal is the only part that depends on that.

**The fix subtracts $\bar s$ from every score, and $p$ cancels.**

$$
s_v - \bar s = (e_v - p) - (q - p) = e_v - q, \qquad \sum_v q_v\,(e_v - q) = q - q = 0
$$

The centered score has mean zero under the engine, so the drift term is $\bar A \times 0$ for any advantage and any
baseline. The covariance is unchanged, because it already subtracts the mean score: replace $s$ by $s - \bar s$ and
the factor $(s_v - \bar s)$ becomes $(s_v - \bar s) - 0$, the same term.

$$
\mathbb{E}_q[A\,(s - \bar s)] = \mathrm{Cov}_q(A, s)
$$

**In code it is one line.** Where the gradient at the logits was $A\,(p - e_y)$, write $A\,(q - e_y)$. The engine has
to return its distribution at each position. No ratios, no clipping, no hyperparameter, and it composes with any
importance weight.

**Why it matters.** The engine is rebuilt from the trainer after every step with its error added back, so the drift
accumulates instead of converging. In the paper's worst setting, a 30B model with an INT8 engine and INT4 KV cache,
centering reaches 30% training accuracy, truncated importance sampling 12%, everything else under 5%.

## One position, three tokens

The trainer gives the next token probabilities $p$, the engine gives $q$, and the engine draws. Only token 0 leads to
a correct answer. The baseline $b$ is the mean reward of this prompt's rollouts.

```python
p = [0.5, 0.3, 0.2]       # trainer
q = [0.6, 0.3, 0.1]       # engine; it draws
R = [1.0, 0.0, 0.0]       # reward if token 0, 1, 2 is drawn
b = 0.25
A = [0.75, -0.25, -0.25]  # R - b
```

The score of each token, and the update if that token is drawn:

```python
S = [[ 0.5, -0.3, -0.2],   # row v: e_v - p
     [-0.5,  0.7, -0.2],
     [-0.5, -0.3,  0.8]]

U = [[ 0.375, -0.225, -0.150],   # row v: A[v] * S[v]
     [ 0.125, -0.175,  0.050],
     [ 0.125,  0.075, -0.200]]
```

The engine draws token $v$ with probability $q_v$, so the expected update is the $q$-weighted sum of the rows: the
matrix product of $q$ with $U$.

```python
q @ U = 0.6 * U[0] + 0.3 * U[1] + 0.1 * U[2] = [0.275, -0.18, -0.095]   # q = [0.6, 0.3, 0.1], the engine's draw probabilities
```

## The expected update is drift plus signal

Split it with the covariance identity $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y] + \mathrm{Cov}(X, Y)$, where $X$
is the advantage of the drawn token and $Y$ its score:

```python
A_avg = q @ A = 0.6*0.75 + 0.3*(-0.25) + 0.1*(-0.25) = 0.35          # mean advantage under the engine's draws
s_bar = q @ S = 0.6*S[0] + 0.3*S[1] + 0.1*S[2]       = [0.1, 0.0, -0.1]   # mean score under the engine's draws, = q - p

drift  = A_avg * s_bar     = [0.035,  0.00, -0.035]
signal = q @ U - drift     = [0.240, -0.18, -0.060]
```

The drift raises token 0 and lowers token 2: it moves $p$ toward $q$, scaled by the average advantage, with no regard
for which token earned the reward. The signal is what remains, and it is the only part that depends on that.

Why $\bar s = q - p$: split the sum that defines it. The one-hot part collects $q$; $p$ is the same vector in every
term and comes out with the probabilities summing to one.

$$
\bar s = \sum_v q_v (e_v - p) = \sum_v q_v e_v - \Big(\sum_v q_v\Big) p = q - p
$$

## Centering the score removes the drift

Use $e_y - q$ as the score. Its mean under the engine's draws is $q - q = 0$, so the drift term is $\bar A \times 0$.
The covariance is untouched, because a covariance already subtracts the means.

```python
Sc = [[ 0.4, -0.3, -0.1],   # row v: e_v - q, the centered score of token v
      [-0.6,  0.7, -0.1],
      [-0.6, -0.3,  0.9]]
Uc = [[ 0.300, -0.225, -0.075],   # row v: A[v] * Sc[v], the centered update if token v is drawn
      [ 0.150, -0.175,  0.025],
      [ 0.150,  0.075, -0.225]]

q @ Sc = 0.6*Sc[0] + 0.3*Sc[1] + 0.1*Sc[2] = [0.0,  0.0,   0.0 ]    # mean centered score under the engine: zero
q @ Uc = 0.6*Uc[0] + 0.3*Uc[1] + 0.1*Uc[2] = [0.24, -0.18, -0.06]   # expected centered update = signal, no drift
```

In a training loop the change is the vector the one-hot is subtracted from. With `p` the trainer's probabilities and
`q` the engine's at each position, both of shape `[B, T, V]`:

```python
y_onehot = F.one_hot(y, V)                        # [B, T, V]
dlogits  = w * A * (p - y_onehot) / (B * T)       # plain: w is the method's per-token weight, 1 for REINFORCE
dlogits  = w * A * (q - y_onehot) / (B * T)       # score centering
```

The same holds for every baseline, because the drift is gone before the baseline enters:

| $b$ | $\bar A$ | plain update | drift | centered update |
|---|---|---|---|---|
| 0 | 0.60 | [0.300, -0.18, -0.120] | [0.060, 0, -0.060] | [0.24, -0.18, -0.06] |
| 0.25 | 0.35 | [0.275, -0.18, -0.095] | [0.035, 0, -0.035] | [0.24, -0.18, -0.06] |
| 0.6 | 0 | [0.240, -0.18, -0.060] | [0, 0, 0] | [0.24, -0.18, -0.06] |
| 1.0 | -0.40 | [0.200, -0.18, -0.020] | [-0.040, 0, 0.040] | [0.24, -0.18, -0.06] |

A baseline equal to this position's expected reward under the engine, 0.6, would also zero the drift. A group mean is
one number per prompt while the expected reward changes at every position, so in practice it does not.

In a real model the engine does not ship its full distribution. The paper uses the engine's top-$k$ probabilities and
models the tail from the trainer's, which brings the correction back onto the $k$ tokens the trainer can see; 128 or
even 32 candidates matched full centering.

## The drift compounds

Toward a fixed teacher the drift would be harmless: it is a step of distillation and converges. But the engine is not
fixed. After each step it reloads the trainer's new weights and adds its error again, so the target stays one error
ahead and the pushes accumulate. The paper's 30B runs show how far that goes [[1]](#ref-1):

| engine | plain policy gradient | truncated IS | score centering |
|---|---|---|---|
| FP8 | stable, 58% | | |
| FP8, FP4 KV cache | collapses within 200 steps | 51% | 52% |
| INT8, INT4 KV cache | under 5% | 12% | 30% |

Every other method tested lands under 5% in the last row. The paper's reading is that clipping was designed for
ratios produced by the policy moving between steps, not by numerical error, and that what breaks training is not the
size of the mismatch but that the teacher tracks the student.

The same thing shows on a toy: a two-layer network learning to sort four digits, with a copy of itself as the engine
and a fixed offset on the engine's logits, trained by REINFORCE with no baseline ([code](https://github.com/dzorlu/research-collections/tree/main/rl-corrections/pglab)).
Mean reward over 40-rollout windows; chance is 0.10.

| offset | gradient | 0-40 | 40-80 | 80-120 | 120-160 | 160-200 |
|---|---|---|---|---|---|---|
| none | either | .105 | .108 | .116 | .124 | .137 |
| 1.0 | plain | .103 | .101 | .105 | .101 | .103 |
| 1.0 | centered | .104 | .111 | .121 | .131 | .140 |

With all-positive rewards the average advantage is the expected reward itself, so every step carries a full push
toward the offset engine, and here it is as large as the signal: plain REINFORCE never leaves chance. Centered, the
curve under the offset matches the curve without it. The offset has to be persistent for this to happen. Fresh noise
each rollout points $q - p$ a different way every step and the pushes average out, which is why quantization and
staleness, both deterministic, are the cases that collapse.

## References

1. <a id="ref-1"></a>Marek and Ryabinin. Score Centering Stabilizes Off-policy Reinforcement Learning. 2026.
   [arXiv:2609.20807](https://arxiv.org/abs/2609.20807).
