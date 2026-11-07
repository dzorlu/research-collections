# Score centering

## TLDR

- Each update is advantage times score. The score of the drawn token is onehot(y) − p, the direction that makes it
  more likely.
- Average the score over the engine's draws:

  ```
  Σ_v q_v (onehot(v) − p) = q − p
  ```

  If the trainer draws, q = p and this is zero. If the engine draws, it is the gap between the two.
- Any product of two random variables averages to mean times mean plus a covariance. Applied to advantage times score
  under q, that gives two terms: A_avg × (q − p), the drift, and a covariance, the signal. The drift is a push of the
  trainer toward the engine, blind to which token was good. On-policy the same split exists but the drift is zero
  because q − p is; the mismatch is what makes it survive.
- Fix: subtract q − p from the score.

  ```
  onehot(y) − p − (q − p) = onehot(y) − q
  Σ_v q_v (onehot(v) − q) = q − q = 0
  ```

  Drift becomes A_avg × 0, for any baseline. The covariance does not change because every row got the same vector
  subtracted.
- In code, one line: where the gradient at the logits was A × (p − onehot(y)), write A × (q − onehot(y)). The engine
  has to return its distribution q at each position (top-k in practice, with the tail modeled from p). No ratios, no
  clipping, no hyperparameter, and it composes with any importance weight w.
- Why it matters: the engine is rebuilt from the trainer every step with its error re-added, so the drift accumulates
  instead of converging. Importance sampling corrects by reweighting toward the trainer and pays in variance.
  Centering corrects by measuring against the engine and pays nothing.
- Result: in the paper's worst setting (30B, INT8 engine, INT4 KV cache) centering reaches 30% training accuracy,
  TIS 12%, everything else under 5%; in pglab with a fixed engine offset and no baseline, plain reinforce stays at
  chance and centered reinforce learns at the noise-free pace.

## One prefix, three tokens

Everything below happens at one prefix: the prompt plus the tokens generated so far. The trainer has a distribution over
the next token, the engine has a slightly different one, and the engine draws. Ascent sign throughout, so an update is
added to the logits.

```python
p = [0.5, 0.3, 0.2]      # (V,)  trainer's next-token probabilities
q = [0.6, 0.3, 0.1]      # (V,)  engine's next-token probabilities; it does the drawing
R = [1.0, 0.0, 0.0]      # (V,)  reward if token 0, 1, 2 is drawn
b = 0.25                 # ()    baseline, the mean reward of this prompt's rollouts
A = [0.75, -0.25, -0.25] # (V,)  R - b
```

## The policy gradient update is the advantage times the score

The score of a token is the direction that makes it more likely. At the logits it is one-hot minus p. In the paper the
score is taken with respect to all d weights and is a d-vector, but backprop is linear, so every statement here holds
there too.

```python
S = [[ 0.5, -0.3, -0.2],   # (V, V)  row v is s_v = onehot(v) - p, the score of token v
     [-0.5,  0.7, -0.2],
     [-0.5, -0.3,  0.8]]
```

One draw of token y gives the update A[y] times its score. The engine draws y with probability q[y], so the expected
update is the q-weighted sum of the three possible updates.

```python
U = [[ 0.375, -0.225, -0.150],   # (V, V)  row v is A[v] * S[v], the update if token v is drawn
     [ 0.125, -0.175,  0.050],
     [ 0.125,  0.075, -0.200]]

E_q[A s] = 0.6 * U[0] + 0.3 * U[1] + 0.1 * U[2] = [0.275, -0.18, -0.095]   # (V,)
```

## The expected update splits into drift plus signal

The update is a product of two random things, the advantage A[y] and the score S[y], both random through the draw.
For any such product the expectation splits into the product of the means plus a covariance.

```python
E[X Y] = E[X] E[Y] + Cov(X, Y)          # X = A[y] scalar, Y = S[y] vector

E[X] = A_avg = 0.6*0.75 + 0.3*(-0.25) + 0.1*(-0.25) = 0.35                    # ()
E[Y] = s_bar = 0.6*S[0] + 0.3*S[1] + 0.1*S[2]       = [0.1, 0.0, -0.1]        # (V,)

drift  = A_avg * s_bar       = [0.035, 0.00, -0.035]                           # (V,)
signal = E[XY] - drift       = [0.240, -0.18, -0.060]                          # (V,)  the covariance
```

The drift pairs the average advantage with the average score. It has forgotten which token earned which reward. The
signal is what is left: whether a high advantage comes with a particular score direction. Only the signal says which
token was good.

## s_bar is q minus p, and zero only when the trainer draws

Split the sum that defines s_bar. The one-hot part collects q. The p part is the same vector in every term, so it comes
out with the probabilities summing to one.

```python
s_bar = 0.6*[1,0,0] + 0.3*[0,1,0] + 0.1*[0,0,1]  -  (0.6 + 0.3 + 0.1) * p
      =        [0.6, 0.3, 0.1]                   -   1.0 * [0.5, 0.3, 0.2]
      =        q - p                             =   [0.1, 0.0, -0.1]           # (V,)
```

If the trainer did the drawing the weights would be p, and p minus p is zero. So on-policy there is no drift. Under the
engine the weights are q while the score still subtracts p, and the drift is a push of p toward q at every prefix,
scaled by the average advantage. Nothing about which token was good enters it.

```python
on-policy: 0.5*S[0] + 0.3*S[1] + 0.2*S[2] = [0.0, 0.0, 0.0]
```

In weight space s_bar has no closed form; it stays the q-weighted sum of the d-dimensional scores.

## The fix subtracts s_bar from the score

Use the centered score, the score minus s_bar, in the update. Its mean under q is zero, so the drift term is the average
advantage times zero. The covariance does not change, because a covariance already subtracts the means.

```python
E_q[A (s - s_bar)] = E_q[A s] - A_avg * s_bar = (drift + signal) - drift = signal
```

At the logits the subtraction is one line, and p cancels.

```python
(onehot(y) - p) - (q - p) = onehot(y) - q

Sc = [[ 0.4, -0.3, -0.1],    # (V, V)  row v is onehot(v) - q, the centered score
      [-0.6,  0.7, -0.1],
      [-0.6, -0.3,  0.9]]

0.6*Sc[0] + 0.3*Sc[1] + 0.1*Sc[2]                 = [0.0,  0.0,   0.0 ]       # (V,)  mean zero under q
0.6*A[0]*Sc[0] + 0.3*A[1]*Sc[1] + 0.1*A[2]*Sc[2]  = [0.24, -0.18, -0.06]      # (V,)  = signal
```

The update changes; this is a different estimator, chosen because the signal is the only term that depends on which
token won. In code the engine returns q, the trainer has every log p, and the loss is the usual term minus a q-weighted
sum of all log-probabilities with the gradient stopped through q.

```python
loss = -A * (log_p[y] - sum(stop_grad(q[v]) * log_p[v] for v in V))     # gradient: A * (s_y - s_bar)
```

The centered update is the same for every baseline, because the drift is zero before the baseline enters.

| `b` | `A_avg` | plain update | drift | centered update |
|---|---|---|---|---|
| 0 | 0.60 | [0.300, -0.18, -0.120] | [0.060, 0, -0.060] | [0.24, -0.18, -0.06] |
| 0.25 | 0.35 | [0.275, -0.18, -0.095] | [0.035, 0, -0.035] | [0.24, -0.18, -0.06] |
| 0.6 | 0 | [0.240, -0.18, -0.060] | [0, 0, 0] | [0.24, -0.18, -0.06] |
| 1.0 | -0.40 | [0.200, -0.18, -0.020] | [-0.040, 0, 0.040] | [0.24, -0.18, -0.06] |

A baseline equal to this prefix's expected reward under q, 0.6, would also zero the drift. GRPO's baseline is one
number per prompt and the expected reward changes at every position, so in practice it does not.

## In pglab the drift cancels the learning, and centering restores it

[pglab](pglab/) trains a two-layer network to sort four digits, with a copy of it as the engine. The swap of p for q
is one line in `model.backward` (`--center true`). The engine's error is a `[V]` logit offset drawn once per run and
added at every rollout (`--noise-fixed true`); fresh noise each rollout had no effect, because then q − p points a
different way every step and the pushes average out. Mean reward of the engine's rollouts in 40-rollout windows,
reinforce with no baseline (A = R), two seeds averaged:

| engine offset std | gradient | 0-40 | 40-80 | 80-120 | 120-160 | 160-200 |
|---|---|---|---|---|---|---|
| 0 | plain or centered | .105 | .108 | .116 | .124 | .137 |
| 1.0 | plain | .103 | .101 | .105 | .101 | .103 |
| 1.0 | centered | .104 | .111 | .121 | .131 | .140 |

Chance is 0.1. With all-positive rewards A_avg is the expected reward itself, so every step carries a full push of p
toward the offset engine, and it is as large as the signal: plain reinforce stays at chance. Centered, the curve under
the offset matches the curve without it.

With GRPO advantages the same runs show no drift at all, and the reason is a quirk of pglab: a position's input is the
prompt and the slot, not the tokens written so far, so q − p is identical for all eight responses of a prompt, and
group-centered advantages sum to zero against it. In a real model the prefixes diverge after the first token and that
cancellation is gone. Reproduce:

```
python -m pglab.train --methods reinforce --advantage none --noise 0,1.0 --noise-fixed true --seeds 2 --steps 200 --center true
```

## Why drift matters

Drift is a small fine-tuning step toward the engine, its errors included. Toward a fixed teacher that would be harmless.
But after each step the engine reloads the trainer's new weights and adds its error again, so the target stays one error
ahead and the pushes accumulate instead of settling. In the paper, Qwen3-30B-A3B-Base on INTELLECT-2 math with an INT8
engine and an INT4 KV cache reached 30% training accuracy with score centering, 12% with TIS, and under 5% with every
other method. With an FP8 engine and an FP4 KV cache, plain REINFORCE collapsed within 200 steps; score centering ended
at 52%, TIS at 51%.

## Cost and limits

- The engine must return its top candidates per position; 128 in miles (`--loss-type score_centering`,
  `--rollout-top-logprobs-num 128`), about 1 KiB each. The rest of q is modeled from the trainer's tail. In the paper
  128 or even 32 candidates matched full centering.
- It does not reweight tokens inside the signal (TIS, MIS and IcePop do; it composes with them), does not allow several
  optimizer steps per batch (no PPO clip), and does not remove the mismatch itself.
- Drift is small when the engine matches the trainer closely. It grows under heavy quantization or a stale engine.

## Sources

- Marek and Ryabinin, "Score Centering Stabilizes Off-policy Reinforcement Learning",
  [arXiv 2609.20807](https://arxiv.org/abs/2609.20807): Eq. 3-5, sections 4-5.
- [Score Function Centering blog](https://huggingface.co/blog/NormalUhr/sfc) (no experiments).
- miles main at 8855b832e4: `miles/backends/training_utils/loss/hub/score_centering/estimator.py` lines 83-90.
- Runnable version with every update checked by autograd: [score_centering.ipynb](score_centering.ipynb).
