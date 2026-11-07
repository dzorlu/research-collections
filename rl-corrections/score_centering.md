# Score centering

- **Problem:** the engine draws tokens with probabilities `q`, but REINFORCE's score subtracts the trainer's `y_hat`.
  The score `onehot(k) - y_hat` averages to `s_bar = q - y_hat`, not 0.
- **Drift:** averaged over the engine's draws, the update carries `A_avg * s_bar`, a step of the trainer toward the
  engine (away if `A_avg < 0`), unrelated to which token was good. The engine is rebuilt from the trainer after every
  step with its error added again, so the pushes add up until training collapses.
- **Fix:** subtract `s_bar` from each draw's score. The score becomes `onehot(k) - q`, `dz = A[k] * (q - onehot(k))`,
  the average score is 0, drift is 0, and the signal (which token was better than expected) is unchanged.
- **Result:** in the paper's worst quantization setting (30B model, INT8 engine, INT4 KV cache), score centering
  reached 30% training accuracy, TIS 12%, everything else under 5%.
- **Cost:** the engine must return its top candidates per position (128 in miles).

## Notation: notes stop at the logits, the paper goes to the weights
Everything happens at one prefix: the trainer has one next-token distribution, the engine has another, one token gets
drawn. The notes stop at the logits, so every gradient is `(V,)`; the paper differentiates with respect to all `d`
weights, so its vectors are `(d,)`. Backprop maps one to the other and is linear, so every identity below holds in both.

| notes | paper | meaning | shape |
|---|---|---|---|
| `y_hat` | `p` | trainer's next-token probabilities, `p_v` one entry | `(V,)` |
| `q` | `q` | engine's next-token probabilities; the draw uses these | `(V,)` |
| `k` | `y_t` | the token that was drawn | `()` |
| `R`, `b`, `A = R - b` | `R` | reward per possible token, baseline, advantage | `(V,)`, `()`, `(V,)` |
| `onehot(k) - y_hat` | `s_{y_t} = ∇_θ log p_{y_t}` | score of the drawn token: nudge weight `j`, `log p` moves by entry `j` | `(V,)` here, `(d,)` in the paper |
| `s_bar = q - y_hat` | `s_bar = Σ_v q_v s_v` | score averaged over the engine's draws | same as the score |
| `A_avg` | `E_q[R]` | advantage averaged over the engine's draws, `Σ_v q_v A_v` | `()` |
| `dz` | `-R s_{y_t}` | the update for one draw; notes use the loss sign, paper the ascent sign | same as the score |

In real training these are indexed per position: `y_hat`, `q` are `(B, T, V)`, `A` is `(B, T)`, and every sum below
runs at each of the `B × T` prefixes separately. Toy state for every step (each `dz` checked by autograd in
[score_centering.ipynb](score_centering.ipynb)):
```python
y_hat = [0.5, 0.3, 0.2]   # (V,)  trainer's probabilities; q differs because the engine is quantized or stale
q     = [0.6, 0.3, 0.1]   # (V,)  engine's probabilities; it draws token 0 60% of the time
R     = [1.0, 0.0, 0.0]   # (V,)  reward if token 0, 1, 2 is drawn: only token 0 leads to a correct answer
b     = 0.25              # ()    GRPO baseline: mean reward of this prompt's 4 rollouts (1 correct), used at every position
A     = R - b             # (V,)  = [0.75, -0.25, -0.25]
```

## Step 1. One draw moves the logits by advantage times score
The paper's update for one draw is `R s_y` with `s_y = ∇_θ log p_y`, which at the logits is `onehot(y) - p`. The notes'
`dz = A[k] * (y_hat - onehot(k))`, applied as `z = z - lr * dz`, is the same vector in loss sign.
```python
# draw token 0:  dz =  0.75 * ([0.5, 0.3, 0.2] - [1, 0, 0]) = [-0.375,  0.225,  0.150]    # (V,)
# draw token 1:  dz = -0.25 * ([0.5, 0.3, 0.2] - [0, 1, 0]) = [-0.125,  0.175, -0.050]
# draw token 2:  dz = -0.25 * ([0.5, 0.3, 0.2] - [0, 0, 1]) = [-0.125, -0.075,  0.200]
```

## Step 2. The expected update weights each token's update by q_v
The draw is the only random thing and lands on `v` with probability `q_v`; `R_v` is the reward if token `v` is drawn.
```python
E_q[R s] = Σ_v q_v R_v s_v                      # (V,) at the logits, (d,) in the paper
average_dz = 0.6*[-0.375, 0.225, 0.150] + 0.3*[-0.125, 0.175, -0.050] + 0.1*[-0.125, -0.075, 0.200]
           = [-0.275, 0.180, 0.095]
```

## Step 3. The expected update is drift plus signal
For a scalar `X = R_{y_t}` and a vector `Y = s_{y_t}` drawn together, `E[XY] = E[X] E[Y] + Cov(X, Y)`. This is Eq. 3.
`E[X]E[Y]` pairs the average advantage with the average score and has forgotten which token earned which reward; the
covariance is what is left.
```python
E_q[R s] = E_q[R] · s_bar  +  Cov_q(R, s)       # drift + signal;  s_bar = Σ_v q_v s_v,  E_q[R] = A_avg = Σ_v q_v A_v
```
At the logits `s_bar` has a closed form; split the sum:
```python
s_bar = Σ_v q_v (onehot(v) - y_hat)
      = Σ_v q_v onehot(v)  -  (Σ_v q_v) y_hat
      =        q           -     1 · y_hat
```
The first sum puts `q_v` in the `v`-th slot and gives the vector `q`; `y_hat` is the same vector in every term, so it
comes out with `Σ_v q_v = 1`. In weight space `s_bar` stays `Σ_v q_v s_v` with no closed form.
```python
s_bar     = 0.6*([1,0,0] - y_hat) + 0.3*([0,1,0] - y_hat) + 0.1*([0,0,1] - y_hat) = q - y_hat = [0.1, 0.0, -0.1]
A_avg     = 0.6*0.75 + 0.3*(-0.25) + 0.1*(-0.25) = 0.35        # ()  expected reward here (0.6) minus b
A - A_avg = [0.40, -0.60, -0.60]
drift     = -A_avg * s_bar       = [-0.035, 0.00, 0.035]        # (V,)  loss sign
signal    = -(q * (A - A_avg))   = [-0.240, 0.18, 0.060]        # (V,)  drift + signal = average_dz
```
Training subtracts the drift: token 0 up, token 2 down, so `y_hat` moves toward `q` regardless of which token was good.
Drift is 0 only when `q = y_hat` or `A_avg = 0` (baseline equals this state's expected reward, 0.6); GRPO's `b` is one
number per prompt while the expected reward changes per position, so `A_avg` is rarely 0.

## Step 4. On-policy the average score is zero; under q it is not
```python
Σ_v p_v ∇ log p_v = Σ_v ∇ p_v = ∇ 1 = 0        # at the logits: Σ_v p_v (onehot(v) - p) = p - p = 0
```
Under `q` the weights no longer match the `p` inside the score, so `s_bar = q - p ≠ 0`.

## Step 5. Centering the score removes the drift and keeps the signal
Use `s - s_bar` instead of `s` (Eq. 4). The same algebra in two forms (Eq. 5):
```python
E_q[R (s - s_bar)] = E_q[R s] - E_q[R] s_bar = (drift + signal) - drift = signal
Cov(R, s - s_bar)  = Cov(R, s)      # covariance already subtracts the means, so the shift lands only on the
                                    # first term, which becomes E_q[R] · E_q[s - s_bar] = E_q[R] · 0
```
This changes the update (a design choice, not an identity); it is legitimate because the signal is the only term that
depends on which token won. At the logits the `y_hat` cancels and the centered score is `onehot(k) - q`:
```python
(onehot(k) - y_hat) - (q - y_hat) = onehot(k) - q

dz = A[k] * (y_hat - onehot(k))     # (V,)  plain REINFORCE
dz = A[k] * (q     - onehot(k))     # (V,)  score centering; per token in miles: dz = w * A * (q - y_onehot) / norm

# draw token 0:  centered dz = plain dz - 0.75*(y_hat - q) = [-0.375, 0.225, 0.150] - [-0.075, 0, 0.075] = [-0.300, 0.225, 0.075]
# centered score averaged over draws: 0.6*([1,0,0] - q) + 0.3*([0,1,0] - q) + 0.1*([0,0,1] - q) = q - q = [0, 0, 0]
# centered average dz = [-0.24, 0.18, 0.06] = signal
# loss form: -A (log p_k - Σ_v sg[q_v] log p_v), gradient A (s_k - s_bar); the engine returns q, the trainer has every log p_v
```
The subtracted term averages to `A_avg * (y_hat - q)`, the drift of Step 3. If `q = y_hat`, nothing changes.

## Step 6. The centered update is the same for any baseline
Drift is 0 for every `b`, so the centered average `dz` is the signal in every row. The state's expected reward is 0.6.

| `b` | `A_avg` | plain average dz | drift | score-centered average dz |
|---|---|---|---|---|
| 0 | 0.60 | [-0.300, 0.18, 0.120] | [-0.060, 0, 0.060] | [-0.24, 0.18, 0.06] |
| 0.25 | 0.35 | [-0.275, 0.18, 0.095] | [-0.035, 0, 0.035] | [-0.24, 0.18, 0.06] |
| 0.6 | 0 | [-0.240, 0.18, 0.060] | [0, 0, 0] | [-0.24, 0.18, 0.06] |
| 1.0 | -0.40 | [-0.200, 0.18, 0.020] | [0.040, 0, -0.040] | [-0.24, 0.18, 0.06] |

## Why drift matters: the target moves with the trainer
Drift is a small fine-tuning step toward the engine, errors included. Toward a fixed teacher that would be harmless, but
after each step the engine reloads the trainer's new weights and adds its error again, so the target stays one error
ahead and the error builds up instead of settling. In the paper (REINFORCE with group-centered rewards, one optimizer
step per batch, about 800 steps), Qwen3-30B-A3B-Base on INTELLECT-2 math with an INT8 engine and an INT4 KV cache
reached 30% training accuracy with score centering, 12% with TIS, and below 5% with every other method. With an FP8
engine and an FP4 KV cache, plain REINFORCE collapsed within 200 steps; score centering ended at 52%, TIS at 51%.

## Cost and limits
- miles needs the engine's top 128 candidates per position (`--loss-type score_centering`,
  `--rollout-top-logprobs-num 128`), about 1 KiB each, and models the rest of `q` from the trainer's tail. In the
  paper, 128 or even 32 candidates matched full centering. The trainer still runs its forward pass to backpropagate `dz`.
- It does not reweight tokens inside the signal (TIS, MIS and IcePop do; it combines with them), does not limit
  several optimizer steps per batch (no PPO clip), and does not remove the mismatch itself.
- Drift is usually small: GRPO advantages average to about 0 per group and `y_hat - q` is small when the engine
  matches the trainer closely. It grows under heavy quantization or a stale engine.

## Sources
- Marek and Ryabinin, "Score Centering Stabilizes Off-policy Reinforcement Learning",
  [arXiv 2609.20807](https://arxiv.org/abs/2609.20807): Eq. 3-5, sections 4-5.
- [Score Function Centering blog](https://huggingface.co/blog/NormalUhr/sfc) (no experiments).
- miles main at 8855b832e4: `miles/backends/training_utils/loss/hub/score_centering/estimator.py` lines 83-90.
