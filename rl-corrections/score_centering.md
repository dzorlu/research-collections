# Score centering

- **Problem:** the engine picks tokens with probabilities `q`, but REINFORCE subtracts the trainer's `y_hat`. So the
  score `onehot(k) - y_hat` does not average to 0; its average is `s_bar = q - y_hat`.
- **Drift:** averaged over the engine's picks, the update carries `A_avg * s_bar`, a step of the trainer toward the
  engine (away if `A_avg < 0`), unrelated to which token was good. Toward a fixed teacher that would be harmless. But
  the engine is rebuilt from the trainer after every step, with its error added again, so the target stays one error
  ahead and the pushes add up step after step until training collapses.
- **Fix:** subtract `s_bar` from each pick's score, so the score becomes `onehot(k) - q` and
  `dz = A[k] * (q - onehot(k))`. The average score is 0, so drift = 0. The signal (which token was better than
  expected) is unchanged.
- **Result:** in the paper's worst quantization setting (30B model, INT8 engine, INT4 KV cache), score centering
  reached 30% training accuracy, TIS 12%, everything else under 5%.
- **Cost:** the engine must return its top candidates per position (128 in miles).

## 0. The formulation

Everything happens at one prefix: the prompt plus the tokens already generated. There the trainer has one
distribution over the next token, the engine has another, and one token gets drawn. The paper's Eq. 3-5 are
statements about that one draw, averaged over how it could have gone.

**Notation.** The notes stop at the logits, so every gradient is a `(V,)` vector. The paper differentiates with respect
to all `d` weights, so its vectors are `(d,)`; backprop turns one into the other and is linear, so every identity below
holds in both.

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

In real training these are indexed per position: `y_hat`, `q` are `(B, T, V)`, `A` is `(B, T)`, and the sums below run
at each of the `B × T` prefixes separately.

**The expected update is a sum over the vocabulary.** The draw is the only random thing, and it lands on `v` with
probability `q_v`, so averaging over draws means weighting each token's update by `q_v`:

```
E_q[ A_{y_t} s_{y_t} ]  =  Σ_v q_v · A_v · s_v                              (V,): a scalar q_v A_v times a vector s_v
```

**Covariance identity.** For a scalar `X` and a vector `Y` drawn together, `E[XY] = E[X] E[Y] + Cov(X, Y)`. With
`X = A_{y_t}` and `Y = s_{y_t}` this is the paper's Eq. 3:

```
Σ_v q_v A_v s_v  =  A_avg · s_bar   +   Σ_v q_v (A_v - A_avg)(s_v - s_bar)
                    drift               signal
```

`E[X]E[Y]` pairs the average advantage with the average score, so it has forgotten which token earned which reward.
The covariance is what is left: whether a high advantage tends to come with a particular score direction.

**On-policy the drift is zero.** With `q = y_hat`, `s_bar = Σ_v p_v ∇ log p_v = Σ_v ∇ p_v = ∇ 1 = 0`. At the logits:
`Σ_v p_v (onehot(v) - p) = p - p = 0`. Under the engine's `q` the weights no longer match the `p` inside the score,
`s_bar = q - p ≠ 0`, and the drift `A_avg · s_bar` is a step of the trainer toward the engine.

**Centering.** Replace the score by `s_tilde = s_{y_t} - s_bar` (Eq. 4). Then `E_q[s_tilde] = s_bar - s_bar = 0`, so the
drift term is `A_avg · 0` for any baseline, and the covariance is unchanged because shifting `Y` by a constant does not
change `Cov(X, Y)` (Eq. 5). `s_bar` is computable: the engine returns `q`, the trainer has `log p_v` for every `v`, and
in code it is one extra term in the loss, `-A (log p_k - Σ_v sg[q_v] log p_v)`, whose gradient is `A (s_k - s_bar)`.

The rest of this page computes each of these on one state with numbers. Runnable version in PyTorch, with every `dz`
checked by autograd: [score_centering.ipynb](score_centering.ipynb).

## 1. The state

One position in one response. The engine is about to pick the next token from 3 options.

```python
y_hat = [0.5, 0.3, 0.2]          # (V,)  trainer's probabilities at this position (softmax of its logits)
q     = [0.6, 0.3, 0.1]          # (V,)  engine's probabilities at this position: it picks token 0 60% of the time
R     = [1.0, 0.0, 0.0]          # (V,)  reward if token 0, 1, 2 is picked: only token 0 leads to a correct answer
b     = 0.25                     # ()    baseline: mean reward of this prompt's rollouts (4 rollouts, 1 correct)
A     = R - b                    # (V,)  = [0.75, -0.25, -0.25], the advantage the update uses for each pick
```

`y_hat` and `q` should be equal; they differ because the engine is quantized or holds older weights. `b` is GRPO's
baseline: one number per prompt, used at every position.

## 2. The update for each possible pick

`dz = A[k] * (y_hat - onehot(k))`, and training does `z = z - lr * dz`.

```python
# pick token 0:  dz =  0.75 * ([0.5, 0.3, 0.2] - [1, 0, 0]) = [-0.375,  0.225,  0.150]    # (V,)
# pick token 1:  dz = -0.25 * ([0.5, 0.3, 0.2] - [0, 1, 0]) = [-0.125,  0.175, -0.050]    # (V,)
# pick token 2:  dz = -0.25 * ([0.5, 0.3, 0.2] - [0, 0, 1]) = [-0.125, -0.075,  0.200]    # (V,)
```

## 3. The average update at this state

Training sees only one pick. Imagine the engine picking at this same state many times: 60% token 0, 30% token 1,
10% token 2. The average `dz` is the direction training moves these logits once the luck of the draw is removed.

```python
average_dz = 0.6 * [-0.375, 0.225, 0.150] + 0.3 * [-0.125, 0.175, -0.050] + 0.1 * [-0.125, -0.075, 0.200]
           = [-0.275, 0.180, 0.095]                                                        # (V,)
```

## 4. Drift and signal

The score of a pick is `onehot(k) - y_hat`: the direction that makes token `k` more likely (`dz = -A[k] * score`).
Its average over the engine's picks is `s_bar`:

```python
s_bar = Σ_i q_i * (onehot(i) - y_hat)                                                    # (V,)  i runs over the 3 tokens
      = 0.6*([1,0,0] - y_hat) + 0.3*([0,1,0] - y_hat) + 0.1*([0,0,1] - y_hat)
      = q - y_hat = [0.1, 0.0, -0.1]
```

`s_bar` is not 0 because the engine picks with `q` while each score subtracts the trainer's `y_hat`. It is the
direction one fine-tuning step toward the engine would move the logits.

Split each advantage into its average and each pick's difference from it:

```python
A_avg     = Σ_i q_i * A[i] = 0.6*0.75 + 0.3*(-0.25) + 0.1*(-0.25) = 0.35     # ()  advantage averaged over the engine's picks: expected reward here (0.6) minus b
A - A_avg = [0.40, -0.60, -0.60]                             # (V,)  how much better or worse each pick is

drift  = -A_avg * s_bar         = [-0.035, 0.00, 0.035]                     # (V,)
signal = -(q * (A - A_avg))     = [-0.240, 0.18, 0.060]                     # (V,)
# drift + signal = [-0.275, 0.18, 0.095] = average_dz
```

- Drift uses only `A_avg`, so it ignores which token was good. Training subtracts it: token 0 goes up, token 2
  goes down, so `y_hat` moves toward `q`. With `A_avg < 0` it moves away.
- Signal uses only `A - A_avg`: the better-than-expected token 0 goes up, the worse tokens go down. The paper calls
  it a covariance because it is the average of (advantage minus its average) times (score minus its average).

Drift is 0 when `q = y_hat` (no mismatch) or when `A_avg = 0` (the baseline equals this state's expected reward,
0.6). GRPO's `b` is one number per prompt, and the expected reward changes from position to position, so `A_avg` is
rarely 0.

## 5. Score centering removes the drift

Subtract `s_bar` from every pick's score:

```python
centered score = (onehot(k) - y_hat) - s_bar = (onehot(k) - y_hat) - (q - y_hat) = onehot(k) - q    # (V,)
```

In `dz` form this is the one-line change:

```python
dz = A[k] * (y_hat - onehot(k))     # (V,)  plain REINFORCE
dz = A[k] * (q     - onehot(k))     # (V,)  score centering
```

The difference is one subtracted term, the same vector `y_hat - q` for every pick, scaled by the pick's `A[k]`:

```python
centered dz = A[k] * (y_hat - onehot(k))  -  A[k] * (y_hat - q)       # (V,)  = plain dz - A[k] * (y_hat - q)

# pick token 0 (A[0] = 0.75):
plain dz          = [-0.375, 0.225, 0.150]                            # (V,)
subtracted term   = 0.75 * [-0.1, 0.0, 0.1] = [-0.075, 0.0, 0.075]    # (V,)
centered dz       = [-0.300, 0.225, 0.075]                            # (V,)
```

Averaged over the engine's picks, the subtracted term is `A_avg * (y_hat - q)`, exactly the drift of section 4.
The centered score averages to 0 under the engine's picks:

```python
0.6*([1,0,0] - q) + 0.3*([0,1,0] - q) + 0.1*([0,0,1] - q) = q - q = [0, 0, 0]         # (V,)
```

So `drift = A_avg * 0 = 0` for any baseline, and the average update is just the signal, `[-0.24, 0.18, 0.06]`.
Plain REINFORCE's score averages to 0 only if the trainer does the picking; centering on `q` makes it 0 for the
engine that actually picks. If `q = y_hat`, nothing changes. In the full per-token formula:
`dz = w * A * (q - y_onehot) / norm`.

**Table 1. Average dz at this state for four baselines.** The state's expected reward is 0.6.

| `b` | `A_avg` | plain average dz | drift | score-centered average dz |
|---|---|---|---|---|
| 0 | 0.60 | [-0.300, 0.18, 0.120] | [-0.060, 0, 0.060] | [-0.24, 0.18, 0.06] |
| 0.25 | 0.35 | [-0.275, 0.18, 0.095] | [-0.035, 0, 0.035] | [-0.24, 0.18, 0.06] |
| 0.6 | 0 | [-0.240, 0.18, 0.060] | [0, 0, 0] | [-0.24, 0.18, 0.06] |
| 1.0 | -0.40 | [-0.200, 0.18, 0.020] | [0.040, 0, -0.040] | [-0.24, 0.18, 0.06] |

## 6. Why drift matters

The engine is a slightly wrong copy of the trainer. Drift is a small fine-tuning step toward that copy, errors
included. After the step the engine reloads the trainer's new weights and adds its error again, so the error builds
up instead of settling. In the paper (REINFORCE with group-centered rewards, one optimizer step per batch, about 800
steps), Qwen3-30B-A3B-Base on INTELLECT-2 math with an INT8 engine and an INT4 KV cache reached 30% training accuracy
with score centering, 12% with TIS, and below 5% with every other method. With an FP8 engine and an FP4 KV cache,
plain REINFORCE collapsed within 200 steps; score centering ended at 52%, TIS at 51%.

## 7. Cost and limits

- miles needs the engine's top 128 candidates per position (`--loss-type score_centering`,
  `--rollout-top-logprobs-num 128`), about 1 KiB each, and models the rest of `q` from the trainer's tail. In the
  paper, 128 or even 32 candidates matched full centering.
- The trainer still runs its forward pass to backpropagate `dz` to the weights.
- It does not reweight tokens inside the signal (TIS, MIS and IcePop do; it combines with them), does not limit
  several optimizer steps per batch (no PPO clip), and does not remove the mismatch itself.
- Drift is usually small: GRPO advantages average to about 0 per group and `y_hat - q` is small when the engine
  matches the trainer closely. It grows under heavy quantization or a stale engine.

## Sources

- Marek and Ryabinin, "Score Centering Stabilizes Off-policy Reinforcement Learning",
  [arXiv 2609.20807](https://arxiv.org/abs/2609.20807): Eq. 3-5, sections 4-5.
- [Score Function Centering blog](https://huggingface.co/blog/NormalUhr/sfc) (no experiments).
- miles main at 8855b832e4: `miles/backends/training_utils/loss/hub/score_centering/estimator.py` lines 83-90.
