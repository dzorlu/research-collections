# Score centering

`E_q[R s_{y_t}]` is the policy gradient update.

```
one draw:   R × s_y          = R × (onehot(y) − p)        paper's ascent sign; notes' dz = A × (p − onehot(y)) is the loss sign
left side:  E_q[ R s_{y_t} ] = Σ_v q_v R_v s_v            the one-draw update averaged over which token the engine draws
```

No separate r and A: the paper writes R, and with a baseline R becomes A. The weight w is not there either; this is plain
policy gradient. `p` `(V,)` is the trainer's next-token distribution, `q` `(V,)` the engine's, `y_t` `()` the drawn token,
`s_v = ∇_θ log p_v` `(d,)` over all parameters; at the logits it is `onehot(v) − p` `(V,)`, and since backprop is linear
every identity below holds in both.

```
E[X Y] = E[X] E[Y] + Cov(X, Y)      with  X = A_{y_t}  ()    scalar
                                          Y = s_{y_t}  (d,)  over all parameters
```
both random through the draw `y_t ~ q`. `E[X] = A_avg = Σ_v q_v A_v` `()`, `E[Y] = s_bar = Σ_v q_v s_v` `(d,)`,
`Cov = Σ_v q_v (A_v − A_avg)(s_v − s_bar)`.

```
# policy gradient = drift + signal
# all averaged over q
E_q[A s] = A_avg · s_bar  +  Σ_v q_v (A_v − A_avg)(s_v − s_bar)
```

## s_bar is q − p at the logits, and zero only on-policy
`s_bar = Σ_v q_v s_v` `(d,)` has no closed form in weight space. At the logits, split the sum:
```
s_bar = Σ_v q_v (onehot(v) − p)
      = Σ_v q_v onehot(v)  −  (Σ_v q_v) p
      =        q           −     1 · p               (V,)
```
The first sum puts `q_v` in the `v`-th slot and gives `q`; `p` is the same vector in every term, so it comes out with
`Σ_v q_v = 1`. On-policy, `q = p` and `Σ_v p_v ∇ log p_v = ∇ Σ_v p_v = ∇ 1 = 0`. Under the engine's `q` the weights no
longer match the `p` inside the score, so `s_bar = q − p ≠ 0` and the drift `A_avg · s_bar` is a step of the trainer
toward the engine, unrelated to which token was good.

## The fix: subtract s_bar from the score
Use `s − s_bar` instead of `s`. Its mean under `q` is `s_bar − s_bar = 0`, so
```
E_q[A (s − s_bar)] = E_q[A s] − E_q[A] s_bar = (drift + signal) − drift = signal
Cov(A, s − s_bar)  = Cov(A, s)        covariance already subtracts the means, so the shift lands only on the
                                      drift term, which becomes A_avg · E_q[s − s_bar] = A_avg · 0
```
At the logits it is one subtraction: `(onehot(k) − p) − (q − p) = onehot(k) − q`, so `dz = A × (q − onehot(k))`. In code,
the engine returns `q` and the trainer has every `log p_v`, so the loss is `−A (log p_k − Σ_v sg[q_v] log p_v)`, whose
gradient is `A (s_k − s_bar)`. If `q = p`, nothing changes.

## The numbers on one state
```python
y_hat = [0.5, 0.3, 0.2]   # (V,)  trainer's p; q differs because the engine is quantized or stale
q     = [0.6, 0.3, 0.1]   # (V,)  engine draws token 0 60% of the time
R     = [1.0, 0.0, 0.0]   # (V,)  reward if token 0, 1, 2 is drawn: only token 0 leads to a correct answer
b     = 0.25              # ()    GRPO baseline: mean reward of this prompt's 4 rollouts (1 correct), used at every position
A     = R - b             # (V,)  = [0.75, -0.25, -0.25]

average_dz = Σ_v q_v * A[v] * (y_hat - onehot(v))  = [-0.275, 0.18, 0.095]   # (V,)  loss sign
s_bar      = q - y_hat                              = [0.1, 0.0, -0.1]
A_avg      = Σ_v q_v * A[v]                         = 0.35                    # expected reward here (0.6) minus b
drift      = -A_avg * s_bar                         = [-0.035, 0.0, 0.035]    # training subtracts it: y_hat moves toward q
signal     = -(q * (A - A_avg))                     = [-0.24, 0.18, 0.06]     # drift + signal = average_dz
centered   = Σ_v q_v * A[v] * (q - onehot(v))       = [-0.24, 0.18, 0.06]     # = signal
```
Drift is 0 only when `q = y_hat` or `A_avg = 0` (baseline equals this state's expected reward, 0.6); GRPO's `b` is one
number per prompt while the expected reward changes per position, so `A_avg` is rarely 0. Centering gives the same
update for every baseline (runnable, each `dz` checked by autograd: [score_centering.ipynb](score_centering.ipynb)):

| `b` | `A_avg` | plain average dz | drift | score-centered average dz |
|---|---|---|---|---|
| 0 | 0.60 | [-0.300, 0.18, 0.120] | [-0.060, 0, 0.060] | [-0.24, 0.18, 0.06] |
| 0.25 | 0.35 | [-0.275, 0.18, 0.095] | [-0.035, 0, 0.035] | [-0.24, 0.18, 0.06] |
| 0.6 | 0 | [-0.240, 0.18, 0.060] | [0, 0, 0] | [-0.24, 0.18, 0.06] |
| 1.0 | -0.40 | [-0.200, 0.18, 0.020] | [0.040, 0, -0.040] | [-0.24, 0.18, 0.06] |

## Why drift matters
Drift is a small fine-tuning step toward the engine, errors included. Toward a fixed teacher that would be harmless, but
after each step the engine reloads the trainer's new weights and adds its error again, so the target stays one error
ahead and the error builds up instead of settling. In the paper (REINFORCE with group-centered rewards, one optimizer
step per batch, about 800 steps), Qwen3-30B-A3B-Base on INTELLECT-2 math with an INT8 engine and an INT4 KV cache
reached 30% training accuracy with score centering, 12% with TIS, and below 5% with every other method. With an FP8
engine and an FP4 KV cache, plain REINFORCE collapsed within 200 steps; score centering ended at 52%, TIS at 51%.

Cost and limits:
- miles needs the engine's top 128 candidates per position (`--loss-type score_centering`,
  `--rollout-top-logprobs-num 128`), about 1 KiB each, and models the rest of `q` from the trainer's tail. In the
  paper, 128 or even 32 candidates matched full centering. The trainer still runs its forward pass to backpropagate `dz`.
- It does not reweight tokens inside the signal (TIS, MIS and IcePop do; it combines with them), does not limit
  several optimizer steps per batch (no PPO clip), and does not remove the mismatch itself.
- Drift is usually small: GRPO advantages average to about 0 per group and `y_hat − q` is small when the engine
  matches the trainer closely. It grows under heavy quantization or a stale engine.

Sources: Marek and Ryabinin, "Score Centering Stabilizes Off-policy Reinforcement Learning",
[arXiv 2609.20807](https://arxiv.org/abs/2609.20807), Eq. 3-5, sections 4-5; the
[Score Function Centering blog](https://huggingface.co/blog/NormalUhr/sfc) (no experiments); miles main at 8855b832e4,
`miles/backends/training_utils/loss/hub/score_centering/estimator.py` lines 83-90.
