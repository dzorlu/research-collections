# Learnings

## Drill 1: policy losses

### 1. Tensor mechanics

1. **Masking.** `.mean()` counts padded slots. A masked mean is `(x * mask).sum() / mask.sum()`; per response,
   `.sum(-1) / mask.sum(-1)`.
2. **Broadcasting starts from the last dimension.** `(4, 3) - (4,)` fails; `(4, 3) - (4, 1)` works. Use
   `keepdim=True` or `unsqueeze(-1)` to line up per row. With a square matrix the wrong version runs silently.
3. **`view` vs `reshape`, strides.** A stride is the memory jump per dimension. `view` never copies and fails on
   non-contiguous tensors (e.g. after `.T`); `reshape` copies when it must.
4. **`view(-1, G)` groups rewards per prompt.** `-1` infers `N`; `view(G, -1)` groups the wrong rewards.
5. **`gather` picks one entry per row by index.** `logits.gather(-1, y.unsqueeze(-1))` is
   `logits[b, t, y[b, t]]` for every position: `(B, T, V)` with index `(B, T, 1)` gives `(B, T, 1)`; then
   `squeeze(-1)`.

### 2. Numerics

6. **Ratios in log space.** `log a - log b = log(a / b)`, so `r = exp(logp_new - logp_old)`. Dividing log-probs is
   meaningless.
7. **Geometric mean = exp of the mean of the logs.** `[2, 0.5]` has geometric mean 1 (`log 2 = -log 0.5`) and
   arithmetic mean 1.25. GSPO's ratio is `exp(masked mean of log-ratios)`.
8. **Clean padding before `exp`.** Garbage `-100` becomes `inf`, and `inf * 0 = nan`. Masking before the `exp`
   works for finite garbage; `torch.where` is safe for any garbage.

9. **log-softmax = `z - logsumexp(z)`** over the vocabulary, `keepdim=True`. `logsumexp` = log of the softmax
   denominator: one number per position `(B, T, 1)` that shifts the scores so the probabilities sum to 1. It is
   computed as `m + log(sum(exp(z - m)))` with `m` the max, so nothing overflows (the max trick). Never
   `log(softmax(z))`: tiny probabilities underflow to 0 and `log 0 = -inf`.
   **To remember:** a log-prob is your score minus the soft max. `logsumexp` is a little above the largest score,
   so a log-prob is how far below the leader you are: the top token is near 0, a token 3 points behind is near -3
   (`[10, 0, 0]` gives about `[0, -10, -10]`). Max trick: subtract the leader first, so the leader sits at 0 and
   `exp` cannot blow up.

### 3. Gradients

10. **Minus sign in front of what you want to increase.** The optimizer always minimizes: loss = `-A * log p`. Terms
   to decrease (KL) go in with a plus.
11. **∇(-log p) w.r.t. the logits is `p_hat - onehot`.** So `dz = w * A * (p_hat - onehot) * mask / norm`. A negative
   gradient on the loss means the value goes up (`w - lr * grad`).
12. **`.detach()`** makes the weight a constant. Without it, autograd differentiates `w = r` too and the product rule
    adds a meaningless `log p` term. It also avoids in-place writes into the autograd graph.
13. **The loss value is meaningless in policy gradient.** It hovers around 0 and can have either sign. Only the
    gradient matters; watch reward.

### 4. The methods

14. **The four modes differ only in the per-token weight `w`.**

| Mode | `w` |
|---|---|
| REINFORCE | 1 |
| PPO | `r`, zeroed once a token moved far enough in the direction `A` wants (`A > 0, r > 1.28` or `A < 0, r < 0.8`) |
| CISPO | `clip(r, 0, 4)`: never zero, only capped |
| GSPO | one ratio per response, PPO's zeroing applied to it: keep or drop the whole response |

15. **PPO's two forms are equal.** Weight form `-w * A * log p` with `w` detached; textbook form
    `-min(r * A, clip(r) * A)` with no `log p` and no detach.
16. **`norm` only changes the divisor.** Total real tokens (`"token"`) or `B * len_b` (`"seq"`). GSPO always uses
    per response.
