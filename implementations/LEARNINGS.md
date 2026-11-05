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

6. **Ratios in log space.** The policy stores log-probs, not probs. Subtracting two logs is the log of their ratio:
   `log a - log b = log(a / b)`, so `r = p_new / p_old = exp(logp_new - logp_old)`. Example: `p_new = 0.5`,
   `p_old = 0.25`: `exp(-0.69 - (-1.39)) = exp(0.69) = 2`. Dividing the log-probs instead gives
   `-0.69 / -1.39 = 0.5`, which means nothing. Staying in log space also avoids forming tiny probabilities and
   dividing them.
7. **Geometric mean = exp of the mean of the logs.** The geometric mean of `n` numbers is the n-th root of their
   product, `(a * b * c)^(1/3)`. Logs turn the product into a sum, so `log(geo mean) = (log a + log b + log c) / 3`,
   the plain mean of the logs; `exp` brings it back. Ratios multiply (a response's probability is the product of its
   token probabilities), so the geometric mean is the right average for them: `[2, 0.5]` has geometric mean 1
   (`log 2 = 0.693 = -log 0.5`, doubling and halving cancel) but arithmetic mean 1.25. GSPO's per-response ratio is
   `exp(masked mean of (logp_new - logp_old))`.
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

## Drill 3: attention

### 5. Forward pass and KV cache

- **Forward:** `S = q kᵀ / sqrt(dh)`, mask, softmax over keys, `@ v`. The output of each query is a weighted average
  of the values it may see. `Wo` is applied after merging the heads (it mixes the heads), never to `x`.
- **Split heads:** `(B, T, D)` → `reshape(B, T, H, dh)` → `transpose(1, 2)` → `(B, H, T, dh)`. Cut `D` first,
  then move the head dimension. **Merge:** `transpose(1, 2)` then `reshape(B, T, D)`: `view` fails here because
  the tensor is not contiguous after the transpose.
- **Scale by the head dim** `sqrt(dh)`, not `sqrt(D)`: each head's dot product sums over `dh` numbers.
- `k.T` on an N-D tensor reverses every dimension; use `k.transpose(-1, -2)`.
- **Mask before the softmax with `-inf`.** `exp(-inf) = 0`, so the softmax renormalizes over the visible keys.
  Zeroing after the softmax (`tril`) leaves rows summing to less than 1. Masking replaces values, it does not
  multiply (`0 * -inf = nan`): `S.masked_fill(mask, float("-inf"))` with a **bool** mask.
- `triu(diagonal=1)` is strictly above the diagonal (the future). Without the 1 a token cannot see itself.
  A `(T, T)` mask broadcasts over `B` and `H`.
- **KV cache:** past keys and values never change, so store them and concatenate the new ones along time (`dim=2`).
  The cache starts as `None`. The past length is `cache["k"].shape[-2]`.
- **With a cache the mask is not square:** `(T, T_past + T)`. Query `i` sits at absolute position `T_past + i`,
  so the mask is `triu(diagonal=T_past + 1)`. Check with a small case: 5 cached + 1 new → `(1, 6)`, nothing masked.

### 6. Backward pass

Attention is a chain of three operations; save `q, k, v, P` on the way forward and go back up:

| Forward | Backward |
|---|---|
| `out = P v` | `dP = dout vᵀ`, `dv = Pᵀ dout` (matmul rule: `Y = A B` → `dA = dY Bᵀ`, `dB = Aᵀ dY`) |
| `P = softmax(S)` | `dS = P ⊙ (dP - rowsum(dP ⊙ P))` (each row of `P` depends on the whole row of `S`) |
| `S = q kᵀ / sqrt(d)` | `dq = dS k / sqrt(d)`, `dk = dSᵀ q / sqrt(d)` |

The mask needs no backward handling: masked entries have `P = 0`, so their `dS` is 0.

### 7. FlashAttention

Same math, exact result; it is **IO-aware**. GPU main memory (HBM) is large but slow; on-chip SRAM is tiny but
fast, and attention is limited by moving data, not by arithmetic. FlashAttention loads blocks of `Q, K, V` into SRAM,
does all the work for a block there, and keeps a running max and running sum per query (online softmax) so the
`T × T` matrix never goes to HBM. Memory `O(T²)` → `O(T)`; compute stays `O(T²)`. The backward pass recomputes the
scores blockwise instead of storing them: more FLOPs, still faster.


## Backprop practice

### 8. Chain rule

Gradient of an input = incoming gradient × local derivative. Walk the forward lines in reverse. A variable used
twice gets the **sum** of its gradients (product rule: hold one factor fixed per path, then add).

Example, GateNet (`h2 = h1_scaled * S`, `S = h1_scaled.sum(1)`):

```
loss
 ↓  dlogits = (p − onehot) / B                  [B, C]
logits = h2 @ W2
 ↓  dW2 = h2ᵀ @ dlogits                         [H, C]
 ↓  dh2 = dlogits @ W2ᵀ                         [B, H]
h2 = h1_scaled * S
 ↓  dS = (dh2 * h1_scaled).sum(1, keepdim)      [B, 1]   S path: h1_scaled held fixed
 ↓  dh1_scaled = dh2 * S  +  dS                 [B, H]   direct path (S held fixed) + S path
h1_scaled = h1 * gamma
 ↓  dgamma = (dh1_scaled * h1).sum(0)           [H]
 ↓  dh1 = dh1_scaled * gamma                    [B, H]
h1 = relu(z1)
 ↓  dz1 = dh1 * (z1 > 0)                        [B, H]
z1 = X @ W1
 ↓  dW1 = Xᵀ @ dz1                              [D, H]
```

Three patterns cover everything: matmul `Y = A @ B` → `dA = dY @ Bᵀ`, `dB = Aᵀ @ dY`; elementwise `y = a * b` →
`da = dy * b` (the other factor is the multiplier); reuse → add.

### 9. Broadcast ↔ sum

A broadcast forward is a sum backward; a sum forward is a broadcast backward. Track the shapes and the operation:
**a gradient always has its variable's shape**, so reduce or expand until it does.

| Forward | Shapes | Backward |
|---|---|---|
| `y = x.sum(1, keepdim=True)` | `[4, 3] → [4, 1]` | `dx = dy * ones(4, 3)`: copy `[4, 1] → [4, 3]` |
| `y = x.mean(1, keepdim=True)` | `[4, 3] → [4, 1]` | same copy, divided by 3 |
| `y = a * s`, `s` is `[4, 1]` | `s` silently copied to `[4, 3]` | `ds = (dy * a).sum(1, keepdim=True)`: `[4, 3] → [4, 1]` |
| `y = a * g`, `g` is `[3]` | `g` copied over the batch to `[4, 3]` | `dg = (dy * a).sum(0)`: `[4, 3] → [3]` |
| `y = x + b`, `b` is `[3]` (bias) | `b` copied to `[4, 3]` | `db = dy.sum(0)` |

Whenever a `[B, 1]` or `[H]` meets a `[B, H]` in the forward pass, there is a hidden broadcast; its gradient needs a
sum.

### 10. Loss calculation shapes

Example: `B = 4` examples, `C = 3` classes.

```
logits               [4, 3]
logp = logits − logsumexp(logits, 1, keepdim)    [4, 3]   stable log-softmax
logp[arange(4), y]   [4]       one entry per row: the label's log-prob   (y = [0, 2, 1, 1])
loss = −mean(...)    scalar
p = logp.exp()       [4, 3]    keep it for backward

dlogits = p.clone()                     [4, 3]
dlogits[arange(4), y] −= 1              [4, 3]   p − onehot, without building the one-hot
dlogits /= 4                            [4, 3]   the loss is a mean over B
```

Tokens `(B, T, V)` with labels `(B, T)`: flatten to `(B*T, V)` and `(B*T,)`, and it is the same 2-D case.

### 11. Credit assignment

Each backward op hands every input its share of the output's gradient.

| Forward | Who contributed | Backward |
|---|---|---|
| sum over n | each input, weight 1 | copy the gradient to all n |
| mean over n | each input, weight 1/n | copy, divided by n |
| broadcast to n places | one value, n uses | sum the n gradients |
| max over n | the winner only | all to the argmax, 0 to the rest |

### 12. gather

`x.gather(dim, index)`: in each row, pick the entry `index` points to, along dimension `dim`. Gather along the
dimension the index counts: labels and token ids count classes, so `dim=-1`:
`logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)` = `logp[arange(B), y]`. Picking each sequence's last token from
`(B, T, D)` counts time, so `dim=1`. The index has as many dims as `x`, and the output has the index's shape.
