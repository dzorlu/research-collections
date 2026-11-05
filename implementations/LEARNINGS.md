# Learnings

## Drill 1: policy losses

### 1. Tensor mechanics

1. **Masking.** A plain mean counts the padded slots. Divide by the number of real tokens instead.

```python
x                                                       # (B, T)
mask                                                    # (B, T)  1 = real token, 0 = padding
loss = (x * mask).sum() / mask.sum()                    # ()      over the whole batch
per_row = (x * mask).sum(-1) / mask.sum(-1)             # (B,)    per response
```

2. **Broadcasting starts from the last dimension.** Line things up per row with keepdim or unsqueeze. With a square
   matrix the wrong version runs silently and gives wrong numbers.

```python
x = torch.randn(4, 3)                                   # (4, 3)
x - torch.randn(4)                                      # (4, 3) - (4,)    error: last dims 3 vs 4
x - torch.randn(4, 1)                                   # (4, 3) - (4, 1)  works: stretches across the 3 columns
x - x.mean(-1, keepdim=True)                            # (4, 3) - (4, 1)  -> (4, 3)
```

3. **view vs reshape, strides.** A stride is the memory jump per dimension. view never copies and fails on a
   non-contiguous tensor; reshape copies when it must.

```python
x = torch.arange(6).view(2, 3)                          # (2, 3)  stride (3, 1)
t = x.T                                                 # (3, 2)  stride (1, 3), not contiguous
t.view(-1)                                              # error
t.reshape(-1)                                           # (6,)    works, copies
```

4. **gather picks one entry per row by index.** See section 11.

```python
logits                                                  # (B, T, V)
y                                                       # (B, T)
logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)          # index (B, T, 1) -> (B, T, 1) -> (B, T)
```

### 2. Numerics

5. **Ratios in log space.** The policy stores log-probs. Subtracting two logs is the log of their ratio, so take
   exp of the difference. Dividing two log-probs means nothing. Staying in log space also avoids forming tiny
   probabilities and dividing them.

```python
# log a - log b = log(a / b)
logp_new, logp_old                                      # (B, T) each
r = torch.exp(logp_new - logp_old)                      # (B, T)  = p_new / p_old

# p_new = 0.5, p_old = 0.25
math.exp(-0.69 - (-1.39))                               # 2.0   right
-0.69 / -1.39                                           # 0.5   meaningless
```

6. **Geometric mean = exp of the mean of the logs.** It is the n-th root of the product; logs turn the product into
   a sum. Ratios multiply (a response's probability is the product of its token probabilities), so this is the
   right average for them: doubling and halving cancel.

```python
# (a * b * c) ** (1/3) == exp((log a + log b + log c) / 3)
math.log(2), math.log(0.5)                              # 0.693, -0.693
geo_mean([2, 0.5])                                      # 1.0
arith_mean([2, 0.5])                                    # 1.25

# GSPO's per-response ratio
s = torch.exp(((logp_new - logp_old) * mask).sum(-1) / mask.sum(-1))    # (B, T) -> (B,)
```

7. **Clean padding before exp.** Garbage in a padded slot becomes inf after exp, and inf times 0 is nan. Masking the
   difference before the exp works for finite garbage; torch.where is safe for any garbage.

```python
torch.exp(torch.tensor(100.0)) * 0                      # ()      nan
diff = torch.where(mask.bool(), logp_new - logp_old, 0.0)    # (B, T)  padded slots set to 0
r = torch.exp(diff)                                     # (B, T)  finite everywhere
```

8. **log-softmax = scores minus logsumexp**, over the vocabulary. logsumexp is the log of the softmax denominator:
   one number per position that shifts the scores so the probabilities sum to 1. The max trick (subtract the
   largest score first) keeps exp from overflowing. Never take the log of a softmax: tiny probabilities underflow
   to 0 and log 0 is -inf.

```python
z                                                       # (B, T, V)
lse = torch.logsumexp(z, dim=-1, keepdim=True)          # (B, T, 1)  = m + log(sum(exp(z - m))), m = max(z)
logp = z - lse                                          # (B, T, V)

torch.log(torch.softmax(z * 200, -1))                   # (B, T, V)  -inf entries
```

   **To remember:** a log-prob is your score minus the soft max. logsumexp is a little above the largest score, so a
   log-prob is how far below the leader you are. Max trick: subtract the leader first, so the leader sits at 0.

```python
z = torch.tensor([10.0, 0.0, 0.0])                      # (3,)
torch.logsumexp(z, 0)                                   # ()    10.0001
z - torch.logsumexp(z, 0)                               # (3,)  [-0.0001, -10.0001, -10.0001]
```

### 3. Gradients

9. **Minus sign in front of what you want to increase.** The optimizer always minimizes. Terms to decrease go in
   with a plus.

```python
A                                                       # (B, 1)
logp                                                    # (B, T)
loss = -A * logp                                        # (B, T)  want log p of good tokens up
loss = -A * logp + beta * kl - c * entropy              # (B, T)  KL down, entropy up
```

10. **The gradient of -log p with respect to the logits is p_hat - onehot.** A negative gradient on the loss means
    the value goes up.

```python
p_hat, onehot                                           # (B, T, V) each
w, A, mask, norm                                        # (B, T, 1) each, broadcast over V
dz = w * A * (p_hat - onehot) * mask / norm             # (B, T, V)
w_new = w - lr * grad                                   # grad < 0 -> w goes up
```

11. **detach makes the weight a constant.** Without it, autograd also differentiates the ratio and the product rule
    adds a meaningless term. It also avoids in-place writes into the autograd graph.

```python
w = torch.exp(logp_new - logp_old).detach()             # (B, T)
loss = -w * A * logp_new                                # (B, T)  grad: -w * A * grad(logp_new)
```

12. **The loss value is meaningless in policy gradient.** It hovers around 0 and can have either sign. Only the
    gradient matters; watch the reward.

### 4. The methods

13. **The four modes differ only in the per-token weight w.**

```python
r = torch.exp(logp_new - logp_old).detach()             # (B, T)
A                                                       # (B, 1)

w = torch.ones_like(r)                                  # (B, T)  REINFORCE
w = r.clone()                                           # (B, T)  PPO: stop once moved far enough
w[(A > 0) & (r > 1.28)] = 0
w[(A < 0) & (r < 0.8)] = 0
w = r.clamp(0, 4)                                       # (B, T)  CISPO: never zero, only capped
s = torch.exp(masked_mean(logp_new - logp_old))         # (B, 1)  GSPO: one ratio per response,
                                                        #         PPO's zeroing on s, whole response kept or dropped
loss = -(w * A * logp_new)                              # (B, T)  then the masked mean
```

14. **PPO's two forms are equal.** The weight form detaches; the textbook form has no log p and no detach.

```python
loss = -w.detach() * A * logp_new                       # (B, T)  weight form
loss = -torch.minimum(r * A, r.clamp(0.8, 1.28) * A)    # (B, T)  textbook form, r keeps its gradient
```

15. **norm only changes the divisor.** GSPO always uses per response.

```python
x                                                       # (B, T)  masked per-token losses
x.sum() / mask.sum()                                    # ()      "token": total real tokens
(x.sum(-1) / mask.sum(-1)).mean()                       # ()      "seq": (B,) per response, then over B
```

## Drill 3: attention

### 5. Forward pass and KV cache

**Forward.** Each query's output is a weighted average of the values it may see. Wo is applied after merging the
heads (it mixes them), never to x.

```python
x                                                       # (B, T, D)
Wq, Wk, Wv, Wo                                          # (D, D) each
q, k, v = x @ Wq, x @ Wk, x @ Wv                        # (B, T, D) each, then split -> (B, H, T, dh)
S = q @ k.transpose(-1, -2) / dh ** 0.5                 # (B, H, T, dh) @ (B, H, dh, T) -> (B, H, T, T)
S = S.masked_fill(mask, float("-inf"))                  # (B, H, T, T)
P = torch.softmax(S, dim=-1)                            # (B, H, T, T)
out = P @ v                                             # (B, H, T, T) @ (B, H, T, dh) -> (B, H, T, dh)
out = merge(out) @ Wo                                   # (B, T, D) @ (D, D) -> (B, T, D)
```

**Split and merge heads.** Cut D first, then move the head dimension. view fails on the merge because the tensor
is not contiguous after the transpose.

```python
q.reshape(B, T, H, dh).transpose(1, 2)                  # (B, T, D) -> (B, T, H, dh) -> (B, H, T, dh)
out.transpose(1, 2).reshape(B, T, D)                    # (B, H, T, dh) -> (B, T, H, dh) -> (B, T, D)
```

**Scale by the head dim**, not D: each head's dot product sums over dh numbers. **Transpose the last two dims only**:
.T on an N-D tensor reverses every dimension.

```python
k.transpose(-1, -2)                                     # (B, H, T, dh) -> (B, H, dh, T)
k.T                                                     # (B, H, T, dh) -> (dh, T, H, B)   wrong
S = q @ k.transpose(-1, -2) / dh ** 0.5                 # (B, H, T, T)
```

**Mask before the softmax with -inf.** exp(-inf) = 0, so the softmax renormalizes over the visible keys. Zeroing
after the softmax leaves rows summing to less than 1. Masking replaces values, it does not multiply (0 times -inf is
nan), and the mask must be bool. diagonal=1 means strictly above the diagonal (the future); without it a token
cannot see itself. A (T, T) mask broadcasts over B and H.

```python
mask = torch.ones(T, T, dtype=torch.bool).triu(1)       # (T, T)        True = future
S = S.masked_fill(mask, float("-inf"))                  # (B, H, T, T)
```

**KV cache.** Past keys and values never change: store them and concatenate the new ones along time. The cache
starts as None.

```python
if cache is None:
    cache = {}
    T_past = 0
else:
    T_past = cache["k"].shape[-2]                       # cache["k"]: (B, H, T_past, dh)
    k = torch.cat([cache["k"], k], dim=2)               # (B, H, T_past + T, dh)
    v = torch.cat([cache["v"], v], dim=2)               # (B, H, T_past + T, dh)
```

**With a cache the mask is not square.** Query i sits at absolute position T_past + i, so the triangle shifts.
Check with a small case: 5 cached and 1 new gives a (1, 6) score matrix with nothing masked.

```python
mask = torch.ones(T, T_past + T, dtype=torch.bool).triu(T_past + 1)    # (T, T_past + T)
```

### 6. Backward pass

Attention is a chain of three operations. Save q, k, v and P on the way forward, then go back up. The mask needs no
backward handling: masked entries have P = 0, so their dS is 0.

```python
q, k, v, dout                                           # (T, d) each
P                                                       # (T, T)

# out = P @ v                    matmul rule, see section 7
dP = dout @ v.T                                         # (T, d) @ (d, T) -> (T, T)
dv = P.T @ dout                                         # (T, T) @ (T, d) -> (T, d)

# P = softmax(S)                 each row of P depends on the whole row of S
dS = P * (dP - (dP * P).sum(-1, keepdim=True))          # (T, T) - (T, 1) -> (T, T)

# S = q @ k.T / sqrt(d)
dq = dS @ k / d ** 0.5                                  # (T, T) @ (T, d) -> (T, d)
dk = dS.T @ q / d ** 0.5                                # (T, T) @ (T, d) -> (T, d)
```

## Backprop practice

### 7. Chain rule

Gradient of an input = incoming gradient × local derivative. Walk the forward lines in reverse. A variable used
twice gets the **sum** of its gradients (product rule: hold one factor fixed per path, then add).

Example, GateNet:

```python
# forward
X                                                       # [B, D]
z1 = X @ W1                                             # [B, D] @ [D, H] -> [B, H]
h1 = relu(z1)                                           # [B, H]
h1_scaled = h1 * gamma                                  # [B, H] * [H] -> [B, H]
S = h1_scaled.sum(1, keepdim=True)                      # [B, H] -> [B, 1]
h2 = h1_scaled * S                                      # [B, H] * [B, 1] -> [B, H]
logits = h2 @ W2                                        # [B, H] @ [H, C] -> [B, C]
```

```
loss
 ↓  dlogits = (p − onehot) / B                  [B, C]
logits = h2 @ W2
 ↓  dW2 = h2ᵀ @ dlogits                         [H, B] @ [B, C] -> [H, C]
 ↓  dh2 = dlogits @ W2ᵀ                         [B, C] @ [C, H] -> [B, H]
h2 = h1_scaled * S
 ↓  dS = (dh2 * h1_scaled).sum(1, keepdim)      [B, H] -> [B, 1]   S path: h1_scaled held fixed
 ↓  dh1_scaled = dh2 * S  +  dS                 [B, H]             direct path (S held fixed) + S path
h1_scaled = h1 * gamma
 ↓  dgamma = (dh1_scaled * h1).sum(0)           [B, H] -> [H]
 ↓  dh1 = dh1_scaled * gamma                    [B, H]
h1 = relu(z1)
 ↓  dz1 = dh1 * (z1 > 0)                        [B, H]
z1 = X @ W1
 ↓  dW1 = Xᵀ @ dz1                              [D, B] @ [B, H] -> [D, H]
```

Three patterns cover everything:

```python
# matmul       Y = A @ B         A [m, k], B [k, n], Y [m, n]
dA = dY @ B.T                                           # [m, n] @ [n, k] -> [m, k]
dB = A.T @ dY                                           # [k, m] @ [m, n] -> [k, n]

# elementwise  y = a * b         a, b, y [m, n]; the other factor is the multiplier
da = dy * b                                             # [m, n]
db = dy * a                                             # [m, n]

# reuse        a used twice      add the gradients
da = da_path1 + da_path2                                # [m, n]
```

### 8. Broadcast ↔ sum

A broadcast forward is a sum backward; a sum forward is a broadcast backward. Track the shapes and the operation:
**a gradient always has its variable's shape**, so reduce or expand until it does. Whenever a [B, 1] or [H] meets a
[B, H] in the forward pass, there is a hidden broadcast, and its gradient needs a sum.

```python
x = torch.randn(4, 3)                                   # [4, 3]

# sum forward -> broadcast backward
y = x.sum(1, keepdim=True)                              # [4, 3] -> [4, 1]
dx = dy * torch.ones(4, 3)                              # [4, 1] -> [4, 3]

# mean forward -> broadcast backward, divided by n
y = x.mean(1, keepdim=True)                             # [4, 3] -> [4, 1]
dx = dy * torch.ones(4, 3) / 3                          # [4, 1] -> [4, 3]

# [4, 1] meets [4, 3]: s is silently copied across 3 columns -> sum over dim 1
y = a * s                                               # [4, 3] * [4, 1] -> [4, 3]
ds = (dy * a).sum(1, keepdim=True)                      # [4, 3] -> [4, 1]

# [3] meets [4, 3]: g is silently copied over the batch -> sum over dim 0
y = a * g                                               # [4, 3] * [3] -> [4, 3]
dg = (dy * a).sum(0)                                    # [4, 3] -> [3]

# bias, same pattern
y = x + b                                               # [4, 3] + [3] -> [4, 3]
db = dy.sum(0)                                          # [4, 3] -> [3]
```

### 9. Loss calculation shapes

Example: B = 4 examples, C = 3 classes. Keep p for the backward pass. The backward subtracts 1 at the label
positions, so no one-hot is needed.

```python
logits                                                  # [4, 3]
y = torch.tensor([0, 2, 1, 1])                          # [4]
logp = logits - torch.logsumexp(logits, 1, keepdim=True)    # [4, 3] - [4, 1] -> [4, 3]  stable log-softmax
picked = logp[torch.arange(4), y]                       # [4]     the label's log-prob, one per row
loss = -picked.mean()                                   # []      scalar
p = logp.exp()                                          # [4, 3]

dlogits = p.clone()                                     # [4, 3]
dlogits[torch.arange(4), y] -= 1                        # [4, 3]  p - onehot
dlogits /= 4                                            # [4, 3]  the loss is a mean over B
```

Tokens: flatten, and it is the same 2-D case.

```python
logits.reshape(-1, V)                                   # (B, T, V) -> (B*T, V)
y.reshape(-1)                                           # (B, T)    -> (B*T,)
```

### 10. Credit assignment

Each backward op hands every input its share of the output's gradient.

| Forward | Who contributed | Backward |
|---|---|---|
| sum over n | each input, weight 1 | copy the gradient to all n |
| mean over n | each input, weight 1/n | copy, divided by n |
| broadcast to n places | one value, n uses | sum the n gradients |
| max over n | the winner only | all to the argmax, 0 to the rest |

### 11. gather

In each row, gather picks the entry the index points to, along the given dimension. Gather along the dimension the
index counts: labels and token ids count classes, so the last dimension; a position in a sequence counts time. The
index has as many dims as the input, and the output has the index's shape.

```python
logp                                                    # (B, C)
y                                                       # (B,)
logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)            # index (B, 1) -> (B, 1) -> (B,)
logp[torch.arange(B), y]                                # (B,)    same result

h                                                       # (B, T, D)
last                                                    # (B,)    index of each sequence's last token
h.gather(1, last.view(B, 1, 1).expand(B, 1, D))         # index (B, 1, D) -> (B, 1, D)
```
