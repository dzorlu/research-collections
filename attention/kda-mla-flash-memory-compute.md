# Attention: where memory and compute go

Worked through on GLM-5.3-Flash's dimensions (2026-08-29). Every computation runs line by line in
[`attention_vs_kda.ipynb`](attention_vs_kda.ipynb).

Shapes: T or N tokens, D hidden size (4096), d head dimension (128), 32 heads, 45 layers.

## 1. Standard attention and the KV cache

Quadratic compute in length, linear memory in length.

Each token has a query (what it looks for), a key (what it is about) and a value (what it contributes). A token
scores its query against every earlier key, turns the scores into weights with a softmax, and averages the earlier
values with those weights. During generation the keys and values of past tokens never change, so they are kept:
that growing list is the KV cache.

One head, one generation step, t tokens so far:

```python
x                               # (D,)      the new token
q, k, v = x @ Wq, x @ Wk, x @ Wv    # (d,) each

K_cache.append(k)               # cache grows by one row
V_cache.append(v)
K, V = stack(K_cache), stack(V_cache)    # (t, d) each

scores = K @ q                  # (t,)      one score per earlier token
w = softmax(scores / sqrt(d))   # (t,)
o = w @ V                       # (d,)
```

Every shape with t grows each step:

- **Memory:** one K and one V row per token per layer: 2 x 4096 x 2 bytes x 45 layers ≈ 737 KB per token, about
  70 GB per sequence at 96K tokens.
- **Compute:** token t compares against t earlier tokens; 1 + 2 + ... + N ≈ N²/2. Double the context, four times
  the work.

During training or prefill all N tokens run at once and `scores` is the full N x N table.

## 2. FlashAttention

Same math, same growth laws; less data movement.

A GPU has a large slow memory (HBM) and a small fast memory on chip (SRAM), and attention is limited by moving
data, not by arithmetic. FlashAttention computes the same scores in tiles that fit on chip, keeps a running max and
running sum per query (online softmax), and folds each tile into the output before moving on. The N x N table never
goes to HBM. At 96K tokens that table would be 18 GB per head per layer, so this is how exact attention runs at
all at that length.

## 3. MLA: compressed KV

Same growth laws as standard attention; each cache entry is about 55x smaller.

Instead of a key and a value per head, MLA caches one small latent per token, shared by all heads. Each head
rebuilds its key and value from the latent when it reads.

```python
# standard: per token, per layer
cache.append((k, v))            # 32 heads x 2 x 128 = 8,192 numbers

# MLA: per token, per layer
c = x @ W_down                  # (latent,)   about 55x fewer numbers
cache.append(c)

k_h = c @ W_up_k[h]             # (d,)  rebuilt at read time, for head h
v_h = c @ W_up_v[h]             # (d,)
```

Every token still compares against every earlier token, so compute stays quadratic.

## 4. Sparse attention (DSA)

Linear compute in length for the main attention; memory still linear.

Each query attends to only k earlier tokens (k = 2,048) instead of all of them:

1. A small **indexer** scores every earlier token cheaply.
2. Keep the **top k**.
3. Run normal softmax attention over those k tokens only.

The indexer has its own small weights, separate from the attention's, so that scoring every pair stays cheap.

```python
# weights
Wq, Wk, Wv      # (D, d)    attention
Wo              # (d, D)
Wqi, Wki        # (D, di)   indexer, di small

x                                            # (B, T, D)
q, K, V = x @ Wq, x @ Wk, x @ Wv             # (B, T, d)
qi, Ki = x @ Wqi, x @ Wki                    # (B, T, di)

scores = relu(qi @ Ki.transpose(1, 2))       # (B, T, T)   cheap: small di, no softmax
future = ones(T, T, dtype=bool).triu(1)      # (T, T)      True = later token
scores = scores.masked_fill(future, -inf)    # (B, T, T)

idx = scores.topk(k, dim=-1).indices         # (B, T, k)   chosen earlier tokens

b = arange(B)[:, None, None]                 # (B, 1, 1)
Ks, Vs = K[b, idx], V[b, idx]                # (B, T, k, d)

p = softmax((Ks @ q[..., None]).squeeze(-1) / sqrt(d), dim=-1)    # (B, T, k)
out = (p[..., None] * Vs).sum(-2)            # (B, T, d)
y = out @ Wo                                 # (B, T, D)
```

Early queries have fewer than k earlier tokens; their extra picks land on -inf scores and get zero weight. The real
indexer has several heads, one shared key per token, and a learned weight per head; the sketch above is the
one-head case.

### 4.1 How the indexer is trained

`topk` returns integers, so it has no gradient. In a mixture of experts this is fine: the router's probability
`p_e` also multiplies the expert's output, so the router still gets a gradient through `p_e`. Here the indexer's
scores only choose tokens; the attention weights come from `q` and `K`. The main loss therefore gives the indexer
no gradient, and it gets its own loss: match where the attention actually put its weight.

```python
target = p.detach()                                       # (B, T, k)   attention weights on the chosen tokens
pred = log_softmax(scores.gather(-1, idx), dim=-1)        # (B, T, k)   indexer scores on the same tokens
loss_indexer = kl_div(pred, target)                       # trains Wqi, Wki only
loss = loss_main + loss_indexer
```

DeepSeek-V3.2 trains it in two stages: a dense warm-up where only the indexer learns to imitate full attention,
then sparse training where the forward pass uses the top-k tokens.

**Figure 1. Gradient flow in sparse training.**

```
                      x  (B, T, D)
          ┌───────────┴────────────────────┐
          │                                │  x.detach()
          ▼                                ▼
  q, K, V = x@Wq, x@Wk, x@Wv          qi, Ki = x@Wqi, x@Wki
          │                                │
          │                                ▼
          │                         scores = relu(qi @ Kiᵀ)      (B, T, T)
          │                                │
          │                    ┌───────────┴──────────────┐
          │                    ▼                          ▼
          │              idx = topk(scores)        sel_scores = scores[idx]
          │               (integers: no grad)             │
          │                    │                          │
          ▼                    ▼                          │
     Ks, Vs = K[idx], V[idx]   ◄── idx only says where    │
          │                                               │
          ▼                                               │
     p = softmax(q·Ks / √d)    (B, T, k)                  │
          │          └──────── p.detach() ─────► target ──┤
          ▼                                               ▼
     out = p @ Vs  →  y = out @ Wo              loss_indexer = KL(target ‖ softmax(sel_scores))
          │
          ▼
     loss_main
```

**Table 1. Which loss trains which weights.**

| Weights | From `loss_main` | From `loss_indexer` |
|---|---|---|
| `Wq, Wk, Wv, Wo` and earlier layers | yes, through the k chosen tokens | no (target is detached) |
| `Wqi, Wki` | no (blocked at `topk`) | yes |
| `x` through the indexer branch | none | no (indexer input is detached) |

The attention learns the task; the indexer learns to rank tokens the way the attention would.

### 4.2 Decode

At each step the new token scores all cached indexer keys (small) and reads only k rows of the main cache.

## 5. Linear attention (KDA)

Linear compute in length; fixed memory (one state, no cache).

Drop the softmax and the history collapses into one d x d matrix, the state, updated once per token. KDA (Kimi
Delta Attention) adds two things: a forget gate per channel, and the delta rule (store only the correction, so
writing a key again overwrites instead of adding up).

```python
S                               # (d, d)    the state, carried from token to token
q, k, v = x @ Wq, x @ Wk, x @ Wv    # (d,) each; k normalized
a = sigmoid(x @ Wa)             # (d,)      forget gate per channel

S = a[:, None] * S              # forget: row i decays by a[i]
v_old = k @ S                   # (d,)      what S returns for this key now
S = S + outer(k, v - v_old)     # (d, d)    delta write
o = q @ S                       # (d,)      read
```

No shape contains t: the work per token and the memory are the same at token 10 and token 1,000,000. The price is
exactness: a fixed-size state cannot hold every token precisely.

It looks like an LSTM (a recurrent state with a forget gate), with two differences that make it trainable at scale:
the state is a matrix read with a query, and the update is linear in the state with gates that depend only on the
input, so training can run as matrix multiplications over chunks instead of one step at a time.

## 6. The growth laws

**Table 2. Memory and compute growth by attention type.**

| Type | Memory | Compute | What it changes |
|---|---|---|---|
| Standard | linear (K, V per token) | quadratic (N²/2) | baseline |
| FlashAttention | linear, same cache | quadratic, same math | data movement only |
| MLA | linear, ~55x smaller | quadratic | the memory constant |
| DSA (on MHA or GQA) | linear, full-size cache | linear (N x 2,048) plus a cheap all-pairs indexer | the compute exponent |
| MLA + DSA | linear, ~55x smaller | linear (N x 2,048) plus a cheap all-pairs indexer | both constants: memory (MLA) and compute (DSA) |
| KDA | flat (d x d state) | linear | both, at the cost of exactness |

DSA does not shrink the cache: any earlier token can be chosen later, so all of them stay. MLA does not reduce
compute: every query still meets every token. Paired, each fixes the other's cost. They also fit mechanically:
the indexer picks one set of k tokens shared by all heads, and an MLA cache entry is one latent shared by all
heads, so a decode step gathers k latent rows once for every head.

**Table 3. Per head per layer, d = 128; sparse with k = 2,048.**

| N | Standard: cache numbers | Standard: comparisons | Sparse: attention comparisons | Sparse: indexer pairs | KDA: state | KDA: ops |
|---|---|---|---|---|---|---|
| 1,000 | 256,000 | 500,500 | 500,500 | 500,500 | 16,384 | 3,000 |
| 12,000 | 3,072,000 | 72,006,000 | 22,479,872 | 72,006,000 | 16,384 | 36,000 |
| 96,000 | 24,576,000 | 4,608,048,000 | 194,511,872 | 4,608,048,000 | 16,384 | 288,000 |
| 1,000,000 | 256,000,000 | 500,000,500,000 | 2,045,903,872 | 500,000,500,000 | 16,384 | 3,000,000 |

Sparse attention keeps the full cache (compressed if it uses MLA), so its cache column equals standard's. Below
2,048 tokens it is the same as full attention. Above, the expensive attention grows linearly (about N x 2,048) while
the indexer still scores every pair, but each indexer pair is cheap: a small dimension, no softmax, low precision.

The cache exists only in generation; training has no cache, and its growing memory is activations. The compute
laws apply to training and generation alike. Weights and optimizer state do not depend on context length, and
when training a 320B model they are what runs out first (233 GB per GPU with 24 GPUs, 175 GB with 32).

## 7. GLM-5.3-Flash

45 layers in a 3:1 pattern: 34 KDA layers and 11 MLA + DSA layers (no positional embeddings in the latter),
with a mixture-of-experts feed-forward in every layer (8 of 288 experts per token, 320B total, 18B active). At 96K
tokens a sequence needs a flat 36 MB of KDA state plus about 1.2 GB of compressed cache, against about 70 GB for
standard attention at the same size.

## References

- FlashAttention: Dao et al., 2022. https://arxiv.org/abs/2205.14135
- Kimi Linear / KDA: https://arxiv.org/abs/2510.26692
- DeepSeek-V3.2 (DSA, indexer training, Section 2.1): https://arxiv.org/html/2512.02556v1
- DeepSeek sparse attention overview: https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/
- GLM-5.3-Flash architecture notes: https://sebastianraschka.com/blog/2026/glm-5-3-flash-architecture-notes.html
