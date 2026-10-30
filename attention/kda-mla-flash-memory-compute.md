# Attention mechanisms: where memory and compute actually go

Notes from working through GLM-5.3-Flash's architecture (2026-08-29). Companion
notebook with every computation runnable line by line: [`attention_vs_kda.ipynb`](attention_vs_kda.ipynb).

## The two GPU memories

A GPU has a large slow memory (268 GB on a B300) and a tiny fast memory on the
chip itself (hundreds of KB per compute unit, roughly 10x the bandwidth). All
math happens in the fast memory, so every kernel is a logistics problem: carry
numbers in, compute, carry results out. For attention, the carrying dominates
the arithmetic.

## Standard attention and the KV cache

**Quadratic compute in length; linear memory in length.**

Each token carries three vectors of head dimension d = 128: a query (what I am
looking for), a key (what I am about), and a value (what I contribute). To
produce its output, a token compares its query against the key of every
predecessor, converts the scores to percentages, and blends the predecessors'
values with those weights.

During generation this happens once per emitted token. The keys and values of
past tokens never change, so they are computed once and kept. That kept,
growing list is the KV cache: iteration t holds t entries and compares against
all of them, one more than the iteration before.

Two costs follow:

- Memory: one cache entry per token per layer. For a 45-layer model at hidden
  size 4096 this is about 737 KB per token, so about 70 GB per sequence at a
  96K context.
- Compute: token t compares against t predecessors, and 1 + 2 + ... + N sums
  to about N squared over 2. Doubling the context doubles both the number of
  tokens comparing and the number of predecessors each faces, so the work
  quadruples. That is what quadratic means here.

### The computation with every shape written out

One head, head dimension 128, hidden size 4096; the model runs 32 such heads
side by side and this is one of them. This is generation iteration t: one new
token arrives, the cache already holds t-1 entries. These are the exact lines
from the notebook, which runs them at d = 6 so the matrices are printable.

```
x                                 # (4096,)   the new token's hidden vector
q = x @ W_q                       # (4096,) @ (4096,128) -> (128,)
k = x @ W_k                       # (4096,) @ (4096,128) -> (128,)
v = x @ W_v                       # (4096,) @ (4096,128) -> (128,)

K_cache.append(k)                 # the cache grows by one entry
V_cache.append(v)
K = stack(K_cache)                # (t, 128)   one key per token so far
V = stack(V_cache)                # (t, 128)   one value per token so far

scores  = K @ q                   # (t,128) @ (128,) -> (t,)   one score per predecessor
weights = softmax(scores)         # (t,) -> (t,)   scores become percentages summing to 1
o       = weights @ V             # (t,) @ (t,128) -> (128,)   weighted blend of past values
```

The letter t appears in five of those shapes, and that is the whole story.
The two stacked lists are the memory cost: one more row every iteration. The
score line and the blend line are the compute cost: each walks all t rows, so
every iteration does a little more work than the one before. Summed over a
whole sequence, that is the quadratic.

During training or prefill all N tokens run at once. The score line then
produces the full N x N table, the thing FlashAttention breaks into tiles
below.

## FlashAttention: same math, less carrying

**Quadratic compute in length; linear memory in length. Both laws unchanged;
only data movement improves.**

FlashAttention changes neither cost law. It computes the identical scores and
blends, but in tiles sized to the fast memory, converting each tile to
percentages and folding it into the output while it is still on chip. The
N x N score table exists only as transient tiles; it is never written to the
large memory. At 96K tokens that table would be 18 GB per head per layer,
which does not even fit, so at modern scale this is not an optimization but
the only way exact attention runs at all. The price is that K and V are
re-read once per query tile. Arithmetic unchanged, data movement cut severalfold.

## KDA: replace the cache with a fixed-size state

**Linear compute in length; fixed memory (no KV cache, one fixed-size state).**

The starting observation: in the final blend, the keys and values only ever
appear together. Drop the softmax and regroup the multiplication, and the
entire history collapses into one product of the key list and the value list.
That product is a 128 x 128 matrix no matter how many tokens went in. Keep
that matrix, updated one token at a time, and the growing lists are
unnecessary.

KDA (Kimi Delta Attention) is this idea made workable. Per head it keeps one
128 x 128 matrix, the state: fixed in size, constantly rewritten. Per token,
three touches:

1. Forget: every row fades by its own learned gate, so stale content decays
   per channel.
2. Delta write: read what the state already returns for this token's key,
   store only the correction. Re-mentioning a key overwrites instead of
   piling up.
3. Read: the token's query times the state gives the output.

### The computation with every shape written out

Same head, same 128. S is the state, carried from token to token; there is no
cache. These are the notebook's exact lines, annotated at real size.

```
S                                 # (128, 128)   carried in from the previous token
x                                 # (4096,)      the new token's hidden vector

q = x @ W_q                       # (4096,) @ (4096,128) -> (128,)
k = x @ W_k                       # -> (128,)   then scaled to length 1
v = x @ W_v                       # -> (128,)
a = sigmoid(x @ W_a)              # -> (128,)   forget gates, one per channel, each in (0, 1)

S = a[:, None] * S                # 1. FORGET   (128,1) * (128,128) -> (128,128)
                                  #    row i of the state fades by gate a[i]
v_old = k @ S                     # 2a. what S already returns for this key
                                  #    (128,) @ (128,128) -> (128,)
S = S + outer(k, v - v_old)       # 2b. DELTA WRITE: a column times a row,
                                  #    (128,1) @ (1,128) -> (128,128), added to S
o = q @ S                         # 3. READ   (128,) @ (128,128) -> (128,)
```

No shape contains t. Each line touches at most the 16,384 numbers of the
state, the same at token 10 and at token 1,000,000. Fixed work per token,
times N tokens: linear. (The paper's full rule adds a learned per-token write
strength and folds the decay into the write step; every tensor keeps the
shapes above.)

The state is 16,384 numbers forever. What is lost is exactness: a fixed-size
summary cannot hold 96,000 tokens precisely, so retrieval is approximate.

## MLA and DSA: the two fixes in the exact-recall layers

**MLA alone: quadratic compute in length, linear memory in length, same as
standard attention. MLA + DSA: linear compute in length, linear memory in
length.**

MLA (multi-head latent attention) changes neither growth law. It only shrinks
each cache entry (about 55x smaller), so the memory line rises 55x more
slowly but still rises with every token, and every token still compares
against every predecessor.

What a cache entry is, and what MLA does to it:

```
# Standard: the entry is this token's key and value, one pair per head.
k = x @ W_k                    # (4096,) @ (4096,128) -> (128,)   per head
v = x @ W_v                    # -> (128,)                        per head
cache.append((k, v))           # 32 heads x 2 vectors x 128 = 8,192 numbers
                               # stored per token, per layer, forever

# MLA: the entry is one small latent, shared by all 32 heads.
c = x @ W_down                 # (4096,) @ (4096, latent) -> (latent,)
cache.append(c)                # one small vector per token per layer,
                               # about 55x fewer numbers than the pair above

# At read time each head rebuilds its key and value from the latent:
k_h = c @ W_up_k[h]            # (latent,) -> (128,)
v_h = c @ W_up_v[h]            # (latent,) -> (128,)
# then attention proceeds exactly as before: still one comparison
# per predecessor, which is why compute stays quadratic.
```

It is DSA (DeepSeek sparse attention) that fixes compute: a tiny scorer picks
each token's top 2,048 predecessors and the expensive attention runs only
against those, so compute becomes linear in length while memory stays linear.
A residual all-pairs term survives in the cheap scorer.

## The growth laws in one table

| Attention type | Memory growth (cache/state) | Compute growth | What it changes |
|---|---|---|---|
| Standard | linear, full K,V per token | quadratic (N^2/2) | baseline |
| + FlashAttention | linear, same cache | quadratic, same math | data movement only |
| MLA | linear, ~55x smaller constant | quadratic | the memory constant |
| MLA + DSA | linear, compressed | linear (N x 2048) | the compute exponent |
| KDA | flat (16,384-number state) | linear (3 ops x N) | both laws, at the cost of exactness |

Measured per head per layer (d = 128):

| N | standard: cache numbers | standard: comparisons | KDA: state | KDA: ops |
|---|---|---|---|---|
| 1,000 | 256,000 | 500,500 | 16,384 | 3,000 |
| 12,000 | 3,072,000 | 72,006,000 | 16,384 | 36,000 |
| 96,000 | 24,576,000 | 4,608,048,000 | 16,384 | 288,000 |
| 1,000,000 | 256,000,000 | 500,000,500,000 | 16,384 | 3,000,000 |

## Which phase each concern belongs to

- Cache memory is generation-only. Training sees the whole sequence at once
  and never builds a cache; its growing memory is activations, linear in
  tokens for every attention type.
- The compute laws apply to both phases in full. Training pays the quadratic
  or the linear in every forward and backward pass.
- FlashAttention helps both, training arguably more: recomputing tiles in the
  backward pass keeps the score table out of activation memory.
- None of this touches parameter-scale memory. Weights and optimizer state
  ignore context length entirely, and in practice they are what run out
  first when training a 320B model (measured: 233 GB per GPU at a 24-GPU
  sharding, 175 GB at 32 GPUs).

## How GLM-5.3-Flash assembles the pieces

45 layers: 34 KDA layers (flat memory, linear work) interleaved with 11
MLA + DSA layers (compressed cache, top-2048 lookup) in a 3:1 pattern, no
positional embeddings in the lookup layers, mixture-of-experts feed-forward
throughout (8 of 288 experts per token; 320B total, 18B active). At 96K
context the per-sequence memory is a flat 36 MB plus roughly 1.2 GB of
compressed cache, versus roughly 70 GB for a standard model of the same size.

## References

- FlashAttention: Dao et al., 2022. https://arxiv.org/abs/2205.14135
- Kimi Linear / KDA: https://arxiv.org/abs/2510.26692
- DeepSeek Sparse Attention: DeepSeek-V3.2 lineage; overview at
  https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/
- GLM-5.3-Flash architecture notes: https://sebastianraschka.com/blog/2026/glm-5-3-flash-architecture-notes.html
