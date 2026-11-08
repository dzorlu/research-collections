# Roadmap

Target: post-training role, from data to large-scale RL optimization. The three coding hints stay the anchor
(tensor manipulation; distributed computing and efficiency; components within a training pipeline), scoped to
post-training and RL. Attention, especially the forward pass, gets its own track. Forward and backward passes by
hand are the daily warm-up. Every drill is judged on two yardsticks: correct (it matches a single-process or
unsharded reference) and efficient (bytes, calls, memory per rank or per request, overlap, FLOPs per byte).
No LeetCode, no Q-learning. RL loss corrections and async RL design are existing strengths: they show up as talking
points, not drills.

Format: every new drill is a `.py` file like `distributed/reference/01_ring_allreduce.py`. The docstring states the
task and the yardsticks, the solution goes between `# >>> TODO` and `# <<< TODO`, and the tests under `__main__`
assert correctness first, then efficiency. Distributed drills run on gloo through `_harness.launch` at 2 and 4
ranks (6 for the weight-reshard drill), with sizes not divisible by N. Write blank-page first, then open the
reference listed for the drill. Collectives are hand-rolled on local tensors, as current large-scale stacks do;
DTensor appears only in one optional cross-check.

Budget: 15 must-do drills, about 20 hours, over 9 days plus one buffer day. Optional drills are listed; skip them
unless a day finishes early.

## 0. Day-by-day plan

**Table 1. Plan.** Warm-up 20 to 25 minutes each morning; main work about 2.5 to 3.5 focused hours.

| Day | Warm-up | Main work | Whiteboard |
|---|---|---|---|
| 1 | W1 two-layer net, timed | A1 full MHA + KV cache (timed redo), A2 GQA and sliding window, A3 varlen | KV bytes per token |
| 2 | W3 attention backward | A4 mixed batch + paged KV, A5 multi-turn KV reuse | KV per request, per GPU |
| 3 | W7 GQA backward | A6 LSE merge (local and split across ranks), A7 compressed (MLA) | MHA vs GQA vs MLA cache bytes |
| 4 | W2 LayerNorm, RMSNorm | A8 linear attention (parallel, recurrent, chunked, gated delta), A9 sparse top-k | prefill vs decode FLOPs per byte |
| 5 | W4 SwiGLU | D2 data parallel with buckets, D3 tensor-parallel MLP | ring bytes, NVLink vs cross-rack |
| 6 | W5 top-k gating | D4 FSDP by hand, with global grad-norm clip | Adam and ZeRO memory per rank |
| 7 | W6 policy gradient through logits | T1 MoE routing with router replay, D6 MoE expert parallel, D5 weight reshard | weight-update bandwidth |
| 8 | W1 again, timed | P1 length-balanced packing, P2 chunked log-probs + policy loss | logits memory |
| 9 | W3 again, timed | P3 async rollout buffer with staleness bound and resume | inference share, Little's law, checkpoints |
| 10 | any | Mock: A4 and D5 from a blank page under time; talking points out loud; then optional drills | all of section 6 |

Seven-day version: day 1 as written; day 2 A4 + A7; day 3 A8 + A9 (A5 and A6 become talking points); day 4 D3 + D4; day 5 D5 + D6; day 6 P1 + P2; day 7 P3, then the mock with whatever time is left.

## 1. Attention track (forward pass first)

Shared setup: x (T, D), H query heads, Hkv KV heads, head dim dh, float64 on CPU. Every drill is checked against a
plain loop over requests or sequences that calls the reference `mha` from `reference/03_attention.py`.

**A1. Causal MHA with a KV cache and chunked-prefill offset.** Must, 25 min. Exists in `03_attention.ipynb`; redo
it from a blank page under a timer.
Correctness: token-by-token decode equals the full forward; a 5-token prefill then a 4-token chunk equals one
9-token pass (the mask shifts by the cached length).
Efficiency: state the KV bytes appended per token and why the cache never recomputes past K and V.
Compare afterwards: `reference/03_attention.py`.

**A2. GQA / MQA, sliding window, interleaved local and global layers.** Must, 45 min.
Spec: one attention function with `n_kv_heads` and `window` arguments; a model config that alternates local
(window W) and global layers. Share KV heads with a view (`expand` then `reshape` at the matmul, or a grouped
einsum), never `repeat_interleave` into the cache. A sliding-window decode cache keeps only the last W positions.
Correctness: Hkv = H matches MHA; Hkv = 1 is MQA; GQA matches MHA with KV repeated by hand; window masking matches a
loop; the ring-buffer cache for local layers gives the same decode outputs as a full cache with the window mask.
Efficiency: cache bytes per token per layer = 2 x Hkv x dh x bytes, and a local layer's cache is capped at W
tokens.

```python
q                                                       # (B, H, T, dh)
k                                                       # (B, Hkv, S, dh)    S = cached + new
qg = q.view(B, Hkv, H // Hkv, T, dh)                    # (B, Hkv, G, T, dh) G query heads per KV head
s = torch.einsum("bkgtd,bksd->bkgts", qg, k) / dh ** 0.5    # (B, Hkv, G, T, S)  no KV copy
i = torch.arange(T)[:, None] + (S - T)                  # (T, 1)  absolute query positions
j = torch.arange(S)[None, :]                            # (1, S)
mask = (j > i) | (i - j >= window)                      # (T, S)  True = masked; window = inf for global
```

Compare afterwards: `GQAttention`, `get_sliding_window_mask_mod` in
`~/src/torchtitan/torchtitan/models/common/attention/attention.py`.

**A3. Varlen attention over packed sequences from cu_seqlens.** Must, 45 min.
Spec: sequences of different lengths concatenated into q, k, v of shape (T_total, H, dh) with `cu_seqlens`
(n + 1,). Build per-token sequence ids and positions from cu_seqlens, then attention with a causal document mask;
no padding. RoPE uses the per-document positions.
Correctness: matches attention on each sequence run alone.
Efficiency: report useful score entries sum(L_i^2) against padded B x max(L)^2 (about 4% on a long-tailed
batch of 64, so padding wastes 96% of attention work), and the (T, T) bool mask bytes, which is why kernels take
cu_seqlens instead of a mask.

```python
cu = torch.tensor([0, 3, 5, 9])                         # (n + 1,)  3 sequences: lengths 3, 2, 4
seq = torch.repeat_interleave(torch.arange(3), cu.diff())    # (T,)  [0, 0, 0, 1, 1, 2, 2, 2, 2]
pos = torch.arange(cu[-1]) - cu[seq]                    # (T,)  [0, 1, 2, 0, 1, 0, 1, 2, 3]
allowed = (seq[:, None] == seq[None, :]) & (pos[:, None] >= pos[None, :])    # (T, T)
```

Compare afterwards: `VarlenAttentionMetadata`, `get_document_mask_mod` in
`~/src/torchtitan/torchtitan/models/common/attention/attention.py`.

**A4. Mixed batch of decodes and chunked prefills with a paged KV cache.** Must, 1.5 h. The core inference drill.
Spec: requests with (cached, new) token counts are flattened into one batch of new tokens. Build
`query_start_loc`, `seq_lens`, per-token request id and absolute position, and `slot_mapping` from a `block_table`.
Write the new K, V into a paged cache of shape (num_blocks, block_size, Hkv, dh) with `index_copy_` on slots; for
attention, gather each request's K, V through its block table and mask by absolute position. Order decodes before
prefills, as the runner does.
Correctness: on the example below each request's output equals A1 run on that request alone; the cache contents
equal the contiguous cache of A1.
Efficiency: memory per request = ceil(seq_len / block_size) blocks, waste below one block per request; new-token
compute only (no recompute of cached positions); count blocks allocated against tokens stored.

```python
cached = torch.tensor([16, 3, 0, 7])                    # (R,)  decode, decode, prefill, chunked prefill
new = torch.tensor([1, 1, 6, 3])                        # (R,)
seq_lens = cached + new                                 # (R,)    [17, 4, 6, 10]
qsl = torch.cat([torch.zeros(1, dtype=torch.long), new.cumsum(0)])    # (R + 1,)  [0, 1, 2, 8, 11]
req = torch.repeat_interleave(torch.arange(4), new)     # (T,)    request of each new token
pos = torch.arange(int(new.sum())) - qsl[req] + cached[req]    # (T,)  absolute positions
slot = block_table[req, pos // bs] * bs + pos % bs      # (T,)    where each new K, V is written
```

Compare afterwards: the metadata slides in the attention-backend presentation
(`/private/tmp/claude-501/-Users-dzorlu-repos/deb89c54-86dd-49dc-944b-0ad5a02578bd/scratchpad/pdf/vllm_attn.txt`);
`~/src/torchtitan/torchtitan/models/common/attention/attention.py` for the varlen path.

**A5. KV reuse across multi-turn rollout turns.** Must, 45 min. Builds on A4.
Spec: a rollout alternates model turns and environment turns. On the next turn, keep the request's blocks, append
the tool output and the next prompt as a chunked prefill against the cached context, then decode. Full blocks are
keyed by a hash of (parent hash, block tokens) so identical prefixes across rollouts in a group share blocks with a
reference count. Then the weight-update case: a new policy version invalidates reuse, so the next turn recomputes
the whole context under the new weights (cache key includes the version), while a request already decoding keeps
its cache until its turn ends.
Correctness: every turn's outputs equal a from-scratch recompute of the full context under the weights in force.
Efficiency: tokens computed on turn t equal the new tokens only; shared prefixes are stored once (count blocks for
a group of G rollouts with a common prompt: prompt blocks once, not G times).
Compare afterwards: the prefix-caching notes in A4's sources; your own rollout engine code.

**A6. Online softmax and LSE merge of partial attention, local and across ranks.** Must, 45 min.
Spec: split one request's KV into chunks, attend to each chunk separately returning (out, lse), merge. Then the
distributed version: KV sharded by token across N ranks (decode context parallel), each rank attends with the full
query, all_gather the (out, lse) pairs, merge.
Correctness: matches full attention for decode and for a causal chunk; a chunk where every key is masked
(lse = -inf) merges without NaN.
Efficiency: KV per rank = 1 / N; bytes moved per decode token = N x (H x dh + H) per layer for the gathered partials,
independent of context length; compare with all-gathering the KV itself.

```python
o1, lse1, o2, lse2                                      # (H, Tq, dh), (H, Tq) partials over disjoint KV
lse = torch.logaddexp(lse1, lse2)                       # (H, Tq)
o = torch.exp(lse1 - lse)[..., None] * o1 + torch.exp(lse2 - lse)[..., None] * o2    # (H, Tq, dh)
```

Compare afterwards: `attention_online` in `reference/03_attention.py`; `update_out_and_lse` in
`~/src/picotron/picotron/context_parallel/context_parallel.py`.

**A7. Compressed attention: MLA with absorbed projections for decode.** Must, 1 h.
Spec: cache only a latent c (T, dc) plus a shared RoPE key (T, dr). Naive path: up-project c to per-head K and V.
Absorbed decode path: fold W_uk into the query and W_uv into the output projection, attend directly over the latent.
Correctness: absorbed equals naive. Efficiency: cache bytes per token (dc + dr) x bytes per layer against GQA;
FLOPs per byte rises because all H heads read one latent row.
Compare afterwards: `MLAAttention`, `materialize_mla_kv` in `~/src/torchtitan/torchtitan/models/common/attention/mla.py`;
`~/repos/research-collections/attention/kda-mla-flash-memory-compute.md`.

**A8. Linear attention: fixed-size state instead of a KV cache.** Must, 1 h.
Spec: drop the softmax, so causal attention becomes a running state. Implement three forms and show they agree:
the parallel form (masked q kᵀ without softmax, then @ v), the recurrent form (state S += k vᵀ, out = q S), and the
chunked form (parallel inside a chunk of C tokens, recurrent state across chunks). Then add a per-channel forget gate
and the delta rule (erase what the state returns for k before writing v), which is the KDA / gated DeltaNet update.
Correctness: parallel equals recurrent equals chunked for plain linear attention; the gated delta recurrence matches
a token-by-token loop; overwriting a key with a new value returns the new value (the delta rule), plain accumulation
returns the sum.
Efficiency: state bytes per head = dk x dv x bytes, constant in T, against KV bytes that grow with T; work per token
O(dk x dv) instead of O(T x dh). State why the chunked form exists: the recurrent form is sequential, the parallel
form is O(T²), chunks give matmuls with O(T x C) work.

```python
q, k, v                                                 # (T, dk), (T, dk), (T, dv)
out_par = (torch.tril(q @ k.T)) @ v                     # (T, T) @ (T, dv) -> (T, dv)   parallel, no softmax
S = torch.zeros(dk, dv)                                 # (dk, dv)  the whole memory of one head
for t in range(T):
    S = S + torch.outer(k[t], v[t])                     # (dk, dv)  write
    out_rec[t] = q[t] @ S                               # (dv,)     read
# gated delta rule: S = diag(alpha_t) S;  S = S + beta_t * outer(k_t, v_t - k_t @ S)
```

Compare afterwards: `~/repos/research-collections/attention/attention_vs_kda.ipynb` (your notes, parts 2 to 4);
`~/src/torchtitan/torchtitan/models/common/attention/linear_attention.py`, `kda.py`, `gdn.py`.

**A9. Sparse top-k attention with a lightweight indexer.** Must, 1 h.
Spec: a small indexer (few heads, small dim, ReLU-weighted scores, as in DeepSeek-style sparse attention) scores
every past token for each query; take the causal top-k indices per query; gather those K, V rows and run softmax
attention over only k keys. Decode version: one query, the indexer reads all cached index keys, attention reads k
rows of the main cache.
Correctness: with k >= sequence length it equals full causal attention; for small k it equals full attention with a
mask built from the top-k indices; future positions are never selected. Edge case: early queries have fewer
than k valid keys, so topk returns masked positions; drop selections whose score is -inf.
Efficiency: main attention FLOPs O(T x k) instead of O(T²); KV rows read per decode step = k instead of T; the
indexer still touches every past token, but at a small dimension (count its bytes per token against the main
cache). Talking point: the indexer's top-k must match between inference and training (indexer replay), the same
consistency problem as MoE router replay.

```python
qi, ki                                                  # (T, Hi, di), (T, di)   small indexer query and key
w                                                       # (T, Hi)                per-head weights
score = (torch.relu(torch.einsum("thd,sd->ths", qi, ki)) * w[..., None]).sum(1)    # (T, T)
score = score.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
idx = score.topk(min(k, T), dim=-1).indices            # (T, k)   selected past positions per query
ks, vs = K[idx], V[idx]                                 # (T, k, H, dh)  gathered rows
```

Compare afterwards: `~/src/torchtitan/torchtitan/models/deepseek_v4/attention.py` and `compressor.py` (indexer
and compressed sparse attention), and the sparse-MLA and indexer slides in the attention-backend presentation.

## 2. Distributed computing and efficiency

The harness logs logical tensor bytes for collectives and wire bytes for point-to-point ops.

**D1. Ring all-reduce from send/recv.** Done.

**D2. Data parallel with bucketed, overlapped all-reduce and no_sync accumulation.** Must, 1.5 h.
Spec: per-parameter post-accumulate-grad hooks fill fixed-size flat buckets; launch `all_reduce(async_op=True)` per
full bucket; wait before the step; skip communication on the first k - 1 micro-batches.
Correctness: weights after 3 steps match single-process training on the global batch, loss scaled to the global
token-weighted mean.
Efficiency: calls per step = number of buckets; bytes logged = gradient size once per step for any k; the first
launch happens before the last gradient hook (overlap, from an event list the drill records).
Compare afterwards: `~/src/picotron/picotron/data_parallel/bucket.py`.

**D3. Tensor-parallel MLP with Megatron f/g autograd functions.** Must, 1.5 h.
Spec: W1 column-split, W2 row-split; f is identity forward and all-reduce backward, g the reverse.
Correctness: output, input grad and gathered weight grads match the unsharded MLP.
Efficiency: exactly one all_reduce forward and one backward per MLP, each B x T x D x bytes.
Optional (30 min): sequence-parallel variant (all_gather and reduce_scatter instead; same wire bytes, norm and
residual activations 1 / N per rank).
Compare afterwards: `~/src/picotron/picotron/tensor_parallel/tp_communications.py`.

**D4. FSDP / ZeRO-3 by hand, with global grad-norm clipping.** Must, 2.5 h. The trainer side of every RL stack.
Spec: each rank stores a 1 / N slice of each block's flat params; all-gather before a block's forward, free after,
re-gather in backward, reduce-scatter grads when the block's backward ends; global norm = all_reduce of the sum of
squares of local slices; AdamW on the local slice with fp32 master and m, v.
Correctness: matches single-process AdamW plus `clip_grad_norm_` over 3 steps.
Efficiency: all_gather_into_tensor calls = 2L per step, reduce_scatter_tensor calls = L; wire bytes 1.5x DDP;
optimizer state per rank 1 / N; peak live full params at most 2 blocks.

```python
P = n_params * 8                                        # bytes, fp64 in the harness
ddp_wire = 2 * (N - 1) / N * P                          # one all-reduce
fsdp_wire = 3 * (N - 1) / N * P                         # gather fwd + gather bwd + reduce-scatter
```

Optional (20 min): wrap the same model with `fully_shard` and count collectives under `CommDebugMode`; the counts
should match yours.
Compare afterwards: `~/src/torchtitan/torchtitan/distributed/fsdp.py`, `FSDP.md`;
`~/src/nanotron/src/nanotron/optim/zero.py`.

**D5. Weight resharding from the trainer layout to the inference layout.** Must, 2 h. The post-training-specific
distributed drill.
Spec: 6 ranks. Ranks 0 to 3 are the trainer: every weight is an FSDP dim-0 shard (1 / 4 of the rows). Ranks 4 and 5
are one inference replica with TP = 2: fused QKV and MLP up are column-parallel (split by output rows, grouped by
head so each TP rank gets whole Q heads and whole KV heads), O and MLP down are row-parallel (split by input
columns). For each (trainer rank, inference rank) pair, compute the intersection of what the trainer holds and what
the inference rank needs, send only that slice with `batch_isend_irecv`, and assemble. Tag the push with a policy
version and a checksum per weight; the inference ranks adopt the new weights only if every checksum matches,
otherwise keep the old version.
Correctness: each inference rank's shards equal the TP = 2 shards cut from the full weight; a corrupted message
leaves the old version in place.
Efficiency: bytes received per inference rank = exactly W / 2 (versus (N - 1) / N x W per rank for a naive
all-gather to every trainer rank and then a full copy to each inference rank); count messages: column-parallel
weights need 2 senders per inference rank, row-parallel weights need all 4 (each sends W / 8); optional bucketing of
small weights into one flat buffer to cut message count.

```python
W                                                       # (out, in) full weight, never materialized
trainer_rows = (r * out // 4, (r + 1) * out // 4)       # FSDP shard on trainer rank r
need_col = (j * out // 2, (j + 1) * out // 2), (0, in)  # column-parallel, inference rank j: rows
need_row = (0, out), (j * in // 2, (j + 1) * in // 2)   # row-parallel, inference rank j: columns
send = W_local[lo - r0 : hi - r0, c0:c1]                # intersection only
```

Compare afterwards: `UpdateWeightFromDistributed`, `UpdateWeightFromTensor` in
`~/src/slime/slime/backends/fsdp_utils/update_weight_utils.py` and
`~/src/slime/slime/backends/megatron_utils/update_weight_utils.py`.

**D6. MoE expert parallelism with uneven all_to_all_single.** Must, 1.5 h. Builds on T1.
Spec: E experts, E / N per rank, top-k; exchange per-destination counts with one `all_to_all_single`, dispatch with
split sizes, run local experts, reverse all_to_all, combine with router weights in the original order. Accept
replayed expert indices from T1.
Correctness: output and grads match a single-process MoE layer with the same capacity rule.
Efficiency: 1 count exchange + 2 token all_to_alls per forward; dispatch bytes per rank = T x k x D x bytes; received
tokens per expert never exceed capacity.
Compare afterwards: `~/src/distributed-training/parallelization/model/expert_parallel.py` (your own);
`AllToAllTokenDispatcher` in `~/src/torchtitan/torchtitan/models/common/moe/token_dispatcher.py`.

Optional, pre-training heavy:
- Ring context parallel with zigzag chunks (rank r holds chunks r and 2N - 1 - r), forward only, 1.5 h. Reuses A6's
  merge. Compare: `HeadTailCPLoadBalancer` in `~/src/torchtitan/torchtitan/distributed/context_parallel.py`.
- Pipeline 1F1B schedule simulator (bubble (p - 1) / (m + p - 1), peak in-flight p - s at stage s), 45 min.
  Compare: `~/src/picotron/picotron/pipeline_parallel/pipeline_parallel.py`.

## 3. Components within a training pipeline (RL post-training)

**P1. Length-balanced packing of variable-length rollouts across DP ranks.** Must, 1.5 h.
Spec: given rollout lengths, sample weights, R data-parallel ranks and a token budget per micro-batch, assign
rollouts to ranks to minimize the maximum load (Karmarkar-Karp or largest-first greedy), then first-fit-decreasing
into micro-batches within each rank; pad every rank to the same number of micro-batches (one all_reduce of the max)
so collectives line up. Each rollout keeps its sample weight. Loss normalization uses a global denominator from one
all_reduce, so the result does not depend on how rollouts were packed.
Correctness (harness, 4 ranks): every rollout appears exactly once; no micro-batch exceeds the budget; gradients
after the all-reduce equal a single-process loss over all rollouts with their weights.
Efficiency: packing efficiency = real tokens / (micro-batches x budget), target at least 0.95 on a long-tailed
synthetic length distribution; imbalance = max / mean rank load, target at most 1.05; optional: balance by
sum(L^2) as well to even out attention cost.

```python
lengths, weights                                        # (n,), (n,)
denom = (weights_local * n_tokens_local).sum()          # ()  local weighted token count
dist.all_reduce(denom)                                  # ()  now global, identical on every rank
loss = (w_tok * per_token_loss).sum() / denom           # ()  each micro-batch, then backward
```

Compare afterwards: `get_seqlen_balanced_partitions`, `karmarkar_karp` in `~/src/slime/slime/utils/seqlen_balancing.py`;
`pack_sequences` in `~/src/slime/slime/backends/fsdp_utils/data_packing.py`.

**P2. Chunked log-probs and policy loss over packed sequences.** Must, 1.5 h.
Spec: hidden (T_total, D) of packed rollouts with cu_seqlens, fp32 output head W (V, D), sampled tokens, recorded
inference log-probs, and the sampling nucleus recorded per token as padded token-id lists (T_total, K). In chunks of
1024 tokens: logits in fp32 with the sampling temperature, log-prob of the sampled token normalized over the
recorded nucleus, never forming (T_total, V). Then the policy loss on response tokens only (reuse your PPO / CISPO
weights and a truncated importance weight against the inference log-probs), normalized per sequence through
segment sums over the packed row. Backward by hand through the chunks, accumulating dW.
Correctness: log-probs, loss, dhidden and dW match a full-materialization version with autograd; with K = V the
nucleus version equals the plain log-softmax.
Efficiency: peak logits bytes = chunk x V x 4 instead of T_total x V x 4; FLOPs unchanged.

```python
logits = (h[c0:c1].float() @ W.float().T) / temp        # (C, V)  one chunk, fp32
nuc = logits.gather(-1, nucleus_ids[c0:c1])             # (C, K)  recorded nucleus, padded with -inf
logp = logits.gather(-1, y[c0:c1, None]).squeeze(-1) - torch.logsumexp(nuc, -1)    # (C,)
seq_loss = torch.zeros(n_seq).index_add_(0, seq_id, tok_loss * mask)    # (n_seq,) per-sequence sums
```

Compare afterwards: `compute_logprobs`, `ChunkedLossWrapper` in `~/src/torchtitan/torchtitan/components/loss.py`;
`calculate_log_probs_and_entropy`, `compute_policy_loss` in `~/src/slime/slime/utils/ppo_utils.py`;
`reference/01_policy_loss.py`.

**P3. Async rollout buffer with a staleness bound, policy versions and exact resume.** Must, 1.75 h.
Spec: a seeded discrete-event simulation in pure Python. Inference workers generate rollouts token by token with
random lengths; a trainer step takes fixed time, consumes a batch, and bumps the policy version; a weight update
lands mid-rollout and later tokens record the new version (the cache is kept). The buffer has a capacity (inference
pauses when full: admission control) and a max off-policy bound k (a rollout whose oldest token version is more than
k behind is dropped). The trainer prefetches the next batch during the current step. Checkpoint = buffer contents,
in-flight rollouts, RNG state, version, step.
Correctness: no consumed rollout exceeds the bound; the buffer never exceeds capacity; every token carries a
version; a run checkpointed at step s and resumed consumes exactly the same rollouts afterwards.
Efficiency: sweep capacity and k; report trainer idle fraction, mean staleness and drop rate. The trade-off: a
bigger buffer keeps the trainer fed but raises staleness.
Compare afterwards: `max_off_policy_steps`, `max_async_level` in `~/src/prime-rl/src/prime_rl/orchestrator/config.py`
and `orchestrator.py`; your own async RL code.

Optional:
- Training loop with exact resume (AdamW by hand, warmup-cosine, token-weighted accumulation, checkpoint with
  data-pipeline state), 1.5 h. Compare: `~/src/torchtitan/torchtitan/components/checkpointer/dcp.py`.
- Concat-then-split packing and best-fit bin packing of a corpus (pre-training), 45 min. Compare:
  `~/src/torchtitan/torchtitan/components/data/packing.py`.

## 4. Tensor manipulation

Most tensor work now lives inside the attention track (masks, cu_seqlens, slot mapping, gathers) and P2 (gather,
segment sums). One standalone drill:

**T1. MoE routing and dispatch with router replay, single process.** Must, 1 h.
Spec: router logits (T, E) to top-k, softmax over the selected k, stable argsort by expert, counts with bincount,
capacity drop, run experts, combine with `index_add_`. Router replay: when recorded expert indices (T, k) from
inference are passed, use them instead of top-k, and still compute the weights from the trainer's router logits
gathered at those indices so the router gets gradients.
Correctness: matches a per-token loop, including drops; with replay the chosen experts equal the recorded ones
even when the trainer's logits would pick others, and router grads are non-zero only at recorded indices.
Efficiency: one sort, no (T, E, D) intermediate; expert FLOPs = 6 x T x k x D x F for a SwiGLU expert.

```python
logits                                                  # (T, E)
idx = replay_idx if replay_idx is not None else logits.topk(k, -1).indices    # (T, k)
w = logits.gather(-1, idx).softmax(-1)                  # (T, k)  grads reach the router either way
order = idx.flatten().argsort(stable=True)              # (T * k,)  assignments sorted by expert
```

Compare afterwards: `LocalTokenDispatcher` in `~/src/torchtitan/torchtitan/models/common/moe/token_dispatcher.py`;
routing replay in `~/src/slime/slime/backends/megatron_utils/actor.py`.

Optional: Tensor Puzzles (https://github.com/srush/Tensor-Puzzles) as filler.

Cut: corpus materialization and tail-patching; separate LR-schedule and AdamW drills; ZeRO-1 and ZeRO-2 as separate
drills (D4 covers the ideas); attention variants as a scattered list (now the attention track).

## 5. Daily warm-up: forward and backward by hand

NumPy, finite-difference check on every gradient (the `check` pattern in `reference/backprop_practice_4.py`).
W1 is the phone-screen question: target 15 minutes.

**Table 2. Warm-up rotation.**

| # | Network | What it trains |
|---|---|---|
| W1 | two-layer net, ReLU, stable softmax cross-entropy | p - onehot, the max trick, mean over B |
| W2 | LayerNorm and RMSNorm | mean and variance paths, sum over the normalized axis |
| W3 | causal attention, multi-head | softmax row rule, masked entries get 0, split and merge heads |
| W4 | SwiGLU MLP | product rule through the gate |
| W5 | top-k gating over k experts | grads only to selected experts, softmax over k |
| W6 | log-softmax with temperature, policy gradient weighted by advantages and a mask | w x A x (p - onehot) / T / norm |
| W7 | GQA attention backward | dK, dV of a shared KV head = sum over its G query heads (broadcast forward, sum backward) |

Stable softmax, out loud: exp(1000) overflows; subtracting the row max puts the largest score at 0, every exp is in
(0, 1], and the factor exp(-max) cancels between numerator and denominator.

## 6. Cost arithmetic (whiteboard)

Reference numbers from a large MoE run: 500B total, 23.4B active parameters, 2,250 TFLOP/s dense BF16 peak per GPU.
RL: 1,920 trainer GPUs and 8,576 inference GPUs, 110K concurrent rollouts. Dimensions marked illustrative are not
from the run. HBM bandwidth 8 TB/s is an assumption. All results checked in Python.

**Inference share, and what the trainer demands of it.**

```python
8576 / (1920 + 8576)                                    # 0.817 of RL GPUs do inference (ratio 4.5 : 1)
3.9 / 4.9, 5.4 / 6.4                                    # 0.80 to 0.84 over the reported ratio range
110e3 / 8576                                            # 12.8 concurrent rollouts per inference GPU
tok_s_trainer = 0.21 * 2250e12 / (6 * 23.4e9) * 1920    # 6.46M tokens/s if the trainer ran at 21% MFU
tok_s_trainer / 8576                                    # 753 tokens/s each inference GPU must generate
753 * 2 * 23.4e9 / 2250e12                              # 0.016: decode uses 1.6% of peak FLOPs
```

**KV bytes per token, per request, per GPU.** Illustrative: 64 layers, dh 128, bf16.

```python
2 * 64 * 64 * 128 * 2                                   # MHA, 64 KV heads: 2 MiB/token
2 * 64 * 8 * 128 * 2                                    # GQA, 8 KV heads: 256 KiB/token
2 * 64 * 1 * 128 * 2                                    # MQA: 32 KiB/token
64 * (512 + 64) * 2                                     # MLA latent 512 + rope 64: 72 KiB/token
256 * 1024 * 32768                                      # GQA, 32K context: 8.6 GB per request
12.8 * 256 * 1024 * 16384                               # 55 GB of KV per GPU at 16K average context
15 * 256 * 1024                                         # 3.9 MB: worst-case paged waste per request, block 16
```

**FLOPs per byte: prefill vs decode.** Ridge = 2250e12 / 8e12 = 281 FLOP/byte.

```python
# decode attention, one token over L cached tokens: FLOPs 4 H dh L, bytes read 4 Hkv dh L (bf16 K and V)
64 / 8                                                  # GQA: 8 FLOP/byte, 35x below the ridge
64 / 64                                                 # MHA: 1 FLOP/byte
# prefill attention, causal, L = 8192: FLOPs 2 H dh L^2, bytes 2 L dh (2 H + 2 Hkv)
64 * 8192 / (2 * 64 + 2 * 8)                            # 3,641 FLOP/byte: compute-bound
# decode linear layers: about b FLOP/byte for a batch of b tokens; MoE splits b by k / E
281 * 256 / 8                                           # 8,992 tokens per step before experts reach the ridge
```

This is why inference needs huge concurrency and large expert parallelism, and why MLA (all heads read one latent
row) and sparse attention matter for decode.

**Weight update.** 500B params in bf16 = 1 TB per inference replica.

```python
1e12 / 8, 1e12 / 8 / 12                                 # 125 GB per GPU, 10.4 GB/s in 12 s (8-GPU replica)
1e12 / 64, 1e12 / 64 / 12                               # 15.6 GB per GPU, 1.3 GB/s (64-GPU replica)
# naive: all-gather full weights on each trainer rank, (N - 1) / N x W received; targeted: W / tp per inference rank
```

Per-GPU bandwidth is not the bottleneck at 12 s; resharding, verification and fan-out to many replicas are.

**Staleness by Little's law.** Mean steps a rollout waits = rollouts in flight / rollouts consumed per step.

```python
110e3 / 8192                                            # 13.4 steps at an illustrative 8,192 rollouts per step
110e3 / 2048                                            # 53.7 steps at 2,048: why a staleness bound and admission control
```

**Checkpoints.**

```python
7.7e12 / 500e9                                          # 15.4 B/param pre-training: fp32 master + m + v + bf16 weights
14.5e12 / 500e9                                         # 29 B/param RL, every 2 steps: ask what else is in it
14.5e12 / 1920                                          # 7.6 GB per trainer GPU per checkpoint: must be async
tau = (2 * 10 * 280 * 60) ** 0.5                        # Young/Daly: 579 s for a 10 s stall, MTBF 280 min
```

**Memory and collectives.**

```python
P, n = 7e9, 8                                           # dense 7B, 8 data-parallel ranks
16 * P, (2 + 2 + 12 / n) * P, (2 + 14 / n) * P, 16 / n * P    # 112, 38.5, 26.25, 14 GB: DDP, ZeRO-1/2/3
2 * (8 - 1) / 8 * 14e9                                  # 24.5 GB ring all-reduce per GPU (bf16 grads)
24.5e9 / 50e9, 24.5e9 / 900e9                           # 0.49 s cross-rack vs 0.027 s NVLink: 36x apart
16384 * 131072 * 4, 1024 * 131072 * 4                   # 8.6 GB full fp32 logits vs 0.54 GB per chunk
```

## 7. Talking points

RL systems (existing strengths, keep each to two sentences):

- **Train/inference mismatch.** Different kernels, precision and batch composition give different log-probs for the
  same tokens; correct with truncated importance weights or masking of extreme ratios (TIS, IcePop, CISPO's capped
  weight, GSPO's sequence ratio), and remove sources: fp32 output head, router replay for MoE.
- **Fully async RL.** Inference never waits for the trainer; the cost is staleness, controlled by per-token policy
  versions, a max off-policy bound, admission control and a bounded buffer.
- **Weight push without checkpoints.** Trainer memory to inference memory directly, resharded to the inference TP
  layout, verified before adoption; in-flight requests keep their cache, the next turn recomputes under new weights.
- **Router replay.** Record expert choices at inference and force them in the trainer forward, so the trainer
  computes the probability of the path that actually generated the tokens.
- **Failure domains.** A trainer failure is a full restart from checkpoint (with data-pipeline and buffer state); an
  inference failure is local repair, the orchestrator reissues its rollouts; prefetched rollouts limit loss.
- **Load balance.** Rollout lengths are long-tailed; pack by length and balance across trainer replicas, keeping
  sample weights so the loss is the same however rollouts are packed.

Infra failure reasoning:

- **Hang.** Ranks calling collectives in different orders or counts (uneven micro-batch counts after packing is the
  classic RL version); a dead NIC can also look like a hang, so a watchdog with timeouts and stack dumps.
- **Straggler.** Every synchronous step waits for the slowest rank: in RL often the rank with the longest rollouts,
  fixed by length balancing.
- **Low MFU.** Loader or rollout supply stalls, exposed communication, small GEMMs from too much TP, a kernel taking
  SMs from NCCL; GC pauses at different times per rank act like stragglers, so collect on all ranks together.
- **Loss spike or NaN.** First step and layer from per-layer grad norms; check mismatch metrics (ratio statistics)
  before blaming the optimizer.

Design trade-offs:

- **TP vs FSDP.** TP moves activations every layer and needs NVLink; FSDP moves params once per layer per pass and
  overlaps across racks.
- **Prefill vs decode.** Prefill is compute-bound, decode is memory-bound; batching decodes and chunking prefills into
  one step keeps both busy.
- **Paged KV.** Fixed blocks remove fragmentation and make prefix sharing a block-table operation; the cost is a
  gather in the kernel and a metadata build per step.
- **GQA vs MLA vs sparse.** GQA divides KV by H / Hkv; MLA caches a latent and raises FLOPs per byte; sparse attention
  caps per-step KV reads at top-k but adds an indexer pass.
- **Bucket size.** Too small: latency per call dominates; too large: overlap starts late.
- **Buffer size vs staleness.** Bigger buffers feed the trainer and raise staleness; the bound should follow from
  what the loss correction can tolerate.
