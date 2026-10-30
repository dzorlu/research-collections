# Tensor, pipeline and expert parallelism: how they interact

Notes from bringing up GLM-5.3-Flash training on 48 B300 GPUs (2026-08-30). Every
number here comes from that setup: the model's own spec arguments, the trainer's
parallel configuration, and memory readings taken from a run that ran out of
memory and the runs on either side of it.

## The model

GLM-5.3-Flash, from the arguments the trainer is launched with:

| Property | Value |
|---|---|
| Layers | 45 (first 3 dense, remaining 42 mixture-of-experts) |
| Hidden size | 4096 |
| Attention heads | 64, head dimension 256 |
| Attention type | multi-head latent (query rank 1536, key/value rank 512) |
| Experts per MoE layer | 288 |
| Experts active per token | 8 |
| Expert inner width | 2048 |

Roughly 320B parameters in total, about 18B of which are used for any single
token because only 8 of 288 experts fire.

## The grid

48 GPUs, arranged as 6 pipeline stages of 8 GPUs. Each stage is one physical
node.

```
stage 0  node A   GPU 0..7    layers  0- 6   (7 layers, 3 dense + 4 MoE)
stage 1  node B   GPU 0..7    layers  7-14   (8 MoE layers)
stage 2  node C   GPU 0..7    layers 15-22   (8)
stage 3  node D   GPU 0..7    layers 23-30   (8)
stage 4  node E   GPU 0..7    layers 31-38   (8)
stage 5  node F   GPU 0..7    layers 39-44   (6)  + output head
```

Pipeline parallelism is the vertical axis: it cuts the model by depth. Tensor
and expert parallelism both live on the horizontal axis, inside a single node's
8 GPUs.

## The one idea worth holding onto

The same 8 GPUs mean two different things depending on which part of a layer you
are in.

In the attention block, the 8 GPUs are slices of the same weights. Every GPU
sees every token and owns one eighth of each matrix.

In the mixture-of-experts block, the 8 GPUs are owners of different whole
experts. Every GPU owns 36 complete experts (288 divided by 8) and sees only the
tokens routed to them.

Attention splits weights and shares tokens. The expert block splits tokens and
owns whole weights. Halfway through every layer the meaning of "GPU 3" flips,
and an all-to-all exchange is the handoff between the two arrangements.

## One layer, with shapes

A micro-batch of 8192 tokens arriving at stage 2. Hidden size is 4096.

```python
# Sequence parallelism: tokens are split 8 ways between layers.
x_local = [1024, 4096]          # each of the 8 GPUs holds 1/8 of the tokens

# --- attention region: tensor parallel, 8 ways ---
x = all_gather(x_local, dim=0)  # [8192, 4096]  every GPU now has ALL tokens

# 64 heads total, head dimension 256, so 8 heads per GPU.
q = x @ W_q_a                   # [8192, 1536]   low-rank query compression
q = q @ W_q_b_shard             # [8192, 2048]   8 heads x 256, this GPU's slice
kv = x @ W_kv_a                 # [8192,  512]   the compressed key/value latent
k, v = kv @ W_kv_b_shard        # [8192, 2048] each, this GPU's 8 heads

o = attention(q, k, v)          # [8192, 2048]   only this GPU's heads
y = o @ W_o_shard               # [8192, 4096]   PARTIAL sum, needs all 8 GPUs

x_local = reduce_scatter(y)     # [1024, 4096]   sum across GPUs, re-split tokens

# --- expert region: expert parallel, 8 ways ---
logits = x_local @ W_router     # [1024, 288]    the router is replicated, it is tiny
top8   = topk(logits, 8)        # each token picks 8 of 288 experts

# The arrangement flips here. GPU e owns experts [36e, 36e+36), whole and unsplit.
# Each token must visit 8 experts scattered across the 8 GPUs.
tokens_out = all_to_all(x_local, destinations=top8)   # dispatch
# this GPU receives a VARIABLE number of (token, expert) pairs for its 36 experts

for e in local_experts:         # 36 of them
    t = tokens_for(e)           # [n_e, 4096], n_e varies per expert and per batch
    h = silu(t @ W_gate[e]) * (t @ W_up[e])   # [n_e, 2048]
    out[e] = h @ W_down[e]                     # [n_e, 4096]

results = all_to_all(out)       # combine, send results home
x_local = weighted_sum(results) # [1024, 4096]  8 expert outputs per token, weighted
```

Stage 2 then sends `[8192, 4096]` over the network to stage 3 and starts on the
next micro-batch.

## What each split costs to move

Per token, per layer:

| Split | Collective operation | Bytes moved per token | Where it travels |
|---|---|---|---|
| Tensor, 8 ways | all-gather and reduce-scatter, twice per layer | about 32 KB | inside the node, over NVLink |
| Expert, 8 ways | all-to-all dispatch and combine | about 128 KB, because each token is copied to 8 experts and back | inside the node, over NVLink |
| Pipeline, 6 ways | point-to-point send | 8 KB, and only at the 5 stage boundaries | between nodes, over InfiniBand |

This explains why the configuration is shaped the way it is. Expert
parallelism moves the most data by a wide margin, because routing 8 experts per
token means every token is copied 8 times in each direction. Tensor parallelism
is chatty as well, twice per layer. Both are set to 8 so that neither ever
leaves a node. Pipeline parallelism moves the least data, once per stage
boundary, so it is the only one allowed to cross the network.

## The constraint that couples them

Expert parallelism has to cover the same GPUs that tensor and data parallelism
cover. This configuration satisfies that exactly: tensor 8 times data 1 is 8
GPUs per stage, and expert 8 times expert-tensor 1 is also 8 GPUs per stage.

There is direct evidence of the constraint from a failure. An upstream recipe
row specified expert parallelism of 16 together with 4 pipeline stages, and on
32 GPUs the trainer refused to start with `world_size (32) not divisible by 64`,
where 64 is 16 times 4. Expert 8 times pipeline 6 is 48, which divides the 48
GPUs exactly.

This is also why expert parallelism cannot simply be raised to save memory.
Going to 16 would need 16 GPUs per stage group, which means tensor 8 with data
2, which means 8 times 6 times 2 equals 96 GPUs, or 12 nodes.

## The memory arithmetic

Per mixture-of-experts layer, one GPU owns 36 experts. Each expert is three
matrices:

```
gate [4096, 2048] + up [4096, 2048] + down [2048, 4096] = 25.2M parameters
36 experts x 25.2M = 906M parameters per GPU per MoE layer
```

Attention contributes roughly 16M parameters per layer per GPU after the 8-way
tensor split, so the experts account for about 98% of what each GPU holds. For
this model, memory is entirely a mixture-of-experts story.

Mixed-precision Adam costs 18 bytes per parameter: 2 for the bf16 weight, 4 for
the fp32 master copy, 4 for the fp32 gradient, and 8 for Adam's two fp32 moment
buffers.

```
4 stages:  42 MoE layers / 4 = 10.5 per GPU  ->  9.5B parameters  ->  171 GB
6 stages:  42 MoE layers / 6 =  7.0 per GPU  ->  6.3B parameters  ->  114 GB
```

The measured trainer footprint when it ran out of memory at 4 stages was 225.7
GB per GPU, so those 171 GB of optimizer and weight state plus roughly 55 GB of
activations, attention weights and workspace accounts for it. At 6 stages the
same arithmetic predicts about 169 GB. With the inference engines' resident 39
GB on the same GPUs, that is around 208 GB of the 267.69 GB available, leaving
roughly 60 GB of headroom.

## Why more pipeline stages was the fix

The failing allocation was Adam initialising its two moment buffers. Two
properties of that allocation decided everything:

It is fixed per GPU. It scales with the number of parameters a GPU owns and is
completely indifferent to sequence length or batch size. Shortening the context
would not have moved it at all, which is worth knowing because shortening the
context is the reflex.

It cannot be shared, because data parallelism was 1. Tensor 8 times pipeline 4
is 32, exactly the GPU count, so there was no second replica of the model and
the distributed optimizer had no peer to spread the moments across.

That left one lever that reduces parameters per GPU without adding replicas:
more pipeline stages. Six stages instead of four takes each GPU from about 11
layers to about 7.5, cutting weights, gradients and optimizer state per GPU by
roughly a third.

The general form of the lesson: when a training job runs out of memory, first
work out whether the failing allocation scales with tokens or with parameters.
Activations scale with tokens, and context length or batch size will move them.
Optimizer state scales with parameters per GPU, and only the parallel layout
will move it.

## References

- Expert parallelism from the token-movement point of view, which is where the
  "columns move tokens, rows share weights" framing comes from:
  https://tinkerings.dev/posts/expert_parallel.html
- Companion notes in this repository on where attention memory and compute go:
  [`../attention/kda-mla-flash-memory-compute.md`](../attention/kda-mla-flash-memory-compute.md)
