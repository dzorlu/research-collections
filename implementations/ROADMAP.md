# Roadmap

Three areas: tensor manipulation, distributed computing and efficiency, components within a training pipeline.
Forward and backward passes are the daily warm-up. Every drill has stubs and tests; distributed drills also measure
efficiency (bytes communicated, number of calls, memory per rank, overlap).

## 1. Distributed computing and efficiency (most important)

| # | Drill | Correctness | Efficiency | Status |
|---|---|---|---|---|
| 1 | Ring all-reduce from send/recv | sum on every rank | bytes sent ≤ 2 (N - 1) / N of the tensor | done |
| 2 | Data parallel by hand | weights match single-process training | one all-reduce per parameter (baseline for 5) | |
| 3 | Tensor-parallel MLP (column then row split) | output and grads match the unsharded MLP | exactly one all-reduce forward, one backward | |
| 4 | ZeRO-1 optimizer sharding | matches plain Adam | optimizer state per rank = 1 / N | |
| 5 | Bucketed all-reduce overlapped with backward | grads match drill 2 | fewer calls; first all-reduce starts before backward ends | |
| 6 | MoE expert parallelism with all_to_all | matches a single-process MoE layer | tokens moved per rank | |
| 7 | Context parallelism: split the KV, merge with LSE | matches full attention | KV per rank = 1 / N | |

Cost arithmetic, by hand:

```python
adam_bytes = 16 * n_params                              # bf16 weights + grads, fp32 master + m + v
train_flops = 6 * n_params * n_tokens
mfu = 6 * n_params * tokens_per_sec / (n_gpus * peak_flops)
ring_allreduce_bytes_per_gpu = 2 * (n - 1) / n * nbytes
pipeline_bubble = (p - 1) / (m + p - 1)                 # p stages, m micro-batches
kv_cache_bytes = 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_elem
```

Failure reasoning, out loud: a run at 40% of theoretical FLOPs (where is the bottleneck), a straggler rank, what
NCCL does in an all-reduce and where it breaks at scale, checkpoint frequency against recovery time, loss spikes and
NaNs, hangs from mismatched collectives.

## 2. Components within a training pipeline

| # | Drill | Test |
|---|---|---|
| 1 | Data loader: shuffle and shard across data-parallel ranks, deterministic per epoch | every example seen once per epoch across ranks; same order for the same seed |
| 2 | Sequence packing: documents into fixed-length rows, position ids restart per document, block-diagonal mask | no attention across documents; matches unpacked outputs |
| 3 | Resumable loader and checkpoint/resume (model, optimizer, scheduler, RNG, loader position) | an interrupted run gives the same loss curve as an uninterrupted one |
| 4 | Gradient accumulation | k micro-batches equal one large batch (loss scaled correctly) |
| 5 | Mixed precision and gradient clipping by global norm | matches torch.nn.utils.clip_grad_norm_ |
| 6 | Learning-rate schedule: warmup then cosine | matches a reference curve |
| 7 | AdamW by hand | matches torch.optim.AdamW |

## 3. Tensor manipulation

- Tensor Puzzles (https://github.com/srush/Tensor-Puzzles): one or two a day.
- gather / scatter_add / index_select, masks (causal, padding, block-diagonal for packing), position ids with cumsum.
- MoE dispatch: top-k experts per token, sort by expert, capacity limit, combine back in the original order.
- Attention variants from the vLLM guide: GQA/MQA, sliding window, paged KV cache (block_table, slot_mapping,
  query_start_loc and seq_lens for a mixed batch), MLA, sparse top-k.

## Daily warm-up: forward and backward

backprop_practice_4.ipynb, then new sets in the same format: attention with a mask and heads, LayerNorm and RMSNorm,
softmax over different axes, residual blocks, backward through top-k gating. Be ready to explain the stable softmax
(why exp overflows, why subtracting the max fixes it).
