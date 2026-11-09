# Implementations

Goal: write ML training code from scratch, quickly and correctly, and reason about how it fails at scale. Blank
page, no AI; docs are fine.

Plan: [ROADMAP.md](ROADMAP.md).

## 1. Drills

Each notebook has instructions, stub cells marked `# TODO`, and test cells that print `ok`.

**Table 1. Drills.**

| Notebook | Time | What you write |
|---|---|---|
| `01_policy_loss.ipynb` | 25 min | GRPO advantages; REINFORCE, PPO, CISPO, GSPO losses with masks and two normalizations; the same from logits |
| `02_rl_loop.ipynb` | 45 + 10 min | reward, sampling and a GRPO + PPO loop on a toy task; then a hacked reward and a label-free check that catches it |
| `03_attention.ipynb` | 25 min + bonus | causal multi-head attention with a KV cache (incl. chunked prefill); attention backward by hand; online softmax |
| `04_grader.ipynb` | 45 min | a grader for "write attention by hand" that survives the 12 files in `04_grader_inputs/`, some of which cheat |

Backprop practice (hand-written forward and backward passes in NumPy, written before these drills):

- `backprop_practice_1.ipynb`: a four-layer network (softplus, ReLU, softmax with the max trick, cross-entropy).
- `backprop_practice_2.ipynb`: a set of small networks with unusual losses and layers (mean of L2 norms, cubed
  error, batch-sum broadcasting, product rule, SiLU with a residual), plus debugging a broken two-layer network.
- `backprop_practice_3.ipynb`: more of the same: max-normalization, batch sums, learnable gates, residuals,
  softmax cross-entropy, and fixing broken backward passes.
- `backprop_practice_4.ipynb`: **exercises with tests**: a product with its own row sum, batch norm, max
  normalization with SiLU and a residual, and a backward pass with three planted bugs. Each test checks the forward
  loss against a fixed value and every gradient against finite differences. Reference:
  `reference/backprop_practice_4.py`.

Distributed drills live in `distributed/`: `dayNN_*.py` are the stubs, `reference/` the solutions, `_harness.py`
runs N CPU processes with gloo and logs calls and bytes per rank. Run from that folder with the pglab env:
`cd distributed && python day01_collectives.py`. After editing a reference, regenerate stubs with
`python reference/_make_stubs.py`.

Conventions:

- `XX_name.ipynb` is the unsolved drill; `XX_name_solved.ipynb` is a worked version.
- [LEARNINGS.md](LEARNINGS.md): what each drill taught.
- `reference/`: reference solutions, one script per notebook; every test passes against them
  (`cd implementations && python reference/02_rl_loop.py`). Open after trying.
- `04_grader_inputs/`: the files drill 4's grader must judge (2 honest, 3 buggy, 7 cheating). Open them and
  `ANSWER_KEY.md` only after your grader passes.

Running: torch is in the `pglab` conda env, registered as the Jupyter kernel "Python (pglab, torch)".
`jupyter lab` from this folder, or VS Code with the `pglab` interpreter.

## 2. A component in a training pipeline

Expect a training loop with one piece missing or broken. Write each from scratch:

- **Data:** a collate function with padding and loss masks. Sequence packing: several documents per row, position
  ids that restart per document, a block-diagonal attention mask so documents do not see each other.
- **Loop:** gradient accumulation with the loss scaled by the number of micro-batches; mixed precision (bf16
  autocast, or fp16 with a GradScaler); gradient clipping by global norm; warmup then cosine learning-rate schedule.
- **Checkpoint and resume:** model, optimizer, scheduler, RNG states and dataloader position. Test: a run
  interrupted and resumed gives the same loss curve as an uninterrupted one.
- **Optimizer:** AdamW by hand, checked against `torch.optim.AdamW`.

## 3. Tensor manipulation

Vectorized, no Python loops.

- `view` vs `reshape` and when `contiguous()` matters; `einsum` for attention and batched matmuls; broadcasting.
- `gather`, `scatter_add`, `index_select`: per-token log-probs, a batched top-k mask, `bincount`.
- Masks: causal, block-diagonal for packed documents, padding; position ids with `cumsum`.
- **MoE dispatch:** top-k experts per token, sort tokens by expert, apply a capacity limit, run the experts, combine
  the outputs back in the original order with the routing weights. The classic hard tensor question.
- RoPE; splitting and merging heads.

## 4. Distributed computing and efficiency

### 4.1 Cost arithmetic (whiteboard)

**Table 2. Numbers to derive on the spot.**

| Quantity | Rule |
|---|---|
| Memory for mixed-precision Adam | about 16 bytes per parameter: bf16 weights and grads, fp32 master weights, two fp32 Adam states |
| ZeRO-1 / 2 / 3, FSDP | shard optimizer states / plus gradients / plus weights across the data-parallel ranks |
| Activation memory | grows with batch x sequence x hidden x layers; activation checkpointing trades it for one extra forward |
| Training FLOPs | about 6 x parameters x tokens |
| MFU | achieved model FLOPs per second / peak GPU FLOPs per second |
| Ring all-reduce traffic | about 2 (N - 1) / N x data size per GPU; all-reduce = reduce-scatter + all-gather |
| Pipeline bubble | about (p - 1) / (m + p - 1) idle, p stages, m micro-batches |

Story to rehearse: [../parallelism/](../parallelism/), GLM-5.3 on 48 GPUs: the memory arithmetic behind a real
out-of-memory failure and why more pipeline stages fixed it.

### 4.2 Code that runs on a laptop

`torch.distributed` with the CPU `gloo` backend, 2 to 4 processes via `torch.multiprocessing`:

1. Ring all-reduce from point-to-point `send` and `recv`, checked against `dist.all_reduce`.
2. Data parallel by hand: all-reduce the gradients after `backward`; weights must match single-process training.
3. Tensor-parallel MLP (Megatron): first layer split by columns, second by rows, one all-reduce; must match the
   unsharded MLP.
4. ZeRO-1: each rank owns a shard of the optimizer state, then all-gathers the updated weights.
5. Bonus: bucket gradients and start the all-reduce while backward is still running.

### 4.3 Failure reasoning, out loud

- Loss spikes: what to check, when to roll back to a checkpoint.
- NaNs: finding the layer and step that produced them.
- A slow GPU (straggler): every step waits for the slowest rank.
- Hangs: ranks calling collectives in different orders.
- Checkpoint and restart at scale: frequency, cost, resuming the data loader.
- Deterministic reruns: what breaks determinism across restarts.

## 5. One-day schedule

**Table 3. If there is only one day.**

| Block | Work |
|---|---|
| Morning | tensor drills, including MoE dispatch and packing masks |
| Midday | a training-loop component: accumulation, clipping, LR schedule, exact resume |
| Afternoon | distributed code: ring all-reduce, DDP by hand, tensor-parallel MLP |
| Evening | memory and communication arithmetic; rehearse the parallelism story |
