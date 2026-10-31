# Long-horizon RL on one 8-GPU node: which model, and the principles that make it work

Notes from 2026-10-30. Two questions:

1. Which open-weight model is the strongest one you can train on a single 8×B200 node?
2. What does it take to train it on long-horizon agent episodes: 256K context, screenshot folding, a
   policy-written summary, and exact-token training through the miles TITO session server?

Every number here comes from a public model card, a public upstream recipe, or arithmetic on a model's
config. Section 7 lists the sources.

## 1. Answer

- **Full-parameter training (SFT or RL): Qwen3.8-27B** (dense, 27.8B, Apache-2.0). It is the only strong
  model whose full training state fits one node. Databricks documents full SFT of it on one 8×B300 node.
- **Best quality that physically fits: MiMo-V2.6-Flash with LoRA** (309B total, 15B active, MIT). It leads
  every benchmark the candidates share (Table 2). No public recipe trains it on one node yet.
- **Middle option: Qwen3.8-Flash-Next with LoRA** (180B total, 6B active). Its bf16 weights take 45 GB per
  GPU. Its license requires a separate Qwen license for "Model as a Service" or "AI Work Assistant"
  businesses; internal use is exempt.
- **256K context on one node works on paper only for Qwen3.8-27B on text tasks** (section 4). Screenshot
  episodes at 256K need context parallelism for image positions, which upstream does not have.

## 2. What fits: the memory arithmetic

An 8×B200 node has 180 GB per GPU (1,440 GB of HBM), 2 TB of host RAM (4 TB as an option) and
8 × 3.84 TB NVMe.

Full fine-tuning with Adam in mixed precision costs about 16 bytes per parameter: bf16 weights (2), bf16
gradients (2), fp32 master weights (4) and two fp32 Adam moments (8). LoRA freezes the base, so it costs
about 2 bytes per parameter plus a small adapter.

Table 1. Memory per model against one node's 1,440 GB of HBM.

| Model | Total params | Full fine-tune state | bf16 weights (LoRA) | One node? |
|---|---|---|---|---|
| Qwen3.8-27B | 27.8B | 445 GB | 56 GB | Full, yes |
| Qwen3.8-Flash-Next | 180B (6B active) | 2.9 TB | 360 GB | LoRA easily; full only with host offload, tight |
| DeepSeek-V4-Flash-0731 | 284B (13B active) | 4.5 TB | ~570 GB | LoRA only |
| MiMo-V2.6-Flash | 309B (15B active) | 4.9 TB | 620 GB | LoRA only |
| GLM-5.3-Flash | 320B (18B active) | 5.1 TB | 640 GB | LoRA only |
| Kimi K3, MiMo-V2.6-Pro, GLM-5.3, DeepSeek V4.1 Flash, Ling 3.1 Flash, Muse Spark Open, Hy4, Step 5 | 570B to 2.8T | n/a | 1.1 to 5.6 TB | No |

The models at the top of the open leaderboards (Kimi K3 leads BenchLM's open list at 69.3; MiMo-V2.6-Pro
and GLM-5.3 follow) do not fit: even their bf16 weights fill or overflow the node.

Three ways to stretch the budget, in order of cost:

1. **Shard the state** over GPUs (FSDP, or Megatron's distributed optimizer over the data-parallel and
   context-parallel ranks). Free, but one node has only 8 ranks.
2. **Move Adam to host RAM** (`--optimizer-cpu-offload`, which Megatron allows only with
   `--use-precision-aware-optimizer`). bf16 moments (`--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16`) halve
   the host side to 8 bytes per parameter. Adam then steps on the CPU.
3. **Stream Adam state through NVMe.** Slowest. Upstream uses it for MiMo-V2.6-Flash full training on two
   8×H200 nodes: a 3.7 TB Adam state, 125 GB peak per GPU, 1.05 TB host memory per node, and about 3 of
   every 4 training minutes spent streaming.

Upstream evidence for the large MoE rows:

- MiMo-V2.6-Flash LoRA trained on 8 H200 at about 130 GB per GPU (8 × 4,096-token micro-batch), with
  sampling on 8 other GPUs. On one node, training and sampling must share the GPUs (colocated): the
  trainer moves its state to host RAM while the engines sample, then takes the GPUs back. That is not
  validated for MiMo.
- GLM-5.3-Flash LoRA runs on 3 nodes for the full model; on one node upstream runs only a 4-layer slice.
- Upstream supports no context parallelism for MiMo-V2.6-Flash and caps its tensor parallelism at 4 (the
  number of global-attention KV heads). Its Qwen3.8-Flash-Next script also uses context parallelism 1.
  Neither can hold one 256K episode on one node.

## 3. Quality of the models that fit

Table 2. Vendor-reported scores on benchmarks that several model cards share. Harnesses differ by vendor.

| Model | DeepSWE 1.1 | Toolathlon-Verified | Agents' Last Exam pass@1 | SWE-bench Pro | License |
|---|---|---|---|---|---|
| MiMo-V2.6-Flash | **67.9** | **73.6** | **27.6** | n/a | MIT |
| Qwen3.8-Flash-Next | 58.7 | 73.5 | 24.3 | **62.5** | Qwen community |
| DeepSeek-V4-Flash-0731 | 54.4 | 70.3 | 25.2 | 56.0 | MIT |
| Qwen3.8-27B | 42.2 | 67.1 | 20.4 | 61.7 | Apache-2.0 |

The DeepSeek numbers in Qwen's card match DeepSeek's own card. Qwen3.8-27B reports OSWorld-Verified 84.3;
MiMo-V2.6-Flash reports 80.8 on the same benchmark. Independent indexes disagree with the vendors:
Artificial Analysis (snapshot of 2026-10-19) put GLM-5.3-Flash at 41.8, Qwen3.8-Flash-Next at 39.8 and
MiMo-V2.6-Flash at 37.9.

## 4. 256K context with Qwen3.8-27B

Qwen3.8-27B's native context is 262,144 tokens (`max_position_embeddings`, no rope scaling). It is a
hybrid: 16 full-attention layers and 48 linear-attention (Gated DeltaNet) layers, vocab 248,320.

### 4.1 Training side: one episode is one sequence

In RL on agent episodes, a whole multi-turn episode is one training sequence. Its activations grow with
length, so the longest episode decides peak memory. Two parallelism axes split it:

- **Tensor parallelism (TP) with sequence parallelism** splits weights and per-layer activations by TP.
- **Context parallelism (CP)** splits the sequence itself: each GPU holds tokens / CP.

Tokens per GPU ≈ episode length / (TP × CP) for activation purposes. On 8 GPUs, TP × CP ≤ 8.

Table 3. Ways to place one 262,144-token episode on 8 GPUs.

| Layout (TP × CP) | Tokens per GPU (sequence) | Weight shard per GPU (bf16) | Works for |
|---|---|---|---|
| 8 × 1 | 262,144 | 7 GB | Text and images |
| 4 × 2 | 131,072 | 14 GB | Text only |
| 2 × 4 | 65,536 | 28 GB | Text only |

TP 8 is allowed even though the model has only 4 KV heads: the KV heads are replicated.

**Why images block CP.** Qwen's vision models use mRoPE: each image token carries three position numbers
(time, row, column). CP cuts and reorders the sequence, and upstream has no way to compute those
positions for a slice; its Qwen vision provider raises NotImplementedError when CP is above 1. So a
screenshot episode must sit whole on one TP group.

**The logits trap.** The loss needs logits of shape tokens × vocab. At 64K tokens that is
64K × 248,320 × 2 bytes = 30 GiB in bf16, and twice that if copied to fp32. Two fixes: TP shards the vocab
dimension (loss computed vocab-parallel), and `--log-probs-chunk-size 4096` computes log-probs and entropy
4,096 tokens at a time.

**What the one-node text run needs:**

1. TP 4 × CP 2 (or TP 2 × CP 4).
2. Adam in host RAM: 27.8B × 12 bytes = 334 GB, well inside 2 TB.
3. Chunked log-probs.
4. Colocated offload: the engines release their weights and KV during the train step
   (`--offload-train-target cpu` moves the trainer out during sampling).
5. One short smoke job to measure peak memory before a real run. No public run has measured this layout on
   B200.

### 4.2 Sampling side: KV cache and prefill

- **KV per token** = full-attention layers × KV heads × head dim × 2 (K and V) × 2 bytes
  = 16 × 4 × 256 × 2 × 2 = 64 KB. The 48 linear-attention layers keep a fixed-size state per request
  instead. A full 256K episode holds 17.2 GB of KV.
- **Engines per GPU.** One engine per B200 at TP 1 (55.6 GB of weights) and memory fraction 0.8 leaves
  about 88 GB of KV pool: about 5 full-length episodes per engine, about 40 per node.
- **Prefill dominates.** Each turn appends an observation to a long prefix. Without a prefix cache, every
  turn re-prefills the whole history. Keep SGLang's radix cache on (it also stores the linear-attention
  states along the prefix tree).
- **Route each episode to the same engine.** With the cache on, all turns of an episode must reach the
  engine that holds its prefix: use a routing key per episode (`--sglang-router-policy
  consistent_hashing`). A cache-aware router can send every request to one engine when all episodes share
  the same system-prompt prefix.
- **Health checks vs long prefills.** A queued 250K-token prompt can take minutes to prefill. Default
  liveness probes (tens of seconds) then kill healthy engines. Raise them
  (`--rollout-health-check-interval`, `--rollout-health-check-timeout`, `SGLANG_HEALTH_CHECK_TIMEOUT`).
- **Screenshots between processes.** Pass image features through CUDA IPC
  (`--sglang-mm-feature-transport cuda_ipc`) rather than CPU pickling, which blocks the server's event loop
  on long multi-image requests.
- **Blackwell attention backend** for hybrid GDN models: SGLang allows only `trtllm_mha`, `fa4` and
  `triton`; `triton` compiles per architecture.

## 5. Long horizons: folding, a summary, and the session server

Context alone does not make an agent go long. An episode that fills its window ends and usually scores 0.
Three pieces work together. The principles below are independent of any particular harness.

### 5.1 Exact tokens through the TITO session server

**Principle:** the trainer must score the exact token IDs the engine sampled. Re-tokenizing the text
changes token boundaries, so the trainer's log-probs no longer match the sampler's, and the policy-gradient
ratio is wrong.

miles's session server does this for agent harnesses that speak the OpenAI chat API. Each turn it matches
the incoming messages against the stored token IDs, tokenizes only the new message, and sends the engine
stored IDs plus new IDs. With `--use-session-server v2` the session is a tree: when the harness rewrites
history (a fold, a summary), the session opens a new branch, and the episode returns one training sample per
branch. Screenshot support through the session server is in upstream review; see
[../upstream-prs/tito-token-space.md](../upstream-prs/tito-token-space.md).

### 5.2 Screenshot folding

Screenshots are the largest token cost in computer-use episodes. Folding keeps only the recent ones:

1. Keep at most B screenshots live.
2. When a new one would make B + 1, advance the fold boundary by S turns. Every screenshot before the
   boundary becomes a short text placeholder. Reasoning, actions and tool text stay.
3. Choose S = B. Each fold rewrites the prompt, which starts a new training sample that re-encodes the
   history as its prompt. Trainer tokens per episode grow by about (turns / S) × history length: about 3×
   at B = S = 40, about 10× at B = 20, S = 10.
4. Start the placeholder with a space. SGLang re-tokenizes the decoded prompt of multimodal requests, and
   a placeholder glued to the previous character can merge into a different token.

Folding is a history rewrite, so it needs the training rules in section 5.4.

### 5.3 A policy-written summary (on-policy compaction)

Folding drops images but keeps all text, so long episodes still fill the window. Compaction lets the policy
write its own summary and continue from it:

1. **Trigger.** When the context passes a threshold (for example half the window, 131,072 tokens), the
   next turn is a summary turn. The prompt shows the context used and asks for task state, values found
   and next steps.
2. **Summary as a tool call.** The policy calls a local tool `compact_context(summary)` that the harness
   handles itself. This keeps a one-tool-call-per-turn protocol intact. Cap the summary (for example
   4,096 tokens).
3. **New context.** Continue from system prompt + task + the summary + the last observation (latest
   screenshot only). The environment holds the world state, so only the model's notes need to carry
   over.
4. **Train the summary.** The summary turn ends the old segment with loss mask 1, so a summary that led
   to a pass is reinforced and one that led to a fail is pushed down. The next segment takes the summary as
   prompt with loss mask 0.
5. **Evaluate the same way.** Base and every checkpoint must run with the same compaction setting. A base
   eval without compaction is not a valid baseline for compaction checkpoints.

No upstream harness does on-policy compaction yet. Sweep the trigger (for example 98K and 131K) in the
first test.

### 5.4 Training rules when one episode becomes many samples

Folds and summaries split an episode into segments. Each segment is its own training sample.

- **One reward, one advantage.** Every segment gets the episode's terminal reward. All segments share one
  `rollout_id`; miles normalizes the reward once per rollout and copies the advantage to each segment, so a
  long episode does not count several times in the group mean.
- **Loss divisor.** Divide the loss by episodes, not samples. Dividing by samples makes the loss shrink as
  episodes fold more ([miles#4050](https://github.com/radixark/miles/pull/4050) fixes this).
- **Variable batch size.** The number of samples per step changes with the number of folds. miles derives
  optimizer steps from samples / (global batch / DP), so pad the batch to a multiple of DP and use
  `--use-dynamic-global-batch-size`. Otherwise a step either crashes or takes several optimizer steps.
- **Per-token loss** (`--calculate-per-token-loss`) weighs every trained token equally, so an episode split
  into two segments weighs the same as one long sequence with the same trained tokens.

## 6. Next steps

1. Run Qwen3.8-27B text at 256K on one node (TP 4 × CP 2, Adam on CPU, chunked log-probs) and measure
   peak memory in a short smoke job.
2. Build the summary tool on the session-server path (section 5.3), then a base eval with it on.
3. Test MiMo-V2.6-Flash LoRA colocated on one node at the target context, if quality matters more than a
   proven path. Fallback: Qwen3.8-Flash-Next LoRA, after the license check.
4. Implement context parallelism for mRoPE, if screenshot episodes at 256K must fit one node.

## 7. Sources

Model cards and configs: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL),
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
miles: [MiMo-V2.6-Flash recipe](https://github.com/radixark/miles/blob/main/docs/models/mimo/mimo-v2-6-flash.md),
[LoRA support matrix](https://github.com/radixark/miles/blob/main/docs/advanced/lora.md),
[GLM-5.3-Flash LoRA launcher](https://github.com/radixark/miles/blob/main/scripts/run_glm5_3_flash_lora.py),
[Qwen3.8-Flash-Next launcher](https://github.com/radixark/miles/blob/main/scripts/run_qwen3_8_next.py).
[Databricks: Qwen3.8-27B SFT on one 8×B300 node](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting/cli/b300-qwen3-8-27b-sft).
[NVIDIA DGX B200](https://nvidia.com/en-us/data-center/dgx-b200).
Leaderboards: [BenchLM open-source](https://benchlm.ai/best/open-source),
[leaderboard audit, 2026-10-19](https://dev.to/ward_ed_6b5e6aa8ded94a987/best-open-weight-llm-in-october-2026-auditing-two-leaderboards-row-by-row-1b64),
[Artificial Analysis](https://artificialanalysis.ai/models/open-source).
