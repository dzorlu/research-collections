# Recipes: per-model run settings for miles

Each recipe is the full set of settings that decide whether a model fits and runs correctly at a given
context length and node count. They are written as upstream miles, Megatron, torchtitan and SGLang flags,
with the reason for each non-obvious choice. Build them into the recipe registry of
[README.md](README.md) section 4.1.

**Hardware the shapes were sized for:** 8 GPUs per node, about 268 GB usable per GPU (B300), about
2.7 TB host RAM and about 1 TB local disk per node. On 180 GB GPUs (B200) re-derive the layout from the
memory arithmetic in [../single-node-training/](../single-node-training/README.md).

"Ran" means the shape completed training steps. "Not run" means it is derived and needs the debug-minimal
gate (README section 7, gate 4) first.

## 1. Settings shared by every recipe

GRPO and optimizer:

```text
--advantage-estimator grpo
--eps-clip 0.2 --eps-clip-high 0.28
--entropy-coef 0.00
--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
--adam-beta1 0.9 --adam-beta2 0.98
--micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu <recipe>
--colocate
```

KL: `--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl` measures KL to the reference without
adding it to the loss. The flag alone builds the reference model and runs a reference log-prob pass every
step. On large MoE models that pass is a large share of step time and pins a second copy of the weights in
host memory; recipes that mark "KL off" drop the flag.

Megatron only:

```text
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
--attention-dropout 0.0 --hidden-dropout 0.0
--attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
--qkv-format thd                       # packed multi-turn sequences
--update-weight-buffer-size 1073741824
--offload-train-target cpu
--check-weight-update-equal
--distributed-timeout-minutes 60
```

Megatron with Adam on CPU (marked "CPU Adam" below):

```text
--optimizer-cpu-offload --use-precision-aware-optimizer
--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16
--rematerialize-param-from-master-weight
```

Environment variables: `PYTHONFAULTHANDLER=1`, `TRITON_CACHE_DIR=/tmp/triton_cache`,
`TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache`. Never `expandable_segments` (torch_memory_saver refuses
it).

Router: `--sglang-router-policy consistent_hashing` with the radix cache on (README section 4.2).

## 2. Qwen3.8-27B

Dense hybrid: 64 layers (16 full attention, 48 Gated DeltaNet), 4 KV heads of 256, vocab 248,320, native
context 262,144. Checkpoint `Qwen/Qwen3.8-27B` (a vision-language checkpoint,
`Qwen3_5ForConditionalGeneration`).

Shared by every Qwen3.8-27B row:

| Setting | Value | Why |
|---|---|---|
| Backend | Megatron | FSDP cannot split one sequence across GPUs; a long episode's logits alone (tokens × 248,320 × 2 bytes) overflow one GPU |
| `--tito-model` | `qwen35` family (newer miles: `qwen38small`) | |
| Chat template | Qwen3.8 fixed template ([miles#2760](https://github.com/radixark/miles/pull/2760)) | stock template rejects image-only user turns |
| `--model-name` | `qwen3_5` | selects the weight converter; emits `model.language_model.*` names |
| Engines | 1 GPU each (`--rollout-num-gpus-per-engine 1`), mem fraction 0.8 | SGLang TP > 1 produced garbage for this family on the version used ([sglang#21039](https://github.com/sgl-project/sglang/issues/21039)) |
| Attention backend | `--sglang-attention-backend triton` | on Blackwell, hybrid GDN models allow only `trtllm_mha`, `fa4`, `triton`; `fa3` is Hopper-only |
| Radix cache | on | keeps linear-attention states along the prefix tree; turns resume from cache |
| Health checks | `--rollout-health-check-interval 300 --rollout-health-check-timeout 300` (text), 600 / 1800 (256K) | a busy TP1 engine answers the probe late and gets killed |
| System prompt | "Put all of your reasoning inside `<think>`. After `</think>`, write only the tool call(s)." | the template allows text between `</think>` and the call; it tends to repeat the reasoning |

Table 1. Qwen3.8-27B rows.

| Row | Nodes | TP × SP × CP × PP | Context / response | `max-tokens-per-gpu` | Extra | Status |
|---|---|---|---|---|---|---|
| Text 96K | 1 or 2 | 4 × on × 2 × 1 | 98,304 / 81,920 | 49,152 (per CP rank) | vision off; `--check-weight-update-skip-list visual.` | Ran |
| Text 256K | 4 | 8 × on × 4 × 1 | 262,144 / 245,760 | 65,536 | | Ran (measured at 4 × 4 on 2 nodes) |
| Screenshots 256K | 4 | 8 × on × 1 × 1 | 262,144 / 245,760 | 262,144 (whole episode) | CPU Adam; vision provider; `--mm-tower-sync visual`; log-prob pass multiplier 1 | Ran |
| Text 256K, one B200 node | 1 | 4 × on × 2 × 1 | 262,144 / 245,760 | 131,072 | CPU Adam; `--log-probs-chunk-size 4096` | Not run |

Notes:

- `max-tokens-per-gpu` is per CP rank in Megatron: packing multiplies it by CP size.
- Text rows leave the vision tower out of training; SGLang still serves the full checkpoint, and the tower
  stays at checkpoint values.
- **Screenshots** need CP 1: mRoPE positions cannot be computed for a CP slice, and upstream's Qwen vision
  provider raises at CP > 1. TP 8 is then the only knob that lowers per-GPU memory. Upstream had no
  Megatron vision provider for this family at the time: write one (`--custom-model-provider-path`) that
  builds the text model with a frozen vision tower, or train on torchtitan.
- Screenshot rows also need `--sglang-mm-feature-transport cuda_ipc`, `SGLANG_MM_FEATURE_CACHE_MB=4096`,
  `SGLANG_HEALTH_CHECK_TIMEOUT=600`. A 250K-token prompt prefills at roughly 1,400 tokens/s in 16K chunks,
  so re-prefills take minutes.
- Use `--log-probs-chunk-size 4096` on any row with responses past ~100K tokens.

## 3. GLM-5.3-Flash

320B total, 18B active MoE (288 routed experts, top 8); 45 layers mixing KDA linear attention and DSA
sparse attention; natively multimodal. Use the BF16 checkpoint (`zai-org/GLM-5.3-Flash-BF16`): the FP8
checkpoint crashed the converter's streaming loader (FP8 tensors on CPU into a GPU-only dequant kernel).

Shared by every GLM row:

| Setting | Value | Why |
|---|---|---|
| `--tito-model` | `glm47` (text), `glm53` (images: carries GLM-5.3's image tokens) | |
| `--model-name` | `glm5_next` | weight converter |
| Engines | one per node: `--sglang-tp-size 8 --sglang-ep-size 8 --sglang-dp-size 1`, mem fraction 0.7 (0.65 with images) | the image processor runs on each engine's GPU 0 and needs ~13 GB free |
| SGLang | `--sglang-chunked-prefill-size 8192 --sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang --sglang-kv-cache-dtype bfloat16 --sglang-mm-attention-backend sdpa --sglang-mm-feature-transport cuda_ipc` | FP8 KV widens the train/rollout gap |
| Router health | `--router-health-success-threshold 1 --router-health-check-interval-secs 15 --router-health-failure-threshold 40` | |
| Trainer | `--train-memory-margin-bytes 3221225472`, `--offload-train-target cpu` (not disk: host RAM is large, local disk filled) | |
| Env | `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1`, `SGLANG_HEALTH_CHECK_TIMEOUT=1800`, `SGLANG_MM_FEATURE_CACHE_MB=4096`, `TORCHINDUCTOR_COMPILE_THREADS=1` | the IPC pool is not free: 16 GB starved a KDA backward kernel |
| Config fixes | write `rope_theta: 800000` into the HF config; for Megatron, copy `num_hidden_layers`, `indexer_rope_interleave`, `linear_attn_config` from `text_config` to the top level | the checkpoint has no `rope_theta`; SGLang defaults 800,000, transformers 10,000, and miles's config check fails. miles's DSA setup reads those keys at top level |
| Engine weights during training | released (default), not resident | resident weights leave the trainer short on 268 GB GPUs |
| KL | off on Megatron rows | the reference pass cost about a third of step time |

Megatron cannot use CP on this model: the DSA indexer raises NotImplementedError at CP > 1, and the KDA
layers get no CP group. So one episode sits whole on one rank, and context is capped by one GPU's memory.
torchtitan has CP for both layer types.

Table 2. GLM-5.3-Flash rows.

| Row | Backend | Nodes | Layout | Context | `max-tokens-per-gpu` | Extra | Status |
|---|---|---|---|---|---|---|---|
| Base, 48K, images | Megatron | 6 | TP8 SP, PP6 (first 7, last 6 layers), EP8, CP1 | 48,000 / 40,000 | 49,152 | vision provider ([miles#2792](https://github.com/radixark/miles/pull/2792)) | Ran; superseded by the 128K row |
| Text 96K | Megatron | 4 | TP8 SP, PP4 (first 11, last 12), EP8, CP1 | 98,304 / 81,920 | 98,304 | CPU Adam; no vision provider | Ran |
| Images 128K | Megatron | 8 | TP8 SP, PP8 (first 5, last 4), EP8, CP1 | 131,072 / 114,688 | 131,072 | CPU Adam; vision provider; `--mm-tower-sync visual`; log-prob multiplier 1 | Ran |
| Images 256K | torchtitan | 6 | TP1, PP1, CP8, EP8, FSDP over 48 | 262,144 / 245,760 | 262,144 (whole micro-batch) | see below | Ran |

Why PP6 on the base row: at 3 nodes the 233 GB per-rank trainer state left no room for activations; at 4
nodes Adam's state initialization ran 3 GB short. Six stages cut layers per rank from about 11 to 7.5, and
EP8 × PP6 = 48 GPUs.

**torchtitan 256K row:**

```text
--train-backend torchtitan --titan-model-name glm5_next --titan-model-flavor flash
--seq-length 262144 --gradient-checkpointing
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
--context-parallel-size 8 --expert-model-parallel-size 8
--max-tokens-per-gpu 262144            # the whole micro-batch; torchtitan splits it 8 ways inside
--optimizer-cpu-offload                # Adam moments in host RAM
--log-probs-chunk-size 4096
--use-rollout-routing-replay --pause-generation-mode in_place --sglang-moe-runner-backend triton
--use-tis                              # truncated importance sampling, clip 2.0
--model-name glm5_next --qkv-format thd
--rollout-health-check-interval 600 --rollout-health-check-timeout 1800
```

- CP8 keeps the KDA state hand-off, the DSA key gather and the expert all-to-all on one node's NVLink.
- No PP: torchtitan pads every micro-batch to `--seq-length` under PP. No TP: the GLM model refuses it.
- The episode count per step must be a multiple of DP (6 here): 72 = 9 prompts × 8 samples.
- Routing replay needs the `triton` MoE runner: SGLang's `flashinfer_trtllm` runner skips the top-k capture
  and returns zeros. `in_place` pause lets the session server fetch only new routing rows each turn.
- Host memory per node runs close to the limit (about 2.3 to 2.4 TiB of about 2.6). Watch it.
- Rehearse on one node with a 4-layer slice of the checkpoint (same model class, same code paths) before
  the 6-node job.

## 4. Muse Spark Open (LoRA)

816B total, 42B active, omni MoE: 62 layers (46 sliding-window of 2,048 tokens, 16 full attention), 256
routed experts top 8, 16 KV heads of 64. Full fine-tuning needs about 13 TB of fp32 state, so LoRA.

| Setting | Value |
|---|---|
| Nodes / layout | 4; TP8 SP, PP4 (first 15, last 15 layers), EP8, CP1 |
| Context | 262,144 / 245,760; `max-tokens-per-gpu` 262,144 |
| LoRA | `--lora-rank 32 --lora-alpha 32` on qkv, attention output and every expert's fc1/fc2; base frozen in bf16 |
| Weight sync | `--lora-sync-merged`: SGLang's Muse Spark model serves no LoRA, so each update sends the merged full weights (1.6 TB) |
| Trainer | `--custom-model-provider-path miles_plugins.models.muse_spark.provider.muse_spark_1x_omni_model_provider --model-name muse_spark --qkv-format thd --attention-backend auto --calculate-per-token-loss --skip-actor-forward-only --offload-train-target cpu --pin-rollout-manager-to-head` |
| Engines | one per node, EP8, mem fraction 0.90; `--sglang-kv-cache-dtype bfloat16 --sglang-chunked-prefill-size 8192 --sglang-mm-feature-transport cuda_ipc --sglang-swa-full-tokens-ratio 0.05` |
| Routing replay | on; the engine's default MoE runner (the model captures experts in its own forward) |
| KL | off |
| Status | Ran |

`--sglang-swa-full-tokens-ratio 0.05`: the sliding-window layers only ever read the last 2,048 tokens, but
SGLang's default sizes their KV pool at 0.8 of the full-attention pool. 0.05 moves that memory to the
full-attention pool.

## 5. Models with upstream recipes only

| Model | Upstream launcher | Shape upstream verified |
|---|---|---|
| MiMo-V2.6-Flash (309B-A15B) | [`scripts/run_mimo_v2_6_flash.py`](https://github.com/radixark/miles/blob/main/scripts/run_mimo_v2_6_flash.py) | full: 2 × 8 H200 with NVMe-streamed Adam; LoRA: 8 train + 8 sample GPUs. TP ≤ 4, no CP |
| Qwen3.8-Flash-Next (180B-A6B) | [`scripts/run_qwen3_8_next.py`](https://github.com/radixark/miles/blob/main/scripts/run_qwen3_8_next.py) | full: 32 GPUs, disk offload; 4-layer slice on one node |
| GLM-5.3-Flash LoRA | [`scripts/run_glm5_3_flash_lora.py`](https://github.com/radixark/miles/blob/main/scripts/run_glm5_3_flash_lora.py) | 3 nodes; routed experts frozen |
| DeepSeek-V4-Flash | [`scripts/run_deepseek_v4.py`](https://github.com/radixark/miles/blob/main/scripts/run_deepseek_v4.py) | 8 × 8 H200 or 8 × 4 GB300 |
| DeepSeek-V4.1-Flash LoRA | [`scripts/run_deepseek_v4_1.py`](https://github.com/radixark/miles/blob/main/scripts/run_deepseek_v4_1.py) `--train-mode lora` | attention and shared-expert adapters |
