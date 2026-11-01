# RL infrastructure on a small GPU budget

How to get the most agent RL training out of a few GPUs, written as upstream
[miles](https://github.com/radixark/miles) settings. The worked case is Qwen3.8-27B on one 8×B200 node,
with text agent episodes of up to 98,304 tokens (96K). Sections 2 to 6 cover fully async training,
section 7 is a run plan for one node, and section 8 lists ways to cut memory and time.

References are to radixark/miles `main` as of late October 2026. **Unverified** marks a claim with no
matching measurement or code check yet.

## 1. Summary

- **On one node, training and generation must share the GPUs.** At 96K, one copy of the trainer
  needs all 8 GPUs (tensor parallel 4 × context parallel 2), so no GPUs are left for separate engines.
- **Fully async removes the wait for the slowest episode.** Unfinished episodes keep running into the
  next step instead of holding the current one. miles allows this together with shared GPUs from
  `main` at or after `dfd85b00c5` (2026-10-14).
- **The one-node plan uses 56 of 75 node-hours:** two short smoke runs, then 24 hours of sync and 24
  hours of fully async on the same node, compared on trained groups per hour.
- **Two cheap levers to try first:** bf16 Adam moments (13.9 GB per GPU freed for training) and FP8
  engine weights (13 to 17 full-length episodes in flight per engine).
- **Measured on 4 nodes of B300: fully async trained 2× faster than sync** with no sign of collapse over
  10 steps; see [async-results](async-results/README.md). The one-node B200 plan below has not run yet.

## 2. Sync training wastes time on long episodes

In sync training (the default), a step launches a batch of prompt groups, waits until every group
finishes, trains, updates the engine weights, and starts the next batch. Agent episode lengths vary
widely. One slow episode holds the whole step, and engine slots sit idle while the batch tails off.

## 3. Fully async lets episodes run across train steps

### 3.1 A producer keeps episodes running, and each step takes the first finished groups

A long-lived producer (`miles.rollout.fully_async_rollout.FullyAsyncRolloutFn`, driven by
`train_async.py`) keeps up to `--async-max-concurrent-samples` episodes generating. Whenever slots free
up, it starts a new prompt group. A train step takes the first `--rollout-batch-size` finished groups.
It never waits for the slowest group. Unfinished groups keep running into the next step.

With `--colocate`, training and generation take turns on the same GPUs (Figure 1). Before each train
step, miles pauses generation in `retract` mode: it pulls the running requests out of the engines and
frees their KV cache. After the weight update, those requests resume on the new weights.

```text
           step N                                          step N+1
GPU 0-7  | generate        |P| train            |W|R| generate        |P| train   ...
         | 8 engines, TP 1 | | TP 4 × CP 2      | | | 8 engines, TP 1 | |
         | up to 96 eps    | | on all 8 GPUs    | | | same episodes   | |

P  pause (retract), flush and release KV and engine weights, move the trainer onto the GPUs
W  move the trainer to host RAM, load engine weights, update them to the new version
R  load the KV pool, resume paused requests (each re-prefills its context once)
```

Figure 1. One 8-GPU node over two steps of colocated fully async training at 96K. Not to scale: no
phase has been timed on this node.

Generation stops during the train step and the weight update, so no compute runs in parallel. The time
saved comes from scheduling: an episode spans several steps instead of holding one, and generation never
idles waiting for a straggler. To generate step N+1 while step N trains, you need a split layout, where
the trainer has its own GPUs.

### 3.2 Sharing GPUs usually costs less than giving the trainer its own

A split layout takes the trainer's GPUs away from generation for the whole run. On 4 nodes with 1 for
training, that is 25% of generation capacity. Sharing gives up only the train step's share of wall
clock. Sharing wins when training is a small part of each step, which is the usual case with long agent
episodes. On one node at 96K, sharing is the only option, because the trainer needs all 8 GPUs
(section 1).

### 3.3 A staleness limit bounds how old the sampling weights can be

An episode's early turns can come from weights several updates old. `--max-weight-staleness N` sets a
limit. When a step takes a group, miles computes the current weight version minus the version of the
group's oldest sample. Groups over N are dropped or retried (`--async-unused-samples-handler`, default
`drop`). With no limit (the default), nothing is filtered. `--async-data-buffer-capacity-factor`
(default 2.0 × the rollout batch, in groups) pauses the producer when finished groups pile up.

### 3.4 Every pause forces a re-prefill

Every pause flushes the KV cache, and the radix cache (SGLang's prefix cache) does not survive it. Each
paused request then re-prefills its whole context once after resuming. With many long episodes in
flight, every step starts with a burst of prefill. Its cost is **unverified**.

## 4. Turning it on in miles

### 4.1 You need miles `main` at or after `dfd85b00c5`

Fully async on shared GPUs needs two upstream PRs, both merged on 2026-10-14. Any `main` at or after
`dfd85b00c5` has them. Before them, miles refused this combination.

- [miles#3376](https://github.com/radixark/miles/pull/3376): the driver owns the generation pause around a
  weight update.
- [miles#3377](https://github.com/radixark/miles/pull/3377): `--fully-async` together with `--colocate`.

### 4.2 Flags, and the settings miles refuses

Required:

```text
--fully-async            # or --rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn
--colocate
--pause-generation-mode retract
--train-backend megatron # needs Megatron's IPC weight updater
```

Tuning:

```text
--async-max-concurrent-samples <n>        # default rollout_batch_size × n_samples_per_prompt; ≥ n_samples_per_prompt
--max-weight-staleness <n>                # bound on staleness, in weight versions
--async-unused-samples-handler drop|retry
--async-data-buffer-capacity-factor 2.0
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
```

`--over-sampling-batch-size` does nothing under fully async. miles logs a warning.

Table 1. Settings that `miles/utils/arguments.py` refuses with fully async on shared GPUs (verified in
code).

| Setting | Why miles refuses it |
|---|---|
| `--pause-generation-mode in_place` | the train step takes the KV cache back, so requests cannot stay in place |
| `--pause-generation-mode abort` | generation is always in flight, so every update would kill it |
| FSDP backend | needs the Megatron IPC weight updater |
| Engine weight offload with `--update-weights-interval` > 1 | offloaded weights must be republished before generation resumes; `--offload-rollout-level kv_cache` keeps them |
| `rollout` in `--ft-components` | a replaced engine would not inherit the pause or get its KV back |
| `--partial-rollout`, `--recompute-logprobs-via-prefill`, `--rollout-all-samples-process-path`, multi-LoRA, legacy rollout v1 | not supported under fully async |
| Any other `--rollout-function-path` | fully async picks its own rollout function |
| `--buffer-filter-path` | refused by the producer |

### 4.3 Fully async conflicts with routing replay on MoE models

Routing replay (`--use-rollout-routing-replay`) works best with `--pause-generation-mode in_place`. In
that mode the session server
([../single-node-training/](../single-node-training/README.md) section 5.1) fetches only each turn's
new routing rows. Fully async on shared GPUs requires `retract`. With `retract`, the engine returns
routing rows for the whole sequence on every turn: MoE layers × top-k × 4 bytes per token. For
GLM-5.3-Flash (45 layers, top 8) that is about 1.4 KB per token, or about 290 MB per turn at 200K tokens
(arithmetic; **unverified** on a run).

## 5. Rules that keep training correct

### 5.1 Correct for stale samples in the loss

An episode's early turns were sampled by older weights. PPO divides by the old policy's probability. If
the trainer recomputes that probability with the current weights, clipping anchors to a policy that
never generated those tokens.

miles enforces a correction only when a critic is used. With GRPO nothing is enforced, so pass one of
`--use-rollout-logprobs`, `--use-tis` or `--keep-old-actor` anyway. `--use-tis` (truncated importance
sampling) weights each token's loss by the trainer/engine probability ratio, clipped at 2.0 by default.
IcePop and MIS are alternatives. Which works best when one episode spans several weight versions is
**unverified**. [../rl-corrections/](../rl-corrections/) has the math for each.

### 5.2 Seven rules for the agent harness

1. **Put all rollout logic in the generate and agent functions.** Fully async refuses a custom rollout
   function, so batch-level logic must move into the agent function or the reward function. The
   other option is a miles change that lets `--fully-async` accept a `FullyAsyncRolloutFn` subclass. That
   change is not upstream.
2. **Never let one episode's error escape.** An exception from one group ends the producer and fails the
   step (the worker loop calls `task.result()`). Catch errors inside the agent function and return
   `ABORTED` samples.
3. **Let model calls outlive a train step.** miles's own HTTP client has no read timeout
   (`miles/utils/http_utils.py`). Give your OpenAI client no read timeout, or one longer than the
   longest train step plus weight update.
4. **Exclude paused time from episode time limits,** or raise the limit. Otherwise an episode gets less
   generation time than in sync mode.
5. **Expect a checkpoint to restart in-flight episodes.** `FullyAsyncRolloutFn.save()` stores running
   groups reset for retry, so a resume regenerates them from scratch. The environment adapter must open
   a fresh environment.
6. **Start without a staleness limit.** A tight `--max-weight-staleness` drops long episodes first,
   because their oldest turn is the stalest. That biases training toward short episodes. Watch the
   staleness histogram, then set a limit.
7. **Test two untested paths before a real run:** session server v2 under fully async on shared GPUs (no
   check blocks it), and episodes that return several samples (folds, summaries) with
   `--use-dynamic-global-batch-size`.

## 6. Checks that show a run works

1. `--check-weight-update-equal` passes at the start of a fresh run.
2. `train_rollout_logprob_abs_diff` stays at the sync baseline for the same model. A rise means the loss
   correction or the pause and resume path is wrong.
3. The staleness histogram of trained groups, and the groups dropped as stale per step.
4. Throughput against a matched sync run (same taskset, nodes, batch and in-flight count): step time,
   trained groups per hour, and reward against wall clock as well as against steps.
5. Prefill tokens per second right after each resume.

Failure signs in the logs:

- "No completed rollout groups for 30s", repeating: the producer is starved or stuck.
- "fully-async rollout worker ended": an exception escaped an episode (section 5.2, rule 2).
- Prefill spikes after each pause that take a large share of the step.
- Episodes ending on time limits that sync runs did not hit (section 5.2, rule 4).

## 7. A run plan for one 8×B200 node at 96K

Goal: train Qwen3.8-27B on text agent episodes of up to 98,304 tokens with fully async on one 8×B200
node, and measure the gain over sync on the same node. Budget: 75 node-hours (600 GPU-hours).

The 96K text row of [../miles-harness/recipes.md](../miles-harness/recipes.md) (Table 1) has run sync on
8×B300 nodes. It has never run on B200, and no fully async run has finished a training step yet.

### 7.1 The trainer leaves about 96 GB per GPU for activations

Table 2. Per-GPU memory on a B200 (180 GB), from the model config and recipe flags. Trainer: tensor
parallel 4, context parallel 2, optimizer state sharded over all 8 GPUs (4 ways by tensor parallel, 2 more by the Megatron distributed optimizer over the context-parallel ranks).
Engines: one per GPU, tensor parallel 1, memory fraction 0.8.

| Phase | Item | GB per GPU | Source |
|---|---|---|---|
| Train | bf16 weights (27.8B × 2 bytes / 4) | 13.9 | arithmetic |
| Train | fp32 gradients (`--accumulate-allreduce-grads-in-fp32`) | 27.8 | arithmetic |
| Train | fp32 master weights + 2 Adam moments (12 bytes / 8 ranks) | 41.7 | arithmetic |
| Train | Activations at 49,152 tokens per GPU, full recompute | not measured | smoke run 1 |
| Generate | Engine weights | 55.6 | [../single-node-training/](../single-node-training/README.md) section 4.2 |
| Generate | KV and linear-attention pool (0.8 × 180 − 55.6) | 88 | arithmetic |

Without activations the trainer needs 83.4 GB, which leaves about 96 GB for activations and buffers. If
the first smoke run peaks above about 170 GB, move Adam to the CPU (section 8).

While the engines generate, the trainer waits in pinned host memory: 83.4 GB per GPU, 667 GB per node,
plus Adam if it lives on the CPU. Plan for at least 1 TB of host RAM.

### 7.2 Each engine holds 13 full-length episodes

- KV per token: 16 full-attention layers × 4 KV heads × 256 × 2 (K and V) × 2 bytes = 64 KiB.
- Linear-attention state per request: 48 layers × 48 value heads × 128 × 128 × 4 bytes = 151 MB, whatever
  the length.
- A full 96K episode: 6.4 GB of KV plus 0.15 GB of state, about 6.6 GB.
- The 88 GB pool holds 13 full-length episodes per engine, 104 per node. Shorter episodes fit more.
  Requests beyond the pool wait in a queue and do not fail.

### 7.3 Settings for the run

The 96K text row of [../miles-harness/recipes.md](../miles-harness/recipes.md) and its shared settings,
plus:

```text
# Fully async on shared GPUs (section 4.2)
--fully-async --colocate --pause-generation-mode retract --train-backend megatron
--update-weights-interval 1

# Batch and episodes in flight
--n-samples-per-prompt 8
--rollout-batch-size 8                    # 8 groups = 64 episodes per train step
--async-max-concurrent-samples 96         # 12 per engine, under the 13 full-length episodes that fit
--async-data-buffer-capacity-factor 2.0
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted

# Stale samples (sections 5.1 and 5.2)
--use-tis
# no --max-weight-staleness at first; read the staleness histogram, then set it

# Layout (96K text row)
--tensor-model-parallel-size 4 --sequence-parallel --context-parallel-size 2
--pipeline-model-parallel-size 1 --max-tokens-per-gpu 49152
--rollout-max-context-len 98304 --rollout-max-response-len 81920
--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.8
```

### 7.4 Four steps in 75 node-hours, each gated on a check

Each step must pass its check before the next one starts. Steps 3 and 4 run for the same fixed
wall-clock time, because sync and async step times differ and neither has been measured on this node.
Both use the same taskset, batch, in-flight count (96) and starting checkpoint.

Table 3. Run plan for one 8×B200 node. Hours are wall-clock caps, so they are also node-hours.

| Step | Run | Cap (h) | Pass if |
|---|---|---|---|
| 1 | Sync smoke: 3 steps, 2 groups of 8 | 4 | peak trainer memory under 170 GB per GPU (else Adam on CPU and rerun); `--check-weight-update-equal` passes; `train_rollout_logprob_abs_diff` near about 0.013, the value measured on sync B300 runs of this row |
| 2 | Fully async smoke: 3 steps, section 7.3 at 2 groups per step | 4 | the producer survives every pause (no "worker ended"); paused requests resume on new weights; log-prob gap unchanged from step 1; re-prefill time after each resume recorded |
| 3 | Sync: section 7.3 without the fully async flags | 24 | none (this is the reference run) |
| 4 | Fully async: section 7.3 | 24 | the checks of section 6 |
| | Reserve for a failed step | 19 | |
| | Total | 75 | |

### 7.5 What to report

- Main figure: cumulative trained groups against wall-clock hours, sync and async on one axis.
- Reward against wall-clock hours, and against train steps.
- Step time split into generate, re-prefill, train and weight update.
- Staleness histogram of trained groups, and groups dropped or retried per step.
- Peak GPU memory per phase, and host memory, against Table 2.

## 8. Try bf16 Adam moments first, then FP8 engine weights

Table 4. Levers for the plan of section 7, with savings from the arithmetic of section 7.1. None is
measured on this run yet.

| Lever | Flags or checkpoint | Saves per GPU | Effect on the plan | Risk |
|---|---|---|---|---|
| bf16 Adam moments | `--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` | 13.9 GB (moments go from 8 to 4 bytes per parameter, sharded 8 ways) | more room for activations in the train step | small; part of the "CPU Adam" set in [../miles-harness/recipes.md](../miles-harness/recipes.md) section 1 |
| FP8 engine weights, bf16 trainer | FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`) as `--hf-checkpoint` for SGLang; bf16 `torch_dist` checkpoint as `--ref-load` for the trainer (the setup in the miles low-precision docs) | engine weights from 55.6 to about 28 GB | KV pool from 88 to about 116 GB: 17 full-length episodes per engine instead of 13 | trainer and engine log-probs drift apart; miles recommends TIS, already on in section 7.3. SGLang support for this checkpoint on the pinned version is unverified |
| Adam on the CPU | "CPU Adam" flags in [../miles-harness/recipes.md](../miles-harness/recipes.md) section 1 | the whole 41.7 GB optimizer shard | fallback if smoke run 1 runs out of memory | Adam steps on the CPU; the slowdown is not measured |
| FP8 KV cache | `--sglang-kv-cache-dtype fp8_e4m3` | KV from 64 to 32 KiB per token | a full 96K episode needs about 3.4 GB: 26 per engine instead of 13 | widened the trainer/engine log-prob gap on GLM-5.3-Flash ([../miles-harness/recipes.md](../miles-harness/recipes.md) section 3) |
| FP8, MXFP8 or NVFP4 training | `--fp8-recipe` with the matching engine precision ([miles low-precision docs](https://github.com/radixark/miles/blob/main/docs/advanced/low-precision.md)) | little: master weights and Adam stay in high precision | faster matrix multiplies in the train step | upstream tested Qwen3-4B, Qwen3-30B-A3B and DeepSeek-V4-Flash only (MXFP8 and NVFP4 are beta); not Gated DeltaNet layers |
| INT4 QAT | [miles INT4 QAT docs](https://github.com/radixark/miles/blob/main/docs/advanced/int4-qat.md) | none for the trainer | does not apply: it covers routed MoE experts, and Qwen3.8-27B is dense | n/a |

Check each lever against the log-prob gap of smoke run 1. Leave the FP8 KV cache and low-precision
training until bf16 Adam moments and FP8 engine weights both hold.

## 9. Open questions

- The speed-up of fully async over sync for long agent episodes (section 7.4 measures it).
- The re-prefill cost after each pause.
- Session server v2 and multi-sample episodes under fully async on shared GPUs.
- The best loss correction when one episode spans several weight versions.
