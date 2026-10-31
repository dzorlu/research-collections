# Colocated fully-async RL on miles: principles, build and run

Fully-async training stops a step from waiting on its slowest episode. Colocated means training and
generation share the same GPUs. This page explains what the combination does in upstream miles, what a
harness must do to be compatible, and how to tell whether a run is working. It extends the harness spec in
[README.md](README.md).

References are to radixark/miles `main` (late October 2026). Items marked **unverified** have no matched
measurement or code check yet.

## 1. What it is

**Sync (the default).** A step launches a batch of prompt groups, waits until every group finishes, trains,
updates the engine weights, and starts the next batch. One slow episode holds the whole step, and engine
slots sit idle while the batch tails off.

**Fully async.** A long-lived producer (`miles.rollout.fully_async_rollout.FullyAsyncRolloutFn`, driven by
`train_async.py`) keeps up to `--async-max-concurrent-samples` trajectories generating. Whenever slots free
up it submits a new prompt group. A train step takes the first `--rollout-batch-size` finished groups the
data buffer accepts. It never waits for the slowest group, and unfinished groups keep running into the next
step.

**Colocated fully async: no compute overlap.** With `--colocate`, each step does this on the shared GPUs
(`train_async.py`):

```text
pause_generation(retract) → flush_cache → offload KV (+ weights) → train → offload trainer
  → onload weights → update_weights → onload KV → continue_generation
```

Generation is frozen during the train step and the weight update; paused requests then continue on the new
weights. The gain is in time, not in parallel compute: an episode spans several steps instead of being cut,
and generation never idles waiting for a straggler. Running step N+1's rollouts during step N's training
needs the split layout, where the trainer has its own GPUs.

**How stale the data can get.** `--max-weight-staleness N` bounds it. At consume time, the buffer computes
each group's lag as the current weight version minus the version of its oldest sample. Groups over N are
dropped or recycled per `--async-unused-samples-handler` (`drop` by default, or `retry`). The default (no
limit) applies no filter. `--async-data-buffer-capacity-factor` (default 2.0 × the rollout batch, in groups)
blocks the producer when finished groups pile up.

## 2. Version

Requires both of these upstream PRs (merged 2026-10-14). Any `main` at or after `dfd85b00c5` has them.

- [miles#3376](https://github.com/radixark/miles/pull/3376): the driver owns the generation pause around a
  weight update.
- [miles#3377](https://github.com/radixark/miles/pull/3377): `--fully-async` together with `--colocate`.

Before these, miles asserted that fully async could not be colocated, which forced a split layout.

## 3. Flags and what miles refuses

Required:

```text
--fully-async            # or --rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn
--colocate
--pause-generation-mode retract
--train-backend megatron # colocated fully async needs Megatron's IPC weight updater
```

Tuning:

```text
--async-max-concurrent-samples <n>        # default rollout_batch_size × n_samples_per_prompt; ≥ n_samples_per_prompt
--max-weight-staleness <n>                # off-policy bound, in weight versions
--async-unused-samples-handler drop|retry
--async-data-buffer-capacity-factor 2.0
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
```

`--over-sampling-batch-size` has no effect under fully async; miles logs a warning.

Refused by `miles/utils/arguments.py` (verified):

| Setting | Why |
|---|---|
| `--pause-generation-mode in_place` with `--colocate` | the train step takes the KV cache back, so "keep it in place" cannot hold |
| `--pause-generation-mode abort` | generation is always in flight; every update would kill it |
| FSDP backend with `--colocate` | needs the Megatron IPC weight updater |
| Rollout weight offload with `--update-weights-interval` > 1 | offloaded weights must be republished before generation resumes; use `--offload-rollout-level kv_cache` to keep them |
| `rollout` in `--ft-components` | a replaced engine would not inherit the pause or get its KV back |
| `--partial-rollout`, `--recompute-logprobs-via-prefill`, `--rollout-all-samples-process-path`, multi-LoRA, legacy rollout v1 | not supported under fully async |
| Any other `--rollout-function-path` | fully async selects its own rollout function: "pass only one" |
| `--buffer-filter-path` | refused by the producer |

**Conflict with expert-routing replay.** Routing replay (`--use-rollout-routing-replay`) prefers
`--pause-generation-mode in_place`, which lets the session server fetch only each turn's new routing rows.
Colocated fully async requires `retract`. With `retract`, the engine returns routing rows for the whole
sequence on every turn: about MoE layers × top-k × 4 bytes per token, so roughly 1.4 KB per token and
about 290 MB per turn at 200K tokens for a 45-layer, top-8 model (arithmetic; **unverified** on a run).

## 4. Loss-side correction

An episode's early turns were sampled by older weights. In PPO, the ratio's denominator is the old policy's
probability. If the trainer recomputes it with the current actor, clipping anchors to a policy that never
generated those tokens.

miles enforces a correction only when a critic is used: one of `--use-rollout-logprobs`, `--use-tis` or
`--keep-old-actor`. With GRPO nothing is enforced, so pass one anyway. `--use-tis` (truncated importance
sampling) weights each token's loss by the trainer/rollout probability ratio, clipped (default 2.0).
IcePop and MIS are alternatives. Which works best when one episode spans several weight versions is
**unverified**. [../rl-corrections/](../rl-corrections/) has the math for each.

## 5. Memory

The envelope is the same as sync colocated:

- During generation, the engines hold their weights and their static KV pool. The trainer sits in host
  RAM (`--offload-train-target cpu`, through torch_memory_saver, which refuses `expandable_segments`).
- During training, the KV pool is released (and the engine weights too, with weight offload), and the
  trainer moves back onto the GPUs.

The one difference: every pause flushes the KV cache. Each in-flight request then re-prefills its whole
context once after resume. With many long episodes in flight, that is a prefill burst at the start of
every step (cost **unverified**). The radix cache does not survive the flush.

Sizing:

- **In flight per engine:** in-flight trajectories × context tokens × KV bytes per token, against the static
  pool. Over that, requests queue instead of failing.
- **Colocated vs split:** a split layout permanently gives the trainer's GPUs up (for example, 1 of 4 nodes
  is 25% of generation capacity). Colocated gives up train time / step time of wall clock. Colocated wins
  when training is a small share of each step, which is the usual case for long agent episodes.

## 6. Harness compatibility

1. **Rollout logic lives in the generate and agent functions.** Fully async refuses a custom rollout
   function, so batch-level logic must move into the agent function or the reward function. The
   alternative is a miles change that lets `--fully-async` accept a `FullyAsyncRolloutFn` subclass; that
   change is not upstream.
2. **Never let one episode's error escape.** An exception from one group's generate task ends the producer
   and fails the step (the worker loop calls `task.result()`). Catch per-episode errors inside the agent
   function and return `ABORTED` samples.
3. **Survive the pause.** Agent HTTP calls to the engines must outlive a train step. miles's own HTTP
   client has no read timeout (`miles/utils/http_utils.py`). Give your OpenAI client no read timeout, or
   one longer than the longest train step plus weight update.
4. **Wall-clock caps count the pause.** An episode timeout of T seconds gets less generation time than in
   sync mode. Either exclude paused time or raise the cap.
5. **Checkpoints restart in-flight episodes.** `FullyAsyncRolloutFn.save()` stores running groups reset
   for retry, so a resume regenerates them from scratch. Environment state from the half-done episode is
   lost; the environment adapter must open fresh.
6. **Staleness and long episodes.** A tight `--max-weight-staleness` drops long episodes first, because
   their oldest turn is the stalest. That silently biases training toward short episodes. Start without a
   limit (or with a generous one) and watch the histogram.
7. **Unverified with this harness:** session server v2 under fully-async colocate (no assert blocks it),
   and episodes that return several samples (folds, summaries) together with
   `--use-dynamic-global-batch-size`. Gate both before a real run.

## 7. Run gates

1. `--check-weight-update-equal` passes at the start of a fresh run.
2. `train_rollout_logprob_abs_diff` stays at the sync baseline for the same model. A rise means the
   correction or the pause/resume path is wrong.
3. Staleness: histogram of consumed staleness, and groups dropped as stale per step.
4. Throughput against a matched sync run (same taskset, nodes, batch and in-flight count): step time,
   trained groups per hour, and reward against wall clock, not only against steps.
5. Re-prefill: prefill tokens per second right after each resume.

Failure modes to watch in the logs:

- "No completed rollout groups for 30s", repeating: the producer is starved or stuck.
- "fully-async rollout worker ended": an exception escaped an episode (section 6.2).
- Re-prefill spikes after each pause that eat a large share of the step.
- Episodes ending on wall-clock timeouts that sync runs did not hit (section 6.4).

## 8. Open questions

- The real speed-up over sync for long agent episodes: no matched measurement yet.
- The re-prefill cost after each pause.
- Session server v2 and multi-sample episodes under fully-async colocate.
- The best loss correction under multi-version staleness.

## 9. Plan: colocated fully async on one 8-GPU node at 96K

Goal: train Qwen3.8-27B on text agent episodes of up to 98,304 tokens with colocated fully async on a
single 8×B200 node, and measure the gain over sync on the same node. The budget is 75 node-hours
(600 GPU-hours).

### 9.1 Why colocated is the only option on one node

```text
           step N                                          step N+1
GPU 0-7  | generate        |P| train            |W|R| generate        |P| train   ...
         | 8 engines, TP 1 | | TP 4 × CP 2      | | | 8 engines, TP 1 | |
         | up to 96 eps    | | on all 8 GPUs    | | | same episodes   | |

P  pause (retract), flush and release KV and engine weights, move the trainer onto the GPUs
W  move the trainer to host RAM, load engine weights, update them to the new version
R  load the KV pool, resume paused requests (each re-prefills its context once)
```

Figure 1. One node over two steps. Not to scale: no phase has been timed on this node.

The 96K text row ([recipes.md](recipes.md) Table 1) trains one model copy on TP 4 × CP 2, which uses all 8
GPUs. A split layout would need GPUs for the trainer on top of those 8 for the engines, so it does not fit
one node. Colocated fully async needs no extra GPUs: the trainer and the 8 engines take turns on the same
GPUs (section 1).

The 96K row has run sync on one 8×B300 node. No colocated fully-async run has finished a training step on
any node count yet, and the row has never run on B200.

### 9.2 Memory on a B200 (180 GB)

Table 1. Per-GPU memory, from the model config and the recipe flags. Trainer: TP 4, CP 2, DP 1, Megatron
distributed optimizer sharded over the 2 CP ranks. Engines: one per GPU, TP 1, memory fraction 0.8.

| Phase | Item | GB per GPU | Source |
|---|---|---|---|
| Train | bf16 weights (27.8B × 2 bytes / TP 4) | 13.9 | arithmetic |
| Train | fp32 gradients (`--accumulate-allreduce-grads-in-fp32`) | 27.8 | arithmetic |
| Train | fp32 master weights + 2 Adam moments (12 bytes / 8 ranks) | 41.7 | arithmetic |
| Train | Activations at 49,152 tokens per GPU, full recompute | not measured | smoke gate 1 |
| Generate | Engine weights | 55.6 | [single-node notes](../single-node-training/README.md) section 4.2 |
| Generate | KV and linear-attention pool (0.8 × 180 − 55.6) | 88 | arithmetic |

Without activations the trainer needs 83.4 GB, which leaves about 96 GB for activations and buffers. If
gate 1 measures a peak above about 170 GB, move Adam to the CPU (the "CPU Adam" flags in
[recipes.md](recipes.md) section 1). That frees the 41.7 GB optimizer shard.

How many episodes fit in the engine pool:

- KV per token = 16 full-attention layers × 4 KV heads × 256 × 2 (K, V) × 2 bytes = 64 KiB.
- Linear-attention state per request = 48 layers × 48 value heads × 128 × 128 × 4 bytes (fp32) = 151 MB,
  independent of length.
- A full 98,304-token episode holds 6.4 GB of KV plus 0.15 GB of state, so about 6.6 GB.
- 88 GB per engine holds 13 full-length episodes, so 104 per node. Shorter episodes fit more. Requests
  beyond the pool queue instead of failing.

Host RAM: while the engines generate, the trainer's 83.4 GB per GPU (667 GB per node) sits in pinned host
memory, plus the CPU Adam state if used. Plan for at least 1 TB of host RAM.

### 9.3 Settings

The 96K text row of [recipes.md](recipes.md) Table 1 and its shared settings, plus:

```text
# Fully async, colocated (section 3)
--fully-async --colocate --pause-generation-mode retract --train-backend megatron
--update-weights-interval 1

# Batch and in-flight count
--n-samples-per-prompt 8
--rollout-batch-size 8                    # 8 groups = 64 episodes per train step
--async-max-concurrent-samples 96         # 12 per engine, under the 13 full-length episodes that fit
--async-data-buffer-capacity-factor 2.0
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted

# Off-policy handling (sections 4 and 6.6)
--use-tis
# no --max-weight-staleness at first; read the staleness histogram, then set it

# Layout (96K text row)
--tensor-model-parallel-size 4 --sequence-parallel --context-parallel-size 2
--pipeline-model-parallel-size 1 --max-tokens-per-gpu 49152
--rollout-max-context-len 98304 --rollout-max-response-len 81920
--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.8
```

Version: miles `main` at or after `dfd85b00c5` (section 2).

### 9.4 Steps and budget

Each step must pass its gate before the next starts. The comparison in step 4 uses fixed wall-clock
windows instead of a fixed step count, because sync and async step times differ and neither has been
measured on this node.

Table 2. Run plan for one 8×B200 node. Hours are wall-clock caps, so they are also node-hours.

| Step | Run | Cap (h) | Gate to pass |
|---|---|---|---|
| 1 | Sync smoke: 3 steps, 2 groups of 8 | 4 | peak trainer memory per GPU under 170 GB (else CPU Adam and rerun); `--check-weight-update-equal` passes; `train_rollout_logprob_abs_diff` at the sync baseline (about 0.013) |
| 2 | Fully-async smoke: 3 steps, settings of 9.3 at 2 groups per step | 4 | the producer survives every pause (no "worker ended"); paused requests resume on new weights; log-prob gap unchanged from step 1; re-prefill time after each resume recorded |
| 3 | Sync baseline: settings of 9.3 without the async block | 24 | none; it is the reference |
| 4 | Fully async: settings of 9.3 | 24 | run gates of section 7 |
| | Reserve for a failed gate | 19 | |
| | Total | 75 | |

Steps 3 and 4 use the same taskset, batch size and in-flight count (96), and start from the same
checkpoint.

### 9.5 What to report

- Main figure: cumulative trained groups against wall-clock hours, sync and async on one axis.
- Reward against wall-clock hours, and against train steps.
- Step time split into generate, re-prefill, train and weight update.
- Staleness histogram of consumed groups, and groups dropped or recycled per step.
- Peak GPU memory per phase, and host memory, against Table 1.
