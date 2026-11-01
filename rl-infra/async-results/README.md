# Fully async on shared GPUs trains 2× faster than sync

We trained the same model on the same task set twice: once with normal (sync) steps and once with
miles's fully async mode on the same GPUs. Fully async trained twice as many prompt groups per hour.
The price is that some training data comes from slightly older weights. Over 10 steps that did not hurt
the reward.

## 1. Results

- **2× faster.** After the first step, fully async trained 205 prompt groups per hour; sync trained
  102. All 10 steps took 54 minutes instead of 97 (Figure 1).
- **Why:** a sync step waits for its slowest rollouts and throws away the unfinished ones. Fully
  async never waits: unfinished rollouts pause during training and then continue, and finished ones
  are replaced at once. The engines stayed busy, with 258 rollouts running on average instead of 123.
- **The cost is staleness.** A trained token was 1.5 weight updates old on average, at most 5. More
  than half of the rollouts (54%) were written by two or more versions of the weights (Figures 2 to 4).
- **Staleness widens the engine/trainer gap in a straight line.** The gap between SGLang's and the
  trainer's log-probabilities grows by about 0.01 per update of lag, starting from the 0.011 that sync
  also has (Figures 5 and 6).
- **No sign of collapse over 10 steps.** Reward stayed in the same band: 0.75 (fully async) and 0.76
  (sync) (Figure 7).

## 2. Setup

Table 1. Both runs use every value below; only the last two rows differ.

| Setting | Value |
|---|---|
| Model | Qwen3.8-27B |
| Task set | [EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym): 649 public tasks with MCP tools; reward = share of the task's checks that pass |
| Hardware | 4 nodes × 8 NVIDIA B300; trainer and 32 SGLang engines share the GPUs |
| Batch | 16 prompt groups × 8 rollouts per step, 10 steps, one optimizer step per batch |
| Rollouts | Up to 100 turns, 32,768 new tokens per turn, 262,144-token context |
| Training | Megatron, tensor parallel 8; GRPO; per-token loss; KL to the reference 0.01; learning rate 5e-6 |
| Sync | Each step starts 48 groups and trains on the first 16 that finish with mixed rewards; the rest are aborted |
| Fully async | Up to 384 rollouts running; groups more than 5 updates old are dropped; truncated importance sampling |

## 3. How to run it

Use miles `main` from 2026-10-14 or later, which has
[#3376](https://github.com/radixark/miles/pull/3376) and [#3377](https://github.com/radixark/miles/pull/3377).
Start training with `train_async.py` (`train.py` refuses `--fully-async`) and add these flags to a
normal colocated Megatron setup:

```
--colocate
--fully-async
--async-max-concurrent-samples 384      # rollouts running at once
--max-weight-staleness 5                # drop groups older than 5 updates
--pause-generation-mode retract         # the only mode allowed with --colocate
--use-tis                               # correct for older weights (cap 2.0)
--calculate-per-token-loss
```

Your rollout code must also do four things:

- **Keep model calls waiting through a pause.** Training takes 1 to 2 minutes per step, and calls to
  the engines must not time out in that time.
- **Not abort requests while the engines are paused.** Aborting them ends every rollout in flight
  (pitfall 1 below).
- **Not cap rollouts below `--async-max-concurrent-samples`.** miles already limits how many run at
  once; an extra cap elsewhere can starve the engines (pitfall 2).
- **Catch errors per rollout.** One rollout that raises an exception stops the whole producer.

Table 2. Checks for the first steps, with the values from our run.

| Check | Expected |
|---|---|
| Rollouts aborted or failed at each training pause | 0 |
| Gap between engine and trainer log-probabilities at step 1 | About 0.011, the same as sync |
| Oldest group in a batch | At most the staleness limit |
| Step time after step 1 | About half of sync's (270 s vs 490 s here) |

## 4. Speed

![Figure 1](cumulative_groups.png)

*Figure 1. Prompt groups trained against wall-clock time. Each step trains 16 groups.*

Table 3. Where the time goes. "Generating" excludes the pauses for training.

| Measure | Sync | Fully async |
|---|---|---|
| Median step time after step 1 | 491 s | 270 s |
| Rollouts running, on average, while generating | 123 | 258 |
| Model turns per hour of generating | 16,298 | 33,067 |
| Rollouts aborted unfinished | 590 of 3,835 | 0 of 2,586 |
| Share of wall time spent paused for training | 1% | 39% |
| Prefix-cache hit rate | 95% | 82% |
| Median time between two turns of a rollout | 12.3 s | 16.2 s |

Fully async pays three costs, all in Table 3. Generation stops while the model trains (1 to 2
minutes per step). Each pause clears the engines' prefix cache, so paused rollouts re-read their
context. And busier engines make each turn a little slower. The 2× speed-up is what remains after all
three.

## 5. Staleness and long rollouts

![Figure 2](trajectory_timeline.png)

*Figure 2. One row per rollout; each segment is one model reply, colored by the weight version that
wrote it. A color change inside a row means the rollout continued on new weights.*

![Figure 3](staleness_over_time.png)

*Figure 3. How many updates old the trained tokens were, per step: the average, and the oldest group
in the batch.*

![Figure 4](versions_per_rollout.png)

*Figure 4. How many weight versions wrote each finished rollout's replies.*

Long rollouts are the ones that span several updates, and they are also the hard ones (Table 4).

Table 4. Finished rollouts by the number of weight updates that happened while they ran.

| Updates | Share of rollouts | Median length | Median turns | Pass rate |
|---|---|---|---|---|
| 0 | 22% | 96 s | 6 | 50% |
| 1 | 50% | 238 s | 6 | 49% |
| 2 | 20% | 509 s | 7 | 46% |
| 3 | 6% | 816 s | 9 | 33% |
| 4 to 7 | 3% | 1,318 s | 10 | 21% |

**Long rollouts count more in training.** With `--calculate-per-token-loss`, every token gets the same
weight, so a rollout counts in proportion to its length. Without the flag, every rollout counts the
same. In this run, the rollouts that spanned two or more updates held 40% of the context tokens.

## 6. Engine/trainer gap

`p_engine` is the probability SGLang gave a sampled token. `p_old` is the trainer's probability for
the same token just before the update. miles logs the average of |log p_old − log p_engine| per step.

![Figure 5](mismatch_over_time.png)

*Figure 5. The average gap per step. Sync stays flat; fully async rises after the first update.*

![Figure 6](mismatch_vs_staleness.png)

*Figure 6. The gap for each fully async step against how stale that step's tokens were, with a
straight-line fit.*

Sync's gap of 0.011 is the small numerical difference between SGLang and the trainer on the same
weights. Fully async starts at the same value and adds about 0.01 per update of staleness (Table 5).

Table 5. The gap at typical staleness, from the fitted line.

| Staleness (updates) | Average gap in log-probability | Typical difference in token probability |
|---|---|---|
| 0 | 0.011 | 1.1% |
| 1.5 (fully async average) | 0.027 | 2.7% |
| 2.2 (fully async worst step) | 0.034 | 3.5% |

Truncated importance sampling reweights each token by p_old / p_engine and caps the weight at 2. The
cap applied to fewer than 0.1% of tokens in every step, so at this staleness the correction stays
small.

## 7. Reward

![Figure 7](reward_over_time.png)

*Figure 7. Average reward of the trained groups per step.*

Over all finished rollouts, the average reward was 0.78 (sync) and 0.76 (fully async). Sync throws away
its unfinished long rollouts, which are the harder ones, so its finished set is slightly easier; that
is a likely reason for the small gap. Entropy rose the same way in both runs.

## 8. Pitfalls

1. **Aborting requests during the pause kills the rollouts.** miles pauses the engines, then frees
   their memory for training. A patch that aborted all requests before freeing the memory turned every
   paused request into an aborted one, and every rollout in flight died at each step. Upstream miles
   does not abort there. The fix for the patch is in [#4519](https://github.com/radixark/miles/pull/4519).
2. **An extra cap on running rollouts made fully async slower than sync.** Our environment service
   limited each group's rollouts using a count that treated finished rollouts as still running. Most
   groups ran one rollout at a time, only 76 ran at once, and steps took longer than sync's.
3. **A kernel version mismatch crashed the first training step.** FlashAttention 4 (4.0.0b19) on Blackwell
   with head dimension 256 calls a function that quack-kernels 0.6.5 lacks; 0.6.4 has it.

## 9. Limits

- One model, one task set, 10 steps, one run of each mode.
- The two runs ran about 3 hours apart on a shared cluster.
- Per-token probabilities and tool-call times were not saved, so the gap is split by each step's
  average staleness, not by each token's own age, and Figure 2 shows replies but not tool calls.
