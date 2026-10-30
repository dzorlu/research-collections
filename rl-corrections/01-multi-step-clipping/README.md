# Multi-step clipping: PPO and CISPO

PPO's clip is a trust region: it stops a policy from drifting too far from the policy that generated the data.
It can only act when one rollout gets several optimizer steps. Our runs take one step per rollout, so the clip never fires
(Qwen 0.019% of tokens, GLM 0%). In our runs the step size is limited by the learning rate, Adam and gradient-norm clipping.

## 1. The loss for one token

```
loss = -min(ratio * A, clip(ratio, 0.8, 1.28) * A)
```

- `ratio = p_now / p_old`: the token's probability under the weights being trained, over its probability when the rollout was scored.
- `A` is the advantage: the episode's reward compared with the average reward of the other episodes for the same prompt (GRPO). A > 0 means better than average.
- 0.8 and 1.28 come from `--eps-clip 0.2 --eps-clip-high 0.28`, which our launcher sets.

Table 1. What the clip does to one token. A token counts as clipped only in the "yes" rows (miles `math_utils.py:262-266`, `train/pg_clipfrac`).

| Episode | Ratio | Clipped? | Gradient for this token |
|---|---|---|---|
| good (A > 0) | above 1.28 | yes | 0: stop raising it |
| good (A > 0) | below 0.8 | no | full: keep raising it |
| bad (A < 0) | below 0.8 | yes | 0: stop lowering it |
| bad (A < 0) | above 1.28 | no | full: keep lowering it |

## 2. Why the clip needs several steps per rollout

```
episodes = generate(prompts, weights)            # rollout on the engine
p_old    = prob(weights, episodes)               # scored ONCE, before any step, then frozen
A        = advantage(rewards)

for minibatch in split(episodes, k):             # k = --num-steps-per-rollout
    p_now   = prob(weights, minibatch)           # scored with the CURRENT weights
    ratio   = p_now / p_old[minibatch]
    loss    = -min(ratio * A, clip(ratio, 0.8, 1.28) * A)
    weights = weights - lr * grad(loss)          # the weights change here

send weights to the engine
```

miles does exactly this: `p_old` for the whole rollout first (miles `megatron_utils/actor.py:735`), then one optimizer step per minibatch (`actor.py:676`).

- **k = 1 (our runs).** The loop runs once. `p_now` is computed before the only weight change, so every ratio is 1.0 and the clip does nothing. The gradient is the plain policy gradient: raise good tokens, lower bad ones, scaled by A.
- **k = 4.** Minibatch 1 sees ratio 1.0 and changes the weights. Minibatches 2 to 4 are scored with the changed weights, so their ratios are no longer 1.0.

Example, k = 4 (made-up numbers): a good episode in minibatch 3 contains the token "banana", `p_old` = 0.40.
The steps on minibatches 1 and 2 also raised it, to `p_now` = 0.55. Ratio 1.375 is above 1.28, so its gradient is 0:
earlier steps already raised it enough. Original PPO also reruns the same rollout for several epochs, so every token gets this check.

## 3. What limits each step in our runs

Table 2. Step-size limits, from the configs of our agent training runs.

| Limit | Qwen | GLM | What it bounds |
|---|---|---|---|
| Learning rate (Adam) | 5e-6 | 1e-6 | size of each step |
| Gradient-norm clip | 1.0 | 1.0 | scales the gradient down when its norm is above 1.0 |
| KL penalty to the reference model | 0.01 | 0.01 | drift from the starting model over the whole run, not per step |
| PPO ratio clip | 0.8 to 1.28 | 0.8 to 1.28 | nothing: k = 1 |

k = 1 because our launcher passes `--use-dynamic-global-batch-size`, which sets the batch to the rollout's sample count (miles `rollout_data_conversion.py:81-89`).
How far one step moves the policy (KL between the policies before and after a step) is not logged.

## 4. Measured values

Table 3. Clip fraction and `train/ppo_kl` (mean of `log p_old - log p_now`) per step, from our agent training runs.

| Model | Runs | Steps | Clip fraction, mean | Clip fraction, max | ppo_kl range |
|---|---|---|---|---|---|
| Qwen3.8-27B (Megatron) | 2 | 18 | 0.019% | 0.025% | -1.1e-5 to 2.1e-5 |
| GLM-5.3-Flash (torchtitan) | 2 | 20 | 0% | 0% | exactly 0 |

Qwen is not exactly 0 because Megatron's two passes differ slightly: `p_old` runs without gradients in eval mode, `p_now` with gradients in train mode
(miles `megatron_utils/model.py:279-280, 385, 407`). A token clips only if its log-prob moves by more than 0.247 up or 0.223 down;
about 2 tokens in 10,000 do. Which kernel causes this is unverified; one step with `--dump-details` would list the tokens (miles `debug_dump.py:44-65`).
GLM's two passes are bit-identical (`ppo_kl` exactly 0); why is unverified. GLM uses routing replay and Qwen does not.

GLM's `train/tis_clipfrac` (0.012% to 0.054%) is a different ratio: trainer `p_old` over the engine's sampling probability (miles `corrections.py:22-25`).

## 5. How to turn the clip on

- Several steps per rollout: `--num-steps-per-rollout k` (miles `arguments.py:1393-1399`), the same as `--global-batch-size` = samples / k.
  Our launcher would have to stop passing `--use-dynamic-global-batch-size`; whether the two flags combine is unverified.
- Or stale samples (fully-async training): the episodes come from older weights, so ratios differ from 1.0 at the first step.


## 6. CISPO

Write-up: [cispo.md](cispo.md) (the three losses, where the factor r comes from, backprop table).

Start with [cispo_simple.ipynb](cispo_simple.ipynb) ([open in Colab](https://colab.research.google.com/github/dzorlu/research-collections/blob/main/rl-corrections/01-multi-step-clipping/cispo_simple.ipynb)):
one token, three logits, where `p_old` comes from, and 8 steps of PPO vs CISPO in pure Python.

[cispo_backward.ipynb](cispo_backward.ipynb) ([open in Colab](https://colab.research.google.com/github/dzorlu/research-collections/blob/main/rl-corrections/01-multi-step-clipping/cispo_backward.ipynb))
runs PPO and CISPO (MiniMax-M1, arXiv 2506.13585) by hand through a two-layer network, checked against PyTorch.
Both give the logits the gradient `w * A * (p - y_onehot) / N`; only the per-token weight `w` differs
(PPO: `r`, or 0 when clipped; CISPO: `clip(r, 0, 4)`, frozen). With 16 steps on one rollout, PPO zeroes the
winning rare token from step 1 and stops at p = 0.326; CISPO keeps it at weight 4 and reaches p = 0.972.
At one step per rollout (our runs) the three losses give the same gradient.

Code: miles = radixark/miles main (eedc8c2287 for sections 2 and 5, 24bd5fb716 for the rest).
