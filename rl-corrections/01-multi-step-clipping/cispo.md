# Multi-step clipping: REINFORCE vs PPO vs CISPO

PPO and CISPO both multiply each token's gradient by how much more likely the token is now than when it was
written. PPO sets that factor to 0 once the token has moved too far. CISPO caps it at 4 instead.

```math
\boxed{r = \frac{p_{\text{now}}(y)}{p_{\text{old}}(y)}}
```

for a written token y. `p_old`: the probability the network gave y at the rollout, when it wrote y (recorded once,
never updated). `p_now`: the probability the network gives y now, after some training steps on that same rollout.

Running example (notebook [`cispo_simple.ipynb`](cispo_simple.ipynb)): at the rollout the network gives
p_old = (0.491, 0.464, 0.045) over tokens (So, The, Wait), writes "Wait" (p_old(Wait) = 0.045), and the response
gets advantage A = +1.

## What to remember

**1. One gradient, three factors.** All three losses give the logits the gradient `factor * A * (p - y_onehot)`.
Only the factor differs. The factor is >= 0, so it never flips direction: as in REINFORCE, the written token's
entry in `p - y_onehot` is p - 1 < 0, A > 0 keeps it negative (token up), A < 0 flips it (token down). The factor
only scales the push, or zeroes it (PPO clipped).

**Table 1. The factor in front of `A * (p - y_onehot)`.** Clip bounds: PPO `[0.8, 1.28]` (our runs), CISPO `[0, 4]`
(miles/Tinker defaults).

| Loss | Factor | When r = 1.516 (step 1 below) |
|---|---|---|
| REINFORCE | 1 | 1 |
| PPO | r, but 0 if A > 0 and r > 1.28 (or A < 0 and r < 0.8) | **0**: the token gets no gradient |
| CISPO | clip(r, 0, 4) | 1.516 |

**2. The factor r is the derivative of PPO's unclipped loss.** PPO's unclipped loss is `-A * r`. Since p_old is a
fixed number, the gradient of r is r times the gradient of log p_now:

```math
\boxed{\nabla(-A \cdot r) = -A \cdot r \cdot \nabla \log p_{\text{now}}(y)}
```

That is REINFORCE's gradient times r. CISPO writes the factor out by hand and caps it.

**3. In our runs, r = 1.** We take one optimizer step per rollout, so p_now = p_old at that step and all three
losses give the same gradient.

## 1. The three losses

```math
\boxed{\text{REINFORCE loss} = -A \cdot \log p_{\text{now}}(y)}
```

```math
\boxed{\text{PPO loss} = -\min\big(r \cdot A,\ \mathrm{clip}(r, 0.8, 1.28) \cdot A\big)}
```

```math
\boxed{\text{CISPO loss} = -\,\mathrm{sg}\big(\mathrm{clip}(r, 0, 4)\big) \cdot A \cdot \log p_{\text{now}}(y)}
```

`sg` (stop-gradient, `.detach()` in PyTorch) makes the clipped ratio a fixed number: autograd differentiates only
`log p_now`. The gradient of `-log p_now(y)` on the logits is `p - y_onehot` (the softmax rule), so:

```math
\boxed{\text{REINFORCE gradient on the logits} = A \cdot (p - y_{\text{onehot}})}
```

```math
\boxed{\text{PPO gradient on the logits} = r \cdot A \cdot (p - y_{\text{onehot}}), \text{ or } 0 \text{ when clipped}}
```

```math
\boxed{\text{CISPO gradient on the logits} = \mathrm{clip}(r, 0, 4) \cdot A \cdot (p - y_{\text{onehot}})}
```

**Table 2. The three losses side by side at step 1.** The network has taken one step on the rollout:
p = (0.501, 0.431, 0.068), so r = 0.068 / 0.045 = 1.516. A = +1 for all three.

| | REINFORCE | PPO | CISPO |
|---|---|---|---|
| Loss per token | **−A · log p_now** | **−min(r·A, clip(r, 0.8, 1.28)·A)** | **−sg(clip(r, 0, 4)) · A · log p_now** |
| Factor | 1 | r, or 0 when clipped | clip(r, 0, 4) |
| Intuition | push the written token by A | same, but stop once the token moved past 1.28× | same, but never more than 4× |
| *Example (step 1)* | | | |
| Token written | Wait | Wait | Wait |
| r for that token | 0.068 / 0.045 = 1.516 | same | same |
| p − y_onehot | (0.501, 0.431, 0.068 − 1) = (+0.501, +0.431, −0.932) | same | same |
| Factor | 1 | 1.516 > 1.28 with A > 0: **0** | **1.516** |
| Gradient | (+0.501, +0.431, **−0.932**) | (0, 0, **0**) | (+0.759, +0.654, **−1.413**) |
| Effect (training subtracts the gradient) | Wait goes **up** | Wait **stops** | Wait goes **up**, 1.516× as hard |

## 2. Where the factor r comes from

p_old is a constant, so only p_now moves. Using `∇p = p · ∇log p`:

```math
\boxed{\nabla r = \frac{\nabla p_{\text{now}}}{p_{\text{old}}} = \frac{p_{\text{now}}}{p_{\text{old}}} \cdot \nabla \log p_{\text{now}} = r \cdot \nabla \log p_{\text{now}}}
```

"Wait" was written when the network gave it 0.045. Now it gives 0.068, so it would
write "Wait" 1.516× as often. That one sample should count 1.516× as much (importance sampling). Uncapped, one
token whose probability grew 20× would count 20×; PPO's clip and CISPO's cap both limit that.

When PPO clips, the clipped term `clip(r) · A = 1.28 · A` wins the min. It is a constant, so its gradient is 0.

## 3. Several steps on one rollout

MiniMax-M1 (arXiv 2506.13585) took 16 optimizer steps per rollout. p_old stays 0.045; the weights move; r grows.

**Table 3. p("Wait") and the factor over 8 steps on the same rollout** (lr = 0.1, notebook
`cispo_simple.ipynb`). p_now is measured at the start of each step. Steps 4-6 omitted; the factor stays 4.

| Step | PPO p_now | PPO r | PPO factor | CISPO p_now | CISPO r | CISPO factor |
|---|---|---|---|---|---|---|
| 0 | 0.045 | 1.000 | 1.000 | 0.045 | 1.000 | 1.000 |
| 1 | 0.068 | 1.516 | **0** | 0.068 | 1.516 | 1.516 |
| 2 | 0.068 | 1.516 | 0 | 0.119 | 2.650 | 2.650 |
| 3 | 0.068 | 1.516 | 0 | 0.257 | 5.713 | **4** |
| 7 | 0.068 | 1.516 | 0 | 0.796 | 17.711 | 4 |

PPO's factor is 0 from step 1, so its weights stop moving and every later step is identical. This is the "clipped
out after the first update" the paper describes for rare tokens like "Wait". CISPO's factor follows r up to the
cap and the token keeps learning.

From the paper (arXiv 2506.13585, Eq. 4-5): the CISPO loss is averaged over every token of the group's responses
(`1 / Σ|o_i|`), A is the GRPO advantage, there is no lower bound on the ratio, and only the upper bound is tuned.

## 4. Backprop through a tiny network, step by step

The notebook trains a two-layer network:

```
x -> Z1 = x @ W1 -> h = tanh(Z1) -> logits = h @ W2 -> p = softmax(logits)
```

One prompt `x = [1.0, 0.5]`, weights from seed 4. Below is CISPO's step 3, where the cap is active.

**Forward.**

```python
h = np.tanh(x @ W1)                 # [-0.739, -0.577, 0.908]
p = softmax(h @ W2)                 # [0.431, 0.312, 0.257]
r = p[y] / p_old                    # 0.257 / 0.045 = 5.713
w = np.clip(r, 0.0, 4.0)            # clip(r, 0, 4) = 4  (frozen)
loss = -w * A * np.log(p[y])        # -4 * 1 * log 0.257 = 5.437
```

**Backward.** Training subtracts each gradient, so a negative gradient means "increase" and a positive one means "decrease". This holds per number: a weight with a positive gradient decreases. A logit's gradient reaches each weight multiplied by that weight's input activation, so for the same logit, weights fed by positive activations and weights fed by negative activations move in opposite directions, and both changes move the logit the same way.

**Table 4. Backward pass at step 3 (CISPO, factor 4).**

| Line | Value | What it says |
|---|---|---|
| `y_onehot` | [0, 0, 1] | "Wait" was written |
| `g_logits = w * A * (p - y_onehot)` | 4 × [0.431, 0.312, −0.743] = [1.724, 1.249, **−2.973**] | Wait's logit up, the others down; 4× REINFORCE's push |
| `dW2 = outer(h, g_logits)` | weight from hidden unit j into Wait: h_j × (−2.973); unit 3: 0.908 × (−2.973) = −2.699 | weights into Wait from units with h > 0 go up, from units with h < 0 go down |
| `g_h = W2 @ g_logits` | [4.911, −2.477, 3.676] | which way each hidden unit should move to favor Wait |
| `g_Z1 = g_h * (1 - h**2)` | [2.229, −1.653, 0.646] | same, damped by tanh's slope (unit 3 is saturated: 1 − 0.908² = 0.176) |
| `dW1 = outer(x, g_Z1)` | row 1 (x = 1.0): [2.229, −1.653, 0.646]; row 2 (x = 0.5): [1.114, −0.826, 0.323] | first-layer weights move so the hidden units favor Wait |

**PPO at the same point.** Its factor is 0, so `g_logits = [0, 0, 0]` and every line below it is 0.

**Update.** `W -= lr * dW` with lr = 0.1. p("Wait") goes from 0.257 to **0.502**.

**The sign rule.** The factor only rescales `g_logits`: 4 × REINFORCE here, 0 for PPO. It never flips the sign,
so the written token moves the same way as under REINFORCE, or not at all.

## 5. In our runs and in miles

- One optimizer step per rollout means only step 0 happens: r = 1, factor 1 for all three losses.
- miles has CISPO only on the Tinker path (`tinker_losses.py:192-213`, `loss_fn="cispo"`), and its p_old is the
  SGLang engine's probability (`rollout_log_probs`), not a trainer pass. There r is the trainer-vs-engine gap,
  near 1, and CISPO acts as a cap of 4 on it. That is the same shape as TIS
  ([`../02-train-inference-mismatch/`](../02-train-inference-mismatch/)), which our GLM-5.3 torchtitan recipe runs with a cap of 2.

## Files

- [`cispo_simple.ipynb`](cispo_simple.ipynb): the network above, hand-written forward and backward with every
  input defined, PPO vs CISPO over 8 steps, checked against PyTorch.
- [`cispo_backward.ipynb`](cispo_backward.ipynb): the same comparison on a group of 4 responses with GRPO
  advantages, 16 steps.
- [`README.md`](README.md): why PPO's clip almost never fires in our runs (measured clip fractions).
