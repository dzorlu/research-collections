# Distillation: SFT vs on-policy distillation (OPD)

SFT copies the teacher's essay. OPD has the student write its own essay and the
teacher mark each word. Both use the same per-token number:

```math
\boxed{A = \log p_t(y) - \log p_s(y)}
```

for a written token y (p_t: teacher, p_s: student).

Running example: teacher p_t = (0.6, 0.3, 0.1), student p_s = (0.2, 0.5, 0.3)
over tokens (A, B, C).

## What to remember

**1. The sign rule.** In `p − y_onehot`, the written token's entry is p − 1, always
negative. A sets the direction.

**Table 1. The sign rule.** Training subtracts the gradient.

| A | Written token's gradient | The token |
|---|---|---|
| A > 0 (the teacher wants more) | stays negative | goes **up** |
| A < 0 (the teacher wants less) | flips to positive | goes **down** |

**2. Both KLs are averages of A.** Only who writes the tokens changes. OPD's tokens
come from the student, so it minimizes the reverse KL.

```math
\boxed{\text{forward KL (SFT)} = \sum_y p_t(y)\, A(y) \qquad \text{reverse KL (OPD)} = \sum_y p_s(y)\,\big({-A(y)}\big)}
```

**3. The minus signs.** A is a reward: raise it. The per-token reverse-KL term is
−A, a distance: lower it. The loss carries the minus, so minimizing the loss raises A.

```math
\boxed{\text{loss} = -A \cdot \log p_s(y) \quad\Longrightarrow\quad \text{minimizing it pushes } \log p_s(y) \text{ in the direction of } A}
```

## 1. The two losses

```math
\boxed{\text{SFT loss} = -\log p_s(y), \quad y \text{ written by the teacher}}
```

```math
\boxed{\text{OPD loss} = -A \cdot \log p_s(y), \quad y \text{ written by the student, } A \text{ frozen}}
```

```math
\boxed{\text{SFT gradient on the logits} = (p - y_{\text{onehot}}) / b}
```

```math
\boxed{\text{OPD gradient on the logits} = A \cdot (p - y_{\text{onehot}}) / b}
```

**Table 2. SFT and OPD side by side.** Example rows use the same written token, B,
in both columns, and batch size b = 1.

| | SFT | OPD |
|---|---|---|
| Who writes token y | teacher | student |
| Loss per token | **−log p_s(y)** | **−A · log p_s(y)**, A frozen |
| Intuition | make the teacher's token more likely | make the student's own token more or less likely, by how much the teacher agrees with it |
| Gradient on the logits | **(p − y_onehot) / b** | **A · (p − y_onehot) / b** |
| *Example* | | |
| Token written | B (by the teacher) | B (by the student) |
| A for that token | not used | log 0.3 − log 0.5 = −1.204 + 0.693 = **−0.511** |
| p − y_onehot | (0.2, 0.5 − 1, 0.3) = (+0.2, −0.5, +0.3) | same: (+0.2, −0.5, +0.3) |
| Gradient | (+0.2, **−0.5**, +0.3) | × (−0.511) = (−0.102, **+0.255**, −0.153) |
| Effect (training subtracts the gradient) | B goes **up** | B goes **down** |
| *KL* | | |
| KL it minimizes | KL(teacher ‖ student), forward | KL(student ‖ teacher), reverse |
| KL written with A | Σ p_t · A | Σ p_s · (−A) |
| Example KL | 0.396 | 0.365 |

`p − y_onehot` is the direction that makes token y more likely. SFT always applies
it at full strength. OPD multiplies it by A, the teacher's verdict on the token the
student wrote.

## 2. A is the teacher's verdict

A = log(p_t / p_s): how many times more the teacher wants the token than the student
gives it, on a log scale.

**Table 3. The verdict for each token the student might write.**

| Student writes | p_s | p_t | A | Verdict | OPD push |
|---|---|---|---|---|---|
| A | 0.2 | 0.6 | +1.099 | use this 3× more | strong pull up |
| B | 0.5 | 0.3 | −0.511 | use this less | push down |
| C | 0.3 | 0.1 | −1.099 | use this 3× less | strong push down |

Each verdict reaches 0 when p_s = p_t for that token, so training settles where the
student matches the teacher (0.6, 0.3, 0.1).

Both KLs average the same column A; only the weights and the sign differ:

```math
\boxed{\mathrm{KL}(\text{teacher} \,\|\, \text{student}) = \sum_y p_t(y)\,\big(\log p_t(y) - \log p_s(y)\big) = \sum_y p_t(y)\, A(y)}
```

```math
\boxed{\mathrm{KL}(\text{student} \,\|\, \text{teacher}) = \sum_y p_s(y)\,\big(\log p_s(y) - \log p_t(y)\big) = \sum_y p_s(y)\,\big({-A(y)}\big)}
```

- Forward KL (weights p_t) = 0.6 × 1.099 + 0.3 × (−0.511) + 0.1 × (−1.099) = **0.396**
- Reverse KL (weights p_s, average of −A) = 0.2 × (−1.099) + 0.5 × 0.511 + 0.3 × 1.099 = **0.365**

In KL(P ‖ Q) the first slot is whose tokens are averaged over. OPD averages over the
student's own tokens, so it is the reverse KL. Training uses A per token; the KL is
only a number to monitor.

## 3. SFT loss = teacher entropy + forward KL

Each token's SFT loss splits into what the teacher itself would score plus A:

```math
\boxed{-\log p_s = -\log p_t + (\log p_t - \log p_s) = -\log p_t + A}
```

Averaged over the teacher's tokens (weights p_t):

```math
\boxed{\underbrace{\sum_y p_t(-\log p_s)}_{\text{SFT loss}} = \underbrace{\sum_y p_t(-\log p_t)}_{\text{teacher entropy}} + \underbrace{\sum_y p_t\, A}_{\text{forward KL}}}
```

**Table 4. SFT loss per token, split.** Averages are weighted by how often the teacher
writes each token (p_t).

| Token | p_t | SFT loss −log p_s | = teacher part −log p_t | + A |
|---|---|---|---|---|
| A | 0.6 | 1.609 | 0.511 | +1.099 |
| B | 0.3 | 0.693 | 1.204 | −0.511 |
| C | 0.1 | 1.204 | 2.303 | −1.099 |
| **Weighted average** | | **1.294** | **0.898** (teacher entropy) | **0.396** (forward KL) |

The teacher part is the teacher's own randomness: even a perfect copy of the teacher
cannot predict which token comes next (0.898, fixed). Only the A part can shrink, so
every SFT step is a forward-KL step.

## 4. Backprop through a tiny network, step by step

The notebook [`opd_backward.ipynb`](opd_backward.ipynb)
([open in Colab](https://colab.research.google.com/github/dzorlu/research-collections/blob/main/distillation/opd_backward.ipynb))
trains a two-layer network with the OPD loss:

```
X -> Z1 = X @ W1 -> H = tanh(Z1) -> logits = H @ W2 -> p = softmax(logits)
```

Two contexts, each with its own teacher. Below is context 2: input `X[1] = [0, 1]`,
teacher `P_T[1] = [0.1, 0.2, 0.7]`, weights from seed 0.

**Forward.**

```python
H = np.tanh(X @ W1)                 # [-0.490, 0.347, 0.863, 0.738]
p = softmax(H @ W2)                 # [0.144, 0.223, 0.633]
y = p.argmax()                      # greedy: the student writes C (index 2)
A = np.log(0.7) - np.log(0.633)     # teacher's C minus student's C = +0.101  (frozen)
loss = -A * np.log(p[y])            # OPD loss
```

A > 0: the teacher gives C more than the student does, so C should go up.

**Backward.** Training subtracts each gradient, so a negative gradient means "increase" and a positive one means "decrease". This holds per number: a weight with a positive gradient decreases. A logit's gradient reaches each weight multiplied by that weight's input activation, so for the same logit, weights fed by positive activations and weights fed by negative activations move in opposite directions, and both changes move the logit the same way.

**Table 5. Backward pass for context 2.**

| Line | Value | What it says |
|---|---|---|
| `y_onehot` | [0, 0, 1] | C was written |
| `g_logits = A * (p - y_onehot) / b` (b = 2 contexts) | 0.101 × [0.144, 0.223, −0.367] / 2 = [0.007, 0.011, **−0.018**] | C's logit up, A and B down |
| `dW2 = H.T @ g_logits` | weight from hidden unit j into C: H_j × (−0.018); unit 2: 0.863 × (−0.018) = −0.016 | weights into C from active units (H > 0) go up; from units with H < 0, down |
| `g_H = g_logits @ W2.T` | [−0.008, −0.022, −0.007, −0.017] | which way each hidden unit should move to favor C |
| `g_Z1 = g_H * (1 - H**2)` | [−0.006, −0.019, −0.002, −0.008] | same, damped by tanh's slope (unit 2 is saturated: 1 − 0.863² = 0.256) |
| `dW1 = X.T @ g_Z1` | X[1] = [0, 1], so only W1's second row gets g_Z1 | first-layer weights move so the hidden units favor C |

**Update.** `W -= lr * dW` with lr = 1 (the notebook adds context 1's gradients to the
same dW). p(C) goes from 0.633 to **0.659**, toward the teacher's 0.7. With sampled
training, both contexts reach their teacher exactly by step 100.

**The sign rule.** In `p − y_onehot` the written token's entry is `p − 1`, always
negative. A > 0 keeps it negative, so the token goes up (C: −0.018). A < 0 flips it
positive, so the token goes down (context 1 writes B with A = −0.279: +0.084).

## Files

- [`opd_backward.ipynb`](opd_backward.ipynb): the network above, hand-written forward
  and backward, checked against PyTorch, plus training.
- [`mopd-reverse-kl.md`](mopd-reverse-kl.md): longer notes and the papers (MOPD,
  MiMo-V2-Flash, Nemotron-Cascade 2, Nemotron 3.5).
- [`reverse_kl_example.py`](reverse_kl_example.py): reproduces the three-token numbers.
