# On-policy distillation and reverse KL, on three tokens

Notes from reading the multi-teacher on-policy distillation (MOPD) papers
(2026-10-16). Recent post-training pipelines train several RL specialist
teachers and then merge them into one student by on-policy distillation. This
note works through the loss on a three-token example. Every number comes from
[`reverse_kl_example.py`](reverse_kl_example.py).

## One rule, three choices of A

Every method here uses the same loss on a sampled token:

```
loss = -A * log p_student(token)
```

If A > 0, training makes the token more likely. If A < 0, less likely. The
methods differ only in what A is.

**Table 1. What A is in each method.**

| Method | A | One A per |
|---|---|---|
| REINFORCE | reward of the response (often minus a baseline) | response, copied to every token |
| GRPO | (reward − group mean) / group std | response, copied to every token |
| On-policy distillation | log p_teacher(token) − log p_student(token) | token |

REINFORCE has no teacher. In distillation the teacher is only the source of A,
the same role a verifier plays in GRPO.

## The example: SFT and on-policy distillation on the same numbers

One position, three possible tokens. Teacher p_T = (0.6, 0.3, 0.1), student
p_S = (0.2, 0.5, 0.3).

The student's loss on a token is its surprise, −log p_S. That surprise splits
into two parts:

- teacher surprise, −log p_T. The teacher itself only gives the token some
  probability, so even a perfect copy of the teacher is surprised this much.
- extra surprise, log p_T − log p_S. This is the part caused by the student
  differing from the teacher, and it is exactly the distillation advantage A.

**Table 2. Per-token surprise split, with SFT and OPD averages.** The two
average rows use different weights. SFT averages over tokens the teacher writes
(weights p_T). OPD averages over tokens the student writes (weights p_S). Every
row satisfies student surprise = teacher surprise + extra surprise.

| Token | p_T | p_S | Student surprise −log p_S | Teacher surprise −log p_T | Extra surprise A = log p_T − log p_S | OPD loss −A·log p_S |
|---|---|---|---|---|---|---|
| A | 0.6 | 0.2 | 1.609 | 0.511 | +1.099 | +1.768 |
| B | 0.3 | 0.5 | 0.693 | 1.204 | −0.511 | −0.354 |
| C | 0.1 | 0.3 | 1.204 | 2.303 | −1.099 | −1.323 |
| **Average over teacher samples (SFT)** | | | **1.294** | **0.898** | **+0.396** | n/a |
| **Average over student samples (OPD)** | | | 1.030 | 1.395 | **−0.365** | **−0.220** |

Reading the SFT row:

- 1.294 is the SFT loss, the cross-entropy on teacher tokens.
- 0.898 is the teacher's entropy. It depends only on the teacher, so training
  cannot change it, and the SFT loss can never go below it. If p_S = p_T, the
  loss is 0.898.
- 0.396 is the forward KL, KL(teacher‖student). It is the only part training
  can reduce, so every SFT gradient step is a forward-KL step.

Reading the OPD row:

- The average of A is −0.365, so the average of −A is 0.365. That is the
  reverse KL, KL(student‖teacher).
- The average OPD loss is −0.220. This value means nothing on its own. Its
  gradient is what matters (next section).
- Training pushes token A up (A > 0) and tokens B and C down (A < 0). SFT can
  only push tokens up.

## Why the loss is −A·log p_S and not −A

- −A is the measurement. For one sampled token, −A = log p_S − log p_T is a
  one-sample estimate of the reverse KL. Averaged over the student's samples it
  gives the 0.365 above. Log it to track progress.
- −A cannot be the loss. A is computed from the sampled token and held fixed
  (stop-gradient), so it has no gradient. Even without the stop-gradient, the
  expected gradient of −A under the student is Σ p_S·∇log p_S = ∇Σ p_S = ∇1 = 0,
  because −A ignores that the student also chooses which token gets sampled.
- log p_S carries the gradient. −A·log p_S moves the log-probability of the
  sampled token by an amount set by A. Its expected gradient equals the exact
  gradient of the reverse KL: on this example both are (−0.293, +0.073, +0.220)
  with respect to the student's logits for (A, B, C). One step with learning
  rate 1 moves the student to (0.275, 0.477, 0.247) and the reverse KL drops
  from 0.365 to 0.231.

So −A is the KL value to log, and −A·log p_S is the loss to differentiate.

## Forward vs reverse

KL(P‖Q) averages log(P/Q) over samples from P. The names only say whose samples
are averaged.

**Table 3. KL direction by method.**

| Method | KL | Samples from | Direction |
|---|---|---|---|
| SFT on teacher outputs | KL(teacher‖student) | teacher | forward |
| On-policy distillation | KL(student‖teacher) | student | reverse |
| GRPO/PPO KL penalty | KL(policy‖reference) | policy | reverse |

The standard RL KL penalty is already reverse KL. On-policy distillation uses
the same quantity toward a teacher, as the main signal instead of a penalty.

## Why SFT minimizes the forward KL

**What the order means.** In KL(P‖Q), the first slot is whose tokens you look at,
the second is what you compare them against. KL(teacher‖student) looks at the
teacher's tokens: that is SFT, where the teacher writes the text.

**SFT loss.** Per token: −log p_student(token), where the teacher wrote the token.

**Averaged over the teacher's tokens, that is the cross-entropy.** The teacher
writes A 60% of the time, B 30%, C 10%, so the average weights each token by
how often the teacher writes it:

0.6 × (−log 0.2) + 0.3 × (−log 0.5) + 0.1 × (−log 0.3)
= 0.6 × 1.609 + 0.3 × 0.693 + 0.1 × 1.204 = **1.294**

**−log p_student is not the forward KL.** The forward-KL term per token is
log(p_T/p_S) = (−log p_S) − (−log p_T): the SFT loss minus what the teacher
itself would score. So every token's SFT loss splits into two parts:

**Table 4. SFT loss per token = teacher part + KL part.** Weighted averages use
how often the teacher writes each token.

| Token | Teacher writes it | SFT loss −log p_S | = teacher part −log p_T | + KL part log(p_T/p_S) |
|---|---|---|---|---|
| A | 0.6 | 1.609 | 0.511 | +1.099 |
| B | 0.3 | 0.693 | 1.204 | −0.511 |
| C | 0.1 | 1.204 | 2.303 | −1.099 |
| **Weighted average** | | **1.294** | **0.898** (teacher entropy) | **0.396** (forward KL) |

**Intuition.** SFT loss is how surprised the student is by the teacher's tokens.
Part of that surprise is the teacher's own randomness: it splits 0.6/0.3/0.1,
so even a perfect copy of the teacher cannot predict the next token (0.898,
fixed). The rest comes from the student guessing differently than the teacher
(0.396). Training can only remove the second part, so every SFT step is a
forward-KL step.

**Link to A.** The reverse-KL term per token is log(p_S/p_T); the advantage is
A = log(p_T/p_S), the same fraction upside down, so the same number with the
sign flipped. For token B in a context where the student gives 0.397 and the
teacher 0.3: KL term +0.28, A −0.28. Training uses A per token; the reverse KL
(average of −A over the student's tokens) is only a number to monitor.

## Why on-policy distillation uses reverse KL

1. The student trains on text it writes itself. SFT trains on teacher prefixes
   the student may never reach in a long agent run.
2. The teacher only scores the sampled tokens (one prefill pass, no
   generation), so it fits into the RL loop like a reward model.
3. Tokens the teacher dislikes get pushed down (B and C in Table 2). SFT cannot
   do that.
4. Where student and teacher agree, A = 0, so no update is spent there.

Reverse KL is also mode-seeking. With a two-mode teacher p_T = (0.49, 0.02,
0.49), a student on one mode (0.96, 0.02, 0.02) has reverse KL 0.582 and
forward KL 1.238. A student spread over all tokens (0.34, 0.32, 0.34) has
reverse KL 0.639 and forward KL 0.303. Reverse KL prefers the one-mode student;
forward KL prefers the spread one.

## Implementation details from the papers

- Clipping. A token the teacher thinks is nearly impossible gets a very negative
  A. With p_T = 1e-6 and p_S = 0.5, A = −13.1. The MOPD paper clips A to
  ±A_max.
- Smoothed teacher. Nemotron 3.5 Super (MOPD stage 2) replaces the teacher with
  0.2·p_T + 0.8·p_S, which bounds A below by log 0.8 ≈ −0.22. The same
  −13.1 token gets −0.223.
- Top-k form (MOPD paper, Eq. 5). Distill over the teacher's top-k tokens with
  Σ [p_S·log(p_S/p_T) − p_S + p_T]. The −p_S + p_T term is required. Without it,
  a student with teacher top-2 = {A, B} and p_S = (0.3, 0.15, 0.55) scores −0.31,
  better than matching the teacher (0). With it the score is +0.14 and the
  minimum is at p_S = p_T. The paper reports top-k (k = 64) and the
  sampled-token form perform about the same.
- Token-local update. For a response, each token gets only its own A, not the
  sum of later tokens' terms. MOPD (Eq. 2) and Nemotron 3.5 both treat the
  sampled prefix as fixed. This is an approximation of the sequence-level
  gradient with lower variance.
- Domain shares are set by hand. No report ablates them. Nemotron 3.5 weights
  its first MOPD stage toward domains that improve fast and regress later
  (agentic 35%, reasoning 25%), and its second toward slow domains (chat,
  vision), with 5-10% of prompts routed to a frozen copy of the student.

## Sources

| Paper | Link | Used for |
|---|---|---|
| MOPD: Multi-Teacher On-Policy Distillation for Capability Integration (Ma et al., Xiaomi + PKU) | [arXiv 2606.30406](https://arxiv.org/abs/2606.30406) | loss formulas (§3.2, Eqs. 1-5), clipping, top-k form |
| MiMo-V2-Flash (Xiaomi) | [arXiv 2601.02780](https://arxiv.org/abs/2601.02780) | first report to name MOPD; adds an outcome-reward term to A |
| Nemotron-Cascade 2 (NVIDIA) | [arXiv 2603.19220](https://arxiv.org/abs/2603.19220) | teachers = best validation checkpoint per domain; MOPD vs GRPO step counts |
| Nemotron 3.5 Super and Lightning (NVIDIA) | [report PDF](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf) | two-stage MOPD, domain shares, smoothed teacher |
| GLM-4.5; DeepSeek-V3.2 | [arXiv 2508.06471](https://arxiv.org/abs/2508.06471); [arXiv 2512.02556](https://arxiv.org/abs/2512.02556) | earlier recipe: RL specialists merged by SFT on their outputs |
