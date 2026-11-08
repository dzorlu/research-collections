# TTT-Discover: test-time RL for discovery

"Learning to Discover at Test Time" (Yuksekgonul, Sun et al., Stanford, Together AI, NVIDIA; arXiv 2601.16175, Jan 2026; ICML 2026). Code: github.com/test-time-training/discover.

## TLDR

Run RL on one hard problem while solving it, so the model learns that problem's tricks instead of searching with frozen weights.

- Archive of scored solutions, PUCT picks a parent, gpt-oss-120b writes 64 new versions, they are scored, a rank-32 LoRA takes one RL step. 50 steps × 512 rollouts, about $500 per problem on Tinker. The adapter is discarded afterwards.
- Claimed records: Erdős minimum overlap and the first autocorrelation inequality (fifth-decimal margins), GPU kernels up to about 2x faster than the best human, two AtCoder heuristic contests, single-cell denoising. Lost or tied: second autocorrelation inequality, circle packing, MLA-Decode.
- Mental model: evolutionary search whose mutation operator is fine-tuned on the fitness score, for one problem. Not continual learning: nothing persists past the problem.

## The loop

```python
archive = [initial_solution]
for step in range(50):
    batch = []
    for g in range(8):                                   # 8 groups per step
        parent   = puct_select(archive)                  # (1) select
        rollouts = [model.sample(prompt(parent)) for _ in range(64)]   # (2) roll out
        R = [score(run(r)) for r in rollouts]            # (3) score, add to archive, update Q and visit counts
        archive.add(rollouts, R)
        beta = bisect(lambda b: KL(softmax(b*R) || uniform) - ln2)     # (4) sharpen, one beta per group
        adv  = 64*softmax(beta*R) - 1 - lam*log(pi/pi_base)
        batch += zip(rollouts, adv)
    model.lora_step(batch)                               # (5) backprop, one step on all 512
return archive.best()
```

Steps 1 and 3 are the evolutionary search. Steps 4 and 5 are the learning.

- **State** s: a candidate solution (code, a construction, a kernel).
- **Action** a: thinking tokens plus new code, written freely by the model. Nothing is pre-proposed.
- **Reward** R: the verifier's continuous score, 0 if invalid.
- **Prompt**: only the parent's code, its score before and after, the gap to the target, and the last 500 characters of its output. No other past attempts.

## Why naive test-time RL fails (section 3.1)

| Problem | What goes wrong | Fix |
|---|---|---|
| Objective | expected reward is indifferent to the record: a 1900 µs kernel against a 2000 µs record scores about like an ordinary one | entropic objective, chases the max |
| Short effective horizon | one attempt of about 26k tokens from empty cannot hold many rounds of improvement | state reuse: start from archived solutions, chaining attempts |
| Exploration | the policy collapses onto safe edits; naive reuse keeps mutating the same few states | KL penalty to base model; PUCT visit bonus and lineage blocking |

## The reuse ladder (section 2.2), all with a frozen model

```python
a = llm(prompt(problem, EMPTY))                    # Best-of-N: nothing reused
a = llm(prompt(problem, pick(buffer)))             # state reuse: warm-start from a past solution
a = llm(prompt(problem, pick(buffer), examples))   # state-action reuse (AlphaEvolve): past attempts and scores in context
```

TTT-Discover sits on the state-reuse rung and carries the lessons of past attempts in the LoRA weights instead of the prompt. AlphaEvolve shows the model its history; TTT-Discover trains it in.

## Entropic objective

Not an entropy bonus. Entropy is never added to the reward. The objective is a log-mean-exp of reward (the entropic risk measure), which is risk-seeking:

```python
import math
R = [0.50, 0.52, 0.55, 0.90]
def entropic(beta): return math.log(sum(math.exp(beta*r) for r in R)/len(R)) / beta
# mean 0.6175; entropic(1) ≈ 0.63, entropic(5) ≈ 0.72, entropic(50) ≈ 0.87, max 0.90
```

Small β gives the mean, large β tends to the max. The gradient weights each rollout by w = exp(βR) / mean(exp(βR)), and the advantage is w − 1.

Concrete kernel example. Record 2000 µs, reward 1000 / runtime:

```python
us = [2400, 2300, 2200, 1900]          # only 1900 beats the record
R  = [0.417, 0.435, 0.455, 0.526]
```

| | advantages |
|---|---|
| expected reward, R − mean | −0.041, −0.023, −0.004, +0.068 (nearly flat) |
| GRPO, divided by std | −1.0, −0.56, −0.08, +1.64 (linear in reward) |
| entropic, β = 28.6 | −0.86, −0.77, −0.59, +2.21 (winner dominates, losers still ordered) |

What gets sharpened is the policy π_θ(a | s), the model's distribution over programs given one parent. The target is q ∝ π₀ · exp(βR); one gradient step moves partway toward it and the KL penalty to the base model (λ = 0.1) limits the move. For discovery, collapsing onto the winning edit is the goal; in continual learning the same sharpening is what causes forgetting.

## Choosing β: one per group, KL = ln 2

The 64 rollouts of a group share one parent. Each group gets its own β, solved by bisection on the batch just sampled, before any weights change. β is not learned.

The KL compares two weightings of the same 64 rollouts: uniform (each 1/64) and tilted q = softmax(βR). β is set so that KL(q ‖ uniform) = ln 2.

Why per group: reward spread depends on the parent.

| Group | Children (µs) | β for KL = ln 2 | Weights |
|---|---|---|---|
| far from record | 2600, 2400, 2300, 1900 | 22.8 | 0.13, 0.26, 0.40, 3.21 |
| near record | 1260, 1255, 1250, 1240 | 279 | 0.09, 0.22, 0.52, 3.17 |

Same shape, twelve times the β. Swap them and it breaks: the near-record group at β = 22.8 gets 0.88, 0.94, 1.01, 1.17 (nothing learned where it matters most), and the far group at β = 279 gets 0, 0, 0, 4.0 (one-sample gradient). It is GRPO's per-group std normalization, done for exponential weights.

Why ln 2 means "about half the batch": put equal weight on k of n rollouts.

```python
uniform = [1/4, 1/4, 1/4, 1/4]
q       = [1/2, 1/2, 0,   0  ]         # best 2 of 4
KL = 1/2*ln(2) + 1/2*ln(2) = ln 2      # each survivor's weight doubles: ln(2) each
# in general KL = ln(n/k); ln(n/k) = ln 2  ->  k = n/2
```

| KL budget | effective rollouts of 64 | meaning |
|---|---|---|
| 0 | 64 | plain expected reward |
| ln 2 | 32 | about the top half drives the update |
| ln 8 | 8 | top eighth |
| ln 64 | 1 | pure max, very noisy |

Two KL-like terms, not to confuse:

| | compares | purpose |
|---|---|---|
| ln 2 budget | tilted vs uniform weights over this batch | sets β |
| λ penalty | current LoRA model vs base model | stops drift and collapse |

## PUCT: which parent to improve

```
score(s) = Q(s) + c · scale · P(s) · sqrt(1 + T) / (1 + n(s))
```

Q is the best child reward (max, not AlphaZero's mean), P a fixed rank prior, n how often s was picked, T total picks, scale = R_max − R_min, c = 1.

| Solution | Q | P | n | bonus | score |
|---|---|---|---|---|---|
| A, 1900 µs | 0.530 | 0.50 | 6 | 0.022 | 0.552 |
| **B, 2000 µs** | 0.505 | 0.33 | 1 | 0.051 | **0.556** |
| C, 2400 µs | 0.417 | 0.17 | 0 | 0.051 | 0.468 |

B wins over the faster A because A has been tried six times. Visit counts propagate to ancestors, and a picked parent's lineage is blocked for the rest of the step so the 8 groups start from different branches. Archive capped at 1000, top 2 children kept per parent.

## Where the gain comes from (Table 8, TriMul H100, best µs, lower is better)

| Objective | Reuse | µs |
|---|---|---|
| adaptive entropic | PUCT | **1203** |
| adaptive entropic | ε-greedy | 1329 |
| constant β = 2 | PUCT | 1484 |
| expected reward | PUCT | 1986 |
| none (frozen model) | PUCT | 2061 |
| adaptive entropic | none | 5274 |
| expected reward (naive RL) | none | 5329 |
| Best-of-N | none | 5352 |

Reuse does most of the work; without it RL is no better than sampling. Weight updates then take 2061 to 1203. Adaptive β matters more than PUCT vs ε-greedy. One task, one seed.

## What it is not

- **Not evolution of weights.** Solutions are selected by fitness; weights get a policy-gradient step.
- **Not a Gödel machine.** The learning procedure (RL rule, PUCT, objective) is fixed by hand. Only the LoRA and the archive change, for one problem. CLE's definition: fixed-procedure TTT is not recursive self-improvement.
- **Not the first RL-in-the-loop.** EvoTune, MiGrATe and ThetaEvolve (2511.23473) did it before; TTT-Discover beats ThetaEvolve by small margins on Qwen3-8B. New here: the entropic objective with adaptive β and PUCT reuse.

## The main follow-up lineage: separate the strategist from the executor

The strongest follow-ups keep TTT-Discover's loop (PUCT reuse, entropic objective, LoRA) but stop training the model that writes the solution. A small model is trained to propose what to try; a frozen, stronger model implements it.

```python
parent = puct_select(archive)                      # unchanged
idea   = guide.sample(prompt(problem, parent))     # small model, trained: "fuse the two matmuls, keep the
                                                   #   intermediate in shared memory"
code   = executor.generate(parent, idea)           # big frozen model writes the kernel
R      = score(run(code))                          # same verifier
archive.add(code, R)
guide.lora_step(entropic_advantage(R))             # only the strategist learns
```

| Paper | Who | Trained at test time | Frozen executor | Result vs TTT-Discover |
|---|---|---|---|---|
| Guidance-TTT (2610.06269, Oct 2026) | MIT, Paul Pu Liang | 8B guidance model, LoRA r=32, proposes edits in language | GLM-5.2 | wins all 4 tasks (Polyomino 91.89 vs 83.72, TriMul 1129 vs 1262 µs re-evaluated) at about $120 vs $283 |
| PACEvolve++ (2605.07039, May 2026) | Google / GDM | 4B or 8B advisor, phase-adaptive RL | Gemini-3.1-pro | beats it on load balancing, recommender, Multi-Evolve; calls the entropic objective "the least stable overall" |
| HarnessTTT (2610.17370, Oct 2026) | Zou and Choi (original authors), NVIDIA | LoRA proposer that writes prompts, tools, middleware | yes | wins Erdős and AC1, loses AC2, ties CP26, with about 1,280 vs 25,600 trajectories |
| MELD (2610.24454) | GWU | coordinator and scientist policies, alternating GRPO | shared frozen executor | abstract only |

### Why split the roles

1. **Credit assignment.** The verifier scores only the final program, so one number must judge both the strategy and its implementation. With one model, a good idea implemented badly gets R = 0 and the update pushes down the idea tokens too.

   ```python
   rollouts = [("fuse matmuls",    "correct code", 0.53),
               ("fuse matmuls",    "off-by-one",   0.00),   # best idea, most negative advantage
               ("tweak tile size", "correct code", 0.42)]
   ```

   If an idea is hard to implement and the model fails half the time, training can steer it away from its best idea. With the split, the gradient reaches only the idea tokens; implementation errors are noise on a fixed, competent coder, not training signal.

2. **Sharpening a cleaner target.** TTT-Discover deliberately sharpens, but with one model it sharpens the joint distribution over ideas and code, including lucky implementations. The split sharpens only the distribution over ideas. The executor's distribution over code stays broad, so collapsing onto one strategy still leaves many implementations of it. This also eases the entropy collapse PACEvolve++ reports.
3. **Smaller action space.** There are far fewer ideas than programs, so the RL signal per idea is less noisy.
4. **Cost.** RL on an 8B proposer is much cheaper than RL on a 120B solver.
5. **No damage to the strong model.** Sharpening touches only the proposer, so the frontier executor keeps its general coding skill. This is the continual-learning concern in miniature: confine the update to a small, separable component.

### Caveats

- With a matched gpt-oss-120b executor, Guidance-TTT scores 83.19 vs TTT-Discover's 83.72 on Polyomino. Much of the headline win is the stronger frozen executor, not a better learning rule.
- Training the 8B model to write solutions directly fails: 41.63 on Polyomino and 9665 µs on TriMul. Small models can learn strategy but not implementation at this budget.
- This is the weights-based version of harness learning (`../continual-learning-benchmarks/METHODS.md`, harness state): the strategy lives in a small model's LoRA instead of a prompt or skill file. HarnessTTT sits between the two.

## Since publication (as of Nov 2026)

- About 118 citing papers. The extensions that beat it train a small strategist and keep a stronger executor frozen (Guidance-TTT, PACEvolve++, HarnessTTT).
- Frozen-weight methods match or beat its numbers for less compute: TTME (same framework, about 84% fewer FLOPs), EurekAgent (under $17 per math task). Frozen systems now hold its Erdős, AC1 and TriMul records.
- Weight updates still win where frozen search finds nothing valid: Alpha-RTL hardware design, 65.1% vs 26.1% PPA reduction.
- No independent rerun of the full 120B recipe; nanodiscover reimplements it at 8B without Tinker.
- Core Automation: Saroufim cites it once ("When AI Starts Writing Systems Code") as promising evidence; GPU MODE reviewed its kernels.

Full report with sources: `~/repos/reports/Learning to discover at test time.md`.
