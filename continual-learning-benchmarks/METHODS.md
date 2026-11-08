# Continual learning methods for LLM agents

The learning half of this directory. `README.md` is the evaluation half: how benchmarks score an agent that learns from a stream. This file covers the update itself, organized by where it lands. Literature window is September to November 2026, with earlier papers named only as anchors. Everything is an arXiv preprint unless a venue is given; "ICLR 2027 submission" means under review on OpenReview. Raw notes with URLs for every number are in `~/repos/research_notes/Test time learning weight updates/` and the longer report in `~/repos/reports/Test time learning weight updates.md`.

## 1. Where an update can land

The README writes an agent as (θ, s) and learning as two updates:

```
θ' = Uθ(θ, s, e)     # weights
s' = Us(θ, s, e)     # harness state
```

Methods differ by which slot they write and by what gates the write. From fastest and most reversible to slowest and most permanent:

| Substrate | What changes | Write rule | Benchmarks that measure it |
|---|---|---|---|
| Context, KV cache | nothing persists past the window, or a cached KV prefix | append | every benchmark's stateless or full-context baseline |
| External text memory | a store of notes, episodes, facts | write on task end, retrieve top-k | AgentCL, MemPABench, CL-Bench |
| Harness state | prompts, skills, tools, orchestration, source | an optimizer edits the harness on a cadence | ServeLearnBench, EvoAudit |
| Fast weights inside the block | a per-layer memory matrix or MLP | one gradient or delta-rule step per token or chunk | long-context recall suites only |
| Adapters or full weights | LoRA or θ itself | SFT, RL, distillation, continued pretraining | SCLATE's 4B student, agent tasks, knowledge-injection streams |

One finding recurs at every level and is the thing to carry out of this file: an update applied on every token, every trajectory, or every self-generated sample degrades the system, and an update gated by something external to the learner (a verifier, a frozen teacher, a held-out real-data check, a protected subspace, a reuse structure in the stream) is what works. Section 7 collects the evidence.

## 2. Context and KV cache

Context is not a learning method, but it is the baseline every parametric method must beat, and in this window it usually wins on acquisition.

**Data efficiency.** To teach a model one document, LoRA needed about 40,000 synthetic self-study examples to reach 64.5 single-document QA accuracy, against 73.6 for the document in context and 70.6 for a trained KV-cache prefix (Cartridges). On multi-document retrieval Cartridges led the parametric methods by 29 points (2609.17346, Amazon).

**Composition.** Facts stored in weights do not chain. Two-hop questions whose facts were both learned into weights succeed 37 to 64% of the time across 15 models up to 2.8T parameters; the same facts in context give 100% (ImplicitHop, 2610.41169). A hypernetwork that writes a LoRA from a fact gets 0.25% on two-hop against 83% when the fact is in context (2610.15677).

**Theory on the exchange rate.** Any in-context estimator that is linear in the labels needs d² / (r(2d − r)) demonstrations to match one LoRA example of rank r on a d-dimensional task, and the rate diverges below a critical depth (ICLR 2027 submission NT9WMkWqlG, linear transformers). A hypernetwork that produces a mid-layer additive update matches 10-shot ICL on productive rules (lemmatization 0.78 to 0.92 against 0.13 to 0.48 for ICL) but plateaus near half of ICL on arbitrary pairings such as antonyms (2610.00526). The practical boundary: rules amortize into weights, pairings do not.

**The limit.** Tworek's description of the problem (Sequoia Training Data, July 2026): in-context learning "doesn't have fundamental problems of catastrophic forgetting. It is pretty data efficient," but "roughly after around 20 minutes of usage, I need to compact it." Context is an acquisition substrate, not a retention one.

## 3. External text memory

The method family every agentic benchmark in the README tests: RAG, Mem0, MemOS, LangMem, File memory, Agent Workflow Memory, Dynamic Cheatsheet, ExpRAG, ReMem, MemProbe, ReasoningBank. All share one loop: write a note at task end, retrieve top-k at task start.

**When it helps.** Only when the stream contains reuse the retriever can find. AgentCL (2606.02461) on the dependent coding stream, where a subtask's solution is needed by a later task: MemProbe +21.9, ExpRAG +17.7, ReMem +13.5 over memoryless ReAct. The same methods on the conventional stream of the same domain: −9.4 to −0.3. On held-out tasks every method is below memoryless. The README's finding 1 is this result.

**Where it fails.** At the retrieve stage, not the store stage. MemPABench (2610.30485): all four stores held the user's objection and only File memory surfaced it; retrieval returned an applicable interaction preference in 7 of 334 probes. CL-Bench's "naive ICL outperforms dedicated memory architectures" and its "spurious generalizations and stale beliefs" are the same failure.

**Memory as a compute substitute.** TTME evolves an external memory with frozen weights and beats weight-updating test-time training on all five math-discovery tasks with 83 to 85% fewer FLOPs (2610.32046, ICLR 2027 submission). NoteBank, a training-free sparse episodic text memory bolted onto a pretrained 1.3B Gated DeltaNet, lifts RULER at 1M tokens from 1.7% to 70.3% at under 0.025% extra FLOPs (2610.15715). Text memory keeps the detailed evidence a recurrent state compresses away.

## 4. Harness state

Prompts, skills, tools, orchestration, or the harness source itself. Same five stages as memory, larger blast radius: one harness change touches every later task. The README's ServeLearnBench and EvoAudit cards and archive section 16 hold the numbers; this is the method map.

**By what they evolve.** Prompt and context: GEPA (2507.19457), ACE, Dynamic Cheatsheet, ReasoningBank (2509.25140). Skills: SkillOpt (2605.23904), PolicyBank (2604.15505), Agent Workflow Memory, Voyager. Whole harness state: Continual Harness (2605.09998; a Refiner that revises prompt, memories, skills and subagents on a cadence while the acting agent never sees scores), Prime Agent (2608.23552), Recursive Harness Self-Improvement (2607.15524), Adaptive Auto-Harness (2606.01770), JIT-Agent (2608.25593), HCL (2608.19013), RSIAgent (2609.15364), Meta-Harness. Code: Darwin Gödel Machine, AlphaEvolve, STOP.

**What the audits say.** Harness-Delta Attribution (2610.22466): 54% of reported gain is overfitting to the search set and 14% is test-time compute. CODA (2610.20845): skill-bank drift erases 52 to 93% of the gain. 2611.03219: Meta-Harness falls from 94.0 to 32.0 on SWE-Bench after five evolution rounds on a different task set. EvoAudit (2611.07624): all 16 evolved agents gain and lose tasks; selecting the best-scoring parent is worse than random on general tasks; a stricter admission gate drops the best candidate in 9 of 15 pools. On drifting streams with Gemini 3.6 Flash, ReasoningBank, GEPA and Meta-Harness gain +2.2 to +6.2 plasticity and lose −1.9 to −4.3 stability (2610.25858).

**Reading.** Harness evolution is the frozen-weight version of fine-tuning and shows the same stability failure. The operator-level fix EvoAudit proposes (regression-aware selection, trace-grounded diagnosis, a preservation-tracking gate) is the harness analogue of the gated weight updates in section 6.

## 5. Fast weights inside the block

Here the architecture carries the learning rule: a memory inside each layer is updated by a gradient or delta-rule step as tokens arrive, and the update is part of the forward pass. This is the line Core Automation's public statements point toward (archive of their quotes in the report), and the line with the most engineering progress and the least retention evidence in the window.

### The delta rule is online gradient descent

A linear fast-weight memory S reads as y = S k. One step of gradient descent on half the squared error between S k and the target v:

```python
S = [[0.0, 0.0], [0.0, 0.0]]
k, v, beta = [1.0, 0.0], [0.5, -0.5], 0.5
pred = [S[0][0]*k[0] + S[0][1]*k[1], S[1][0]*k[0] + S[1][1]*k[1]]   # S k
err  = [pred[0] - v[0], pred[1] - v[1]]                               # S k - v
S = [[S[i][j] - beta*err[i]*k[j] for j in range(2)] for i in range(2)]
```

In matrix form S_t = S_{t−1}(I − β k kᵀ) + β v kᵀ: erase the old association along k, write the new one. Gated DeltaNet adds a decay in front. Because this is linear in S, a whole chunk of tokens can be processed with one scan, which is why delta-rule linear attention trains as fast as attention. Replace S with a two-layer MLP and the step is no longer a scan; the from-scratch papers below are all attempts to parallelize that.

### From-scratch architectures, Sep to Nov 2026

| Paper | Memory | Rule | Parallelization | Scale | Result vs baseline |
|---|---|---|---|---|---|
| ParaTTT (2610.38255; MIT CSAIL, Liquid AI) | 2-layer MLP per head | decay, then one GD step on reconstruction with learned step size | iterate the chunkwise gradient trajectory twice as a fixed-point solve | 1.3B, 100B tokens | WikiText PPL 15.21 (GDN 15.78, Transformer 15.04); NIAH 4K to 32K falls 53.8 to 14.7; trains 4.9x faster than TTT-MLP; about one third of a linear model's throughput |
| DeltaTTT (2610.08553; Tsinghua) | f(x) = B φ(A x), two matrices | each matrix regresses its own local target with a delta rule | reuses DeltaNet's chunk scan per layer | 0.54B, 92.5B tokens | RULER single-needle 50.47 (full attention 72.42, GDN 47.60) |
| uTTT (2610.05484; UW-Madison, Adobe) | one SwiGLU fast-weight set shared by all layers | dot-product loss, momentum, Muon orthogonalization, no forget gate | 4,096-token chunks | 760M | RULER avg 27.85 (Transformer 34.16, LaCT 23.21) |
| GradLev (2609.34174; UT Austin) | final MLP of the Transformer | GD on the true next-token loss with retention toward initial weights | exact token-parallel scan given predicted activation gradients | 300M | gains inside confidence intervals; authors call it preliminary |

Three things to retain. The recall gap to softmax attention did not close at any scale tested. Nonlinear memories cost 3 to 5x the throughput of linear recurrences. DeltaTTT's motivating experiment is itself a negative result: serial state-dependent nonlinear TTT scored 28.25 on single-needle against 37.87 for a parallel fixed-base baseline.

### Retrofits onto frozen pretrained models

| Paper | Host | What is written | Result |
|---|---|---|---|
| Titan-Llama (2610.19210) | Llama-3.2-1B, first 4 layers | 3-layer MLP per KV head, 17M params, GD with momentum and decay per chunk | QuALITY 23.9 to 40.5 (full attention 43.3); degrades past the 16K training length |
| Select-TTT (2610.36794; USTC, Huawei) | Llama-3.1-8B, Qwen3-4B | MLP down-projections at 6 layers, write strength weighted by query relevance | LongMemEval-S 33.4 to 37.4; uniform write strength (In-Place TTT) gives 33.6 and hurts Qwen (27.2 to 25.4) |
| Dynamic evaluation study (2610.33025) | Qwen3 1.7B to 27B, Gemma-4, diffusion LMs | one SGD step per 128-token chunk on MLP output projections, no training | next-token loss −5 to −8% on softmax models, under 1% on hybrid GDN models; recovers none of the retrieval accuracy lost to truncation; costs 36 to 38 GSM8K points |

The answer to "can a deployed transformer learn facts in its weights by gradient steps on the stream" is, in this window, that it learns style and not facts. Writing facts via next-token loss gave 30% recall under chat prompting against 80% in context.

### Linear attention and memory-native models

The delta-rule line scales further. KronLSTM (2610.26834; Hochreiter) at 1.5B reaches LAMBADA PPL 11.72 against 12.54 for GDN-2 but is weak on single-needle at 16K (11.6 vs 99.2). Muir (2611.08376) reports passkey near 100% to 1M tokens and extrapolation from 4K training to 10M-token streams, self-reported with no variance. MemT-1.0 (2610.26825; MemTensor) is the largest deployment: a 288B/17B-active post-trained DeepSeek-V4-Flash with GDN-2 memory blocks whose dynamic state persists across turns within a trajectory. It averages 50.6 across 11 memory benchmarks against 38.1 for the best parametric baseline, but 68.6 for MemOS and 65.0 for full context. The Google Titans/MIRAS/HOPE line published nothing new in the window.

### Negative results

- In released TTT-Linear and TTT-MLP checkpoints the prediction-error part of the gradient cancels at initialization; removing it costs 2.8 to 2.9% perplexity while freezing all fast-weight updates costs 4.2 to 4.5x (2610.16268). The fast weights help; the "gradient on the error" story explains little of why.
- Retaining updates computed on the model's own generated text worsens prediction on independent human text at 125M to 3B and on Qwen3-4B; a frozen generator or a held-out "settlement" check on real text removes over 98% of the damage (2610.05076).
- VisionHOPE's unconstrained self-modifying update "proved unstable" and needed a step cap and spectral clamp (2609.33325).
- Retrieval collapses past training length for every bounded-state model tested beyond 32K.
- No paper in the window measures cross-session forgetting for any of these architectures.

## 6. Adapters and full weights at deployment

### Why fine-tuning forgets, at the level of a toy

The window converged on a mechanism: forgetting is drift from the base model measured on out-of-domain inputs, and the drift concentrates in identifiable places. Two of them have cheap fixes.

**Adam's second moment on absent tokens.** In continued pretraining, a token that never appears in the new corpus gets a tiny one-sided gradient on its output row at every step (always "push this logit down"). Adam normalizes it to a full-size step:

```python
g, lr = 1e-6, 1e-4                       # tiny one-sided gradient on an absent token's output row
for eps in (1e-8, 1e-4):
    m, v = g, g*g                        # Adam moments after one bias-corrected step
    print(eps, lr * m / (v**0.5 + eps))
# eps=1e-8 -> 9.9e-05   (full learning rate from a 1e-6 gradient)
# eps=1e-4 -> 9.9e-07   (step now scales with the gradient)
```

Adam's step is lr · m / (√v + ε). When |g| is far below ε the step is lr · g / ε, plain SGD; when |g| is far above ε it is sign-like. Raising ε to 1e-4 on the output projection only removes 39 to 68% of forgetting across Pythia 160M to 12B, Qwen2.5-0.5B, TinyLlama and OLMo-2-1B with no loss on the target; adding 1% replay reaches 80% (2610.09835).

**Project the applied update off the dominant subspace.** Subspace Tuning keeps the top singular directions of each weight matrix untouched:

```python
import numpy as np
W = np.random.randn(64, 64); U, s, Vt = np.linalg.svd(W)
k = 48                                                # protect the directions holding ~99% of the energy
Pu = np.eye(64) - U[:, :k] @ U[:, :k].T
Pv = np.eye(64) - Vt[:k].T @ Vt[:k]
dW = adam_step(grad_W)                                # the optimizer's applied update, not the raw gradient
W += Pu @ dW @ Pv
```

Order matters: projecting the raw gradient and then running Adam leaked 69% of the change back into the protected directions, because Adam's per-coordinate scaling does not commute with the projection. On Qwen3-8B, plain SFT on science data reaches 77.6 in-domain and drops a five-benchmark prior-capability average from 81.0 to 55.5; SFT with the projection reaches 80.1 and holds 80.0 (2610.22293; ICLR 2027 submission).

### What worked, and under what rule

| Paper | Models | What is updated | Rule | Result | Forgetting measured |
|---|---|---|---|---|---|
| Time-incremental CPT (2609.23916; Öncel et al.) | OLMo2 1B/7B, Llama 1B to 8B, Gemma-3-1B | full weights (LoRA r=256 matches) | continued pretraining on 6B curated post-cutoff tokens | post-cutoff knowledge +0.043 macro-F1; 13-task average within 0.01 of base | yes; forgetting ≈ 0.60 × acquisition; the optimal learning rate for knowledge is about 10x below the one for capability |
| Subspace Tuning (2610.22293) | Qwen3 1.7B to 32B | full weights, projected | above | prior-capability loss 25 to 57 points becomes 1 to 2 | yes, five benchmarks plus KL to base |
| iSDFT (2609.24646; Imperial, UCL) | Qwen2.5-7B, Qwen3-1.7B, Gemma-4-E2B, OLMo3-7B | full policy | KL to a per-token tilt of student and in-context teacher, plus a KL anchor to base | beats SDFT and SFT on tool use and science | median change −0.05 over 16 benchmarks vs −0.25 for SDFT |
| OPDArena (2610.25651; Microsoft) | Qwen2.5 and Qwen3, 1.5B to 8B | full | RLVR vs SFT vs on-policy distillation on shared math data | RLVR +5 to +14 in-domain with out-of-domain ≈ 0; SFT out-of-domain −2 to −4.5 | yes; out-of-domain distribution shift predicts forgetting |
| TSRL (2610.17726; CASIA) | Qwen2.5-7B/32B | full, GRPO | expert SFT loss only on states where all 8 rollouts fail | WebShop retained after ALFWorld: SFT 29.4, GRPO 65.0, TSRL 68.3 | prior task only |
| ASCENT (2610.05303; UNSW) | Qwen3.5-4B/9B | persistent LoRA | after each verifier-accepted episode, distill from a frozen copy that sees the verified trajectory as context | ALFWorld 46.4 to 69.5 (best memory method 63.6); WebShop 17.8 to 41.9 | no general benchmark |
| Local Support Learning (2610.02126; MIT, Tel Aviv) | Qwen2.5-7B | per-phase LoRA with input gates | adapter active only where its gate fires | retention 76.6% to 98.8%; plain LoRA loses 44% | yes |
| SCLATE student (2609.32391; Apple) | Qwen3.5-4B | full, on-policy distillation through unmodified harnesses | teacher sees the benchmark's corrective feedback | held-out days +7.6 to +11.8 in 5 of 6 configurations | no |

**The through-line is drift geometry, not on-policy data.** RLVR forgets less than SFT because it moves the policy less on inputs outside the training set, not because it samples from itself. The test: on-policy self-distillation against a drifting EMA teacher specializes faster than GRPO and then collapses after three continual stages (2607.01763 v2). A theory paper gives the account: self-distillation yields sparse edge updates, SFT dense overwrites (2610.05200). Every successful online method anchors to something that does not drift: the frozen base in ASCENT, an RLVR teacher in OPDArena, an explicit KL term in iSDFT. Muon spreads update energy over about 7 singular directions against 1.5 for AdamW and matches dedicated CL methods on TRACE (2609.24678), though the effect reverses on Adam-pretrained models (2605.10468), so it is conditional on optimizer consistency.

### Test-time RL on reasoning

The one clean matched-compute win for weight updates over more sampling: TTT-Consolidate (2611.02104; EPFL; NeurIPS 2026). A meta-learned procedure encodes eight parallel reasoning trajectories into a LoRA and the adapted model answers once. At about 1.05e15 FLOPs on Qwen3-1.7B, majority vote at 8, 13 or 18 samples sits at 22.2 to 22.7 and consolidation reaches 35.4; on AIME25 with all eight inputs wrong it scores 69.5 where majority vote scores 0. Costs: 1.9 s per problem on one H100, about 22 GB against 15.8 GB, and the adapter is reset per problem. It loses on AIME26.

Everything else is TTRL (majority-vote pseudo-reward, 2504.16084) and repairs to it: votes collapse on small answer spaces (PROSE, 2609.16660), suppress legitimate multiple solutions (MS-TTRL, 2610.32234, +19 recovery), are gamed by spurious consensus on code (ERPO, 2609.09135), and have analytically derived collapse regimes where an exploration bonus prevents collapse but lowers pass@1 (2611.01307). Entry-policy entropy predicts the gain with ρ = −0.90 (2610.00903). Updating only about 100K bias parameters already gives 76.67% MATH-500 on Qwen2.5-7B (2609.18587). Transfer out of the adapted set is about +2 points (2610.41296).

### Agents that update weights during deployment

Research prototypes on open models up to 12B. ASCENT above is the clearest parametric win over in-context experience on the same stream, at about 1.2x episode time, gated by a binary environment verifier. ReTTT (2610.23942; Shanghai AI Lab, Fudan) trains an episode-local LoRA on in-episode reflections: Qwen3.5-9B SWE-bench Lite 33.3 to 43.3, ALFWorld 47.1 to 63.6, at about 3x worker time; the same reflections kept as text score 53.6. ServeLearnBench abandoned SFT and GRPO on 14B/32B models because no trajectories succeeded under sparse reward. Every frontier-model agent result in the window (GPT-6 Astra on ARC-AGI-3, PILOT, StepLearn) is frozen-weight. No primary source describes a production system that fine-tunes from live traffic; Thinking Machines' Tinker retrospective describes folding deployment experience back into the recipe offline.

### Negative results

- Surprise-gated updates learn lies. With LoRA updates triggered by high per-token loss on a stream of claims, Qwen2.5-1.5B absorbs 84 to 93% of planted false claims; an oracle verifier gate learns 0.86 of true facts and 0 lies, a judge-model gate 0.51 and 6.4% (2610.39840).
- Training a model against what it believes induces misalignment: supervised LoRA on believed-false QA raises the misaligned-answer rate from about 4% to over 60% on Qwen3.6-27B with MMLU unchanged (2611.05287).
- Direct SFT on experience barely consolidates. Consolidating in-context gains into weights keeps at least 64.8% of them on 749 SWE tasks; SFT on the same experience keeps 3.8% (ICLR 2027 submission 3y70lKgk9Y).
- General-benchmark averages hide capability-specific loss: EWC cuts a general drop from −11.0 to −1.7 while instruction following on the target task still falls 35 points (2609.28395).
- Rankings flip with adapter rank and evaluation date: a method leads rank-8 LoRA replay by 5.0 and trails rank-72 replay by 11.6 on a 24-month Wikidata stream (2609.03900). Report the rank.
- Plasticity loss onset scales sublinearly with model size, so scale delays and does not prevent it (2606.24752, ICLR 2027). Ever-growing subspace protection saturates and kills plasticity (ICLR 2027 submission uXkO3yaysv). No continual-backprop or reinitialization result exists on a model of 1B or more.
- No agent paper measures general-capability forgetting after a long adapter stream; retention is on prior agent tasks only.

## 7. The gating pattern

Lined up, the successes and failures sort by one variable.

| Level | Ungated (fails) | Gated (works) |
|---|---|---|
| Text memory | write every episode, retrieve top-k on any stream (AgentCL conventional: below memoryless) | reuse exists in the stream and the retriever can find it (AgentCL dependent: +14 to +22) |
| Harness | admit any mutation that raises the search-set score (68% of gain is overfitting and compute) | regression-aware select, trace-grounded diagnosis, preservation gate (EvoAudit AGE) |
| Fast weights | uniform write strength per chunk (In-Place TTT hurts Qwen3-4B); retain updates from own generations (2610.05076) | query-relevance weighted writes (Select-TTT); settle only after a held-out real-text check |
| Weights | surprise-triggered LoRA (84 to 93% of lies absorbed); EMA self-teacher (collapse after 3 stages); SFT on own attempts (below base) | verifier-accepted episodes distilled from a frozen teacher (ASCENT); KL anchor to base (iSDFT); update projected off the dominant subspace (SST); expert loss only on states where every rollout failed (TSRL) |

The reading of the window: context and KV representations are the acquisition substrate (fast, reversible, compositional, 10 to 29 points ahead of parametric injection on document QA). Weights are the consolidation substrate (slow, cheap at inference, able to hold arbitrary bindings that in-context learning cannot amortize). What survives the negative results is periodic gated consolidation, not weights that learn on every token. The frozen-weight benchmarks in `README.md` and the parametric methods here fail the stability test in different places, and no paper measures a system that combines the two routes against both pure routes at matched compute on one long stream.

## 8. Where Core Automation sits

Core Automation (Tworek, Anil, Saroufim, Amid; founded March 2026) is the lab whose stated bet is this file's section 5: architectures where, in Saroufim's words, "the distinction between training and inference doesn't really exist." Their public artifacts as of November 2026 are about the search apparatus rather than a mechanism: a two-site Shampoo-trained 32B MoE (compute-islands post, 2 Nov), agent-written kernels and a four-role author/competitor/cheater/auditor self-play recipe (systems-code post, 28 May), agent-driven architecture search with continual learning named as a fitness criterion (architecture-discovery post, 2 Sep), and One Layer Deeper (9 Sep), a competition on learned depth extrapolation whose result was negative: at 500M parameters and one GPU-hour, no end-to-end-trained model extrapolated past training depth, and all three perfect scorers broke the rules. Their codebase still runs autoregressive transformers. Their diagnosis matches the window (context is bounded by horizon, fine-tuning forgets) and their stated end state is the closed-loop regime that 2610.05076, VisionHOPE and ASCENT found unstable without an external anchor. Nothing public says how they intend to gate it. The full quote archive, artifact table and a list of questions to ask are in section 5 of the report named at the top of this file.

## 9. Open problems on the methods side

- A from-scratch fast-weight model at 3B or more, with cross-session forgetting measured.
- Any agent that updates weights from a long stream, with general-capability forgetting measured alongside task gain.
- The hybrid: context for acquisition, gated consolidation into weights, compared against both pure routes at matched compute on one stream. AgentCL's dependent stream and SCLATE's calendar are the right substrates.
- Continual backprop or reinitialization tested on an LLM.
- Seeds. Most fine-tuning papers in the window report three or fewer; ICLR 2027 submissions are under review; ARC Prize 2026 results land 4 Dec 2026.
