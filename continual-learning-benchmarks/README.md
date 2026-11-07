# Continual learning benchmarks: a survey (Nov 2026)

How agents are evaluated on learning from experience: what the benchmarks measure, how their environments are built, where their controls are missing, and a taxonomy. Sections 1 to 10 are the field map and catalog; 11 is environment construction; 12 and 13 are notes on SCLATE and MemPABench; 14 is the taxonomy. ApprenticeBench numbers come from NeoCognition's blog and site, which are JS-rendered; the rest from the papers cited.

## 1. What "continual learning" means in the 2026 benchmark literature

Nearly every benchmark below freezes the weights. "Learning" means: a *system* (model + harness + memory/notes/skills) processes a **stream** of tasks and must do better on later tasks because of earlier ones. The harness state (Markdown notes, memory DB, skill files, or just the un-truncated context) is the only thing that persists. CL-Bench's operational definition is the cleanest: a system continually learns if it "does better on tasks given prior experience than if running from a fresh start." CLE defines experience as the tuple (action, next observation, reward) and distinguishes this from RSI: "recursive self-improvement concerns retained changes that improve how subsequent improvements are generated," and "online learning, TTT, and CBR can all operate with fixed learning procedures; their use alone does not establish RSI."

So three different things get called continual learning. Know which one a benchmark measures:

1. **Test-time learning from experience** (frozen weights, stream of tasks, notes/memory persist). ApprenticeBench, CLE, CL-Bench, AgentCL, AgentStream, Evo-Memory, StreamBench.
2. **Memory recall / personalization** (retrieve facts from a long history; no improvement over a stream expected). LoCoMo, LongMemEval, MemoryAgentBench, LifeVal. Not CL, but interviewers conflate them.
3. **Parametric continual learning** (weights change: SFT/RL on the job, test-time training, sparse memory finetuning). RL's Razor, Meta sparse memory finetuning, TTT on ARC. No agentic benchmark above evaluates this; CL-Bench and AgentCL explicitly scope it out.

## 2. The axes that distinguish benchmarks

This is the thing to be able to reproduce on a whiteboard.

| Axis | Options seen | Why it matters |
|---|---|---|
| Is reusable structure guaranteed? | Hidden latent structure (CL-Bench), constructed dependencies (AgentCL dependent, CLE ladders, ApprenticeBench traceability), or unknown (AgentCL "conventional", AgentStream) | Without guaranteed reuse you can't tell "no learning" from "nothing to learn" |
| Baseline that isolates learning | Stateless same-system run (CL-Bench gain), memoryless ReAct (AgentCL), novice ablation (ApprenticeBench, one model only), none (CLE Trial1 vs Trial2 is confounded) | The score vs the *gain* over a fresh start are different claims |
| Feedback signal | Dense mentor feedback then sparse (ApprenticeBench), diagnostic rubric feedback on failure (CLE K=2), native reward only (CL-Bench), none/self (AgentCL, AgentStream) | SkillLearnBench: self-feedback alone causes "recursive drift"; external feedback is what moves scores |
| Resettable? | One non-resettable sequence (ApprenticeBench), fixed order one run (CLE), 5 rollouts in shuffled orders (CL-Bench), 3 seeds (AgentStream) | Single runs on a path-dependent sequence have unreported variance |
| Transfer split | Independent held-out tasks after freezing memory (AgentCL), none (most) | Shows whether memory hurts on out-of-distribution tasks |
| Stability vs plasticity | AgentCL PG/SG/GG; CL-Bench stab/plas decomposition | "Did it learn" vs "did it keep what it learned" |
| Human baseline | ApprenticeBench (n=2), LifeVal (100% by construction) | Only ApprenticeBench claims agent > human |
| Realism | One simulated job with GUI (ApprenticeBench), 19 mock apps + 1,024 APIs (CLE), real stateful environments (CL-Bench), repackaged QA sets (AgentCL, Evo-Memory, AgentStream) | Ecological validity vs controllability |
| Cost reporting | $/task + human wage (ApprenticeBench), $/run and hours (CLE, CL-Bench), per-task (AgentStream) | Several papers find cost does not buy gain |

## 3. Benchmark cards

### ApprenticeBench (NeoCognition, Sept 2026, no paper/code)
- One simulated accounts-payable job in Odoo (Acme Home Builders, CA construction). 6 months of history, then 100 bills in one non-resettable sequence; dense AP-manager feedback in month 1, month-end only after.
- State = Markdown notes via Claude Code / Codex Read/Write tools. 500K context. GUI vs API ("CUA tax").
- Score: cumulative success over 100 bills, $/task (tokens vs BLS wage).
- Numbers: Fable 5.1 72% GUI ($18.23) / 70% API; GPT-6 Astra 68%/65%; best human 51% ($7.21), second human 40%; Fable 5 43%; Opus 5 36%; GPT-5.6 Sol 26%; Kimi K3 18%.
- Learning attribution: "smart novice" ablation on Fable 5 only: 11 novice, 31 history only, 35 feedback only, 43 both. Verbatim cost-code lookup works for all models; inferred codes only for Fable 5.1 and GPT-6 Astra.
- Behavior: agents slowed down as notes grew (Fable 5.1 192K words / 122 files; humans ~2,750 words); humans sped up. Opus 5 "corrected 17 times... never changed its mind."
- Gaps: single runs, n=2 humans, novice ablation on one model, no held-out split, no forgetting/slope metric, private artifact built by the company whose pitch depends on it.

### Continual Learning Exam, CLE (UCL, arXiv 2610.33956, Oct 2026)
- 120 expert-authored tasks x 5 connected levels = 600 levels, 89 roles, 12 domains, 19 stateful mock apps, 1,024 APIs, 16,125 rubric criteria. Files/app state persist across the 5 levels; later levels reuse Tools & Skills, Data Dependency, Strategy, User Context, Domain Knowledge. Passing a level unlocks the next; a level passes only if every rubric item passes.
- Protocol: all 120 tasks in one fixed order; harness state (memories, skills) persists; evaluator is DeepSeek V4 Flash (96.86% agreement with humans). Trial 1 = one submission; Trial 2 = failed first submission gets diagnostic feedback naming unmet rubric categories, then one revision.
- Metrics: Completion = levels cleared / 600. L_k = share of tasks whose first k levels all pass. Δ = Trial 2 − Trial 1 (the page itself says this "does not isolate a causal effect of memory or learning").
- Numbers (Trial 2 completion, L5 full-task, Δ): Opus 5 / Claude Code 47.67%, 23.33%, +5.50 (Trial 1 42.17%; $721, 58.9 h). Sonnet 5 40.17%, 19.17%, +13.17. Gemini 3.7 Flash / Hermes 33.67%, +8.33. GPT-5.6 Sol and Terra / Codex 33.17%, +5.33. Open models via OpenClaw 10–24%. Median Δ = 5.33 pp, range −1.33 to +13.17.
- Findings: pass rate falls at every level; L1→overall ρ=0.75 but L1→L5 ρ=0.42 (Qwen3.8-Max 74% L1, 2.5% L5); in every Trial 1 config more than half the tasks end at 0 or 1 levels; harness matters (Hermes beats OpenClaw on all six shared backbones, +5.56 pp mean in Trial 2); newer model generations do not help monotonically (Gemini 3.8 Flash < 3.7 Flash).
- Gaps: no failure taxonomy, no analysis of the memory/skill files agents wrote, no harness-reset control, single run.

### CL-Bench (Berkeley / Snorkel / UW-Madison, arXiv 2606.05661, June 2026)
- Six real stateful domains with hidden latent structure: codebase adaptation (19 issues), blind spectrum monitoring (90 scans), cohort studies (20), database exploration (40 questions, 15-query budget, schema migration at q20 = concept drift), exploitable poker (120 hands vs 3 fixed opponents), sales prediction (12 yearly instances). Structure "must be inferable from experience."
- Metric: gain g_t = r_t^stateful − r_t^stateless for the *same system*; normalized gain = (r̄^sf − r̄^sl)/(r_max − r̄^sl); normalized reward uses stateless GPT-5.4 ICL as the external zero. Decomposed into stability (first instance of each new variant) and plasticity (within variant). 5 rollouts in different orders.
- Systems: full-context ICL, ICL Notepad, Mem0, ACE, Claude Code, Codex. Models Opus 4.7, Sonnet 4.6, Gemini 3.1 Pro/3 Flash, GPT-5.4.
- Numbers (norm gain, cost): ICL Sonnet 4.6 25.4% $30; Claude Code Sonnet 23.9% $39; Mem0 GPT-5.4 20.2%; ICL GPT-5.4 20.1%; ICL Opus 4.7 19.5% $50; Notepad Sonnet 18.2%; ACE 8.6% $63; ICL Gemini 3.1 Pro 6.2%.
- Findings in their words: "Naive ICL outperforms dedicated memory architectures on most tasks"; "the choice of learning medium matters as much as the underlying model"; "memory modules introduce spurious generalizations and stale beliefs"; expensive systems "do not translate that cost into proportionally higher gain"; ACE and GPT-5.4 ICL show "virtually no stable learning."
- Gaps: tens of instances per sequence, 6 tasks, wide overlapping CIs, context-based methods only.

### AgentCL (OSU, Huan Sun & Yu Su, arXiv 2606.02461, June 2026) — NeoCognition's own lab
- Three stream types: conventional (same domain, reuse unknown), dependent (subtasks precede complex tasks that need them; dependencies hidden, ~1% of stored episodes relevant, half a stream earlier), independent (held-out, different source).
- Metrics: B_i memoryless ReAct; F_i first pass (memory read/write); S_i second pass (memory frozen); H_j held-out. Plasticity Gain = F−B, Stability Gain = S−F, Generalization Gain = H−B.
- Methods: ReAct, LangMem, Mem0, AWM, Dynamic Cheatsheet, ExpRAG, ReMem, MemProbe (theirs). Backbone Qwen3.5-35B-A3B.
- Numbers: CodeEval-Pro dependent first pass ReAct 44.8 → MemProbe 66.7 (+21.9); independent held-out ReAct 72.5, every memory method below it (−0.8 to −11.7). BrowseComp+ dependent MemProbe +40.0. Conventional streams: spread across methods is 2–6 points vs 9–16 on dependent streams.
- Findings: "conventional streams offer limited ability to distinguish memory designs"; memory "frequently induce[s] memory interference or performance degradation on conventional streams and independent tasks." The central challenge is "deciding which past experience to abstract, retrieve, or ignore."

### AgentStream (Microsoft / CASIA, arXiv 2608.00155, July 2026)
- 6 benchmarks x 50 tasks (AppWorld, BFCL, BrowseComp+, HLE, SWE-bench Verified, Tau2). Three streaming scenarios: Isolated (per-domain state), Sequential (one state across domains in fixed order: forward transfer), Interleaved (shuffled, one state: retrieval under interference). 3 seeds.
- Methods: ACE, A-Mem, ReasoningBank, AutoSkill, Harness. Δ = Perf(with state) − Perf(vanilla).
- Numbers: only 34/45 (Isolated), 28/45 (Sequential), 28/45 (Interleaved) configs beat vanilla; mean gains +1.37 / +0.75 / +0.90. GPT-5.4 net negative (−0.35 to −0.78); Gemini 3.1 Pro +2–2.7; Opus 4.7 +0.9–1.75.
- Findings: "the weakest model exhibits negative evolution gains" (a low solve rate fills the buffer with failed trajectories); benefit is non-monotonic in model strength; "context-integrated methods favor Isolated while retrieval-based methods favor Interleaved, and the optimal method varies across models."

### Evo-Memory (arXiv 2511.20857, Nov 2025) and StreamBench (2024)
- Evo-Memory: restructures MMLU-Pro, GPQA, AIME, ToolBench into streams; compares ~15 memory methods; long-context baselines "remain highly competitive." StreamBench is the 2024 precursor: online input-feedback sequences.

### LifeVal (OpenAI, arXiv 2611.06687, Nov 2026) — the personalization neighbor, not CL
- 169 tasks (702 planned) weighted by a willingness-to-pay survey (n=3,837; $1.7T/yr across 148 activity categories). Each task: prompt, the writer's real ChatGPT conversation dump + memories, reference files, extra context, rubric, RL environment with offline web. GPT-5.6 grader (r=0.864 with humans); hack grader zeroes any internal-API use.
- Numbers (Figure 5): GPT-5.2 ~28–49%, GPT-5.4 ~63–73%, GPT-5.5 ~57–78%, GPT-5.6 ~64–83%, GPT-6 Astra ~67–83% across reasoning effort; humans 100% by construction at ~120 min.
- The CL-relevant finding: removing past conversations and memories barely hurt; removing extra task context hurt a lot. Criteria sourced from past conversations had a 33% win rate vs 48% from memories vs 56% from explicit context, and the model's conversation search "relied on keywords and topical cues" (literal grep). Every 100%-win conversation criterion was also answerable from elsewhere. So even the frontier does not reliably mine its own history, which is the retrieval half of any notes-based CL system.

## 4. Metric intuition in plain Python

Gain vs score: the stateless baseline is what makes "learning" a claim.

```python
stateful  = [0.2, 0.4, 0.6, 0.7]   # reward per instance, memory on
stateless = [0.2, 0.3, 0.3, 0.3]   # same system, fresh start each time
gain = [s - b for s, b in zip(stateful, stateless)]   # [0.0, 0.1, 0.3, 0.4]
r_max = 1.0
norm_gain = (sum(stateful)/4 - sum(stateless)/4) / (r_max - sum(stateless)/4)  # 0.275
```

AgentCL's three gains on one task: `PG = F - B` (did the stream help), `SG = S - F` (did a second pass with frozen memory help or interfere), `GG = H - B` (held-out).

```python
B, F, S = 0.45, 0.67, 0.60      # ReAct, first pass, frozen-memory second pass
PG, SG = F - B, S - F           # +0.22 plasticity, -0.07 interference
```

CLE's level metric is a survival curve, not an average:

```python
levels_cleared = [5, 1, 0, 3, 2]            # per task
L = [sum(x >= k for x in levels_cleared)/5 for k in range(1, 6)]  # [0.8, 0.6, 0.4, 0.2, 0.2]
completion = sum(levels_cleared) / (5*5)   # 0.44
```

ApprenticeBench's learning attribution is a 2x2 ablation, not a gain curve: novice 11, +history 31, +feedback 35, +both 43 (interaction is small: 31+35−11 = 55 > 43, so history and feedback overlap).

## 5. What holds across all of them (interview-grade claims)

1. **Plain long context is the baseline to beat, and it usually isn't beaten.** CL-Bench (ICL best), Evo-Memory, AgentCL independent split, AgentStream (only ~2/3 of configs beat vanilla). ApprenticeBench's notes-only result is consistent: a stock coding harness with Markdown files is the strongest public system.
2. **External feedback moves scores; self-reflection drifts.** CLE median +5.3 pp from one diagnostic revision; ApprenticeBench feedback-only +24 over novice; SkillLearnBench recursive drift.
3. **Learning gain scales with base capability and vanishes below a threshold.** ApprenticeBench "positive for strong, vanish for weak"; AgentStream GPT-5.4 negative; CLE L5 collapse for Qwen/GLM. Yu Su's "escape intelligence" framing is the vendor version of this.
4. **Interference is the main failure, not forgetting.** AgentCL SG negative, CL-Bench "stale beliefs," ACE "context collapse," ApprenticeBench 192K-word notes slowing the agent down. The research problem is curation: what to abstract, retrieve, or ignore.
5. **Harness is a first-class variable.** CLE Hermes vs OpenClaw +5.6 pp on identical models; CL-Bench Claude Code is the one non-ICL point on the Pareto frontier; ApprenticeBench's open-model gap may be partly harness mismatch.
6. **Cost does not buy learning.** ACE most expensive and near bottom on CL-Bench; CLE Trial 2 nearly doubles hours for +5 pp.
7. **Most results are statistically thin.** Single fixed-order runs (CLE, ApprenticeBench), 5 rollouts with overlapping CIs (CL-Bench), n=2 humans.
8. **Nobody benchmarks weight updates on the job.** Every benchmark above scopes out TTT/SFT/RL. This is the open lane and the one NeoCognition's MTS [Research] posting hints at.

## 6. Vocabulary to have ready

- Stability–plasticity (classic CL), reliability–plasticity (Su's version), catastrophic forgetting vs memory interference, forward/backward transfer, concept drift (CL-Bench DB migration), context collapse (ACE), recursive drift (SkillLearnBench), escape intelligence / smartest novice / local physics (Su), stateful vs stateless system, non-parametric vs parametric CL, RSI vs experiential learning (CLE), CUA tax (ApprenticeBench), survival-style level metric (CLE), normalized gain over own headroom (CL-Bench), PG/SG/GG (AgentCL), Isolated/Sequential/Interleaved streams (AgentStream).

## 7. Questions worth asking a continual-learning lab

- Which of the three definitions in §1 are you optimizing, and what is your stateless control?
- Do you measure gain on a held-out independent split, or only within the stream?
- How do you handle interference: curation, consolidation, forgetting policy, or retrieval gating?
- What is your variance across task orders? Is the sequence resettable?
- Where does external feedback come from in deployment, and what happens when it is month-end-sparse?
- Is anything parametric, and if so how do you control regression (RL's Razor, sparse memory finetuning)?

## 8. Catalog: every other CL / lifelong / memory benchmark found, through Oct 2026

Grouped by what they measure. Dates are arXiv v1 unless noted. Abstract-level only; numbers appear only where the abstract gives one.

### 8a. Learning over a task stream (frozen weights) — the same family as ApprenticeBench / CLE / CL-Bench / AgentCL / AgentStream
| Benchmark | Date | Setup | What persists | Metrics | Headline |
|---|---|---|---|---|---|
| **ContinualSkillBench** (2608.03874) | Aug 2026, rev Oct 15 | 5 domains x 100 tasks ordered by difficulty and skill-transfer opportunity | context, feedback, explicit skill library | mean performance, skill accumulation/reuse counts | "retaining context and feedback without skill maintenance yields nearly the same mean performance as maintaining an explicit skill library" |
| **PATH-Bench** (2608.01149) | Aug 2026 | probe-centered sequences with controlled helpful vs interfering histories; code gen + multi-turn tool use; 8 agents | memories or skills | avg perf, forward transfer, backward transfer, forgetting | their Selective Experience Use harness "reduces forgetting while improving forward transfer" |
| **SkillLearnBench** (2604.20087) | Apr 2026 | 20 verified skill-dependent tasks, 15 sub-domains | generated skills | skill quality, trajectory, outcome | all methods beat no-skill, "consistent gains remain elusive"; self-feedback alone = recursive drift |
| **SkillFlow** (cited in 2608.29596) | 2026 | 20 workflow families, 166 tasks, lifelong skill evolution | skills | — | — |
| **StuLife** (2508.19005) | Aug 2025, rev Jan 2026 | simulated college journey, 3 phases, 10 sub-scenarios, interdependent tasks | memory, skills, internalized knowledge | — | self-evolving-agent framing; ProactAgent later reports up to 32% relative SR gain on it |
| **LifelongAgentBench** (2505.11942) | May 2025 | DB, OS, Knowledge Graph; interdependent skill-grounded tasks | accumulated experience | lifelong performance | experience replay "limited"; group self-consistency helps |
| **SWE-Bench-CL** (2507.00014) | Jun 2025 | SWE-Bench Verified issues in chronological per-repo sequences | FAISS semantic memory | avg accuracy, forgetting, FWT/BWT, composite CL score | CL-Bench calls it closest prior work but saturated/binary |
| **StreamBench** (2406.08747) | 2024 | online input-feedback sequences | — | — | precursor |
| **MemoryBench** (2510.17281) | Oct 2025, rev Jun 2026 | simulated user-feedback streams, multi-domain, multilingual | accumulated feedback | effectiveness, efficiency | SOTA "far from satisfying" |
| **AgentMemoryBench** (ICLR 2026 Lifelong Agents wkshp) | Apr 2026 | interleaved streams across code tool-use, embodied, web, dialogue; system + personal memory | both memory stores | five modes: improvement, retention, forgetting, generalization, conflict resolution | single-memory designs fail under mixed streams; MEMs router |

### 8b. Memory benchmarks that include a "test-time learning" axis but are mostly recall
| Benchmark | Date | Notes |
|---|---|---|
| **MemoryAgentBench** (2507.05257) | Jul 2025, v4 Jun 2026 | four competencies: accurate retrieval, test-time learning, long-range understanding, selective forgetting; "current methods fall short" on all four |
| **MemoryArena** (2602.16313, ICML 2026) | Feb 2026 | agentic tasks with interdependent subtasks across sessions (web nav, preference planning, progressive search, formal reasoning); LoCoMo-saturated agents "perform poorly" here |
| **EvoMemBench** (2605.18421, NeurIPS 2026) | May 2026 | 2x2: in-episode vs cross-episode, knowledge vs execution memory; 15 methods; long-context "remains highly competitive" |
| **Evo-Memory** (2511.20857) | Nov 2025 | MMLU-Pro/GPQA/AIME/ToolBench as streams; ReMem method |
| **Harness the Memory** (2608.15008, NeurIPS 2026 D&B) | Aug 2026 | 11 memory substrates, 3 backbones, 4 suites, 26 metrics, cost attributed to write/read/manage; "no single substrate consistently dominates" |
| **StreamMemBench** (2606.14571, EMNLP 2026 Findings) | Jun 2026 | EgoLife egocentric streams, initial task then follow-up; metrics: evidence recall, initial use, feedback incorporation, follow-up reuse; systems "fail to turn feedback into reliable follow-up behavior" |
| **MemTrace** (2606.17328) | Jun 2026 | per-fact probing by memory age, question type, evidence condition; evidence was retrievable 10x more often than missing, so failures are usage not retrieval |
| **AMA-Bench** (2602.22769) | Feb 2026 | QA over real agent trajectories; AMA-Agent 57.22% |
| **LifeVal** (2611.06687) | Nov 2026 | see §3 |

### 8c. Parametric / weight-update side (no agentic CL benchmark exists yet)
| Benchmark | Date | Notes |
|---|---|---|
| **ForgetBench** (2607.26455) | Jul 2026, rev Oct 2026 | forgetting of parametric memory under sequential knowledge editing; temporal decay, retention strength, cross-instance stability |
| **Beyond Perplexity** (2607.00368) | Jul 2026 | evaluation framework for TTT "deployment memory" claims; one-step LoRA lowers loss across three Qwen3 scales "while generated free-form recall stays at zero" (the sharp result: loss drops are not memory) |
| **Mapping Post-Training Forgetting** (2510.17776) | Oct 2025 | sample-wise 1→0 / 0→1 transitions across post-training stages; "model merging does not reliably mitigate forgetting" |
| **TiC-LM** (2504.02107, Apple) | Apr 2025 | 114 Common Crawl dumps revealed monthly; replay schedules match retrain-from-scratch loss at 2.6x less compute |
| RL's Razor (2509.04259), sparse memory finetuning (2510.15103) | 2025 | method papers, not benchmarks; the forgetting numbers to cite |

### 8d. Within-episode adaptation (not cross-task)
| **ARC-AGI-3** (2603.24621) | Mar 2026 | interactive 64x64 grid environments; efficiency vs human action baselines; humans 100%, frontier <1% as of March 2026. Tests learning inside one episode; nothing persists. |

### 8e. Surveys worth skimming
"Continual Learning in Transition" (2608.06216): When / How / Where axes, thesis "a transition from parameter-centric learning toward system-level adaptation." "Always-On Agents" (2606.30306): persistent memory, state, governance. "Towards a Systems Foundation for Agentic Skills" (2608.29596): skill lifecycle, lists SkillsBench / SkillLearnBench / SkillFlow.

### What the catalog adds to §5
- Three independent Aug 2026 papers reach the CL-Bench conclusion from different angles: ContinualSkillBench (context+feedback ≈ skill library), Harness the Memory (no substrate dominates), EvoMemBench (long context competitive).
- The failure is usage, not retrieval: MemTrace (evidence present 10x more often than missing), StreamMemBench (feedback not turned into follow-up behavior), LifeVal (33% win rate on conversation-sourced criteria).
- Forgetting metrics (FWT/BWT) from classic CL only appear in PATH-Bench and SWE-Bench-CL; the big agentic benchmarks dropped them in favor of gain-over-stateless.
- On the parametric side, Beyond Perplexity is the paper to cite when someone claims TTT gives agents memory.

## 9. New since September 2026 (arXiv v1 on or after 2026-09-01, verified against abstracts)

The field moved from "does memory help" to three sharper questions in these two months: does the agent keep experience *current* when the environment drifts, does harness/skill self-evolution *forget*, and how much of a reported self-improvement gain is real.

### 9a. Learning over a task stream, frozen weights
| Benchmark | v1 | Setup | Metric | Headline (abstract wording) |
|---|---|---|---|---|
| **CLE** (2610.33956) | Oct 24 | see §3 | levels cleared | 47.7% of levels, 23.3% full ladders, median +5.3 pp from feedback |
| **ServeLearnBench** (2610.07792) | Oct 6 | retail support, banking, sales-pitch; 53 environment windows, 7,718 tasks; hidden policies evolve; 5 harnesses (RAG, Mem0, SkillOpt, Continual Harness, Prime) x 6 models, 252 runs | not in abstract | "substantial gap between task capability and learning from experience"; "continual adaptation is costly and can degrade already-correct behavior" |
| **CLR-Bench** (2610.40758) | Oct 29 | 20 chains x 10 domains; each level adds one hidden rule; practice feedback reveals one member and one non-member; 6 self-evolution methods vs frozen agent | accuracy on unseen members | with the certifying model, best gain ≤5 points over frozen; a fixed lesson lifts the same methods 14.5–16.5; with GPT-6-Sol ACE and Dynamic Cheatsheet reach 78% vs 53% |
| **ReScopeBench** (2611.05450) | Nov 3 | 18 interactive environments, streams built around shared experience with silent environment changes; structural vs rule experience; 20 models | task perf + tracking of saved-experience edits | "retained experience improves task performance, but agents struggle to keep it current as environments change"; correct recording does not guarantee success |
| **AhaBench** (2609.05435) | Jun 30 v1, Sep 21 v2 | three tracks: Puzzle (hidden state), Euler (math teaching and transfer), Vending (year of daily ops, delayed feedback); fixed weights | advantage over matched cold targets, held-out accuracy, profit | trace support adds 36.0–53.5 points; worked procedures give 80–100% held-out; Doubao Seed 2.0 Pro ends +495 nominal but averages −10 over the year |
| **FilingLoop** (2610.37905) | Oct 27 | 380 SEC IPO review tasks, 96 companies, 6,099 requirements; feedback distilled into role-specific guidance across cases, no weight updates | requirement fulfillment, full comment resolution, cost | 24.28% → 27.96% fulfillment, 16.90% → 20.75% full resolution, +5.8% cost |
| **AAArena** (2610.12341) | Oct 8 | 12 adversarial games, 1,920 archived human programs; agents revise executable policies from replays under fixed budgets | medals vs human ladders | Opus 5.5 + Claude Code earns 6 golds; no config tops the other 6 human ladders |
| **EmbodiedCLBench** (2610.13021) | Oct 9 | embodied sim; learn basic tasks, compose for unseen advanced tasks | not in abstract | basic-task experience helps but "does not scale with the number of learning samples" |
| **EdgeBench** (2607.05155, ByteDance Seed) | Jul 6 (pre-window, but the scaling-law framing is the one people cite now) | 134 day-scale tasks, 6 domains, ≥12 h continuous operation each, 51 public, ~38K hours of interaction | log-sigmoid scaling-law fit | R² = 0.998; "agent learning speed roughly doubles every three months" |
| **LatentGym** (2606.15306) | Jun 13 v1, Nov 4 rev | controllable latent variable shared across a task sequence; separates exploration (did actions gather info about the latent) from exploitation | exploration vs exploitation metrics | frontier models fail to adapt across related tasks |
| **VidLearn** (2611.01460) | Oct 31 | learn chess openings from 72 tutorial videos into a growing non-parametric "tutorialbook" | QA accuracy, head-to-head games | +113% train / +91% test relative over vanilla MLLMs |

### 9b. Non-stationarity: does experience stay current?
| **EvoHarnessBench** (2609.04280) | Sep 3 | 17 harness streams, 802 tasks, 520 tools, 42 skills, 62 agents; drift is in the harness, not the tasks | retention + adaptation | "harness expansion alone can degrade performance on previously solved tasks, producing harness-induced forgetting" |
| **Δτ-Bench** (2610.34238) | Oct 25 | 1,588 tasks, 42 domains, six successive tool-ecosystem states; procedural knowledge must be revised | task success | strongest model/harness 42.3%; proposes Steward |
| **SkillShift-Bench** (2610.33528, EMNLP 2026) | Oct 24 | 3 self-hosted web systems with Contextual, Perceptual, Execution shifts applied after skills are learned | success, interaction cost, skill failure | systematic degradation; Execution Shift strongest |
| **RSE-Bench** (2610.19910) | Oct 14 | 4 domains, 6 benchmarks; non-adversarial noise injected into instructions, evidence, execution records, outcome pairings during learning | fraction of clean self-evolution gain retained | noisy learning retains only 25.5–42.1% of the clean gain |

### 9c. Does self-evolution forget, and is the gain real?
| **Can Coding Agents Sustain Continual Evolution without Forgetting?** (2611.03219) | Nov 2 | harness evolution across 3 SE datasets, 6 evolution directions, DeepSeek-V4-Flash | SWE-Bench after sequential evolution | Meta-Harness 94.0% → 32.0% by round five, a 66% relative drop |
| **EvoAudit** (2611.07624) | Nov 5 | 4 evolution methods, 16 evolved agents, general + coding | task coverage gained/lost, reliability, cost | all 16 gain uncovered tasks while losing covered ones; +5.2 reliability while losing 10 tasks |
| **Harness-Delta Attribution** (2610.22466) | Oct 15 | 11 executors, 3 harness-evolution frameworks; split search-set gain into overfitting, test-time scaling, residual | share of gain | overfitting 54% and test-time scaling 14% of gain on average across 21 significant settings |
| **CODA / skill-policy co-evolution trap** (2610.20845) | Oct 14 | 4 methods, 5 benchmarks; simulate skill-bank drift with policy fixed | worst-case-drift success | worst case 5–13 points below peak, "erasing 52–93% of the delivered bank's gain" |
| **AlphaDiana** (2610.20675) | Oct 14 | matched-condition interventions on tools, skills, retained memory; 3 models x 3 harnesses | accuracy, entropy, tokens per correct | "neither skill guidance nor retained history reliably improves accuracy" |
| **Memory Tax** (2610.34073) | Oct 25 | dedicated memory systems vs naive RAG over raw history, multi-benchmark | memory benefit vs memory tax | "many dedicated memory systems underperform naive RAG" |
| **ConsolidationBench** (2609.33013) | Sep 26 | oracle-by-construction scoring of keep/compress/abstract/forget decisions | correlation with transfer accuracy | +22.7% task success at 7x compression; production retrieval systems score zero on cross-level transfer |
| **SCLATE** (2609.32391) | Sep 26 | execution substrate on a simulated clock ("month-long scenario into hours"), 7 benchmarks ported, 10 harness/memory configs x 10 models, plus post-training Qwen3.5-4B | pass rate, file-line reads | trained model reads 6.8x fewer lines with +16.7 pt SWE-bench Verified |

### 9d. Memory benchmarks in the window (recall-oriented, listed for completeness)
PraxisMemBench (2610.20022; personal + procedural memory, Claude Code native memory 20.6% vs PraxisMem 40.8% session success), MemU-Bench (2611.00191; 118 history-dependent mobile GUI tasks, best SPL 16.4%), PDEU-Bench (2609.34930; 214 tasks, preferences fail to propagate through plan lifecycle), Bemula (2610.18052; no system passes >1.5% of forget requests), MemCut (2610.21405; 39–76% residual recovery after deletion), AUBench (2610.17659), PA-Bench (2609.05767), MemRiskBench (2609.14976), ECG-Scroll (2609.33117), ReaLMem (2609.19167), TrackMem/FinLoCoMo (2611.02001), 4DMemory (2610.32332), StuLife v2 (Jan 2026).

### Takeaways from the Sep–Nov wave
1. **Drift is the new axis.** Five benchmarks (ReScopeBench, EvoHarnessBench, Δτ-Bench, SkillShift-Bench, ServeLearnBench) make the environment, tools, or policies change mid-stream. Stale experience is now the headline failure, not missing experience.
2. **Self-evolution gains are being audited and shrinking.** Harness-Delta Attribution puts 68% of reported gain down to overfitting plus test-time compute; CODA, EvoAudit, and 2611.03219 show the gains are fragile or come with forgetting.
3. **Capability threshold again.** CLR-Bench: ≤5 points with a mid model, 78% with GPT-6-Sol. Matches ApprenticeBench and AgentStream.
4. **Feedback quality beats method.** CLR-Bench's single fixed lesson (+14.5–16.5) outperforms every self-evolution method (+5); AhaBench's worked procedures reach 80–100% held-out.
5. **Still nothing parametric except SCLATE's small post-training run.** The agentic CL benchmark for weight updates does not exist yet.

## 10. Can CLE be run locally? (checked 2026-11-07)

Partially. The full 120-task set is not released; only **CLE Lite** (20 tasks x 5 levels = 100 levels, 3,309 rubric criteria) is runnable, and only after HF access approval.

- **Public, no gate:** `Continual-Learning-Exam/CLE-Lite` (task cards: instruction, inputs, outputs, software, file inventory) and `CLE-Lite-Workspaces` (agent-facing files). Enough to read tasks, not to run or grade them.
- **Gated (click "request access" on HF):** `CLE-Lite-Rubrics` (rubrics + reference answers) and `CLE-Lite-Full` (task packages, the mock-app environments, `runtime/cle_lite.py`, `runtime/codex-worker.Dockerfile`, judge code). The GitHub repo itself holds only docs and citation, no code.
- **Harness:** the packaged runtime drives the solver through **Codex only** (`npm i -g @openai/codex@0.160.0`, `codex login --device-auth`). The leaderboard's Claude Code / Hermes / OpenClaw runs used unreleased glue. Judge is DeepSeek V4 Flash via OpenRouter or GPT-6.1-Sol via Codex.
- **Memory across tasks:** the Codex conversation plus a declared `/app/cle-memory/` directory persist; files and app services are fresh per task.
- **Requirements as written:** Linux x86-64, Docker + Compose, Python 3.10+, uv, Node 22+, ≥30 GB disk per example task, "hours" per full-budget run. This Mac is arm64 with Docker 28.4, Node 22.20, uv, and hf 1.1.2 present; Codex CLI is not installed. x86 images under Docker Desktop emulation may work but are slow and unsupported.
- **Run shape:** `cle_lite.py download --task 008-operations-research-analyst` → `setup` → `run --setting cle --trial 1|2` (or `--setting direct`) → `stream --task-list runtime/lite-example-tasks.json` for a two-task stream. Outputs `run.json`, `summary.json`.
- **Revisions to pin:** runtime `26ab77e5085476d59dc8c243b928e5b8ac490ef4`, data `803301ec01b26ab594a5ab3cdcd855a14dcc26e1`.
- **License:** per-task notice; do not train on tasks, do not expose rubrics to the agent, do not redistribute rubrics. No leaderboard submission path; GitHub Issues for reproducibility reports.

## 11. How continual-learning environments are constructed (four routes, read in depth)

Every CL environment has to solve the same five problems: (a) make sure there is something to learn that the model does not already know, (b) make later tasks depend on it, (c) give the agent a way to discover it, (d) make re-discovery cost something, and (e) grade without leaking the answer. The four benchmarks below solve them differently.

### 11a. ReScopeBench: experience-first, coded simulations (HKUST/Huawei, 2611.05450)

**Order of construction is the trick.** Knowledge first, tasks second, changes third.

1. **Environment preparation.** Each of the 18 environments is a simulated application written in code, modeled on public docs of a real system (SAP purchase-to-pay, Kubernetes, REDCap, Oracle Clinical One, railway ticketing...). Entities, workflows, and operations are invented, so nothing is in pretraining. Each exposes 3 to 18 tools (10.1 mean) plus 8 shared experience/history tools. Tool descriptions say how to call; what the data is must be discovered. Each tool call has a declared cost and each environment a budget (e.g. Oracle Clinical: 17 backend work units per task, a context read costs 5, a correction write costs 2).
2. **Experience discovery.** Authors read the environment code and pick facts that (i) support multiple goals and (ii) can be discovered via the agent's tools. Each fact is recorded with its scope, the observations that reveal it, and the decisions that depend on it. This forms a private dependency graph. Facts are grouped into reference units of claims, each claim with a rubric stating what saved content must establish. Two kinds: structural (resource relationships, IDs, locations) and rule (conditions on valid actions).
3. **Joint stream synthesis.** Using the dependency graph, build 16/24/32 ordered tasks that share the facts but vary the requested objects. Schedule 1 to 3 silent changes immediately before chosen tasks; each change alters selected claims and updates their reference versions. Changes are detectable only through tool observations (e.g. a `DELIVERY_POLICY_MISMATCH` error).
4. **Reference solutions and graders.** Authors complete every task in order with the agent's own tools, carrying state and applying changes. Those tool-call sequences become private reference solutions; their artifacts and final states define delivery requirements for a programmatic grader that accepts equivalent outcomes. Budgets are set from the cost of the reference solution plus local correction, below the cost of exhaustive rediscovery.
5. **Verification.** Replay reference solutions from the initial state, confirm expected scores, confirm that update-revealing call sequences fit the budget, inspect for leaked answers, then freeze.

**Worked example (Filesystem, Riverlight Ledger).** Task 4: agent learns workspace `workspace-g9c-k4` + manifest ID + lane `amber` at `node-lane-k2d8s4` + routing token + policy `overwrite_existing false` and saves it. Before task 7 the amber lane node is silently replaced. At task 7 the old call fails with `DELIVERY_POLICY_MISMATCH`; GPT-5.5 saves a warning. At task 9 it calls `list_delivery_lanes` (new handle) and `get_delivery_policy` (unchanged), rewrites the binding keeping the five valid fields, and reuses it for tasks 11 to 14 and 16 with zero further lookups. The experience judge scores the claim 0 at T07, 1 at T09; the task grader gives 25/25 on that request.

**Second example (Oracle Clinical One, T10).** All four models have the correct bindings saved. Budget 17 units; four corrections cost 8. GPT-5.4 spends 15 on three context reads and completes one correction; GPT-5.5 and Opus 5.5 write all four for 8. Same memory, different outcome: the budget is what makes "using" experience distinct from "having" it.

### 11b. CLE: expert-authored ladders over mock apps (UCL, 2610.33956)

**Order: expert workflow first, then review gates.**

1. **Expert design.** A domain expert drafts a five-level ladder from a workflow in their own practice: role and working context, authentic input files (real public data with its noise, duplicates, conflicts), expected deliverable per level, and a checklist rubric per level. Rubric item format: concrete target, expected outcome, evidence required; one unambiguous pass/fail each; items grouped into categories for diagnostic feedback. Mean items per level rise 18.7 → 36.8 from L1 to L5; 16,125 total.
2. **Linking levels.** Later levels reuse earlier work via Tools & Skills, Data Dependency, Strategy, User Context, Domain Knowledge. 88 of 120 ladders reference earlier-level files. The agent gets `workspace/t1..tN/` inputs plus its own `tK/results/`.
3. **Mock applications.** 19 stateful apps with 1,024 APIs "covering software used by the represented roles," reached through tool calls via the benchmark API. Each app has its own database, initialised per run, shared across the five levels of one ladder, reset between tasks. How the apps were built is not described.
4. **Admission.** A review agent with a "review skill" audits the full ladder (format, file authenticity, answer leakage, rubric verifiability, CL design) and runs it end to end as an actor, returning a conference-style verdict (Strong Accept / Accept / Borderline / Major Revision / Reject) with evidence-grounded findings. Accept or better goes to two random human reviewers; both must accept. Rejections return to the author with feedback; loop until accepted. Acceptance rates not reported.
5. **Grading.** Evaluation agent (DeepSeek V4 Flash) checks submitted artifacts plus an app-state snapshot against the private rubric and human-verified reference materials, in a fresh container. Agreement with human labels 96.86% on 732 items; 99.46% self-consistency.

**What is weak:** no stateless control, no exploration budget, no experience grading, and the whole app layer is unverifiable from public material (Lite deliverables are all files).

### 11c. CL-Bench: latent structure in real data (Berkeley/Snorkel, 2606.05661)

**Order: admission criteria first, then one bespoke environment per domain.**

1. **Three admission criteria.** Headroom (initial performance far below max, structure not recoverable from offline training, otherwise "a stronger base model and a learning system are indistinguishable"); Shared latent structure (discoverable, shared across instances, never stated); Learning mechanism (a feedback loop such as test failures or error messages that a learner can turn into knowledge). These "jointly preclude" chaining existing benchmark instances.
2. **Expert review.** Two authors review against the criteria, then 2 to 3 domain experts per task score realism, reusable knowledge, and learning improvement, 2.14 h per cycle, iterated.
3. **Per-environment mechanics**, each a different way to hide structure:
   - *Database Exploration*: real Amazon reviews, 150K products, 10 SQLite tables. Column names abbreviated (`prc`, `ts`, `vrf`), units differ per group (integer cents vs float dollars), tables present/absent per group, distractor columns populated for 30% of rows. 40 expert-written questions, 15-query budget each, reward 1 − queries/15 if correct. After Q20 a live migration adds `prc_v2`, partially populates a new column, splits timestamps, renames a table `_legacy`. Latent structure = the schema conventions.
   - *Exploitable Poker*: heads-up Hold'em, three deterministic rule-based opponents (calling station, fit-or-fold, loose-aggressive), hands from a seeded PRNG, stacks reset each hand. 120 hands in five stages with opponents returning (CS 1–20, LAG 21–50, CS 51–60, FoF 61–95, LAG 96–120) to test retention. Reward = bb/hand; r_max 9.49 from an oracle that knows each opponent's policy.
   - *Blind Spectrum Monitoring*: 90 scans of a 168 MHz band, three stages where more of a fixed 13-channel grid becomes active; "the drift is epistemic rather than real." Reward = IoU vs ground truth. Latent structure = the channel grid, to be carried through silence.
   - *Cohort Studies*: five fictional-disease studies, four instances each, per-study coding conventions and sampling bias, no cross-study dictionary, 20 tool calls, reward = information gain over flat baseline, no score feedback between instances.
   - *Codebase Adaptation*: 19 real SWE-bench issues from two repos in chronological order, fresh container per issue, hidden regression tests, reward 1 − (steps−1)/40.
   - *Sales Prediction*: fictional retailer, 12 yearly instances, visible store rotates, distractor columns added per stage, workspace persists so code and models carry over.
4. **Control.** Same system run stateless on each instance; gain per instance; 5 stateful rollouts with permuted order within variant.

**What is weak:** no sandboxing described, tens of instances, wide CIs.

### 11d. MetaClaw: authored workday scenario (UNC/UCSC, 2603.17187), the SCLATE training target

1. **Scenario.** Part I: 30 simulated workdays, 346 rounds (10 to 15 per day), two round types. `multi_choice`: procedural questions on domain rules, scored max(0, 1 − (FP+FN)/n_options). `file_check`: structured edits producing files, graded by a checker script (exit 0 = pass). Difficulty rises with day index; workspace persists within a day. Part II: 14 days, 588 questions, a persona (Orion Tech backend lead) via IDENTITY.md and SOUL.md.
2. **Hidden rules.** Five procedural conventions never stated in the system prompt, each with a checker, introduced by failure: ISO timestamps with +08:00 (day 1), `YYYYMMDD_description.ext` naming (day 4), metadata fields (day 6), `.bak` before modify (day 8), append to `done.log` (day 10).
3. **Learning signal.** Each round carries the previous round's evaluation outcome as corrective feedback, e.g. "Time/date fields must use ISO 8601 with +08:00 timezone."
4. **Interface.** One tool, `run_command`, one command per turn, no clarifying questions. Tasks are "authored to simulate realistic deployment," method unspecified.
5. **What SCLATE adds.** Replaces prompt-injected dates with a simulated clock so unmodified harnesses and their crons run on scenario time; runs 40 calendar days in 5.9 h; captures every model call for post-training; privilege-separates the agent user from the grader and ground truth.

### 11e. The common skeleton

| Step | ReScopeBench | CLE | CL-Bench | MetaClaw / ApprenticeBench |
|---|---|---|---|---|
| Source of "unfamiliarity" | invented entities in coded sims | real data but real-world specific firm context | obfuscation, fictional entities, deterministic opponents | invented company conventions |
| What is learnable | pre-selected claims with rubrics | implicit in ladder design | latent structure chosen per task | hidden procedural rules / cost codes |
| Dependency guarantee | dependency graph, tasks built from it | expert judgment + review agent | admission criteria + expert review | rules introduced on schedule |
| Discovery channel | tool calls with costs | files + app APIs | queries, errors, tests, scores | failure feedback / mentor feedback |
| Re-discovery cost | explicit per-tool budget | none | query/step budgets in 3 of 6 | none / token cost only |
| Reference solution | authors play the stream with agent tools | human-verified reference materials | oracle policies, ground truth | checker scripts |
| Grader | programmatic on artifacts + LLM on memory | LLM on artifacts + app snapshot | programmatic reward | programmatic |
| Leak control | private references, frozen after verification | review agent checks leakage, fresh grading container | hidden tests only | SCLATE user separation |
| Change | silent, scheduled, claim-level blast radius | none | variant switches, one migration | none |

### 11f. Design choices to be able to argue

- **Invent vs obfuscate vs use real.** Invented (ReScopeBench) gives clean isolation but a thin world. Obfuscated real (CL-Bench DB) keeps realistic messiness. Real (CLE, ApprenticeBench) is most ecologically valid but mixes prior knowledge with learning, so you need a novice ablation.
- **Who writes the dependency.** Authors with a graph (ReScopeBench) or experts with reviewers (CLE). The first guarantees reuse; the second guarantees realism. Nobody has both.
- **Budget or not.** Without a budget, "learning" and "re-exploring every time" are indistinguishable. Only ReScopeBench and parts of CL-Bench charge for lookups.
- **Grade the memory or not.** Only ReScopeBench writes an answer key for what should be in memory. It costs a private claim inventory per environment version, but it is the only way to see stale notes.
- **Reference solutions by humans playing the stream** (ReScopeBench) double as budget calibration and leak detection. This is the cheapest high-value step.
- **Freeze then evaluate.** All four freeze tasks, tools, graders before any model run.

## 12. SCLATE notes (Apple, arXiv 2609.32391, Oct 2026)

**Mental model, two ideas**

(i) **Time becomes a simulated quantity instead of a turn counter, so memory is exercised the way it is in deployment.** Facts are written on one day, survive restarts and consolidation, and must be retrieved on a later day. The same recall dataset moves from "find it in a 100K context" to "did your memory pipeline keep it and surface it." Done with a hybrid clock: flows at wall-clock while any process is busy, jumps to the next scheduled event when everything is idle. `libfaketime` (LD_PRELOAD) makes every harness and memory process read the simulated clock, so crons and nightly jobs fire on scenario time without code changes.

(ii) **The agent is an event source, not only a responder.** Session start/stop, crons, memory consolidation, state save/restore register on the same timeline as benchmark events. A personal-assistant agent that idles most of the day and occasionally schedules something is representable; in a turn-based harness it is not.

Plus the training plumbing: snapshot/restore of agent + memory state across containers, and an in-container proxy that captures tokens and log-probs per call and tags which component made the call (agent turn vs memory extractor). Neither piece is new on its own; the combination is what lets a multi-session run double as training data.

**Results that bear on (i) and (ii)**

- Compression: 40-calendar-day MetaClaw arc in a median 5.9 real hours (162x); 70 jumps skip 39.5 simulated days in 22 real seconds. GAIA-2 Time scenarios up to 4.3x.
- Fidelity: all 13,050 GAIA-2 events delivered in order, median 0.9 s (max 2.0 s) from schedule; 98.7% of checkable events at the scenario's specified delay. SCLATE vs upstream GAIA-2 disagree as often as two upstream reruns (same agent, same judge).
- Simulated time reaches the harness: Claude Code's system-prompt date, shell `date`, Mem0 record timestamps, Hindsight consolidation logs, GBrain "dream" at sim 02:00 all agree (Fig. 8).
- Memory benchmarks under session cuts: LongMemEval-s Claude Code native 85/91 vs Hermes native 66/51 (size-capped memory drops facts); Hermes + Hindsight 77/74. PersonaMem-128k roughly flat 52 to 59.
- GAIA-2 Adaptability and Time categories, the ones that depend on events arriving mid-run, stay lowest in every cell (≤37.5%, ≤34.4%).
- MetaClaw grid (10 models x 10 configs): every memory config beats no-memory; added memory loses to native in 11/50 cells (8 under Hermes); Gemini-3.7F leads (79–82%) and is the only model that searches prior session transcripts everywhere (7.6% of calls).
- Post-training a 4B student through unmodified harnesses: held-out days +7.6 to +11.8 pp in 5/6 configs (3 significant); memory writes shift to procedural rules (23.6 → 40.6%) and purpose clauses (1.1 → 19.6%). Caveat: teacher sees the benchmark's corrective feedback.

**Definitions and scope.** A continual-learning agent, per the paper, is the triple memory layer + harness + model training; the grid varies all three. The two recall benchmarks (LongMemEval, PersonaMem) stay recall tests after the session cuts: the memory layer is exercised, but the agent reproduces a fact rather than improving at a task. All grading is outcome-only (final answer or final state); the memory-write analysis in §4.3 is descriptive, not scored. ReScopeBench is the only benchmark that grades memory contents against an answer key (revision vs retention). SCLATE's timeline plus ReScopeBench's claim rubrics is the obvious combination nobody has built.

**What is not shown**: no ablation isolating agent-scheduled events (e.g. consolidation on vs off) on task score; no human baseline; no SCLATE code URL in the paper; one run per cell.

## 13. MemPABench notes (NYU Abu Dhabi, arXiv 2610.30485, Oct 21 2026) — the personalization benchmark

**What it measures.** Whether a personal agent's memory learns *interaction preferences* (how the user wants to be helped: tone, verbosity, reasoning visibility, autonomy level, proactive outreach, how to elicit missing info; 14 attributes in 4 dimensions) rather than content facts. Preferences are a state indexed by (user, context ∈ {work, personal}, attribute, time): 28 cells per user, 112 total, 18 of 56 user–attribute pairs differ between contexts, 20 cells change once via scheduled events.

**Construction.**
- Four users grounded in sitcom characters (Big Bang Theory, The Office): preferences extracted from transcripts, verified against cited scenes, then anonymized (names, workplaces, places swapped). Check: a backbone guessing the 28 settings gets 53.2% from the real identity, 46.4% anonymized, vs 45.2% chance.
- Each user has a four-month world with 83 executable tools across 26 domains (calendar, messages, documents). 99 sessions: 94 live accumulation sessions + 5 pre-written preference-change events replayed through the same interface so every memory condition experiences the same change. Each stable cell appears in 3 sessions; each changing cell in 3 before and 2 after.
- Each session targets one cell and has a built-in decision point: the task cannot be finished without choosing a setting for that attribute, every setting is feasible and observably different, and the user never names the setting, only reacts.
- User simulator (Gemini-3.8-Flash) keeps its own cross-session memory of impressions ("tends to over-explain"), so reactions escalate like a person's. Rule checks + a Gemma judge block any message that names a setting; post-hoc leakage audit found none.
- PA runs on nanobot; at a decision point it must call an interaction tool (e.g. `IX_autonomy_level`) with one setting; an IX Wrapper enforces that setting mechanically (hold actions for approval, hide rationale, suppress outreach) without seeing the target. This makes selection directly observable.
- Probes: 33 per user, run on isolated branches from frozen checkpoints with world + memory but no raw conversation, 3 repeats. 28 final probes cover every cell; 5 pre-event probes test changing cells just before their change.

**Metrics, by stage.**
- Selection: PSA (exact match on final probes), CS (both work and personal settings right for pairs that differ), PET (right before and after the change).
- Realization: BRS, LLM judge 1–5 on the delivered reply and executed actions vs the target definition; PRS, the user simulator's own 1–5 rating.
- Reference conditions: No Memory (backbone defaults) as floor; Oracle (target setting injected in place of memory) as ceiling that removes acquisition and retrieval, isolating realization.

**Results (4 backbones x 4 memory systems: File memory, Simple RAG, Mem0, MemOS).**
- PSA: No Memory 49.6% (chance 45.2%). Mem0 50.0, MemOS 50.0, RAG 49.9. File memory 59.2 (+9.6, CI [+3.9, +15.6]). Only the "everything in the system prompt" memory helps.
- Why retrieval fails: in 334 judged probes, retrieval-based inputs contained an applicable interaction preference 7 times; File memory 54%. Retrieval returns content (what was done, with whom) and leaves the user's reactions in the store. Case study: all four stores held the user's objection to an unsolicited reminder; only File memory showed it to the PA. Bottleneck is retrieval, not storage.
- CS ~20% and PET ~11–25% everywhere. Most wrong selections match the backbone's default, including all 8 wrong selections under Oracle. Memory has to override a strong interaction prior.
- Information Flow and Initiative & Autonomy (preferences about actions) are hardest, 34–44% PSA; Expression Style and Disclosure (preferences about the reply) 55–69%.
- Under Oracle, five settings (Hide rationale, Terse, Iterative/Structured elicitation, Low breadth) pass BRS≥4 in only 60% of correctly selected probes vs 95% for the rest. The model keeps adding "Why that day:" sections after choosing Hide. Saying is easier than doing.

**Mental model.** Course and exams. The 99-session timeline is the course: the agent interacts, its memory fills, nothing is graded. A probe is an exam: snapshot memory and world, run one test session on the snapshot, record the pick and the reply, delete the snapshot. The real memory never sees the exam, so 33 exams do not become 33 extra lessons and the 28 final exams do not contaminate each other. Learn on the timeline, test on disposable copies.

**Vocabulary.** An *interaction attribute* is one knob on the agent's behavior (14 of them: verbosity, autonomy level, proactive outreach, ...), each with 2 or 3 *settings* (Terse/Detailed; Reactive/Self-directed/Autonomous; Low/High). A *cell* is one attribute in one context, work or personal; 14 x 2 = 28 cells per user. A cell is *stable* if the user's setting never changes over the four months (23 per user) or *changing* if it flips once at a scripted event (5 per user). The agent never sees the table; it must infer each cell from the user's reactions. At a *decision point* the agent must *pick* a setting by calling a tool such as `IX_verbosity("Terse")`, then write its reply; a wrapper enforces the pick where it can (hold actions, hide a rationale field, suppress outreach). The pick is scored against the table; the delivered reply is scored by a judge and by the simulated user.

**Session types, per user**

| Session type | Count | Purpose | Scripted or live | Scored |
|---|---|---|---|---|
| Accumulation, stable cell | 69 (23 x 3) | agent learns the cell from reactions | live, on-policy | no |
| Accumulation, changing cell, before event | 15 (5 x 3) | agent learns the old setting | live | no |
| Preference-change event | 5 | a consequence makes the user revise one cell | scripted, replayed identically for every memory system | no |
| Accumulation, changing cell, after event | 10 (5 x 2) | new setting reinforced | live | no |
| Pre-event probe | 5 | did the agent hold the old setting just before the change | isolated branch, 3 repeats | PET (first half) |
| Final probe | 28 | does the agent hold the current setting at the end | isolated branch, 3 repeats | PSA, CS, PET (second half), BRS, PRS |

Live = the simulator generates the user's lines at run time in reaction to the agent. On-policy = the history the agent later relies on came from its own actions. Time is narrative (sessions carry storyline dates) but execution is turn-based: no simulated clock, nothing fires between sessions. A probe runs on a frozen copy of world + memory (no raw conversation), and the copy is discarded, so probes neither teach the agent nor affect each other.

**The five metrics on a worked example.** One user, four cells, end-of-timeline probes:

| Cell | User wants | Pick | Pick right? | Reply the user saw | BRS (judge) | PRS (user) |
|---|---|---|---|---|---|---|
| work, outreach | Low | High | no | pinged the user about an unrequested event | 1 | 1 |
| work, verbosity | Terse | Terse | yes | answer, then three paragraphs of reasons | 2 | 2 |
| personal, verbosity | Detailed | Detailed | yes | answer with background and a caveat | 5 | 4 |
| personal, autonomy | Reactive (after the change) | Self-directed | no | sent the rebooking message without asking | 1 | 1 |

- **PSA**, preference selection accuracy: picks right over all cells. 2/4 = 50%.
- **CS**, context sensitivity: only attributes whose work and personal settings differ (here verbosity). A pair scores 1 only if both contexts are picked right. 1/1. Had the agent picked Detailed at work too, 0/1: it failed to keep contexts apart.
- **PET**, preference evolution tracking: only changed cells (here personal autonomy), probed before the change and at the end. Scores 1 only if the old setting was picked before and the new one after. Before: wanted Self-directed, picked Self-directed. End: wanted Reactive, picked Self-directed. 0/1: the agent kept the old setting after the user changed it.
- **BRS**, behavioral realization: a judge reads the delivered reply and executed actions against the definition of the user's target setting, 1 to 5, without seeing the pick. Row 2 is the split that matters: pick right, reply wrong, BRS 2.
- **PRS**, perceived realization: the simulated user, memory cleared, rates the same reply 1 to 5 as "did this match what I wanted." Row 3: judge gives 5 for meeting the definition, the user gives 4 because this persona is hard to please. BRS applies one rubric to everyone; PRS carries the persona.

Three scores on the pick (PSA, and its stricter cuts CS and PET), two on the reply (BRS, PRS). Why force a pick at all: without it a wrong reply cannot be split into "did not know the preference" (memory problem) and "knew it, could not do it" (model habit); and settings like "hold this action for approval" need a runtime signal to enforce. The cost is that no real assistant announces its setting.

**Personas.** Four, from the first three seasons of The Big Bang Theory (A, B) and The Office (C, D). A two-pass LLM extraction tags transcript scenes with an attribute and context, then decides a setting per cell with a confidence level; reactions to others' help outweigh the character's own speech style. Human review checks every setting against its cited scenes: of 112 cells, 53 confirmed at high confidence, 34 reviewed and kept, 23 reset by the reviewer, 2 set by hand (User A is 27/28 transcript-derived; User D is mostly reviewer judgment). Names, workplaces and places are then swapped by an approved substitution table; a reader check confirms the anonymized persona reacts to preference violations the same way (14/14), and a shortcut check shows a backbone guessing the 28 cells drops from 53.2% (real identity) to 46.4% (anonymized), chance 45.2%.

**TLDR.** Memory systems built for facts do not learn manners, and models do not follow manners they are given. Four simulated users with 28 hidden behavior settings each live with an agent for 99 sessions; the user only reacts, never states a setting, and five settings flip after a scripted incident. Retrieval memories score at the no-memory level (~50% vs 45% chance) because they retrieve events and leave reactions in the store; prompt-dumped memory gets 59%; context separation and change tracking are 10–25% for everyone; and with the right setting handed over, default habits still win 40% of the time on the hard settings. Task completion is never scored.

**Why it belongs in the survey.** It is the only benchmark here with (a) on-policy history (the agent's own actions shape what it later has to remember), (b) a context axis (same preference, different setting at work vs home), (c) scheduled preference changes replayed identically across conditions, (d) a three-stage failure decomposition with floor and ceiling controls, and (e) selection made observable through a forced tool call. Weaknesses: 4 users, sitcom personas, a fixed harness (nanobot), LLM-simulated user and judge, no memory-content grading beyond case studies.

**What is missing.**
- *Scale and variance.* Four users, three repeats, one harness. CS and PET are computed over 18 pairs and 20 cells; a few probes move them several points.
- *The learning signal is one-sided.* Every signal is a reaction after the fact. Real users also state preferences outright, tolerate violations silently, and contradict themselves. A tolerated-violation condition, where the user says nothing but PRS drops, would separate "no signal" from "signal ignored."
- *No memory-content grading.* Finding 3 (stored but not retrieved) is a case study on two probes. ReScopeBench-style claim rubrics over the memory store would make it a metric and would show whether the preference was never written, written but lost, or retrieved but ignored.
- *Task completion is not scored.* Tasks exist only to create decision points; no metric checks whether the dinner was rebooked. So the tradeoff between behaving as the user wants and getting the job done is invisible: an agent that always holds for approval scores well on a Reactive user and would never finish a task that needed a decision.
- *No cost of asking.* Reactive is always feasible, so there is no price for over-asking except the user's irritation score. A budget on clarifying turns, or a task that fails if the agent waits for approval that never comes, would make autonomy a real tradeoff.
- *Time is narrative, not simulated.* Nothing happens between sessions, so proactive outreach is tested only at decision points inside tasks. The attribute is about unprompted contact; a SCLATE-style clock with idle gaps and agent-side crons is where it would be tested for real. Also, no context arrives between sessions. Real memory is mostly unrelated traffic with preference evidence buried in it, so retrieval here is easier than deployed (compare LifeVal's 33% on conversation-sourced criteria). And real preference changes often show up only as a shift in ambient context, a new manager, a busier month, with no incident and no complaint; here every change comes with a scripted incident and an explicit objection.
- *The forced pick is unnatural.* No deployed assistant announces its setting. A version that infers the setting from behavior alone (judge-only) would show how much the explicit pick helps or hurts.
- *Simulated everything.* User, judge, and persona source are all LLM-mediated; the shortcut check and second judge are good, but there is no human rating of any delivered interaction.
- *Stable cells only ever reinforce.* Three identical lessons per cell. No noise, no conflicting evidence, no preference that depends on a third variable (who the message is to, how urgent). The contextual axis has two values; real life has more.

## 14. Taxonomy, built from the five benchmarks read in full

Five anchors: **ApprenticeBench** (AB), **CLE**, **ReScopeBench** (RSB), **SCLATE** (substrate, with MetaClaw as its workload), **MemPABench** (MPB). CL-Bench (CLB) is included where it is the cleanest instance of a dimension.

### 14a. The stage model every benchmark is a slice of

An agent that learns on the job passes through five stages. Each benchmark instruments some and ignores others.

```
acquire  ->  store  ->  retrieve / select  ->  realize / act  ->  revise on change
```

| Stage | What can go wrong | Who measures it |
|---|---|---|
| Acquire | the signal is never turned into a note (rules inferred from errors, preferences from reactions) | RSB rule coverage (RC); MPB diagnostic cases; AB "never changed its mind" |
| Store | note is written but lossy, capped, or junk | RSB SC/RC against a claim inventory; SCLATE capped Hermes memory on LongMemEval |
| Retrieve / select | note exists but is not surfaced at decision time | MPB Finding 3 (stored yet not retrieved), PSA; LifeVal 33% on conversation-sourced criteria |
| Realize / act | right knowledge, wrong behavior | MPB BRS under Oracle (60% vs 95%); RSB "correct experience, failed request"; RSB budget case (GPT-5.4 spends budget re-reading) |
| Revise | environment or user changed, note is stale | RSB UC/RT (37.6% best); MPB PET (~11–25%); CLB variant switches |

Reading a result against this table is the fastest way to say what a headline number does and does not show. CLE's Δ sits across all five stages at once; AB's 72% is outcome-only; MPB and RSB are the two that split stages with controls.

### 14b. Dimension matrix

| Dimension | AB | CLE | RSB | SCLATE/MetaClaw | MPB | CLB |
|---|---|---|---|---|---|---|
| **Object of learning** | firm rules, cost codes, vendor quirks | prior artifacts, tools, strategy, domain rules | structural facts + rules of a system | procedural conventions (MetaClaw) | user interaction preferences, per context | hidden latent structure (schema, opponents, dynamics) |
| **Source of experience** | own interaction + history dump + mentor | own interaction over files/apps | own interaction via tools | own interaction across sessions | own interaction, on-policy by design | own interaction |
| **Learning signal** | dense human-style feedback month 1, sparse after | rubric-category diagnostics on failed submit (K=2) | tool errors and outcomes only | previous round's corrective feedback | user reactions only, never stated | native reward (tests, IoU, chips) |
| **Where learning lives** | Markdown notes in harness | harness memory/skills (unspecified) | memory file (structural) + skills (rules), Hermes | any harness memory, or weights via OPD/GRPO | File memory / RAG / Mem0 / MemOS, nanobot | context, notepad, Mem0, ACE, Claude Code |
| **Time model** | 100-bill sequence after 6-month history | 5-level ladder, one fixed order of 120 | 16–32 task stream per env | simulated calendar, sessions, crons, idle jumps | 4-month calendar, 99 sessions, probes at checkpoints | tens of instances, order permuted |
| **Non-stationarity** | one policy notice | none | 26 silent changes, 6 mechanisms, claim-level blast radius | none in MetaClaw | 5 preference-change events per user, replayed identically | variant switches, one schema migration |
| **Context dependence** | none | none | none | none | work vs personal, same attribute | none |
| **Autonomy level tested** | act on every bill, no approval loop | act on every level, no approval loop | act; permission changes alter who may authorize an op | act (MetaClaw); harness crons act unprompted | explicit: Autonomy Level, Proactive Outreach, Capability Boundary settings; wrapper holds actions for approval; user sim rates unwanted outreach | act |
| **Re-discovery cost** | tokens only | none | explicit per-tool budget | none | none | query/step budgets in 3 of 6 |
| **Control for "learning"** | novice ablation, one model | none (Δ confounded) | paired stateless stream | no-memory column | No Memory floor + Oracle ceiling | same-system stateless per instance |
| **What is graded** | bill outcome | files + app-state snapshot | artifacts + final state (code) **and memory contents (LLM vs claim rubrics)** | final answer / file checker | selected setting (tool call) + delivered behavior (judge) + user rating | reward |
| **Stage decomposition** | novice / history / feedback | none | acquire, retain, revise, use | none | acquire, select, realize | stability vs plasticity |
| **Realism** | real ERP, simulated firm, real-looking traps | real public data, expert workflows | coded sims of real app types, invented entities | authored workday rounds | persona-grounded sims, 83 tools | real data obfuscated, real repos, sims |
| **Human baseline** | yes, n=2 | no | no | no | no (user sim rates) | no |
| **Harness as variable** | Claude Code / Codex | 4 harnesses | fixed (Hermes) | 3 harnesses x 3 memories x 10 models | fixed (nanobot), 4 memories | 6 systems |
| **Weights updated** | no | no | no | yes (4B, OPD and GRPO) | no | no |
| **Runs** | 1 | 1 | 1 per condition | 1 per cell | 3 probe repeats | 5 |
| **Judge validation** | none | 96.9% vs humans on 732 items | judge-vs-judge 99% | n/a | second judge on 418 probes | n/a |
| **Public** | no | Lite, gated | yes | paper only | yes | yes |

### 14c. Axes that separate the field, in order of how much they change the conclusion

1. **Is there a stateless or no-memory control?** Without it, a score is capability plus learning. AB (partial), RSB, MPB, CLB have one; CLE does not.
2. **Is the environment non-stationary?** The 2026 result is that retention is easy and revision is hard. Only RSB, MPB, CLB test it.
3. **Is the stored experience itself graded?** Only RSB. Everyone else infers from outcomes.
4. **Is the learning signal explicit or implicit?** Explicit feedback (CLE diagnostics, AB mentor, MetaClaw corrections) moves scores by 5–25 points. Implicit (RSB errors, MPB reactions) is where retrieval collapses to chance.
5. **Is re-discovery charged?** Only RSB. Without a budget, "learned" and "looked it up again" are indistinguishable.
6. **Is the harness held fixed or varied?** SCLATE shows no stable ranking of harness, memory, or model; CLE shows +5.6 pp from harness alone. Single-harness results (RSB, MPB) carry an unmeasured harness term.
7. **On-policy or replayed history?** MPB and SCLATE insist the agent's own actions shape what it must later remember. Static-log benchmarks (LongMemEval) test retrieval, not learning.
8. **Does anything update weights?** Only SCLATE.
9. **Is autonomy a variable or a constant?** Every benchmark except MemPABench fixes the agent at "act on everything." MemPABench makes when-to-act a learned, per-context preference (confirm before messaging family, act freely on work tasks, suppress outreach after an objection) and finds it the hardest class: preferences about actions select at 34–44% vs 55–69% for preferences about replies, and below chance with no memory. ReScopeBench touches it from the environment side with permission changes (who may authorize a dispatch). For personal agents this axis is the product: an agent that learns the user's rules but not the user's autonomy boundary is the one that sends the message it should have held.

### 14d. The one-paragraph version

Continual learning for agents in 2026 means a frozen model plus a harness plus a memory, run over a sequence of tasks whose later parts depend on earlier ones. The benchmarks differ in what the dependency is (firm rules, system structure, user preferences, prior artifacts), how the agent finds out (explicit feedback vs errors vs reactions), whether the world moves under it, and which stage of acquire–store–retrieve–realize–revise is instrumented. Across all of them the same three findings recur: plain context competes with dedicated memory, explicit feedback beats self-reflection, and stale notes are the dominant failure once the environment changes. The open cells are a benchmark that grades memory contents on a simulated calendar with weight updates allowed, and a human baseline anywhere other than ApprenticeBench.

## 15. ServeLearnBench notes (CMU / Amazon, arXiv 2610.07792, Oct 6 2026)

**TLDR.** Agents serve tickets in retail, banking, and sales-pitch environments whose rules are hidden, never stated, and change every window. Feedback is a score only. Oracle (rule disclosed) scores 95.4 on the hidden-dependent tasks and Blind (no memory) 14.1, so the tasks are easy once the rule is known; the best learning harness reaches 61.7, the median model-harness pair on the hardest retail tier 22.0. Learning costs 4 to 101x following a disclosed rule and damages tasks that needed no learning. Dropping a withdrawn rule is as slow as learning it. How many distinct answers a harness tries on a new rule predicts its reward with rank correlation 1.0.

**Construction (evolving-environment streaming dataset, EESD).**
- A stream is a chronological sequence of windows. A latent state (the active rule set) is fixed within a window and may change at boundaries: rules introduced, revised, withdrawn, or reactivated. Nothing about the state or its changes is disclosed.
- Each window = serving tasks with score feedback, then held-out test tasks with no feedback, covering the same active rules on disjoint instances. Test tasks run from the frozen post-serving state and are discarded (course-and-exam, as in MemPABench §13).
- Each task is tagged Hidden-Dependent (visible information is insufficient) or Fully Specified (the manual suffices). The second tag measures collateral damage from adaptation.
- Domains. Retail: cancel/modify/return/exchange against an isolated database; hidden categorical rules over payment form, product category, destination, refund method, order composition ("block exchange when destination is Florida", "permit return of a clearance item"). Banking: authorize transactions, review credit-limit increases, screen transfers; hidden numeric thresholds and scopes ("electronics instant-approval ceiling $650, later $460", "overseas-merchant restriction applies to jewelry, later electronics"). Pitch: 40–80 word pitch from a product sheet; hidden state is one of six buyer profiles (value, identity, anti-marketing, risk-averse, eco, conformist).
- Tiers. L1 acquisition: new rules appear. L2 revision and composition: rules tightened, relaxed, re-scoped; multi-item and ranked tasks. L3 non-monotonic: removal, reversal, re-entry. Retail uses 7/14/12 atomic rules per tier, Banking 5/9/11 rule values; the appendix prints the full rule dictionaries and window-by-window activation matrices.
- Scale: 9 scenarios, 53 windows, 4,508 serving and 3,210 test tasks.
- Grading. Retail and Banking: SHA-256 of the canonicalized terminal database against a reference produced by replaying ground-truth actions, AND one structured finish action (refusal with an accepted reason code, or a typed value). No LLM. Pitch: a GLM-5.3 Flash judge returns per-attribute verdicts (mentioned / contradicted / absent) plus a list of unsupported claims; deterministic code maps that to a score with multiplicative penalties (0.5 per negative attribute mentioned, 0.3 per red-line, 0.5 per contradiction, 0.5 per fabricated claim, capped at 5).
- Learner-visible: task input, own trajectory, scalar score, monotonic timestamp, own state. Never: rule text, corrections, slice labels, window boundaries in prompts (though SkillOpt's scheduler uses window segmentation for replay).
- Controls: Blind (no retained experience) and Oracle (active rule handed over per window) bracket the learnable headroom.

**Harnesses, not matched by design.** RAG (BM25 over served cases with scores, top 8), Mem0 (memories + a post-outcome memory turn), SkillOpt (prompt optimizer, batches of 12, strict acceptance gate, replays served tasks), Continual Harness (acting agent never sees the score; a Refiner revises prompt, memories, skills, subagents every 25 then 100 steps), Prime (its own code-interpreter runtime, multiple calls per turn, writes its own memory/skill/prompt files after each score). The paper says outright that aligning update cadence, state budget or compute "would change the methods themselves," so Prime vs CH includes the runtime.

**Results (Table 2, 6 models x 5 harnesses, 28 of 30 pairs, one run each).**
- Hidden average across the four open-weight models: Prime 61.7, CH 57.6, RAG/Mem0/SkillOpt 38.3–43.4. CH beats Prime on GLM-5.3 (69.8 vs 63.7); Prime leads elsewhere.
- Fully Specified damage: every harness loses vs Blind in Retail (98.9 vs 89.5–95.5); all but Prime and Mem0 lose in Banking. GLM-5.3 Flash + CH: Hidden 14.6 → 49.7, Fully Specified 96.7 → 82.8. Prime: Hidden 61.1 with Fully Specified 95.2 retained.
- Cost: Prime $0.195/task for 59.0 reward, CH $0.083 for 55.1; RAG/Mem0/SkillOpt cheaper and 13–20 points behind. Kimi K3 + Prime total $2,479 vs Oracle $102.
- Learning speed (AULC over first 10 tickets of a new rule): Prime 54.9 and CH 49.9 in Banking, 37.7 and 36.2 in Retail; SkillOpt 18.2 and 11.1. After five examples of a new banking rule Prime and CH are near 60% correct.
- Dropping a rule (31 rule withdrawals, pairs where the harness had learned it): same curve shape as learning. First ticket after withdrawal, most harnesses still apply the old rule, although a harness that had learned nothing would be correct. Mem0 learns faster than it drops (70% vs 43% on tickets 4–6); append-only stores (Mem0, RAG) drop slowest.
- Exploration (distinct answers over the first 8 tickets of a rule, minus one): Prime 1.16, CH 1.02, RAG 0.88, Mem0 0.65, SkillOpt 0.51, the exact order of Hidden reward (ρ = 1.00). Across models exploration spans only 0.77–0.92, so exploration is a harness property.
- Knowledge acquisition (correctness on the 5 tickets after first success): Prime 78%, CH 76%, SkillOpt 68%, Mem0 41%, RAG 38%. Curves are flat: a success becomes reusable immediately or not at all. RAG finds and fails to reuse; SkillOpt reuses but rarely finds.
- Oracle is not a ceiling everywhere: on Pitch L1, Opus 5 + CH (99.7) beats Oracle (90.8) because Oracle, told the buyer's dislikes, fabricates claims to appeal to them.
- Training-based methods (Appendix D.5): SFT and GRPO on 14B/32B models abandoned; too few successful trajectories under sparse outcome reward to filter or contrast.

**Caveats stated by the authors.** One run per cell, no CI. Pitch L1 has 6–8 effective attribute combinations per window. Harness runtimes differ. SkillOpt's replay is impossible in live serving. Opus 5 under Blind speculates about undocumented rules (Fully Specified banking 53–69 vs 100 under Oracle), so damage relative to Blind is not purely learning-induced.

**Where it sits.** The cleanest signal design among the drift benchmarks: score-only feedback, explicit withdrawal and re-entry, a Fully Specified slice that prices collateral damage, and Blind/Oracle brackets. Adds two analyses nobody else has: learning-speed curves per rule and an exploration metric. Does not grade memory contents (ReScopeBench does) and has no human baseline. It is the controlled version of ApprenticeBench's story, with the Oracle ceiling ApprenticeBench lacks.

## 16. Self-evolving harnesses: the adjacent field

Learning can live in weights, in a memory store, or in the harness itself (prompts, skills, tools, orchestration, source). The third is "agent harness evolution" or recursive self-improvement without weight updates. Same stage model, larger blast radius: one harness change touches every future task, so gains and regressions are both bigger than for a memory write.

**Methods, by what they evolve** (ids from the ServeLearnBench and EvoAudit bibliographies)
- Prompt/context: GEPA (2507.19457, ICLR 2026; reflective prompt evolution with Pareto selection), ACE (ICLR 2026; evolving playbooks, source of "context collapse"), Dynamic Cheatsheet, ReasoningBank (2509.25140, ICLR 2026).
- Skills/procedures: SkillOpt (2605.23904), PolicyBank (2604.15505), Agent Workflow Memory, Voyager.
- Whole harness state: Continual Harness (2605.09998; acting agent never sees scores, a Refiner revises prompt, memories, skills, subagents on a cadence), Prime Agent (2608.23552; self-improving code-runtime harness), Recursive Harness Self-Improvement (2607.15524), Adaptive Auto-Harness (2606.01770), JIT-Agent (2608.25593), HCL (2608.19013; retention during harness updates), RSIAgent (2609.15364), Meta-Harness.
- World-model side: self-evolving world models for planning (2606.30639), Memento 3 (2610.11794).
- Code/program: Darwin Gödel Machine, AlphaEvolve, STOP.

**Audits**
- Harness-Delta Attribution (2610.22466): 54% of reported gain is overfitting to the search set, 14% test-time compute.
- Beneath the Surface / EvoAudit (2611.07624): below.
- CODA (2610.20845): skill-bank drift erases 52–93% of the gain.
- Can coding agents sustain continual evolution without forgetting (2611.03219): Meta-Harness 94.0 → 32.0 on SWE-Bench after five evolution rounds on a different task set.
- Evolving agents in the dark (2606.05922, EMNLP 2026): retrospective harness optimization via self-preference, no verifier.
- EvoAgentBench (2607.05202): self-evolution as ability transfer.

**Where ServeLearnBench lands on these.** Prime and Continual Harness (whole-harness state, with a post-score reflection step) reach 57–62 on hidden-rule tasks; SkillOpt, Mem0, RAG 33–43. Exploration (distinct answers tried on a new rule) is a harness property and ranks the five identically to their reward.

### EvoAudit / "Beneath the Surface: What We Overlook When We Talk About Agent Evolution" (USTC, Berkeley, BIGAI, SJTU; arXiv 2611.07624, Nov 5 2026)

**Setup.** Agent = frozen model + harness. Four recipes: AHE and HyperAgents (implementation-centric), A-Evolve and GEPA (reflection-centric). Two harnesses: MiniSWE 2.4.5 (source-editable) and Codex 0.143.0 (outer layer only). Two benchmarks: τ3-Banking (97 tasks) and Terminal-Bench 2 (89), split train/val/test with evolution seeing only train/val. GPT-5.4. Seed and Champion each evaluated 8 times, giving reliability (Avg@8) and coverage (Pass@8) separately, plus per-task gained/lost counts and deployment cost.

**RQ1, what improves.** 11/16 lineages improve reliability, 10/16 coverage, 4 move them in opposite directions. All 16 Champions gain and lose tasks: best case +15/−4 (AHE on MiniSWE, τ3), worst +6/−9 (GEPA on Codex, τ3). Reflection-centric methods barely move (A-Evolve and GEPA within ±2.7 everywhere). 8 of 11 reliability gains cost more per trial. Trace inspection: gains come from state preservation and dependency handling; losses from exact-grounding errors and premature handoff.

**RQ2, transfer.** Test-split gains survive in 10/16, 7/8 for implementation-centric. Cross-benchmark: coverage falls for 7/8 Champions on τ3 → τ2-bench and rises for 7/8 on Terminal-Bench → DeepSWE. Gains are target-specific.

**RQ3, per operator.**
- Select: choosing the best-scoring Parent to evolve further beats random (25%) only on coding (55–60%) and is worse than random on general tasks (9–18%). Current score does not predict improvement potential.
- Analyze: specific trace-grounded components beat generic reminders ("check tools, verify results") in 15/16 cases; generic reminders are flat or negative.
- Mutate: 6/8 Champions beat either isolated component, so changes interact and must be tested combined.
- Gate: a stricter admission rule (mean improvement and regression ≤ 0.05) drops the best candidate in 9 of 15 pools.

**AGE (Audit-Guided Evolution).** Four revisions installed in A-Evolve: regression-aware tie-breaking in select, behavior-grounded diagnosis that excludes rewards and hidden tests from the judge's view, dependency-closed mutations with stated activation conditions, and a preservation-tracking gate. Beats A-Evolve on reliability and coverage in all four settings (τ3 MiniSWE 34.0 vs 26.3; τ3 Codex 38.1 vs 28.1; TB2 MiniSWE 71.8 vs 68.0; TB2 Codex 73.5 vs 68.0) and loses fewer seed-covered tasks in three. Eight experts endorsed the revisions 6.75/8.

**Caveats.** One model, 20 generations, one seed. AGE is compared against A-Evolve, the weakest recipe; its margin over AHE and HyperAgents is 1–2 points. The durable contribution is methodological: report gained and lost tasks separately, price deployment, and evaluate each search operator by the decision it makes rather than by one aggregate score.

**Relation to the rest of the survey.** Store and revise at the harness level. "Coverage is a flow" is the harness analogue of ServeLearnBench's Fully Specified damage, and the operator-level audit is the analogue of MemPABench's stage split: the same score can hide a bad select, a bad mutate, or a bad gate.

## Sources
ApprenticeBench: neocognition.io/blog/apprentice-bench, apprenticebench.com (JS-rendered; numbers from the Nov 4 report). CLE: haoyu-zhao.github.io/CLE, arXiv 2610.33956, github.com/haoyu-zhao/Continual-Learning-Exam, HF Continual-Learning-Exam/CLE-Lite. CL-Bench: arXiv 2606.05661. AgentCL: arXiv 2606.02461 (v3, Sep 2026). AgentStream: arXiv 2608.00155, github.com/microsoft/Sico. Evo-Memory: arXiv 2511.20857. StreamBench: arXiv 2406.08747. LifeVal: arXiv 2611.06687 (local PDF ~/Documents/papers/2611.06687v1.pdf). SkillLearnBench 2604.20087, EvoMemBench 2605.18421, LifelongAgentBench 2505.11942, MemoryAgentBench 2507.05257, ACE 2510.04618, RL's Razor 2509.04259, sparse memory finetuning 2510.15103. Subagent-fetched paper sections marked "summary" in the fetch were not read verbatim; spot-check Table 1 of CLE and CL-Bench before quoting cost figures.
