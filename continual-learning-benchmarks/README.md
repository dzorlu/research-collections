# Continual learning benchmarks for LLM agents

A survey of how agents are evaluated on learning from experience, written from the benchmark side. Eight benchmarks read in full or by section (Nov 2026): ApprenticeBench, Continual Learning Exam, ServeLearnBench, ReScopeBench, MemPABench, SCLATE with MetaClaw, EvoAudit, AgentCL. Everything else is one row in Appendix A. The working notes this was condensed from are in `archive/`.

## 1. What continual learning means for an agent

In every benchmark here the model's weights are frozen. The agent is a pair (θ, s): weights and harness state, where s is whatever persists between tasks: a memory file, a retrieval store, skill files, the system prompt, in SCLATE's case the harness code. One task produces an experience e = (action, observation, reward or feedback). Learning is the update

```
θ' = Uθ(θ, s, e)     # identity in every benchmark except SCLATE's post-training run
s' = Us(θ, s, e)     # write a note, save a skill, edit the harness
```

Three things get called continual learning and should be kept apart:

1. **Test-time learning over a task stream.** Later tasks depend on earlier ones; s is the only carrier. ApprenticeBench, CLE, ServeLearnBench, ReScopeBench, AgentCL, MemPABench.
2. **Memory recall.** Retrieve a fact from a long history; no improvement over a stream expected. LongMemEval, LoCoMo, LifeVal. SCLATE converts two of these into type 1 by cutting the history into sessions.
3. **Parametric updates.** Weights change on the job. Only SCLATE evaluates this, with a 4B student.

### The five stages

Any agent that learns on the job passes through five stages, and each benchmark instruments some of them.

```
acquire  ->  store  ->  retrieve / select  ->  realize / act  ->  revise on change
```

| Stage | Failure | Measured by |
|---|---|---|
| Acquire | the signal is never turned into a note (a rule inferred from errors, a preference from a reaction) | ReScopeBench rule coverage; MemPABench diagnostic cases; ServeLearnBench exploration |
| Store | the note is written but lossy, capped, or junk | ReScopeBench coverage against a claim inventory; SCLATE's capped Hermes memory |
| Retrieve | the note exists but is not surfaced at decision time | MemPABench "stored yet not retrieved"; AgentCL stability gain |
| Realize | right knowledge, wrong behavior | MemPABench realization under Oracle; ReScopeBench budget case |
| Revise | the world or user changed, the note is stale | ReScopeBench update coverage; ServeLearnBench dropping curve; MemPABench change tracking |

### A caution on vocabulary

AgentCL borrows plasticity, stability, and generalization from parametric CL. With frozen weights, "plasticity" is retrieval plus in-context use: the model's capacity did not change, a relevant example was put in front of it. AgentCL's own reference row, where the subtask solution is handed over with no retrieval, scores 60; the best memory method scores 67. The borrowed terms describe system-level adaptation, where the system is model plus memory plus retriever. The one quantity with no parametric counterpart is stability gain: retrievability of a record after many later writes.

## 2. Dimensions

| | ApprenticeBench | CLE | ServeLearnBench | ReScopeBench | MemPABench | SCLATE / MetaClaw | EvoAudit | AgentCL |
|---|---|---|---|---|---|---|---|---|
| **What is learned** | firm rules, cost codes, vendor quirks | prior artifacts, tools, strategy, domain rules | hidden policy rules and thresholds | structural facts and rules of a system | user interaction preferences, per context | procedural conventions | nothing: the harness itself is what changes | solutions to earlier subtasks |
| **Learning signal** | dense mentor feedback, then month-end | rubric-category diagnostics on failed submit | score only | tool errors and outcomes | user reactions, never stated | corrective feedback per round | benchmark score | none, self-judged |
| **Drift** | one policy notice | none | every window: add, withdraw, re-scope, re-enter | 26 silent changes, claim-level blast radius | 5 scripted preference flips per user | none | none; the agent drifts | none |
| **Control** | novice ablation, one model | none; Δ is two separate runs | Blind floor, Oracle ceiling | paired stateless stream | No Memory floor, Oracle ceiling | no-memory column | seed agent, 8 passes | memoryless ReAct per task, two passes, held-out |
| **What is graded** | bill outcome | files and app-state snapshot | terminal database hash plus finish action | artifacts by code, memory contents by LLM vs claim rubrics | the pick by exact match, the reply by judge and user | file checkers, multiple choice | task success, gained and lost tasks, cost | accuracy, success rate |
| **Collateral damage measured** | no | no | yes, Fully Specified slice | no | no | no | yes, lost tasks | yes, held-out GG |
| **Autonomy** | act on everything | act on everything | act on everything | permission changes alter who may authorize | learned per context; wrapper holds actions | act | act | act |
| **Re-discovery cost** | tokens only | none | per-task tool budget | per-tool budget, calibrated from reference solutions | none | none | none | none |
| **Human baseline** | 2 testers | no | no | no | simulated user rates | no | 8 experts review revisions | no |
| **Harness varied** | Claude Code, Codex | 4 | 5, unmatched | fixed, Hermes | fixed, nanobot; 4 memories | 3 harnesses x 3 memories x 10 models | 2 harnesses x 4 recipes | 8 memory methods |
| **Weights updated** | no | no | tried, abandoned | no | no | yes, 4B student | no | no |
| **Runs** | 1 | 1 | 1 per cell | 1 per condition | 3 probe repeats | 1 per cell | 8 passes | 3 for coding |
| **Judge validation** | none | 96.9% vs humans, 732 items | none needed except Pitch | judge vs judge, 99% | second judge on 418 probes | n/a | n/a | n/a |
| **Public** | no | Lite, gated | yes | yes | yes | paper only | yes | yes |

The rows that change a conclusion, in order:

1. **Was reuse built into the stream?** AgentCL: the same memory methods gain 14 to 40 points on streams with guaranteed reuse and 0 or less on ordinary streams. A flat result on an ordinary stream is uninformative.
2. **Is there a stateless or no-memory control?** Without it a score is capability plus learning. CLE has none; its Δ is the difference between two separate runs with different budgets.
3. **Does the world move?** Retention is easy, revision is hard. ServeLearnBench, ReScopeBench, MemPABench.
4. **Is the stored experience graded?** Only ReScopeBench writes an answer key for memory. Everyone else infers from outcomes.
5. **Explicit feedback or implicit signal?** Explicit corrections (ApprenticeBench mentor, CLE diagnostics, MetaClaw) move scores by 5 to 25 points. Implicit signal (ServeLearnBench scores, MemPABench reactions) is where retrieval collapses to chance.
6. **Is collateral damage measured?** ServeLearnBench's Fully Specified slice and EvoAudit's gained/lost counts show every learner loses something. Benchmarks that report one average hide this.
7. **Is re-discovery charged?** Without a budget, "learned it" and "looked it up again" are indistinguishable. ReScopeBench and ServeLearnBench charge.
8. **Is autonomy a variable?** Only MemPABench. When to act is the hardest class of preference to learn and the one a personal agent gets fired for.

## 3. Cards

Template: setup, protocol, metrics, numbers, control, weakness, where it sits.

### ApprenticeBench (NeoCognition, Sept 2026; no paper, code, or data)

**Setup.** One simulated construction company in a real Odoo ERP. The agent joins as an accounts-payable clerk with six months of historical bills, the handbook, and an Odoo tutorial, then processes 100 incoming bills in one non-resettable sequence. A mentor corrects every bill in month one, then only at month end. Each bill's answer is traceable to history, feedback, or both. Traps: rebar billed in gallons, a 0% tax line where the rate is 9.125%, an expired workers' comp certificate, a sign-off threshold changed by a notice posted in September and effective in October.

**Protocol.** Stock Claude Code or Codex harness with computer-use tools and Markdown notes the agent reads and writes. GUI and API variants.

**Metrics.** Cumulative success over 100 bills; cost per task (tokens for agents, BLS wage for humans).

**Numbers.** Fable 5.1 72% GUI at $18.23 per task; GPT-6 Astra 68%; best human tester 51% at $7.21, second human 40%; Fable 5 43%; Opus 5 36%; Kimi K3 18%. Novice ablation on Fable 5: 11 with nothing, 31 with history only, 35 with feedback only, 43 with both. The additive prediction is 55 against 43 observed, so history and feedback overlap heavily. Agents slowed as notes grew (Fable 5.1 wrote 192K words in 122 files) while humans sped up.

**Control.** The novice ablation, on one model only.

**Weakness.** Single runs, two human testers, one drift event, no held-out split, no memory grading, private artifact built by the company that sells the agent.

**Where it sits.** The acquisition benchmark with a human baseline. ServeLearnBench is its controlled, score-only, drifting counterpart.

### Continual Learning Exam, CLE (UCL; arXiv 2610.33956, Oct 2026)

**Setup.** 120 expert-authored tasks, each a ladder of five levels (600 levels), 89 roles, 12 domains, 23,981 input files of real public data, 19 stateful mock applications with 1,024 APIs. Later levels reuse earlier work through tools and skills, data dependency, strategy, user context, domain knowledge. 88 of 120 ladders reference earlier-level files. Admission: a review agent audits and runs each draft (five verdicts), then two human reviewers must both accept.

**Protocol.** All 120 tasks in one fixed order; harness state persists; each task starts with fresh files and app databases. A level passes only if every rubric item passes. Trial 1: one submission. Trial 2: a failed first submission gets diagnostic feedback naming unmet rubric categories, then one revision.

**Metrics.** Completion = levels cleared / 600. L_k = share of tasks whose first k levels all pass. Δ = Trial 2 minus Trial 1.

**Numbers.** Opus 5 in Claude Code: Trial 1 42.17%, Trial 2 47.67%, 23.3% of ladders completed, $721 and 58.9 hours for Trial 2 against 32.8 hours for Trial 1. Sonnet 5 40.17%, Δ +13.17. Open models through OpenClaw 10 to 24%. Median Δ across 17 configurations +5.33, range −1.33 to +13.17. Hermes beats OpenClaw on all six shared backbones by 5.56 points mean. Evaluator agrees with humans on 96.86% of 732 rubric items.

**Control.** None. The authors say Δ "does not isolate a causal effect of memory or learning."

**Weakness.** No stateless run, no failure taxonomy, no analysis of what agents wrote to memory, single run, full set unreleased. The Lite release is 20 tasks with files only; the app layer is gated.

**Where it sits.** The widest-coverage acquisition benchmark and the weakest on controls.

### ServeLearnBench (CMU, Amazon; arXiv 2610.07792, Oct 2026)

**Setup.** Three domains: retail support (cancel, modify, return, exchange against an isolated database), banking case review (authorize, deny, escalate), sales-pitch generation (40 to 80 words from a product sheet). Hidden rules govern correct behavior and change at window boundaries: introduced, withdrawn, re-scoped, re-entered. Retail rules are categorical ("block exchange when destination is Florida"), banking rules numeric ("electronics ceiling $650, later $460"), pitch rules a hidden buyer profile. Three tiers: L1 introduces, L2 revises and composes, L3 withdraws and re-enters. 53 windows, 4,508 serving tasks, 3,210 test tasks. Every task tagged Hidden-Dependent or Fully Specified.

**Protocol.** Each window: serving tasks with score-only feedback, then test tasks from the frozen state with no feedback, same active rules, disjoint instances. Learners see task, trajectory, scalar score, timestamp, own state. Never the rule, never a correction. Retail and banking graded by hashing the terminal database against a replayed reference plus a structured finish action; pitch by an LLM checklist fed to deterministic scoring.

**Metrics.** Hidden and Fully Specified reward per tier; cost per task; learning speed (mean correctness over the first 10 tickets of a new rule); exploration (distinct answers tried on a rule); knowledge acquisition (correctness over the 5 tickets after first success).

**Numbers.** Oracle 95.4 on Hidden, Blind 14.1. Best harness averages across open-weight models: Prime 61.7, Continual Harness 57.6, RAG, Mem0, SkillOpt 38 to 43. Hardest retail tier: best pair 59.6, median 22.0. Fully Specified: every harness loses in retail (Blind 98.9, harnesses 89.5 to 95.5); GLM-5.3 Flash with CH goes 14.6 → 49.7 on Hidden and 96.7 → 82.8 on Fully Specified. Cost: 4 to 101x Oracle; Kimi K3 with Prime $2,479 against $102. Dropping a withdrawn rule is as slow as learning it; append-only stores (Mem0, RAG) drop slowest. Exploration ranks the five harnesses identically to reward, Spearman 1.0, and varies 0.51 to 1.16 across harnesses against 0.77 to 0.92 across models. SFT and GRPO on 14B and 32B models were tried and abandoned: too few successful trajectories under sparse reward.

**Control.** Blind and Oracle bracket the headroom; the Fully Specified slice prices damage.

**Weakness.** One run per cell. Harnesses are unmatched by design (Prime has its own runtime). SkillOpt replays served tasks, impossible live. Opus 5 under Blind speculates about undocumented rules, so damage relative to Blind is not purely learning-induced.

**Where it sits.** The cleanest drift design: quiet signal, explicit withdrawal, damage slice, floor and ceiling. Adds learning-speed and exploration analyses no one else has.

### ReScopeBench (HKUST, Huawei; arXiv 2611.05450, Nov 2026)

**Setup.** 18 coded simulations of business systems (SAP purchase-to-pay, Kubernetes, REDCap, railway and cinema ticketing, carbon credits, emergency dispatch, robot logistics) with invented entities, 3 to 18 tools each. Built experience-first: authors read the environment code, pick facts an agent could discover through tools, write a rubric per fact, build 16 to 32 tasks that reuse those facts, then schedule 26 silent changes that invalidate specific facts while others stay true. Two kinds of fact: structural (where things are, how they connect) and rule (conditions on valid actions). Tool calls cost budget units calibrated from the authors' own reference solutions, so exhaustive rediscovery is unaffordable.

**Protocol.** Paired streams per model and environment: one keeps memory, skills, and history between tasks, one wipes them. Environment state persists in both. Agents are told to store structural facts in Hermes memory and rules in Hermes skills. Fresh conversation per task.

**Metrics.** Task: BP (no experience), EP (with), DP (first task after a change). Experience: SC and RC, coverage of current structural and rule claims, judged by GPT-6 Astra over the whole memory and skill corpus against the private rubrics; UC, fraction of changed claims recorded after the first affected task; RT, how early all updates and still-valid claims are correct together.

**Numbers.** 20 models. Gemini-4-Argon BP 25.9 → EP 81.1; Claude Opus 5.5 21.9 → 75.0; GPT-5.4 0.1 → 15.0. Best UC 37.6%. Four tasks after a change, agents record 64.6% of changed facts while keeping 91% of still-valid ones. Rule coverage below structural for 17 of 20 models, mean gap 4.8; Opus 5.5 reverses it (RC 73.2). Resource changes cause the largest immediate drop (40 points of advantage); structural changes recover within four tasks; tool-path changes stay 8.7 down. Second judge agrees on 99% of claim decisions.

**Control.** The paired stateless stream, though the two streams' environment states diverge after task 1.

**Weakness.** One run per condition. Judge validated against another LLM, not humans. Coverage, not precision: junk in memory costs nothing. One harness, so the structural-versus-rule gap is entangled with memory being loaded in full and skills fetched on demand.

**Where it sits.** The only benchmark that grades what the agent wrote. First to say "finished the task but left the wrong answer in its notes."

### MemPABench (NYU Abu Dhabi; arXiv 2610.30485, Oct 2026)

**Setup.** Four simulated users built from sitcom characters, anonymized so a model cannot guess them (shortcut check: 53.2% from the real identity, 46.4% anonymized, chance 45.2%). Each has 28 hidden settings: 14 interaction attributes (verbosity, autonomy level, proactive outreach, reasoning visibility, and ten more) times work and personal contexts. Each user lives in a four-month world with 83 tools. The user never states a setting; it only reacts, and remembers past annoyances.

**Protocol.** 99 timeline sessions: 94 live, where each session targets one cell and is built so the task cannot finish without a decision on that attribute, plus 5 scripted preference-change events replayed identically for every memory system. At a decision point the agent must call a tool naming its chosen setting; a wrapper enforces it mechanically. Exams are probes on frozen copies of world and memory, discarded afterward: 5 before change events, 28 at the end, 3 repeats each. Course and exams; the exam never counts as a lesson.

**Metrics.** On the pick: PSA (exact match over 28 cells), CS (both contexts right for attributes that differ), PET (old setting before the change and new one after). On the reply: BRS (judge, 1 to 5, against the target definition, without seeing the pick), PRS (the simulated user's own rating).

**Numbers.** PSA: No Memory 49.6, Mem0 50.0, MemOS 50.0, RAG 49.9, File memory 59.2. Retrieval returned an applicable interaction preference in 7 of 334 judged probes against 54% for File memory; case study shows all four stores held the user's objection and only File memory surfaced it. CS about 20% and PET 11 to 25% for everyone. Action-side attributes (autonomy, outreach, elicitation) select at 34 to 44% against 55 to 69% for reply-side ones, and below chance with no memory. Under Oracle, five settings (Hide, Terse, Iterative, Structured, Low breadth) pass realization only 60% of the time.

**Control.** No Memory floor and Oracle ceiling, which isolate realization from acquisition and retrieval.

**Weakness.** Four users, one harness, everything simulated. Task completion never scored. No cost of asking, so always-confirm is free. No context between sessions, so retrieval is easier than deployed and every change comes with an incident and a complaint.

**Where it sits.** The only benchmark with on-policy history, a context axis, scheduled preference change, and a stage split with both floor and ceiling. The only one where autonomy is a learned variable.

### SCLATE with MetaClaw (Apple; arXiv 2609.32391, Sept 2026)

**Setup.** Not a benchmark but the substrate benchmarks run on. Three parts. An open event scheduler on a hybrid simulated clock: benchmark events (tasks, environment mutations, grading) and agent events (session start and stop, crons, memory consolidation) register on one timeline; the clock flows at wall speed while anything is busy and jumps across idle gaps otherwise; `libfaketime` feeds the simulated date to every harness and memory process. Adapters that run unmodified harnesses (Claude Code, Hermes, Codex) with native or add-on memory (Mem0, Hindsight, GBrain). An in-container proxy that records every model call's tokens and log-probs and tags which component made it, so a multi-session run doubles as training data. Workload: MetaClaw, 30 simulated workdays over 40 calendar days, 346 rounds of file-check and multiple-choice tasks with hidden procedural conventions, corrective feedback carried into the next round.

**Protocol.** Session per workday; process stops and restarts; memory persists. LongMemEval and PersonaMem are cut into sessions on consecutive days with the question in the last session, so recall must pass through the memory system.

**Metrics.** Pass accuracy; parity with upstream GAIA-2; compression ratio; post-training gains on held-out days.

**Numbers.** 40 calendar days in a median 5.9 real hours. GAIA-2 events delivered in order within 0.9 s median. MetaClaw grid, 10 configurations x 10 models: every memory configuration beats no-memory; an added memory loses to native in 11 of 50 cells; Gemini-3.7F leads at 79 to 82% and is the only model that searches prior session transcripts everywhere. LongMemEval-s under session cuts: Claude Code native 85 to 91, Hermes native 51 to 66 because its memory is size-capped, Hermes with Hindsight 74 to 77. GAIA-2's Time and Adaptability categories stay lowest everywhere. Post-training a Qwen3.5-4B student through unmodified harnesses by on-policy distillation: held-out days +7.6 to +11.8 in five of six configurations, three significant; the trained model writes procedural rules into memory in 40.6% of facts against 23.6% before. GRPO on SWE-Gym: SWE-bench Verified 9.8 → 26.5 under Claude Code.

**Control.** The no-memory column.

**Weakness.** The distillation teacher sees the benchmark's corrective feedback, so "learned to use memory" is partly "learned the answer's shape." One run per cell. No ablation that turns agent-side scheduling on and off. Outcome-only grading. No code release found.

**Where it sits.** The only place weights change, and the only place time is a quantity rather than a turn counter. Pairing its calendar with ReScopeBench's memory rubrics is the obvious unbuilt benchmark.

### EvoAudit, "Beneath the Surface" (USTC, Berkeley, BIGAI, SJTU; arXiv 2611.07624, Nov 2026)

**Setup.** The harness is what learns. Four evolution recipes (AHE, HyperAgents, A-Evolve, GEPA) on two harnesses (MiniSWE, source-editable; Codex, outer layer only) and two benchmarks (τ3-Banking, 97 tasks; Terminal-Bench 2, 89), split train/val/test with evolution seeing only train/val. GPT-5.4. Seed and evolved Champion each evaluated 8 times.

**Protocol.** Every recipe is seven operators: select a parent, roll out, analyze, mutate, validate, gate, record. The audit tests select (does the best-scoring parent improve most?), analyze (specific components vs generic reminders), mutate (do changes interact?), gate (does stricter admission keep the best candidate?).

**Metrics.** Avg@8 reliability, Pass@8 coverage, tasks gained and lost against seed, deployment and evolution cost.

**Numbers.** 11 of 16 lineages improve reliability, 10 coverage, 4 move them oppositely. All 16 Champions gain some tasks and lose others (best +15/−4, worst +6/−9). Engineering-centric recipes gain 2 to 7 points; reflection-centric ones sit within ±2.7 of seed. 8 of 11 reliability gains cost more per trial. Transfer: coverage falls on the general-task target for 7 of 8, rises on the coding target for 7 of 8. Select: best-scoring parent beats random only on coding (55 to 60% vs 25%) and is worse than random on general tasks (9 to 18%). Analyze: specific components beat generic reminders in 15 of 16. Gate: a stricter rule drops the best candidate in 9 of 15 pools. AGE, four operator revisions built from the audit, beats A-Evolve in all four settings (34.0 vs 26.3; 38.1 vs 28.1; 71.8 vs 68.0; 73.5 vs 68.0) and loses fewer seed tasks in three. Eight experts endorsed the revisions 6.75 of 8.

**Control.** The seed agent, eight passes, per-task transitions.

**Weakness.** One model, one seed, 20 generations. AGE is compared against the weakest recipe; its margin over AHE and HyperAgents is 1 to 2 points inside the pass-to-pass noise.

**Where it sits.** Store and revise at the harness level, where one change touches every future task. "Coverage is a flow" is the harness analogue of ServeLearnBench's Fully Specified damage. Related audits: Harness-Delta Attribution (2610.22466) attributes 54% of harness-evolution gain to overfitting and 14% to test-time compute; CODA (2610.20845) finds skill-bank drift erases 52 to 93% of the gain; 2611.03219 finds Meta-Harness dropping from 94% to 32% on SWE-Bench after five rounds of evolution elsewhere.

### AgentCL (OSU; arXiv 2606.02461, v3 Sept 2026)

**Setup.** Three cross-task relations. Conventional: same domain, reuse unknown (how every "stream an existing dataset" benchmark works). Dependent: subtasks placed before complex tasks that need their solution, relation hidden, about 1% of stored episodes relevant and half a stream back. Independent: held out from a different source. Coding from CodeEval-Pro self-invoking pairs (48 pairs, reuse guaranteed by construction; 120 HumanEval-Pro held out). Deep research from 100 BrowseComp+ tasks with 308 subtasks synthesized by GPT-5.2 sharing evidence documents, filtered so none are paraphrases. MMLU-Pro, ScienceWorld, BabyAI as conventional streams. Qwen3.5-35B-A3B, top-2 retrieval, no ground-truth feedback during memory construction.

**Protocol.** Two passes. B: memoryless ReAct. F: first pass, memory read and write. S: second pass over the same tasks, memory frozen. H: held-out with frozen memory.

**Metrics.** PG = F − B (did earlier experience help this task), SG = S − F (is this task's experience still usable after later writes), GG = H − B (does memory transfer or interfere).

**Numbers.** CodeEval-Pro, dependent stream PG: MemProbe +21.9, ExpRAG +17.7, ReMem +13.5, Mem0 −1.0, AWM −5.2. Same methods on the conventional stream: −9.4 to −0.3, all at or below ReAct. Held-out GG: every method below ReAct, −0.8 to −11.7. Std across methods 9.4 on dependent vs 3.0 on conventional; BrowseComp+ 14.9 vs 2.3. The three methods with large PG have SG of 0.0, −2.0, −2.1. No forgetting window: PG holds at +18 with the subtask 76 tasks back, retrieval Hits@2 above 90%. On dependent streams PG rises with position (+8 → +22); on conventional streams it falls.

**Control.** The best in the field: per-task memoryless baseline, a second pass that isolates consolidation, a held-out split.

**Weakness.** Synthetic reuse pairs are an upper bound. Held-out is a mild shift (coding to coding). No drift, no budget, no human baseline, no memory grading. Stream-position analysis is single-run.

**Where it sits.** Explains the "memory does not beat context" results elsewhere: they were measured on streams without built-in reuse.

## 4. How the environments are built

Every benchmark has to solve five problems: make sure there is something to learn the model does not already know, make later tasks depend on it, give the agent a channel to discover it, make rediscovery cost something, and grade without leaking the answer.

| | ApprenticeBench | CLE | ServeLearnBench | ReScopeBench | MemPABench |
|---|---|---|---|---|---|
| Unfamiliarity | invented company conventions in a real ERP | each expert's own workflow with real public data | hidden rules that override a visible manual | invented entities in coded sims modeled on real apps | sitcom personas, anonymized |
| What is learnable | cost codes, policies, vendor quirks | implicit in ladder design | a rule dictionary with a per-window activation matrix | pre-selected claims, each with a rubric | a 28-cell preference table per user |
| Dependency guarantee | each bill traceable to history or feedback | expert judgment plus review agent and two reviewers | serving and test cover the same active rules | dependency graph, tasks built from it | each session targets one cell, three sessions per cell |
| Discovery channel | history dump plus mentor | files plus app APIs | score after each ticket | tool calls with declared costs | user reactions only |
| Rediscovery cost | tokens | none | per-task tool budget | per-tool budget below exhaustive search | none |
| Reference solution | mentor's answer per bill | human-verified reference materials | replayed ground-truth tool actions, hashed | authors play the stream with the agent's tools | the preference table itself |
| Grader | outcome | LLM on artifacts plus app snapshot | database hash plus finish action; LLM checklist for pitch | code on artifacts, LLM on memory | exact match on the pick, LLM and user sim on the reply |
| Leak control | private | review agent checks leakage; fresh grading container | learners never see rule text or corrections; audit of visible prompts | private references, frozen after replay | rule checks and a judge block any line naming a setting |
| Change | one notice | none | scheduled per window | scheduled, claim-level | five scripted events, replayed identically |

Two patterns worth copying. ReScopeBench's order of construction: pick the facts first, then write tasks that need them, then schedule changes against specific facts. It is the only way to end up with an answer key for memory. ServeLearnBench's rule dictionary with a window-by-window activation matrix: it makes drift a table you can print, and withdrawal and re-entry become rows you can test. ReScopeBench's budget calibration is the cheapest high-value step: authors play the stream with the agent's own tools, which yields the reference solutions, the budgets, and a leak check at once.

Two worked examples. In ReScopeBench's Filesystem environment, GPT-5.5 saved a lane binding at task 4; at task 7 the call failed with `DELIVERY_POLICY_MISMATCH` and it saved a warning; at task 9 it re-listed lanes, found the new handle, rewrote the binding keeping five valid fields, and reused it for five later tasks with no lookups. The memory judge scored the claim 0 at task 7 and 1 at task 9. In ServeLearnBench's retail L3, hidden rule Z1-A ("block exchange of audio items") is off in window 0, on in window 1, off in window 2, and on again in window 6. A headphone exchange is correct, then wrong, then correct, then wrong, and the agent only sees the score.

## 5. Metrics

Gain over a stateless run is what every careful benchmark reports. The stateless baseline is what makes "learning" a claim.

```python
stateful  = [0.2, 0.4, 0.6, 0.7]   # reward per task, memory on
stateless = [0.2, 0.3, 0.3, 0.3]   # same system, fresh start each task
gain = [s - b for s, b in zip(stateful, stateless)]          # [0.0, 0.1, 0.3, 0.4]
```

AgentCL's three gains on one task. The memoryless score B, first pass F, second pass S with frozen memory, held-out H.

```python
B, F, S, H = 0.45, 0.67, 0.60, 0.70
PG = F - B    # +0.22  earlier experience helped
SG = S - F    # -0.07  the experience did not survive consolidation
GG = H - 0.72 # held-out vs its own memoryless score
```

CLE's level metric is a survival curve, and its Δ is a difference of two whole runs.

```python
levels_cleared = [5, 1, 0, 3, 2]
L = [sum(x >= k for x in levels_cleared) / 5 for k in range(1, 6)]   # [0.8, 0.6, 0.4, 0.2, 0.2]
completion = sum(levels_cleared) / 25                                  # 0.44
delta = completion_trial2 - completion_trial1                         # confounded with a second attempt
```

ReScopeBench's update coverage and recovery timeliness, per affected memory unit.

```python
changed = {"lane": False, "policy": True}      # claims the change touched, correct after first affected task?
UC = sum(changed.values()) / len(changed)      # 0.5
# RT = 100 * (L - tau + 1) / L, tau = first task where all changed and still-valid claims are right together
```

MemPABench's three selection metrics.

```python
cells = {("work","verbosity"): ("Terse","Terse"), ("personal","verbosity"): ("Detailed","Detailed"),
         ("work","outreach"): ("Low","High"), ("personal","autonomy"): ("Reactive","Self-directed")}
PSA = sum(t == p for t, p in cells.values()) / len(cells)                     # 0.5
CS  = int(cells[("work","verbosity")][0] == cells[("work","verbosity")][1] and
          cells[("personal","verbosity")][0] == cells[("personal","verbosity")][1])   # 1
PET = int(pre_pick == old_target and final_pick == new_target)                 # 0 if either half fails
```

ServeLearnBench's learning speed and exploration.

```python
a = [0.1, 0.3, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8]   # correctness on the k-th ticket of a new rule
AULC10 = sum(a) / 10                                       # 0.58
answers = ["A", "A", "B", "C", "C", "C", "C", "C"]         # distinct answers over the first 8 tickets
c = [len(set(answers[:k])) for k in range(1, 9)]
AUCE = sum(x - 1 for x in c) / 8                           # 0.0 for an agent that never varies
```

EvoAudit's reliability versus coverage.

```python
Y = [[1,1,0,1,1,1,0,1],   # task 1 over 8 passes
     [0,0,0,0,1,0,0,0]]   # task 2
Avg8  = sum(map(sum, Y)) / 16            # 0.5  reliability
Pass8 = sum(max(r) for r in Y) / 2       # 1.0  coverage: both tasks solved at least once
```

Judge validation is uneven. CLE validated its evaluator against humans on 732 items (96.86%). ReScopeBench validated judge against judge (99%). MemPABench used a second judge on 418 probes. ServeLearnBench's retail and banking need no judge. ApprenticeBench and SCLATE report none. Where most rubric items fail, raw agreement is inflated by the base rate; none of these papers report a chance-corrected agreement.

## 6. Findings

1. **Memory helps only when the stream contains reusable structure and the method can find it.** AgentCL: +14 to +40 on dependent streams, 0 or less on conventional ones and on held-out tasks. CL-Bench's and MemPABench's "memory does not beat context" results were measured on streams of the second kind.
2. **Explicit feedback moves scores; implicit signal does not.** ApprenticeBench's mentor adds 24 points over novice; CLE's diagnostics a median 5.3; MetaClaw's corrections are what SCLATE trains on. With score-only or reaction-only signal, retrieval collapses to chance (MemPABench PSA 50% against 45% chance; ServeLearnBench median pair 22% on retail L3).
3. **Retention is easy, revision is hard.** ReScopeBench keeps 91% of still-valid facts and records 65% of changed ones four tasks after a change, 38% on the first. ServeLearnBench agents drop a withdrawn rule as slowly as they learned it. MemPABench change tracking is 11 to 25%.
4. **Learning damages what was already correct.** ServeLearnBench: every harness loses Fully Specified accuracy in retail. EvoAudit: all 16 evolved agents lose previously solved tasks. AgentCL: every memory method is below memoryless on held-out tasks.
5. **Exploration is a harness property.** ServeLearnBench: distinct answers tried on a new rule varies 0.51 to 1.16 across harnesses and 0.77 to 0.92 across models, and ranks harnesses exactly by reward. CLE: Hermes beats OpenClaw by 5.6 points on identical models. SCLATE: no harness or memory is best for every model.
6. **Knowing is not doing.** MemPABench: with the preference handed over, five settings pass realization 60% of the time. ReScopeBench: correct bindings in memory, and GPT-5.4 still spends its budget re-reading. ServeLearnBench: Oracle fabricates claims on pitch when told the buyer's dislikes.
7. **Capability threshold.** Gains are positive for strong models and vanish for weak ones: ApprenticeBench 72% vs 18%, ReScopeBench Gemini-4-Argon 81 vs Qwen3.5-9B under 1, ServeLearnBench's abandoned 14B training because no trajectories succeeded.
8. **Cost does not buy learning.** ServeLearnBench 4 to 101x Oracle for a fraction of its accuracy; CLE Trial 2 nearly doubles hours for +5; EvoAudit 8 of 11 reliability gains cost more per trial; Harness-Delta Attribution attributes 68% of harness-evolution gain to overfitting and test-time compute.
9. **Most results are single runs.** ApprenticeBench, CLE, ReScopeBench, ServeLearnBench, SCLATE: one run per cell. EvoAudit's eight passes show pass-to-pass standard deviations of 2 to 5 points, which is the size of most reported gains.

## 7. Open problems

- A benchmark that grades memory contents on a simulated calendar: ReScopeBench's rubrics on SCLATE's clock.
- Autonomy as a trade-off: MemPABench makes when-to-act learnable but never charges for asking or scores the task. A benchmark where over-confirming fails tasks would make it real.
- Between-session context: every benchmark's memory holds only relevant material. Deployed memory is mostly noise.
- Human baselines: ApprenticeBench's two testers are the only ones. Nobody knows what a competent human scores on ServeLearnBench or ReScopeBench.
- Parametric learning on a stream: SCLATE's 4B distillation and ServeLearnBench's failed GRPO are the whole literature here.
- Chance-corrected judge agreement and run-to-run variance, reported as a matter of course.

## Appendix A. Benchmarks screened but not read

Abstract-level only. Dates are arXiv v1.

| Benchmark | id | Date | One-line claim |
|---|---|---|---|
| CL-Bench | 2606.05661 | Jun 2026 | Six real stateful domains with hidden latent structure; same-system stateless control; ICL beats every memory system, best normalized gain 25.4% |
| AgentStream | 2608.00155 | Jul 2026 | Six benchmarks as Isolated, Sequential, Interleaved streams; only 34/45 configurations beat vanilla; GPT-5.4 net negative |
| CLR-Bench | 2610.40758 | Oct 2026 | Hidden-rule curriculum; self-evolution gains ≤5 over frozen, a fixed lesson +14.5 to +16.5; GPT-6-Sol 78% vs 53% |
| AhaBench | 2609.05435 | Sep 2026 (v2) | Puzzle, math-teaching, year-long vending; worked procedures 80 to 100% held-out |
| FilingLoop | 2610.37905 | Oct 2026 | 380 SEC review tasks; feedback distilled into guidance; 24.3 → 28.0% at +5.8% cost |
| AAArena | 2610.12341 | Oct 2026 | 12 games vs 1,920 archived human programs; Opus 5.5 earns 6 golds, loses 6 ladders |
| EdgeBench | 2607.05155 | Jul 2026 | 134 day-scale tasks; log-sigmoid scaling law, R² 0.998 |
| EvoHarnessBench | 2609.04280 | Sep 2026 | 17 harness-evolution streams; harness expansion causes forgetting |
| Δτ-Bench | 2610.34238 | Oct 2026 | 1,588 tasks across six tool-ecosystem states; best 42.3% |
| SkillShift-Bench | 2610.33528 | Oct 2026 | Learned web skills under contextual, perceptual, execution shifts |
| RSE-Bench | 2610.19910 | Oct 2026 | Noise during skill learning keeps 25 to 42% of the clean gain |
| ContinualSkillBench | 2608.03874 | Aug 2026 | Context plus feedback ≈ an explicit skill library |
| PATH-Bench | 2608.01149 | Aug 2026 | Forward and backward transfer on controlled helpful and interfering histories |
| SkillLearnBench | 2604.20087 | Apr 2026 | Self-feedback alone causes recursive drift |
| EvoMemBench | 2605.18421 | May 2026 | 15 memory methods; long-context baselines competitive |
| MemoryAgentBench | 2507.05257 | Jul 2025 | Four competencies incl. test-time learning; current methods fall short on all |
| MemoryArena | 2602.16313 | Feb 2026 | Agentic tasks with interdependent subtasks across sessions |
| Harness the Memory | 2608.15008 | Aug 2026 | 11 memory substrates, 26 metrics; none dominates |
| Harness-Delta Attribution | 2610.22466 | Oct 2026 | 54% of harness-evolution gain is overfitting, 14% test-time compute |
| CODA | 2610.20845 | Oct 2026 | Skill-bank drift erases 52 to 93% of gain |
| Coding agents and continual evolution | 2611.03219 | Nov 2026 | Meta-Harness 94 → 32 on SWE-Bench after five rounds |
| Memory Tax | 2610.34073 | Oct 2026 | Many memory systems underperform naive RAG |
| LifeVal | 2611.06687 | Nov 2026 | Personalized consumer tasks; models win 33% of criteria sourced from past conversations |
| Memento 3 | 2610.11794 | Oct 2026 | Frozen LLM writes and repairs a code world model; ARC-AGI-3 public set 100 RHAE |
| ARC-AGI-3 | 2603.24621 | Mar 2026 | Within-episode adaptation; humans 100%, frontier under 1% |

## Appendix B. Running CLE Lite

The full 120-task CLE is not released. CLE Lite (20 tasks, 100 levels) runs only after access approval on Hugging Face. Public: task cards (`Continual-Learning-Exam/CLE-Lite`) and agent-facing workspaces (`CLE-Lite-Workspaces`, 59K files). Gated: `CLE-Lite-Rubrics` and `CLE-Lite-Full`, which holds the environments, `runtime/cle_lite.py`, the Codex worker Dockerfile, and the judge. The packaged runtime drives Codex only; judge is DeepSeek V4 Flash via OpenRouter or GPT-6.1-Sol via Codex. Requirements: Linux x86-64, Docker Compose, Python 3.10+, uv, Node 22+, 30 GB per task, hours per full-budget run. Memory persists as the Codex conversation plus `/app/cle-memory/`. Pin runtime revision `26ab77e5085476d59dc8c243b928e5b8ac490ef4` and data revision `803301ec01b26ab594a5ab3cdcd855a14dcc26e1`. Per-task license; no training on tasks, no exposing rubrics. The Lite deliverables are all files; the 19 mock applications are visible only in the gated package.
