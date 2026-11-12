# Continual learning benchmarks for LLM agents

Notes on how agents are tested on learning from experience. We read eight benchmarks and sort them into four buckets by what changes over time. Longer working notes are in `archive/`. Which of these can be run today, and which fit a weight-update experiment, is in [REPRODUCIBLE.md](REPRODUCIBLE.md). The fast-weight reading list is in [PAPERS.md](PAPERS.md).

| Bucket | What changes over time | Benchmarks |
|---|---|---|
| Sequential tasks | Nothing in the world changes. Later tasks reuse what earlier tasks taught. | ApprenticeBench, CLE, AgentCL, CL-Bench |
| Environment drift | The rules or state of the world change under the agent. | ServeLearnBench, ReScopeBench |
| Preference change | The user's wishes change, and differ by context. | MemPABench |
| Time and lifecycle | Days pass, sessions restart, the agent schedules its own actions. | SCLATE |

A sixth kind, memory recall, sits outside continual learning: find a fact in a long history, with no improvement over a stream expected (LongMemEval, LifeVal). SCLATE turns recall into the time bucket by cutting the history into sessions.

Some benchmarks touch two buckets. ApprenticeBench has one rule change and CL-Bench has two drifting tasks, so both have a little drift. MemPABench learns each preference over three sessions, so it is also sequential.

## 1. What "learning" means here

There is no parametric learning in these benchmarks. The model's weights never change (one exception: SCLATE trains a small 4B model). What changes is the stuff around the model: a memory file, a retrieval store, skill files, the prompt, the harness code.

Three words come from classic continual learning:

- **Plasticity**: learning new information.
- **Stability**: keeping what you learned.
- **Generalization**: using what you learned on a related task.

With frozen weights, "plasticity" is really retrieval. The model did not get better. A useful example was put in front of it. AgentCL shows this: handing the agent the right earlier solution directly scores 60; the best memory system scores 67. So these benchmarks measure the whole system (model plus memory plus retrieval), not the model.

AgentCL measures the three words like this. B is the score with no memory. F is the first pass, with memory being written. S is a second pass over the same tasks with memory frozen. H is a held-out task from somewhere else.

```python
B, F, S, H = 45, 67, 65, 71    # held-out task scores 72 with no memory
plasticity     = F - B         # +22  earlier tasks helped
stability      = S - F         # -2   the agent could not get its own notes back later
generalization = H - 72        # -1   memory did not help elsewhere
```

## 2. Two things to check in any benchmark

**Was reuse built into the stream?** A stream is a sequence of tasks. AgentCL splits them in two:

- **Dependent**: later tasks need what earlier tasks taught. ApprenticeBench (bill 60 needs the cost code learned on bill 12), CLE, ServeLearnBench, ReScopeBench, MemPABench.
- **Conventional**: tasks only share a topic. Most "stream an existing dataset" benchmarks (AgentStream, Evo-Memory, MMLU-Pro).

This decides everything. On dependent streams, memory methods gain 14 to 40 points. On conventional streams and held-out tasks, every memory method is at or below no memory. So when a paper says "memory does not beat plain context," first ask whether the stream had anything to reuse.

**Is the result graded, or the memory?** Almost every benchmark grades only the final result: was the bill right, did the order end in the right state. They never open the memory file.

ReScopeBench is the exception. The authors wrote an answer key for what should be in memory after each task, and an LLM judge checks the memory against it. That is how they can say "the agent finished the task but left the wrong answer in its notes." Example: the agent found a moved folder and finished the job, but its notes still pointed to the old folder.

## 3. Sequential tasks

Later tasks need what earlier tasks taught, and the world stays the same. This is the plainest form of continual learning: acquire, then reuse.

### ApprenticeBench (NeoCognition, Sept 2026)

**Mental model.** A new accounts-payable clerk's first hundred bills.

**Setup.** A simulated construction company running real Odoo accounting software. The agent joins in May. It gets six months of old bills, the company handbook, and a tutorial. Then it processes 100 new bills in order, with no reset. In the first month a mentor corrects every bill. After that, feedback only comes at month end.

**What has to be learned.** Company habits you can only pick up on the job: which cost code each item goes to, which vendor always bills in the wrong units, that the tax line must be 9.125% and not zero, that a safety certificate expired, that a sign-off limit changed in October.

**How the agent learns.** It writes Markdown notes through a normal Claude Code or Codex setup. No special memory system.

**Results.** Best model 72%, best human tester 51%, second human 40%. One model tested with parts removed: 11% with nothing, 31% with old bills only, 35% with mentor feedback only, 43% with both. The agents slowed down as their notes grew to hundreds of thousands of words; the humans sped up. One model was corrected 17 times on the same mistake and never changed.

**Catch.** Private. One run per model. Only two humans. Almost no drift (one rule change). The company that built it sells the agent.

### CLE, Continual Learning Exam (UCL, Oct 2026)

**Setup.** 120 job tasks written by 50+ domain experts across 89 roles: equity analyst, bioinformatician, load forecaster, music engraver. Each task is a ladder of five levels. Each level is an email from a manager asking for a deliverable (a spreadsheet, a report, a patch). Later levels build on earlier work and earlier files. Real public data, often thousands of files per task.

**How it runs.** All 120 tasks in one fixed order. The agent keeps its memory across tasks. A level passes only if every item on a hidden checklist passes, and passing unlocks the next level. Two settings: one try per level, or one try plus feedback naming what failed and a second try.

**Results.** Best agent (Opus 5 in Claude Code) clears 48% of levels and finishes all five on 23% of tasks. Feedback plus a retry adds a median 5 points. The harness matters: on the same models, Hermes beats OpenClaw by about 6 points.

**Catch.** No run without memory, so you cannot separate learning from just getting a second try. The authors say so. Only 20 of the 120 tasks are released. Their task cards and input files are open; the rubrics and the runnable packages need approval.

### AgentCL (Ohio State, Sept 2026)

**Setup.** Builds two kinds of task stream from the same tasks. In the dependent stream, a simpler subtask comes earlier and its solution is needed later. In the conventional stream, the subtasks are left out. Coding and web research tasks, plus a held-out set from a different source.

**How it runs.** Every task is run three ways: with no memory, with memory being written (first pass), and again with memory frozen (second pass). That gives plasticity, stability, and generalization (Section 1).

**Results.** On the dependent stream, the best memory methods gain 18 to 40 points. On the conventional stream, every method is at or below no memory. On held-out tasks, every method is below no memory. Methods that learn fast tend to lose it on the second pass. The memory found the right earlier task even when it was 76 tasks back.

**Catch.** The reuse is built in on purpose, so these gains are a best case.

### CL-Bench (Berkeley and Snorkel, June 2026)

**Mental model.** Six small worlds, each with a hidden pattern. A learner finds the pattern and gets faster or more accurate; a non-learner starts from scratch every time.

**Setup.** Six tasks, 12 to 120 instances each:
- Database: answer questions about an unfamiliar SQLite database with scrambled column names and prices stored in cents in one table and dollars in another. Fewer exploratory queries scores higher.
- Codebase: fix 19 real GitHub issues across two repos. Fewer steps scores higher.
- Poker: play heads-up against three opponents who each follow a fixed, beatable strategy.
- Spectrum monitoring: map 13 radio transmitters from partial scans; some go silent for long stretches.
- Cohort studies: estimate survival across five medical studies that each code their data differently.
- Sales forecasting: predict five years of demand when you only see one store's data at a time.

**How it runs.** Each system runs the same sequence twice: once keeping its memory (stateful), once starting fresh on every instance (stateless). Gain is the difference. Two tasks also drift: the database is migrated after question 20, and poker opponents rotate and come back.

**Results.** Plain long context (keep the whole history in the prompt) beats every memory system: Mem0, ACE, a notepad, Claude Code, Codex. The best system, Sonnet 4.6 with plain context, recovers 25% of its possible headroom. ACE costs the most ($63 a run) and gains among the least. Learning shows up clearly in spectrum monitoring and sales forecasting, not at all in cohort studies, codebase, or poker. After the database migration, agents hold on to old beliefs and keep querying tables that no longer exist.

**Catch.** Tens of instances per task, five runs, wide error bars. Codebase uses real GitHub issues the models may have seen in training. Memory is never graded.

## 4. Environment drift

In drift benchmarks the environment changes under the agent, and nobody says so. A rule that was true is now false.

- **ServeLearnBench**: hidden rules switch on and off between windows. Example: "block exchange of audio items" is off, then on, then off, then on again. The agent only sees its score drop.
- **ReScopeBench**: 26 silent changes. A file moved, a policy changed. The agent finds out when a call fails.

What they find: keeping old facts is easy, updating them is hard. ReScopeBench agents keep 91% of facts that stayed true but record only 65% of facts that changed, four tasks after the change. ServeLearnBench agents drop a withdrawn rule as slowly as they learned it.

ServeLearnBench also probes forgetting in two ways. Can the agent drop a rule when it goes away? And does learning break tasks it already did right? (It does. Every method loses accuracy on tasks the manual fully covers.)

### ServeLearnBench (CMU and Amazon, Oct 2026)

**Setup.** Three jobs: retail support (cancel, return, exchange orders), bank case review (approve or deny transactions), and writing short sales pitches. The agent has a manual. Hidden rules override the manual and change over time.

**How the rules change.** Time is split into windows. Between windows a rule can switch on, switch off, narrow, or come back. Example: "block exchange of audio items" is off, then on, then off, then on again. In banking it is a number: the approval limit for electronics is $650, then $460, then $650. In sales the hidden buyer type changes.

**How it runs.** Each window has practice tasks with a score, then test tasks with no feedback. The agent only ever sees a score. Never the rule, never a correction. Some tasks need the hidden rule; others are fully covered by the manual, which lets them check whether learning breaks things that already worked.

**Results.** Told the rule, agents score 95. With no memory, 14. Best learning system, 62. On the hardest retail level the typical setup scores 22. Every learning system got worse on tasks the manual already covered. Learning costs 4 to 100 times more than being told. Dropping a rule that went away is as slow as learning it. The more different answers a system tries when it keeps failing, the better it learns, and this depends on the harness, not the model. Training a 14B or 32B model on this was tried and abandoned: too few successes to learn from.

**Catch.** One run each. The five learning systems are built very differently, so the comparison is between whole systems.

### ReScopeBench (HKUST and Huawei, Nov 2026)

**Setup.** 18 fake business apps modeled on real ones (SAP, Kubernetes, ticketing, clinical records), with made-up names so the model cannot know them in advance. Streams of 16 to 32 tasks per app. 26 silent changes: a folder moves, a policy changes, a permission is revoked.

**How it was built.** Backwards. The authors first picked the facts an agent should learn, wrote an answer key for each, then wrote tasks that need those facts, then scheduled changes that break specific facts.

**How it runs.** Each lookup costs budget, so rereading everything every time is not affordable. Each model runs twice: once keeping its memory between tasks, once wiping it. After each task, a judge reads the agent's memory and checks it against the answer key.

**Results.** Best model goes from 26 without memory to 81 with it. Weak models barely move. After a change, agents keep 91% of facts that are still true but record only 65% of the new ones four tasks later, and 38% on the first task. Most models write down where things are more completely than the rules for using them.

**Catch.** One run each. The memory judge was checked against another AI, not humans. Extra junk in memory is not penalized.

## 5. Preference change

**Interaction preferences** are how the user wants the agent to act, not what the agent should know. How long to talk, whether to explain, whether to ask first, whether to message the user unprompted.

**Autonomy** is the key axis. For some things the user wants the agent to just do it. For others the agent must ask first. The right answer can differ at work and at home.

MemPABench is the one benchmark built on this.

### MemPABench (NYU Abu Dhabi, Oct 2026)

**Mental model.** A course and its exams. The exams are thrown away.

**Setup.** Four simulated users, based on sitcom characters with the names changed. Each user has 28 hidden preferences: 14 behaviors (how long to talk, whether to explain, whether to act without asking, whether to message unprompted) times work and home. Each lives in a four-month world with calendars, messages, and files.

**How it runs.** 99 sessions. Each one is a normal task built so the agent has to make one behavior choice, and the user reacts but never says the rule. Five times, something goes wrong and the user changes a preference. At the end, the agent is tested on copies of its memory, and the copies are deleted so the tests never teach anything.

**Results.** Fact-oriented memory systems (Mem0, MemOS, RAG) score the same as no memory, about 50% against 45% chance. They stored the user's complaints but did not bring them back at the right moment. Putting all memory in the prompt scores 59%. Keeping work and home apart, and following a changed preference, both sit at 10 to 25%. Preferences about actions (asking first, messaging unprompted) are the hardest. Even when told the right preference, models fall back to their habits 40% of the time on the hard ones.

**Catch.** Never checks whether the task got done. Asking first is always free. Nothing happens between sessions.

## 6. Time and lifecycle

SCLATE adds two things personal agents need:

1. **Time passes.** The clock runs while the agent works and jumps over idle gaps, so 40 days run in 6 hours. Facts written on day 3 must survive restarts and be found on day 28. This tests memory the way real use does, instead of dumping everything into one long prompt.
2. **The agent can schedule things.** It can set a nightly memory clean-up or a reminder, and the clock will fire it. A personal agent is idle most of the time and sometimes acts on its own; turn-based benchmarks cannot represent that.

Still missing everywhere: in real life a lot happens between sessions. Emails arrive, plans change. In these benchmarks nothing happens between sessions, so memory only ever holds useful material.

### SCLATE (Apple, Sept 2026)

**What it is.** Not a benchmark. A way to run benchmarks with real time in them.

**Two ideas.** Time passes: the clock runs while the agent works and jumps over idle gaps, so 40 days of work runs in about 6 hours. The agent can schedule things: a nightly memory clean-up or a reminder fires at the right simulated time. It runs real harnesses (Claude Code, Codex, Hermes) unchanged.

**Results.** On a 30-workday office benchmark, every memory setup beat no memory, but no setup was best for every model. On a recall test split into daily sessions, Claude Code's memory scored 85 to 91 and Hermes's 51 to 66, because Hermes's memory fills up and drops facts. It also trains weights: a 4B model trained this way gains 8 to 12 points on later days and learns to write more useful notes.

**Catch.** Grades the final answer only. The training teacher saw the correct feedback, which flatters the result.

## Related: harness evolution

Not a benchmark bucket. Here the learning method is different: instead of writing notes, the agent rewrites its own harness (prompts, skills, tools, code) and keeps a change only if it scores better. The papers run these methods on fixed benchmarks, so they say nothing about task shape. They belong here because they show the same failure the drift benchmarks show: learning breaks things that already worked. One change affects every future task, so gains and losses are both bigger.

**A-Evolve** is a framework for this. It edits files in the agent's workspace, keeps a change only if it passes held-out tasks, and tags each kept change in git.

### EvoAudit, "Beneath the Surface" (USTC and others, Nov 2026)

**Setup.** Tests four methods that let an agent rewrite its own harness, on a banking benchmark and a coding benchmark, with two harnesses. Each original and evolved agent runs eight times, so they can see which tasks were gained and which were lost.

**Results.** Every evolved agent gained some tasks and lost others. Gains were 2 to 7 points, inside the run-to-run noise for many. Picking the best-scoring version to improve further was worse than picking at random on banking. A stricter rule for keeping changes threw away the best version in 9 of 15 cases. Their improved method beats the weakest of the four by 4 to 10 points and the strongest by 1 to 2.

**Catch.** One model, one random seed.
