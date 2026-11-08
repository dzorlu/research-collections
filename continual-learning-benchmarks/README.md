# Continual learning benchmarks for LLM agents

Notes on how agents are tested on learning from experience. We read eight benchmarks: ApprenticeBench, CLE, ServeLearnBench, ReScopeBench, MemPABench, SCLATE, EvoAudit, AgentCL. Others we only skimmed are listed at the end. Longer working notes are in `archive/`.

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

## 2. Dependent streams vs conventional streams

A stream is a sequence of tasks. AgentCL splits them in two:

- **Dependent**: later tasks need what earlier tasks taught. ApprenticeBench (bill 60 needs the cost code learned on bill 12), CLE, ServeLearnBench, ReScopeBench, MemPABench.
- **Conventional**: tasks only share a topic. Most "stream an existing dataset" benchmarks (AgentStream, Evo-Memory, MMLU-Pro).

This decides everything. On dependent streams, memory methods gain 14 to 40 points. On conventional streams and held-out tasks, every memory method is at or below no memory. So when a paper says "memory does not beat plain context," first ask whether the stream had anything to reuse.

## 3. Drift: the world changes

In drift benchmarks the environment changes under the agent, and nobody says so. A rule that was true is now false, or a user changed their mind.

- **ServeLearnBench**: hidden rules switch on and off between windows. Example: "block exchange of audio items" is off, then on, then off, then on again. The agent only sees its score drop.
- **ReScopeBench**: 26 silent changes. A file moved, a policy changed. The agent finds out when a call fails.
- **MemPABench**: five times per user, something goes wrong and the user changes a preference.

What they find: keeping old facts is easy, updating them is hard. ReScopeBench agents keep 91% of facts that stayed true but record only 65% of facts that changed, four tasks after the change. ServeLearnBench agents drop a withdrawn rule as slowly as they learned it.

ServeLearnBench also probes forgetting in two ways. Can the agent drop a rule when it goes away? And does learning break tasks it already did right? (It does. Every method loses accuracy on tasks the manual fully covers.)

## 4. What gets graded: the result, or the memory?

Almost every benchmark grades only the final result: was the bill right, did the order end in the right state. They never open the memory file.

ReScopeBench is the exception. The authors wrote an answer key for what should be in memory after each task, and an LLM judge checks the memory against it. That is how they can say "the agent finished the task but left the wrong answer in its notes." Example: the agent found a moved folder and finished the job, but its notes still pointed to the old folder.

## 5. Personal agents

**Interaction preferences** are how the user wants the agent to act, not what the agent should know. How long to talk, whether to explain, whether to ask first, whether to message the user unprompted.

**Autonomy** is the key axis. For some things the user wants the agent to just do it. For others the agent must ask first. The right answer can differ at work and at home.

**MemPABench** is the one benchmark built on this. Mental model: a course and its exams. The agent lives with a simulated user for 99 sessions over four months (the course). The user never states a preference, only reacts. Then the agent is tested on frozen copies of its memory, and the copies are thrown away (the exams), so an exam never teaches anything. It never scores whether the task got done, only whether the agent behaved the way the user wanted.

Results: memory systems built for facts (Mem0, MemOS, RAG) score the same as no memory, about 50% against 45% chance. They store the user's complaint but never retrieve it. Putting everything in the prompt gets 59%. Autonomy-type preferences are the hardest. Even when told the right preference, models fall back to their habits 40% of the time on the hard ones.

**SCLATE** adds two things personal agents need:

1. **Time passes.** The clock runs while the agent works and jumps over idle gaps, so 40 days run in 6 hours. Facts written on day 3 must survive restarts and be found on day 28. This tests memory the way real use does, instead of dumping everything into one long prompt.
2. **The agent can schedule things.** It can set a nightly memory clean-up or a reminder, and the clock will fire it. A personal agent is idle most of the time and sometimes acts on its own; turn-based benchmarks cannot represent that.

Still missing everywhere: in real life a lot happens between sessions. Emails arrive, plans change. In these benchmarks nothing happens between sessions, so memory only ever holds useful material.

## 6. Harness evolution (adjacent)

Instead of writing notes, the agent can rewrite its own harness: prompts, skills, tools, code. This is next door to continual learning. One change affects every future task, so gains and losses are both bigger.

**EvoAudit** tested four such methods. Every evolved agent solved some new tasks and lost some old ones. Best case: +15 tasks, −4. Gains were 2 to 7 points. Most cost more to run. A related paper found 54% of reported gains were overfitting to the tasks used during evolution.

**A-Evolve** is a framework for this. It edits files in the agent's workspace, keeps a change only if it passes held-out tasks, and tags each kept change in git.

## 7. The benchmarks

**ApprenticeBench** (NeoCognition). A new accounts-payable clerk's first hundred bills at a simulated construction company. It gets six months of old bills and a month of mentor feedback, then keeps notes in a text file. Best model 72%, best human 51%. Without history or feedback, 11%. Almost no drift. Private, single run, two humans.

**CLE** (UCL). 120 job tasks, each five levels where later levels build on earlier ones. Best agent clears 48% of levels. Feedback plus one retry adds a median 5 points, which is small. No no-memory baseline, so you cannot tell learning from a second try.

**ServeLearnBench** (CMU, Amazon). Retail, banking, and sales tasks with hidden rules that change. Score-only feedback. Given the rule, agents score 95; with no memory, 14; best learner, 62. Learning costs 4 to 100 times more than being told. The more different answers a method tries, the better it learns.

**ReScopeBench** (HKUST, Huawei). 18 fake business apps, 26 silent changes, a lookup budget, and grading of the memory itself (Section 4). Best model goes from 26 without memory to 81 with it.

**MemPABench** (NYU Abu Dhabi). Interaction preferences and autonomy (Section 5).

**SCLATE** (Apple). A simulated clock and agent scheduling (Section 5). Also the only one that trains weights: a 4B model gains 8 to 12 points on later days.

**EvoAudit** (USTC and others). Harness evolution (Section 6).

**AgentCL** (Ohio State). Dependent vs conventional streams, and the plasticity, stability, generalization metrics (Sections 1 and 2).

## 8. Open problems

- Grade memory contents on a simulated calendar: ReScopeBench's answer key on SCLATE's clock.
- Make asking cost something. In MemPABench, always asking first is free.
- Add noise between sessions.
- Get human baselines. Only ApprenticeBench has any.
- Run things more than once. Most results are a single run, and run-to-run noise (2 to 5 points) is as big as most gains.

## Skimmed, not read

| Benchmark | arXiv | One line |
|---|---|---|
| CL-Bench | 2606.05661 | Hidden structure in six domains; plain context beats memory systems on most tasks |
| AgentStream | 2608.00155 | Conventional streams; only 34 of 45 setups beat no memory |
| CLR-Bench | 2610.40758 | Hidden rules added per level; one fixed lesson beats every self-improvement method |
| EvoHarnessBench | 2609.04280 | Adding tools and skills makes the agent forget old tasks |
| Δτ-Bench | 2610.34238 | Tool set changes over time; best 42% |
| Harness-Delta Attribution | 2610.22466 | Most harness-evolution gain is overfitting |
| Memento 3 | 2610.11794 | Model writes code that emulates the environment and plans inside it |
| LifeVal | 2611.06687 | Models rarely use facts buried in past conversations |
| ARC-AGI-3 | 2603.24621 | Learning inside one game; humans 100%, models under 1% |
