# Continual learning benchmarks that can be reproduced

State on 2026-11-11. `README.md` covers how the benchmarks work. This file answers a narrower question: which ones have code and data that can be run today, and which of those fit an experiment where a small open model updates its own weights across a stream.

A benchmark is listed if it passes two tests:

- **Reproducible.** The code and data are public, or gated with approval that can be requested.
- **Continual.** There is a stream, and what happens early affects what is scored later.

Nothing here has been installed or run. "Reproducible" means the repo and data pages were opened and checked.

## The shortlist, by bucket

The buckets are the four from `README.md`, sorted by what changes over time.

### Sequential tasks

Nothing in the world changes. Later items reuse earlier ones.

| Benchmark | Published | One line | Smallest open model run | Weight updates already run |
|---|---|---|---|---|
| [AgentOdyssey](https://github.com/agentodyssey/agentodyssey) (2606.24893) | May 2026 | A 500-step text game where the agent must learn world rules it cannot know in advance. | Qwen3-4B | LoRA fine-tuning |
| [CL-Bench](https://github.com/pgasawa/continual-learning-bench) | June 2026 | About 300 sequential tasks across domains, two of which drift. | frontier models only | none |
| [CLE-Lite](https://huggingface.co/Continual-Learning-Exam) (2610.33956) | Oct 2026 | 20 office jobs of five connected levels each, where each level builds on the last deliverable. | none reported | none |
| [ComposeCL](https://github.com/cozheyuanzhangde/compose-cl) (2609.06986) | Sep 2026 | 100 batches of facts written into weights one after another, then tested on how many are retained. | Qwen3-4B | all methods |
| [MedKIT](https://github.com/bethgelab/MedKIT) (2609.38543) | Sep 2026 | 48 weekly batches of new oncology evidence, tested on recall and on reasoning with it. | 4B | LoRA and model editing |

### Environment drift

The rules or state of the world change under the agent.

| Benchmark | Published | One line | Smallest open model run | Weight updates already run |
|---|---|---|---|---|
| [ServeLearnBench](https://github.com/Infini-AI-Lab/ServeLearnBench) | Oct 2026 | Customer-service and banking jobs where hidden rules switch on and off, and the agent only sees its score. | 552B MoE | none |
| [ReScopeBench](https://github.com/JoeYing1019/ReScopeBench) | Nov 2026 | Tool-use streams in 18 fake business apps with 26 silent changes, under a lookup budget. | Qwen3.5-9B | none |
| [OAKS](https://github.com/kaistAI/OAKS) (2603.07392) | Mar 2026 | A long text arrives in chunks and the same questions are re-asked as their answers change. | Qwen3-4B | none |
| [studying-cl](https://github.com/anneharrington/studying-cl) (2607.07847) | Jul 2026 | Staged streams where Wikipedia facts are overwritten across time slices, plus a domain-shift stream for forgetting. | Qwen3-8B | In-Place TTT and five others |

### Preference change

The user's wishes change, and differ by context.

| Benchmark | Published | One line | Smallest open model run | Weight updates already run |
|---|---|---|---|---|
| [MemPABench](https://github.com/mempabench/mempabench) (2610.30485) | 21 Oct 2026 | A personal agent works with four simulated users over 99 sessions and must infer preferences they never state. | Qwen3.8-27B | none |

### Time and lifecycle

Empty. SCLATE is the only benchmark in this bucket and it has no code release.

## Two kinds of stream

The buckets sort by what changes in the world. They do not say what the model has to pick up, and that splits the shortlist a second way.

- **Knowledge streams** (OAKS, studying-cl, MedKIT, ComposeCL). The model reads a fact and is later asked about it. The thing to remember is stated in the text, and the test is answering a question.
- **Agent task streams** (AgentOdyssey, CL-Bench, CLE-Lite, ServeLearnBench, ReScopeBench, MemPABench). Nobody tells the model the lesson. It has to work it out from its own attempts and errors, then use it by taking the right action in a later task.

The second kind is harder on both ends: finding the lesson and using it. A method could do well on OAKS and still fail on ReScopeBench, though both sit under drift.

## Notes per benchmark

**AgentOdyssey.** Apache-2.0. LoRA fine-tuning, long-context and RAG agents are implemented in one harness on Qwen3-4B. Scoring is programmatic: quest reward plus template questions on world knowledge. On the Mark game, LoRA with short-term memory gets the best quest reward (7, against 1 for RAG and 0 for long context) but its world-knowledge recall is 0.04 against 0.34 for RAG. About 1.2M tokens per run (estimated from tokens per step). No drift. The NeurIPS 2026 workshop challenge resets learned state between games.

**CL-Bench.** Apache-2.0, Python 3.13+, Docker, programmatic scoring. Wired to provider APIs and only tested on frontier models, at $7.6 to $62.8 per rollout. A local model needs an adapter.

**CLE-Lite.** Four datasets on the Hub. Task cards and workspaces (29.6 GB) are open. Rubrics and the full packages with the runner (52.1 GB) need manual approval. The other 100 tasks of the 120 are not released. The runner drives the solver through Codex or Claude Code (Codex only in stream mode) and has no base-URL flag. Scoring is an LLM judge on human-written rubrics, and a level passes only if every criterion passes. Needs Linux x86-64, Docker and about 30 GB of disk per task. No score is published for the 20-task set.

**ComposeCL.** No license file. Qwen3-4B-Base, 100 tasks with no replay data. Naive sequential fine-tuning retains 1.2%; the best method 34.9% on average. Retention only: no overwrite, and no in-context or RAG baseline.

**MedKIT.** Apache-2.0. LoRA, model editing and RAG compared on five 4B and 8B models. Continual post-training gains 49 points on the update probe and about zero on compositional questions. Exact match on closed-form probes; a gpt-4o judge on the rest.

**ServeLearnBench.** Apache-2.0. `slb run --scenario banking_l1 --method rag --model ...` against any OpenAI-compatible endpoint with tool calling. Told the rule, agents score 95.4; with no memory, 14.1. No model under 35B has been evaluated.

**ReScopeBench.** No license shown on the repo; data is Apache-2.0 (352 tasks). Docker, `OPENAI_BASE_URL` supported. Task scoring is programmatic. The memory score reads a text memory with an LLM judge, so it cannot grade a memory held in weights. Qwen3.8-27B scores 46.44 and Qwen3.5-9B scores 5.76, against 60.19 for the best model.

**OAKS.** Code has no license; data is under the Adobe Research License (noncommercial). 128k to 150k tokens in about 65 chunks, answers change 4.7 times on average. Full-context, RAG and memory-agent baselines are published. Qwen3-4B scores 26.4% and the best model 66.3%, so part of what a small model shows here is failure at the reading itself. No weight method has been tried; the authors list it as future work.

**studying-cl.** MIT. `./run.sh <method> <task_family>`. Programmatic scoring, reported as a forgetting matrix. In-Place TTT stays near baseline on the fact-overwrite stream. No full-context or RAG baseline. Each training phase took up to an hour on 8 H100s.

**MemPABench.** MIT; personas CC-BY-4.0. `uv sync`, then `mempabench run` and `mempabench score`. The README names OpenRouter only, so a self-hosted model needs a patch. Share of probes where the agent picks the current preference: random 45.2%, no memory 49.6%, Mem0, MemOS and RAG about 50%, file memory 59.2%, oracle 99.0%. The stored preference reached the agent's input in 7 of 334 judged probes for retrieval systems.

## Left out

**Reproducible but not continual as shipped**

- SE-Bench (2602.04811): train once, then test. No stream.
- WitnessBench (2609.32208): each game is an independent episode. Within a game the rule list carries across 20 levels, and the test set is private.
- MemoryBench (2510.17281): main results feed pre-generated feedback in batches.

**Continual but not reproducible**

- No code or harness: AgentCL (dataset only), SCLATE, ASCENT (2610.05303, the repo holds only a license file), SCoL SQuAD-100 (2605.07076), ForgetBench (2607.26455), "Can a Language Model Learn Facts Continually in Its Weights?" (2607.11020).
- Private: ApprenticeBench.
- Data not available: StudyBench (textbooks not released), Chronos (2604.05096, dataset download not confirmed).

## Where to start

Four of the ten have been run on a model of 8B or smaller with programmatic scoring: AgentOdyssey, studying-cl, OAKS and ComposeCL.

- **AgentOdyssey** is the only one with weight-update, long-context and RAG arms in one harness at 4B.
- **studying-cl** already has In-Place TTT on Qwen3-8B.
- **OAKS** is the densest overwrite test and has no weight method on it yet.

Two results from the search bear on the design:

- One-step LoRA on Qwen3-1.7B to 8B lowers loss but gives 0% generated recall (2607.00368).
- ASCENT keeps LoRA weights across a task stream with no reset and scores each task before updating. On ALFWorld with Qwen3.5-4B it reports 46.4 frozen, 63.6 for the best memory baseline and 69.5 for its method.
