# Continual learning for LLM agents: benchmarks, approaches, papers

State on 2026-11-25.

This page is a map of how language-model agents learn from a stream of experience. It has three parts: the benchmarks that can be run today, the approaches people use, and the papers behind each approach. Each paper gets one or two lines, a year and a link.

Two words used throughout:

- **Stream.** A sequence of tasks or texts given in order, where what happens early affects what is scored later.
- **Fast weights.** Parameters that change while the model is being used, as opposed to slow weights, which change only in training.

One running example: the model reads "Alice lives in Paris", later reads "Alice moved to Rome", and is then asked where Alice lives.

## 1. Benchmarks

A benchmark is listed if it passes two tests. It is **reproducible**: code and data are public, or gated with approval that can be requested. It is **continual**: there is a stream, and early items affect later scores. Nothing here has been installed or run. "Reproducible" means the repo and data pages were opened and checked.

The buckets sort by what changes over time.

### Sequential tasks

Nothing in the world changes. Later tasks reuse what earlier tasks taught.

- **[AgentOdyssey](https://github.com/agentodyssey/agentodyssey)** ([2606.24893](https://arxiv.org/abs/2606.24893), May 2026). A 500-step text game where the agent must learn world rules it cannot know in advance. Smallest open model run: Qwen3-4B. LoRA fine-tuning, long context and RAG are all implemented in one harness.
- **[CL-Bench](https://github.com/pgasawa/continual-learning-bench)** (June 2026). About 300 sequential tasks across domains, two of which drift. Frontier models only, at $7.6 to $62.8 per run. A local model needs an adapter.
- **[CLE-Lite](https://huggingface.co/Continual-Learning-Exam)** ([2610.33956](https://arxiv.org/abs/2610.33956), Oct 2026). 20 office jobs of five connected levels each. Each level builds on the last deliverable. Task cards and input files are open; rubrics and runnable packages need approval. The runner drives Codex or Claude Code only. No open model reported.
- **[ComposeCL](https://github.com/cozheyuanzhangde/compose-cl)** ([2609.06986](https://arxiv.org/abs/2609.06986), Sep 2026). 100 batches of facts written into weights one after another, then tested on how many are retained. Qwen3-4B. Plain sequential fine-tuning retains 1.2%; the best method 34.9%.
- **[MedKIT](https://github.com/bethgelab/MedKIT)** ([2609.38543](https://arxiv.org/abs/2609.38543), Sep 2026). 48 weekly batches of new oncology evidence, tested on recall and on reasoning with it. 4B models. Training on the stream gains 49 points on recall and about zero on questions that combine facts.

### Environment drift

The rules or state of the world change under the agent, and nobody says so.

- **[ServeLearnBench](https://github.com/Infini-AI-Lab/ServeLearnBench)** (Oct 2026). Customer-service and banking jobs where hidden rules switch on and off. The agent only sees its score. Told the rule, agents score 95.4; with no memory, 14.1. No model under 35B has been run.
- **[ReScopeBench](https://github.com/JoeYing1019/ReScopeBench)** (Nov 2026). Tool-use streams in 18 fake business apps with 26 silent changes, under a lookup budget. Smallest open model: Qwen3.5-9B, which scores 5.76 against 60.19 for the best model. The memory score reads a text memory, so it cannot grade a memory held in weights. The task score can.
- **[OAKS](https://github.com/kaistAI/OAKS)** ([2603.07392](https://arxiv.org/abs/2603.07392), Mar 2026). A long text arrives in about 65 chunks and the same questions are asked again as their answers change. Qwen3-4B scores 26.4% against 66.3% for the best model. No weight-update method has been tried on it. The data license is noncommercial.
- **[studying-cl](https://github.com/anneharrington/studying-cl)** ([2607.07847](https://arxiv.org/abs/2607.07847), Jul 2026). Streams where Wikipedia facts are overwritten across time slices, plus a domain-shift stream for forgetting. Qwen3-8B, with In-Place TTT and five other weight methods already run.

### Preference change

The user's wishes change, and differ by context.

- **[MemPABench](https://github.com/mempabench/mempabench)** ([2610.30485](https://arxiv.org/abs/2610.30485), Oct 2026). A personal agent works with four simulated users over 99 sessions and must infer preferences they never state. No memory 49.6%, file memory 59.2%, told the answer 99.0%. Smallest open model: Qwen3.8-27B. The code calls OpenRouter only.

### Time and lifecycle

Days pass, sessions restart, and the agent schedules its own actions. This bucket is empty. SCLATE is the only benchmark of this kind and it has no code release.

### Two kinds of stream

The buckets say what changes in the world. They do not say what the model has to pick up.

- **Knowledge streams** (OAKS, studying-cl, MedKIT, ComposeCL). The fact is stated in the text, and the test is a question. In OAKS the text says a character moved, and the model is asked where they live now.
- **Agent task streams** (AgentOdyssey, CL-Bench, CLE-Lite, ServeLearnBench, ReScopeBench, MemPABench). Nobody states the lesson. In ReScopeBench each task has failed provider calls to restore under a lookup budget. The same providers come back in later tasks, so the agent has to remember what each one needed instead of paying to look it up again.

The second kind is harder at both ends: finding the lesson, and using it. A method can do well on OAKS and fail on ReScopeBench, though both are drift benchmarks.

### Where to start

Four of the ten have been run on a model of 8B or smaller with programmatic scoring: AgentOdyssey, studying-cl, OAKS and ComposeCL. AgentOdyssey is the only one with weight-update, long-context and RAG arms in one harness at 4B.

## 2. Approaches and their papers

An agent is a model plus the things around it: a prompt, a memory store, tools. Learning can land in any of them. The groups below go from the most reversible place (the prompt) to the most permanent (the weights).

One pattern holds in every group. An update applied on every token or every episode makes the system worse. An update that must first pass a check from outside the learner (a verifier, a frozen copy of the model, a held-out test) is what works.

### 2.1 Context: the baseline

Keep the whole history in the prompt. This is not a learning method, but every other method has to beat it, and on picking up new facts it usually wins. It stops working when the history no longer fits. What is unproven is whether anything else matches it on facts that must be combined.

- **Cartridges comparison** (Amazon, 2026, [2609.17346](https://arxiv.org/abs/2609.17346)). To teach one document, LoRA needed about 40,000 synthetic examples to reach 64.5 accuracy. The document in the prompt gives 73.6.
- **ImplicitHop** (2026, [2610.41169](https://arxiv.org/abs/2610.41169)). Two facts stored in weights chain together 37 to 64% of the time across 15 models. The same facts in the prompt chain 100% of the time.
- **Hypernetwork two-hop study** (2026, [2610.15677](https://arxiv.org/abs/2610.15677)). A network that writes an adapter from a fact gets 0.25% on two-hop questions, against 83% with the fact in the prompt.

### 2.2 Text memory and notes

Write a note when a task ends. Fetch the most relevant notes when the next task starts. Every agent benchmark above tests this family (RAG, Mem0, MemOS, file memory, ReasoningBank). It helps only when later tasks reuse earlier ones. The weak step is fetching the right note at the right moment, not storing it.

- **AgentCL** (Ohio State, 2026, [2606.02461](https://arxiv.org/abs/2606.02461)). On a stream where later tasks need earlier solutions, memory methods gain 13.5 to 21.9 points. On a stream that only shares a topic, the same methods lose 0.3 to 9.4.
- **MemPABench** (NYU Abu Dhabi, 2026, [2610.30485](https://arxiv.org/abs/2610.30485)). All four memory stores held the user's objection. Retrieval put the right preference in front of the agent in 7 of 334 probes.
- **TTME** (2026, [2610.32046](https://arxiv.org/abs/2610.32046)). Evolves a text memory with frozen weights and beats weight-updating methods on five math-discovery tasks with 83 to 85% less compute.
- **NoteBank** (2026, [2610.15715](https://arxiv.org/abs/2610.15715)). A small text memory added to a 1.3B fast-weight model lifts recall at 1M tokens from 1.7% to 70.3%.

### 2.3 Harness self-rewriting

The harness is everything around the model: prompts, skills, tools, code. Here the agent edits its own harness and keeps a change if it scores better. One change touches every later task, so gains and losses are both larger. Audits find that much of the reported gain does not hold up.

- **GEPA** (2025, [2507.19457](https://arxiv.org/abs/2507.19457)) and **ReasoningBank** (2025, [2509.25140](https://arxiv.org/abs/2509.25140)). Rewrite the prompt or a bank of lessons from past attempts.
- **SkillOpt** (2026, [2605.23904](https://arxiv.org/abs/2605.23904)) and **PolicyBank** (2026, [2604.15505](https://arxiv.org/abs/2604.15505)). Grow a library of reusable skills.
- **Continual Harness** (2026, [2605.09998](https://arxiv.org/abs/2605.09998)). A second agent revises prompt, memories, skills and helpers on a schedule. The acting agent never sees scores.
- **Harness-Delta Attribution** (2026, [2610.22466](https://arxiv.org/abs/2610.22466)). 54% of reported gain is fitting to the search set and 14% is extra compute.
- **CODA** (2026, [2610.20845](https://arxiv.org/abs/2610.20845)). Drift in the skill bank erases 52 to 93% of the gain.
- **EvoAudit** (USTC, 2026, [2611.07624](https://arxiv.org/abs/2611.07624)). All 16 evolved agents gain some tasks and lose others. Gains are 2 to 7 points, inside run-to-run noise for many.
- **"Learning, Fast and Slow: Towards LLMs That Adapt Continually"** (Tiwari, 2026, [2605.12484](https://arxiv.org/abs/2605.12484)). Not a fast-weight paper, despite the title. The fast part is a set of system prompts rewritten by another model. On its task stream, forgetting did not separate the methods; whether the model could still learn the next task did.

### 2.4 Fast weights: the roots

A slow network decides what to store. A second set of weights, which starts empty, is rewritten as inputs arrive. All later work in 2.5 to 2.8 is one loop: form a key, form a target, check what the memory returns, nudge the memory, read.

- **"Fast weight programming and linear transformers: from machine learning to neurobiology"** (Irie, 2025, [2508.08435](https://arxiv.org/abs/2508.08435)). The primer to read first. Recent models differ only in the write rule: plain add, add after shrinking, or the delta rule.
- **"Linear Transformers Are Secretly Fast Weight Programmers"** (Schlag, 2021, [2102.11174](https://arxiv.org/abs/2102.11174)). Attention without softmax is a matrix that each token adds to. Adding "Rome" leaves Paris and Rome mixed. The delta rule fixes this: look up what the matrix returns for Alice, then write the difference.
- **"Using Fast Weights to Deblur Old Memories"** (Hinton, 1987). Every connection gets a slow weight and a fast weight that decays. Rehearsing 10 of 100 old pairs restores the other 90.
- **"Learning to Control Fast-Weight Memories"** (Schmidhuber, 1992). A slow network outputs changes to a fast network's weights. Two toy tasks.
- **"Using Fast Weights to Attend to the Recent Past"** (Ba, 2016, [1610.06258](https://arxiv.org/abs/1610.06258)). A second matrix on an RNN that decays and adds at every step. The rule is fixed by hand.
- **"HyperNetworks"** (Ha, 2016, [1609.09106](https://arxiv.org/abs/1609.09106)). A small network outputs a larger one's weights, rescaling them at every step. Nothing accumulates, so it is not a memory.

### 2.5 Better write rules

Here the memory replaces attention. The state is one matrix per head, wiped at the start of each sequence. This line is in production models because it makes long context cheap. It says nothing about learning across sessions, and recall still trails full attention.

- **DeltaNet: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"** (Yang, 2024, [2406.06484](https://arxiv.org/abs/2406.06484)). The delta rule computed in chunks of 64 tokens so it trains at scale. At 1.3B it matches a Transformer on perplexity and trails on retrieval (49.5 against 66.6).
- **"Gated Delta Networks: Improving Mamba2 with Delta Rule"** (Yang, 2024, [2412.06464](https://arxiv.org/abs/2412.06464)). Shrinks the whole matrix before each write, so the model can wipe memory when the topic changes. Finding a number in a 4K-token essay goes from 18.6 to 92.2.
- **"Kimi Linear: An Expressive, Efficient Attention Architecture"** (Kimi Team, 2025, [2510.26692](https://arxiv.org/abs/2510.26692)). One shrink factor per row, so some rows last and some fade. In a 48B production model, with up to 75% less cache.
- **"Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"** (Grazzi, 2024, [2411.12537](https://arxiv.org/abs/2411.12537)). Letting the write strength reach 2 flips the sign of what is stored. Parity accuracy goes from 0.017 to 1.000.
- **"DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products"** (Siems, 2025, [2502.10297](https://arxiv.org/abs/2502.10297)). Several delta writes per token. Two flips make a rotation.
- **"Complex KDA"** (Siems, OpenEuroLLM, 2026, [2609.24797](https://arxiv.org/abs/2609.24797), [code](https://github.com/OpenEuroLLM/ComplexKDA)). Lets Kimi's per-row factor go negative, so one write makes a rotation. On par with Kimi's rule on language (perplexity 15.78 against 15.73).
- **"Longhorn: State Space Models are Amortized Online Learners"** (Liu, 2024, [2407.14207](https://arxiv.org/abs/2407.14207)). Derives the write as "the closest memory that also returns the new value". The trained model uses a shortcut that does not subtract what the memory returns.
- **"MesaNet: Sequence Modeling by Locally Optimal Test-Time Training"** (von Oswald, 2025, [2506.05233](https://arxiv.org/abs/2506.05233)). Reads as if the memory were the best fit to every pair so far. The best fit of "Paris" and "Rome" is their average, so overwriting depends on the forget gate (our inference).
- **KronLSTM** (Hochreiter lab, 2026, [2610.26834](https://arxiv.org/abs/2610.26834)). At 1.5B it beats Gated DeltaNet on perplexity but scores 11.6 against 99.2 on finding one fact at 16K tokens.
- **MemT-1.0** (MemTensor, 2026, [2610.26825](https://arxiv.org/abs/2610.26825)). The largest deployment: a 288B model with delta-rule memory blocks whose state lasts across the turns of one task. It averages 50.6 on 11 memory benchmarks, against 65.0 for full context.

### 2.6 Why it works

Theory that ties attention, fast weights and gradient descent together. Mostly toy settings.

- **"Transformers learn in-context by gradient descent"** (von Oswald, 2022, [2212.07677](https://arxiv.org/abs/2212.07677)). One attention layer without softmax can output what one gradient step on linear regression would. Training finds those weights.
- **"Transformers learn to implement preconditioned gradient descent for in-context learning"** (Ahn, 2023, [2306.00297](https://arxiv.org/abs/2306.00297)). Proves the best single layer is that gradient step with a trained rescaling.
- **"Test-time regression: a unifying framework for designing sequence models with associative memory"** (Wang, 2025, [2501.12352](https://arxiv.org/abs/2501.12352)). Each layer fits a function from keys to values. DeltaNet is one gradient step per token; Gated DeltaNet adds weight decay.
- **Miras: "It's All Connected"** (Behrouz, 2025, [2504.13173](https://arxiv.org/abs/2504.13173)). Makes the loss and the forgetting penalty design choices. A capped loss stops one garbled token from wrecking the memory.
- **"Test-Time Training with KV Binding Is Secretly Linear Attention"** (Liu, 2026, [2602.21204](https://arxiv.org/abs/2602.21204)). On two models, more inner steps hurt and gradient ascent works as well as descent. Tested only where the inner loss is a dot product.
- **TTT checkpoint ablation** (2026, [2610.16268](https://arxiv.org/abs/2610.16268)). Removing the prediction-error part of the gradient costs 2.8 to 2.9% perplexity. Freezing the fast weights costs 4.2 to 4.5 times. The fast weights help; the error-correction story explains little of why.
- **"Test-time Memory From The Online Learning Perspective"** (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=hIYLpBqjZL), abstract only). Recursive least squares has a provably small error; a constant-step delta rule can be much worse.

### 2.7 Writing into the model's own weights

This is the group that bears on continual learning. The update lands in one of the model's own MLP layers or blocks, not in a side matrix. Until fall 2026 every paper reset the weights at the end of each document. Several add-ons to pretrained models now keep them across sessions. What is unproven: nobody has measured forgetting across sessions for a model built this way from scratch.

Three levels of work per write: a closed formula (In-Place TTT, TTCD), a backward pass through a small inner model (TTT layers, Titans, ATLAS, LaCT), and a backward pass through real blocks (TTT-E2E).

**Built from scratch, reset per sequence**

- **TTT layers: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"** (Sun, 2024, [2407.04620](https://arxiv.org/abs/2407.04620)). The state is a small model. Reading 16 tokens means one training step on it. Keeps improving to 32K context where Mamba levels off at 16K.
- **"Titans: Learning to Memorize at Test Time"** (Behrouz, 2025, [2501.00663](https://arxiv.org/abs/2501.00663)). An MLP memory beside attention, written harder when the input is surprising, with a forget gate. Removing the forget gate hurts most.
- **"ATLAS: Learning to Optimally Memorize the Context at Test Time"** (Behrouz, 2025, [2505.23735](https://arxiv.org/abs/2505.23735)). Each update fits the last few tokens together, so a stale pair can be down-weighted.
- **LaCT: "Test-Time Training Done Right"** (Zhang, 2025, [2505.23884](https://arxiv.org/abs/2505.23884)). One update per chunk of 2,048 tokens or more. Needle retrieval at 32K: 48.0 against 91.2 for full attention.
- **TTT-E2E: "End-to-End Test-Time Training for Long Context"** (Tandon, 2025, [2512.23675](https://arxiv.org/abs/2512.23675)). Every 1K tokens, one gradient step on next-token loss through the last quarter of blocks. Needle retrieval at 128K: 0.06 against 0.99.
- **ParaTTT** (MIT CSAIL and Liquid AI, 2026, [2610.38255](https://arxiv.org/abs/2610.38255)). A two-layer MLP memory trained 4.9 times faster than TTT-MLP. Recall falls from 53.8 at 4K to 14.7 at 32K.
- **DeltaTTT** (Tsinghua, 2026, [2610.08553](https://arxiv.org/abs/2610.08553)). Two matrices, each updated by its own delta rule. Single-needle 50.47 against 72.42 for full attention.
- **uTTT** (UW-Madison and Adobe, 2026, [2610.05484](https://arxiv.org/abs/2610.05484)). One fast-weight set shared by all layers. RULER average 27.85 against 34.16 for a Transformer.
- **GradLev** (UT Austin, 2026, [2609.34174](https://arxiv.org/abs/2609.34174)). Gradient steps on the true next-token loss in the final MLP. Gains are inside the error bars.
- **"Fast Weights in Fast Weights"** (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=bZXUddyKmp), abstract only). First MLP layer updated per chunk, output layer per token. Claims 8.5 long-context points over LaCT.
- **"Delayed Supervision for Test-Time Language Models"** (Kim, 2026, [2609.32312](https://arxiv.org/abs/2609.32312)). Trains with questions asked at least 4,096 tokens after the evidence, so a fact must survive later writes. LaCT on BABILong goes from 25.60 to 31.08. One seed.

**Added to a pretrained model, reset per sequence**

- **"In-Place Test-Time Training"** (Feng, 2026, [2604.06169](https://arxiv.org/abs/2604.06169)). Adds to an MLP matrix the model already has, by a closed formula, in every sixth layer. Qwen3-4B gains 2 to 4 points on RULER.
- **TTCD: "Learning What to Remember: Test-Time Training via Context Distillation"** (Wang, 2026, [2608.01672](https://arxiv.org/abs/2608.01672)). A teacher sees a long window and a student half of it. The difference is written into the same MLP slot. Gains under one point on 8B models.
- **Titan-Llama** (2026, [2610.19210](https://arxiv.org/abs/2610.19210)). A 17M-parameter MLP memory in the first four layers of Llama-3.2-1B. QuALITY goes from 23.9 to 40.5, against 43.3 for full attention.
- **Select-TTT** (USTC and Huawei, 2026, [2610.36794](https://arxiv.org/abs/2610.36794)). Writes harder on text relevant to the question. LongMemEval-S goes from 33.4 to 37.4, where writing everything equally gives 33.6.
- **Dynamic evaluation study** (2026, [2610.33025](https://arxiv.org/abs/2610.33025)). One gradient step per 128 tokens on untrained Qwen3 models lowers next-token loss 5 to 8% and recovers no retrieval accuracy. The model picks up style, not facts.
- **ACE: "Encoding Context into LLM Weights: What Makes It Context?"** (Wang, Melbourne and RIKEN, 2026, [2610.17686](https://arxiv.org/abs/2610.17686)). Writes each chunk under a random tag, then names the tag in the question. Recall of a short fact on a 27B model goes from 1.5% to 75.6%. So the fact was stored all along, and the failure is access.
- **Anchor** (William, Arizona, 2026, [2611.04327](https://arxiv.org/abs/2611.04327)). Recall through the memory lives in a few layers, so the state is written only there. Cuts state on Llama-3.1-8B from 201 MB to 38 MB.
- **Self-generated text study** (2026, [2610.05076](https://arxiv.org/abs/2610.05076)). Keeping updates computed on the model's own output makes it worse on human text. A check on held-out real text removes over 98% of the damage.
- **"Test-Time Training for Pretrained LLMs: An Empirical Study of Limited Fast-Weight Utilization"** (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=x7actWdqhZ), abstract only). Claims most of In-Place TTT's gain comes from how the base model was trained, not from the writes at inference.
- **"Dynamic Writes, Simple Schedules: An Oracle Study of In-Place TTT"** (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=t2exYd8L0y), abstract only). A perfect choice of where and when to write lifts RULER-16K from 78.38 to 79.77.

**Kept across sessions**

- **"Cortex: Consolidating Context into Parametric Memory"** (Zhang, Zhejiang and ByteDance, 2026, [2610.20399](https://arxiv.org/abs/2610.20399), [code planned](https://github.com/zjunlp/cortex)). The model writes questions a text will matter for, answers them with the text in view, then trains a few new parameters to give the same answers without it. Earlier writes are frozen. After 50 edits in a row on a 14B model it keeps 88.9% of the benefit of having the text in the prompt; the best baseline keeps 28.5%. Each write is a small training run.
- **"LSTMem: Hierarchical Long Short-Term Online Memory for LLMs"** (Shi, USTC, 2026, [2609.33268](https://arxiv.org/abs/2609.33268)). Two small gated matrices per layer on a frozen model, carried across the sessions of one conversation. MemoryAgentBench on Qwen3-4B goes from 29.54 to 45.08. Replacing an old fact, the Alice case, only moves from 14.37 to 18.00.
- **"Delta N-gram"** (Wu, Illinois Tech, 2026, [2610.31125](https://arxiv.org/abs/2610.31125), [code](https://github.com/af-74413592/delta-ngram)). Delta-rule writes, forward passes only, into a lookup table one production model already has. After 35,000 writes earlier tasks lose 0.26 points. Little is forgotten because little is learned: 0.44 F1 on questions about written text, against 56.79 with the text in the prompt.
- **"DPPM: Dual-Path Parametric Memory for Personalized Language Models"** (Chen, USTC and iFLYTEK, 2026, [2610.11776](https://arxiv.org/abs/2610.11776), [code](https://github.com/isyuhaochen/DPPM)). A key-value matrix per user, updated by the delta rule across sessions and turned into an adapter. PersonaMem-v2 on Qwen3-4B: 54.22 against 45.73 for its predecessor. Retention is measured on internal states, not answers.
- **ProTTT** (Park, Seoul National, 2026, [2609.32564](https://arxiv.org/abs/2609.32564)). A per-user adapter updated by one gradient step per chunk of history. Qwen3-0.6B. General ability drops (MMLU .474 to .432).
- **P-TTT** (Wang, Zhejiang and NUS, 2026, [2609.35109](https://arxiv.org/abs/2609.35109)). In-Place TTT with one write per whole response, kept per user. 74.48% on a preference task where In-Place TTT is at chance. 1B models only.
- **δ-mem: "Efficient Online Memory for Large Language Models"** (2026, [2605.12357](https://arxiv.org/abs/2605.12357)). The baseline LSTMem and DPPM compare against.
- **"All Memory Is Local: Measuring the Horizon of Persistent Test-Time Training"** (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=gXFgHs8TiZ), abstract only). Removes the reset from drop-in TTT on Qwen3 0.6B to 4B. The gain tied to a session falls by about four fifths within five sessions and is gone by fifteen. Resetting keeps 60% of the benefit.
- **"Learning Whether to Learn"** (ICLR 2027 submission, [OpenReview](https://openreview.net/forum?id=XYI4FitZL2), abstract only). A second model decides whether each write happens, and only writes that keep proving useful become permanent.
- **"Beyond Update Counts: A Control Map for Evaluating Test-Time Model Updates"** (Zhao, Meta, 2026, [2610.36109](https://arxiv.org/abs/2610.36109)). An evaluation method: score learning, later recall and side damage separately. LoRA edits pass learning in 6 of 6 settings and retention in 0 of 6.

### 2.8 Models that learn to write their own weights

"Writes its own weights" has three meanings, from weakest to strictest.

**A network writes a separate memory.** This is everything in 2.5 and 2.7. A frozen network decides what to write, and a fixed rule writes it.

**A network changes the weights that compute the change.** No frozen part. No paper trains one of these as a language model. The primer gives the reason: merging the slow and fast networks loses parallel training.

- **"A 'Self-Referential' Weight Matrix"** (Schmidhuber, 1993). A network with extra units that read and edit any of its own connections. No experiments.
- **"A Modern Self-Referential Weight Matrix That Learns to Modify Itself"** (Irie, 2022, [2202.05780](https://arxiv.org/abs/2202.05780)). One matrix produces its output and also the key, target and step size for a delta-rule edit of itself. At most 3.6M parameters, images and games, no language.
- **"Metalearning Continual Learning Algorithms"** (Irie, 2023, [2312.00276](https://arxiv.org/abs/2312.00276)). Trains that matrix with "still do task A after learning task B" in the loss. First-task accuracy after the second task goes from 22.9% to 98.2%. Small image tasks.
- **Hope: "Nested Learning: The Illusion of Deep Learning Architectures"** (Behrouz, 2025, [2512.24695](https://arxiv.org/abs/2512.24695)). Small MLPs produce the keys, targets and step sizes for updates to themselves. Trained as a 1.3B language model. Its continual-learning results are figures without numbers.

**The model produces its own update and an optimizer applies it.**

- **SEAL: "Self-Adapting Language Models"** (Zweiger, 2025, [2506.10943](https://arxiv.org/abs/2506.10943)). The model writes study notes from a passage and a fine-tuning step trains on them. Notes that help are reinforced. Accuracy without the passage on Qwen2.5-7B goes from 33.5% to 47.0%. Earlier passages are forgotten as edits pile up.
- **"PAPER2LLM++: Continual Self-Evolution of LLMs from Research Papers"** (Pu, 2026, [2610.02793](https://arxiv.org/abs/2610.02793)). For each paper describing a failure, generate training data, try a temporary adapter, and keep it only if nothing earlier gets worse. Retention 83.7% over 30 steps on Llama-3.1-8B. The data is written by a different, larger model.

### 2.9 Editing weights from outside

Something other than the model changes its weights: a formula, a trained editor, a compiler, or a second language model. The newest version has a large model read a small model's weights as text and write new numbers. It works on a 0.8B model and not yet on a 9B one. Nobody has had a model read and rewrite its own weights this way.

**By formula or trained editor**

- **MEND: "Fast Model Editing at Scale"** (Mitchell, 2021, [2110.11309](https://arxiv.org/abs/2110.11309)). The gradient from one example is the product of two vectors. A small network adjusts those two vectors so the edit fixes the fact and leaves other answers alone.
- **ROME: "Locating and Editing Factual Associations in GPT"** (Meng, 2022, [2202.05262](https://arxiv.org/abs/2202.05262)). Finds the MLP layer that stores a fact and overwrites it with a closed formula.
- **MEMIT: "Mass-Editing Memory in a Transformer"** (Meng, 2022, [2210.07229](https://arxiv.org/abs/2210.07229)). ROME for thousands of facts at once.

**Weights as readable programs**

- **RASP: "Thinking Like Transformers"** (Weiss, 2021, [2106.06981](https://arxiv.org/abs/2106.06981)). A small programming language whose operations match what a transformer layer can do.
- **"Tracr: Compiled Transformers as a Laboratory for Interpretability"** (Lindner, 2023, [2301.05062](https://arxiv.org/abs/2301.05062)). A compiler turns a RASP program into exact transformer weights. Nothing is trained.
- **"Equivariant Architectures for Learning in Deep Weight Spaces"** (Navon, 2023, [2301.12780](https://arxiv.org/abs/2301.12780)) and **"Permutation Equivariant Neural Functionals"** (Zhou, 2023, [2302.14040](https://arxiv.org/abs/2302.14040)). Networks that take another network's weights as input and predict its properties. They read but do not write.

**A language model as reader and editor**

- **MAIA: "A Multimodal Automated Interpretability Agent"** (MIT, 2024, [2404.14394](https://arxiv.org/abs/2404.14394)). A language model inspects another model and runs experiments on it. It reads but does not write.
- **"Sparks of Weight-Space Programming"** (Takehi, 2026, [MIT CSAIL post](https://rikiyat.github.io/blog/weight-space-programming/)). An off-the-shelf frontier model gets a network's weights as text and replies with a diagnosis or with new numbers. No training is involved. An independent rerun found reading holds up and writing gains are smaller than reported.
- **"Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations"** (Hu, Shanghai AI Lab, 2026, [2611.09382](https://arxiv.org/abs/2611.09382), [code](https://github.com/WSP-Scope/WSP-Scope-Project)). GPT-6 Astra edits Qwen3.5-0.8B with 50 forward passes and no gradients: math goes from 8.35 to 37.80. The 9B edit is discarded, and 6 of 16 searches keep the original weights.

### 2.10 Document-to-adapter networks

A trained network reads a text and outputs an adapter (a small set of added weights) in one forward pass. The write is fast and needs no gradient steps at use time. It holds one fact well and loses it once other facts share the adapter.

- **"Text-to-LoRA: Instant Transformer Adaption"** (Charakorn, 2025, [2506.06105](https://arxiv.org/abs/2506.06105)). Reads a task description and outputs an adapter. 67.7 on unseen benchmarks against 55.8 for the base model.
- **"Doc-to-LoRA: Learning to Instantly Internalize Contexts"** (2026, [2602.15902](https://arxiv.org/abs/2602.15902)) and **SHINE** (2026, [2602.06358](https://arxiv.org/abs/2602.06358)). The same idea for documents. The baselines the fall papers compare against.
- **"Cache2LoRA: Understanding and Improving Context Internalization"** (Fang, Virginia Tech and Stanford, 2026, [2610.24107](https://arxiv.org/abs/2610.24107)). Builds the adapter from the document and the question together. Doc-to-LoRA recalls one fact at 0.982 alone and 0.184 once other facts are added.
- **CLIK: "Encode Once, Answer Many"** (Zhong, 2026, [2610.38356](https://arxiv.org/abs/2610.38356), [code](https://github.com/Cypher-Bruce/CLIK)). A 1B generator emits an adapter for Qwen3-4B. SQuAD F1 46.3 against 71.1 with the text in the prompt. Building each new adapter from the previous one fails: 25.2.
- **"HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning"** (Kim, 2026, [2610.03039](https://arxiv.org/abs/2610.03039)). Shifts about 90K bias values per question so the model answers as if it had reasoned first. GSM8K 74.81 to 84.75 on a 3B model. Not a memory.

### 2.11 Fine-tuning and adapters during deployment

Ordinary training, run while the agent is in use: fine-tuning, reinforcement learning, or distilling from a copy of the model that can see more. Slow and permanent. Plain fine-tuning forgets. The versions that work tie the update to something that does not move, such as a frozen copy of the base model.

- **ASCENT** (UNSW, 2026, [2610.05303](https://arxiv.org/abs/2610.05303)). Keeps one adapter across a task stream. After each episode a verifier accepts, it distills from a frozen copy that sees the verified attempt. ALFWorld on Qwen3.5-4B: 46.4 frozen, 63.6 best memory method, 69.5.
- **Subspace Tuning** (2026, [2610.22293](https://arxiv.org/abs/2610.22293)). Keeps each weight matrix's main directions untouched. On Qwen3-8B, plain fine-tuning drops earlier ability from 81.0 to 55.5; with the projection it holds 80.0.
- **Adam epsilon fix** (2026, [2610.09835](https://arxiv.org/abs/2610.09835)). Tokens absent from the new data get tiny gradients that Adam scales up to full steps. Raising one constant on the output layer removes 39 to 68% of forgetting.
- **Time-incremental pretraining** (Öncel, 2026, [2609.23916](https://arxiv.org/abs/2609.23916)). Continued pretraining on 6B tokens of newer text. Forgetting is about 0.60 times what is learned.
- **iSDFT** (Imperial and UCL, 2026, [2609.24646](https://arxiv.org/abs/2609.24646)). Distills from a copy that sees the answer in context, tied to the base model. Median change over 16 benchmarks is −0.05.
- **OPDArena** (Microsoft, 2026, [2610.25651](https://arxiv.org/abs/2610.25651)). Reinforcement learning gains 5 to 14 points in domain with about zero loss elsewhere. Supervised fine-tuning loses 2 to 4.5 elsewhere.
- **ReTTT** (Shanghai AI Lab and Fudan, 2026, [2610.23942](https://arxiv.org/abs/2610.23942)). Trains a per-episode adapter on the agent's own reflections. ALFWorld on Qwen3.5-9B goes from 47.1 to 63.6, at about three times the run time.
- **TTT-Consolidate** (EPFL, 2026, [2611.02104](https://arxiv.org/abs/2611.02104)). Encodes eight reasoning attempts into an adapter, then answers once. 35.4 where majority vote gets 22.2 to 22.7 at equal compute. Reset per problem.
- **Surprise-gated updates** (2026, [2610.39840](https://arxiv.org/abs/2610.39840)). Updating whenever the text is surprising makes Qwen2.5-1.5B absorb 84 to 93% of planted false claims.
- **One-step LoRA study** (2026, [2607.00368](https://arxiv.org/abs/2607.00368)). One LoRA step on Qwen3 1.7B to 8B lowers loss and gives 0% recall when the model has to generate the answer.

## 3. What the NeurIPS 2026 TTCL workshop adds

The Workshop on Towards Test-Time Continual Learning Agents (NeurIPS 2026) accepted 85 papers. We read 40 of them in full: AgentCL (section 1), 18 other benchmark and evaluation papers, and the 21 parametric papers. Ethos is the one exception: we read only its abstract. The other 45 papers (memory, skills, harness, other) are not read yet. Ids are OpenReview ids unless an arXiv id is given. We also add three papers from outside the workshop where they fit.

### 3.1 New benchmarks, by bucket

- **Sequential tasks.**
  - **"Are Self-Evolving Agents Continual Learners?"** ([2610.25858](https://arxiv.org/abs/2610.25858)). The authors run GEPA, ReasoningBank and Meta-Harness on drifting streams. Methods that learn new blocks well lose more on old blocks: plasticity and stability correlate at r = -0.72. GEPA has the highest plasticity (+6.2) and the worst stability (-4.3).
  - **"Is One Score Enough?"** ([2605.15384](https://arxiv.org/abs/2605.15384)). An evaluation protocol for memory that changes over a stream of existing datasets. On final hold-out accuracy, 197 of 292 method pairs (67%) cannot be told apart.
  - **"Useful Memories Become Faulty When Continuously Updated by LLMs"** ([2605.12978](https://arxiv.org/abs/2605.12978)). On 19 ARC-AGI problems that GPT-5.4 solves without memory, rewriting the memory every round drops accuracy from 100% to 52.6% by round 10. In the 400-step ARC-AGI Stream, keeping raw trajectories instead scores 76.6, against 38.6 with no memory, and does not degrade.
- **Environment drift.** **BEAST** (VpjnFuTNT4). A moderation classifier on a stream sorted so that later posts are less familiar. Retraining every 5 steps adds about 0.03 PR-AUC for about 42% more compute. The models are classifiers, not agents.
- **Preference change.**
  - **Replay A-B-A** (ThseoldG1J). Preferences switch from A to B and back to A. When the readout cannot represent B, replaying all past data does not help: LoRe scores 28.8% agreement on B with full replay and 28.7% with recent data only, while a readout that can represent B scores 53.1%.
  - **MemoryVCD** (ZMKKh7HEKB). Vision-language personalization, scored at snapshots as memory grows. The model gets no feedback.
- **Time and lifecycle.** **ReLiveGym** ([2610.00710](https://arxiv.org/abs/2610.00710)). Weeks of replayed real events with deadlines. Seed-to-seed variation is the largest share of variance on every task (52 to 95%). Only 2 of its 8 tasks give feedback the agent can learn from.
- **Recall, not continual learning.** MEME ([2605.12477](https://arxiv.org/abs/2605.12477)), DolphinBench ([2609.24971](https://arxiv.org/abs/2609.24971); built by Mem0, whose product tops two of its three harness groups), EndlessMemory, and RoboMME-Interference ([2606.22338](https://arxiv.org/abs/2606.22338)). In each, the agent answers over a finished history and is never expected to improve.

### 3.2 Audits of the evaluation

- **Null-Policy Audit** (zmPrKIqrXn). On AgentOdyssey, a random policy scores higher action diversity than 29 of 48 published agent set-ups (60%). A diversity metric needs a random-policy baseline before it counts as evidence of exploration.
- **What Self-Evolution Leaves Behind** (N6ouy2iqMu). In Gomoku, Reflexion memory raises a pressure-test score from 8/22 to at most 11/22, yet the agent wins 0 of 60 normal games. Endpoint scores hide what the agent did or did not learn.
- **When Correction Memory Fails** (d7EYjlyvpg). A pairwise judge with a correction memory gains nothing on held-out records (GPT-5.4: 0.712 with and without memory). Its one apparent gain came from retrieving the same MT-Bench question.
- **Cross-phase Knowledge Incorporation** (CzHArWQqRu). A position paper. It proposes two metrics for whether a later training phase keeps earlier knowledge, but runs no experiments.

### 3.3 Learning in the harness, with the optimizer's own memory

- **SIM** (Google, [2611.18856](https://arxiv.org/abs/2611.18856), outside the workshop). The harness optimizer keeps a record of each harness problem, the edits it tried, and their predicted and observed outcomes. It carries this record across a change of benchmark or model. With the record, continuation ends higher than harness-only continuation in all 10 settings (6 dataset changes, 4 Opus-to-Gemini changes). Each of these settings is one run. Only the coding study is repeated: SIM adds 2.36 points over 10 paired runs (p = 0.003).

### 3.4 Harness gains distilled into weights

- **Self-Improving Memory** (Meta, [blog](https://facebookresearch.github.io/RAM/blogs/self_improving_memory/), outside the workshop). Kimi-K2.6 rewrites its own memory prompts on three CL-Bench tasks (Database, Codebase, Spectrum). It then runs with the best prompts. The authors swap those prompts back to generic ones in the recorded runs and fine-tune the model (LoRA) on the result. At test time the model gets only generic prompts. The recorded runs show the model naming memory files, recording facts and running tests before edits. The authors call it an untested hypothesis that the gain comes partly from better memory use.

| Setting | CL-Bench score |
|---|---|
| Generic prompts, no fine-tuning | 36.1 |
| Best hand-written prompts | 44.6 |
| Prompts found by search | 56.0 |
| Fine-tuned on generic-prompt runs | 37.6 |
| Fine-tuned on improved-prompt runs, generic prompts at test | 43.1 |

The hand-written and searched prompts are the best observed per task, picked by score.

- **Offline distillation.** UniSD, PINT, RSD and Projected Distillation train with a teacher that sees information the student does not (gold answers, skills, or later turns). RAD ([2610.14572](https://arxiv.org/abs/2610.14572)) instead distills from a larger teacher that sees the same input. All five train offline. Only Projected Distillation reports forgetting metrics, and only within one training task.

### 3.5 Weights updated during deployment

- **TAHI** ([2609.04141](https://arxiv.org/abs/2609.04141)). The authors train a LoRA with DPO after each of 20 user sessions. Forgetting is not measured.
- **LLS** (6Qyl17csOg). Test-time adaptation of an image classifier on an unlabeled stream. Forgetting is not measured.
- **Hierarchical TTT** ([2610.06269](https://arxiv.org/abs/2610.06269)). RL updates a LoRA on a guidance model (Qwen3 8B or 14B) within one problem, using the verifier's reward on a frozen executor's output. Nothing carries across problems.
- **Cell-TR** (MqMNumTovo). Continual fine-tuning of a quantized model, where only the integer codes that change are shipped. On a 5-task stream the patch is 3.71 MB, against 52.01 MB for plain sequential fine-tuning.
- **OLGA** (Upwork, [2611.03496](https://arxiv.org/abs/2611.03496), outside the workshop). OLGA keeps a memory of 5 synthetic text examples. At each step it fine-tunes phi-1.5 from the base weights on that memory. On text classification it beats the best selection baseline by 6.6 stream-AUC points. The paper does not say the stream drifts, and it does not measure forgetting.

### 3.6 Context compiled into a trained cache or an adapter

- **Accordion** (FTlwDpcS0s). At each compaction, Accordion trains a new KV cache on the frozen model's answers to its own study questions, and freezes the earlier caches. On CL-Bench's spectrum task it scores 0.358 IoU against 0.220 for full in-context learning (Qwen3.5-9B).
- **DyPRAG** ([2503.23895](https://arxiv.org/abs/2503.23895)) and **AdapterBench** (J6lPOxI7V5). A hypernetwork turns a document or task description into a one-off adapter. Nothing accumulates across inputs.

### 3.7 Fast weights

- **"Position: Lifelong In-Context Learning with Transformers Requires Parametric Forms of Attention"** ([2606.25342](https://arxiv.org/abs/2606.25342)). The authors argue that a stream with no end needs attention with a fixed-size state that is updated online, as in linear attention and TTT layers. The paper runs no experiments.

### 3.8 Training procedures, not agent continual learning

- G-Zero, SynCo, Fixed Task Generator, Beam Search as Self-Distillation and Adapter Reuse each train or decode on one fixed distribution. None has a stream of tasks that the model must keep up with.
- Relearning Savings and Metric-Topological Factorization do run task streams and measure forgetting. Their models are a 1.5B LM on synthetic facts and small image classifiers, not agents.

## 4. Open gaps


- **Forgetting across sessions for a model built around fast weights.** Measured for add-ons to pretrained models, never for TTT layers, Titans or Hope.
- **Removing the reset may not help.** The one direct test ("All Memory Is Local", abstract only) finds the gain gone within fifteen sessions.
- **No self-referential matrix trained as a language model.** The closest attempt, building each adapter from the last one (CLIK), fails.
- **Nobody reads and rewrites their own weights as text.** Weight-space programming uses a large model to edit a small one, and the one 9B edit was discarded.
- **Stored is not the same as usable.** ACE shows a fact can be in the weights and still not be used. Facts in weights also do not chain.
- **The time and lifecycle bucket has no runnable benchmark.** No agent benchmark with a real clock can be run today.
- **No agent paper measures general ability after a long stream of weight updates.** Retention is checked on earlier agent tasks only.
