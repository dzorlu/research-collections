# Reading list: fast weights and test-time weight updates

State on 2026-11-11. The papers behind `METHODS.md` section 5, grouped by the question each group answers. Each paper was read in full; the long entry for each is in [PAPER-NOTES.md](PAPER-NOTES.md). Years are arXiv submission years.

Every method here is one loop: form a key, form a target, check what the memory returns, nudge the memory, read. The groups differ in what the memory is and how the nudge is computed.

The running example: the model reads "Alice lives in Paris", later "Alice moved to Rome", and is then asked where Alice lives.

## 1. Start here

- **"Fast weight programming and linear transformers: from machine learning to neurobiology"** (Irie and Gershman, 2025, 2508.08435). A primer. A slow network with frozen weights emits a key, a value and a query per token; a fast matrix starts at zero and is rewritten at every token. Recent models differ only in the write rule: plain add, add after shrinking, or the delta rule.
- **"Linear Transformers Are Secretly Fast Weight Programmers"** (Schlag et al., 2021, 2102.11174). Attention without softmax is a matrix that each token adds `value × key` to. Adding "Rome" leaves Paris and Rome mixed. The fix is to look up what the matrix returns for Alice, then write the difference. A key with 64 numbers holds at most 64 clean facts. WikiText-103 perplexity: Transformer 29.6, linear 33.0, delta rule 31.5.

## 2. Roots

- **"Using Fast Weights to Deblur Old Memories"** (Hinton and Plaut, 1987). Every connection gets a slow weight and a fast weight that decays. After new learning smears 100 old pairs, rehearsing 10 of them restores the other 90. An optimizer writes both; nothing is learned about how to write.
- **"Learning to Control Fast-Weight Memories"** (Schmidhuber, 1992). A slow network outputs changes to a fast network's weights, as a product of a FROM vector and a TO vector. Two toy tasks, three fast weights.
- **"Using Fast Weights to Attend to the Recent Past"** (Ba et al., 2016, 1610.06258). A second weight matrix on an RNN gets `0.95 × old + 0.5 × h hᵀ` every step. The rule is fixed by hand; it equals attention over recent hidden states.
- **"HyperNetworks"** (Ha et al., 2016, 1609.09106). A small network outputs a larger one's weights. The dynamic version rescales an LSTM's weight rows at every step. Nothing accumulates, so it is not a memory.

## 3. Better write rules

The memory replaces attention. The state is one matrix per head, wiped at the start of each sequence. This line is in production models for long-context cost, and has nothing to say about learning across sessions.

- **"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"** (Yang et al., 2024, 2406.06484; DeltaNet). Same write rule as Schlag, computed in chunks of 64 tokens so it trains at scale. At 1.3B parameters it matches a Transformer on perplexity (16.87 against 16.85) and trails on retrieval (49.5 against 66.6).
- **"Gated Delta Networks: Improving Mamba2 with Delta Rule"** (Yang et al., 2024, 2412.06464). Multiplies the whole matrix by a number between 0 and 1 before each write, so the model can wipe the memory when the topic changes. Finding a number in a 4K-token essay goes from 18.6 to 92.2; pass-key retention at 2K drops from 96.8 to 88.4.
- **"Kimi Linear: An Expressive, Efficient Attention Architecture"** (Kimi Team, 2025, 2510.26692). The gate becomes one number per row, so some rows last and some fade. Three such layers per full-attention layer, in a 48B production model. Up to 75% less cache.
- **"Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"** (Grazzi et al., 2024, 2411.12537). Write strength up to 2 flips the sign of what is stored, which parity needs. DeltaNet parity accuracy goes from 0.017 to 1.000 with a one-line change.
- **"DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products"** (Siems et al., 2025, 2502.10297). Several delta writes per token. Two flips make a rotation, so one layer can track shuffles that DeltaNet cannot.
- **"Longhorn: State Space Models are Amortized Online Learners"** (Liu et al., 2024, 2407.14207). Derives the write as "the closest memory that also returns the new value", with an automatic step size. The trained model uses a cell-by-cell shortcut that does not subtract what the memory returns.
- **"MesaNet: Sequence Modeling by Locally Optimal Test-Time Training"** (von Oswald et al., 2025, 2506.05233). Reads as if the memory were the best fit to every pair so far, using an equation solver at read time. Matches a Transformer on perplexity at 1B (13.79). The best fit of "Paris" and "Rome" is their average, so overwriting depends on the forget gate (reader's inference).

## 4. Why it works

- **"Transformers learn in-context by gradient descent"** (von Oswald et al., 2022, 2212.07677). One attention layer without softmax, with the right weights, outputs what one gradient step on linear regression would. Training finds those weights. Toy regression only; nothing is stored.
- **"Transformers learn to implement preconditioned gradient descent for in-context learning"** (Ahn et al., 2023, 2306.00297). Proves the best single layer is that gradient step with a trained rescaling. Beyond one layer the proof covers stationary points only.
- **"Test-time regression: a unifying framework for designing sequence models with associative memory"** (Wang et al., 2025, 2501.12352). Each layer fits a function from keys to values. Linear attention is least squares with the "undo key overlap" factor dropped; DeltaNet is one gradient step per token; Gated DeltaNet adds weight decay.
- **"It's All Connected"** (Behrouz et al., 2025, 2504.13173; Miras). Makes the loss and the forgetting penalty design choices. A capped loss stops one garbled token from wrecking the memory. Three new models beat Gated DeltaNet at 1.3B.
- **"Test-Time Training with KV Binding Is Secretly Linear Attention"** (Liu et al., 2026, 2602.21204). On two TTT models, more inner steps hurt, gradient ascent works as well as descent, and updating only the last layer is best. That last layer is a running sum of key-times-value products. Tested only on models with a dot-product inner loss.

## 5. Writing into the model's own weights

This is the branch that bears on continual learning. The update lands in an MLP or in a block's weights. Every paper here resets the weights at the document or sequence boundary.

- **"Learning to (Learn at Test Time): RNNs with Expressive Hidden States"** (Sun et al., 2024, 2407.04620; TTT layers). The state is a small model; reading 16 tokens means one training step on it. The loss it descends is learned. Training needs gradients of gradients. Keeps improving to 32K context where Mamba plateaus at 16K.
- **"Titans: Learning to Memorize at Test Time"** (Behrouz et al., 2025, 2501.00663). An MLP memory beside attention, written harder when the gradient is large, with momentum and a forget gate. Removing the forget gate is the largest ablation loss.
- **"ATLAS: Learning to Optimally Memorize the Context at Test Time"** (Behrouz et al., 2025, 2505.23735). Each update fits the last few tokens together, with a learned weight per token so a stale pair can be down-weighted.
- **"Test-Time Training Done Right"** (Zhang et al., 2025, 2505.23884; LaCT). One update per chunk of 2,048 tokens or more. The write is a real backward pass through a small gated MLP. Needle retrieval at 32K: 48.0 against 91.2 for full attention.
- **"End-to-End Test-Time Training for Long Context"** (Tandon et al., 2025, 2512.23675). Every 1K tokens, one gradient step on next-token loss through the MLPs in the last quarter of blocks. Training picks starting weights that are good after those steps. Needle retrieval at 128K: 0.06 against 0.99.
- **"In-Place Test-Time Training"** (Feng et al., 2026, 2604.06169). Adds to the MLP output matrix a pretrained model already has, by a closed formula, in every sixth layer. Qwen3-4B gains 2 to 4 points on RULER.
- **"Learning What to Remember: Test-Time Training via Context Distillation"** (Wang et al., 2026, 2608.01672; TTCD). A teacher with a long attention window and a student with half of it; the difference between them is written into the same MLP slot. Gains under one point on pretrained 8B models.

Three levels of gradient at inference: a closed formula (group 3, In-Place TTT, TTCD), a backward pass through a small inner model (TTT-MLP, Titans, ATLAS, LaCT), and backprop through real blocks (TTT-E2E).

## 6. Models that learn to write their own weights

Three senses, from weakest to strictest.

**A network writes a separate memory.** Everything in groups 3 and 5. A frozen network decides what to write; a fixed rule writes it.

**A network changes the weights that compute the change.**

- **"A 'Self-Referential' Weight Matrix"** (Schmidhuber, 1993). A recurrent network with extra units that read and edit any of its own connections. A thought experiment with no experiments.
- **"A Modern Self-Referential Weight Matrix That Learns to Modify Itself"** (Irie et al., 2022, 2202.05780). One matrix produces its output and also the key, target and step size for a delta-rule edit of itself. No frozen slow network. At most 3.6M parameters, few-shot images and games, no language.
- **"Metalearning Continual Learning Algorithms"** (Irie et al., 2023, 2312.00276). Trains that matrix with "still do task A after learning task B" in the loss. Accuracy on the first task after the second goes from 22.9% to 98.2%. Small image tasks; edits last one sequence.
- **"Nested Learning: The Illusion of Deep Learning Architectures"** (Behrouz et al., 2025, 2512.24695; Hope). Small MLPs produce the keys, targets, step sizes and forget rates for updates to themselves. The only one of these trained as a language model (1.3B). Its continual-learning results are figures without numbers and use a variant without the self-modifying block.

No paper was found that trains an Irie-style self-referential matrix as a language model. The primer gives the reason this line stayed small: merging the slow and fast networks loses parallel training.

**The model produces its own update and an optimizer applies it.**

- **"Self-Adapting Language Models"** (Zweiger et al., 2025, 2506.10943; SEAL). The model writes study notes from a passage and a fine-tuning step trains on them. Notes that help are reinforced. No-passage accuracy on Qwen2.5-7B goes from 33.5% to 47.0%. Earlier passages are forgotten as edits accumulate.

## 7. Editing weights from outside

- **"Sparks of Weight-Space Programming"** (Takehi et al., 2026, MIT CSAIL blog). An untrained general LLM reads and writes another network's weights by reasoning. An independent replication found reading holds and writing gains are smaller than reported.
- **"Fast Model Editing at Scale"** (Mitchell et al., 2021, 2110.11309; MEND). A small trained network turns the fine-tuning gradient for one correction into a better update. Trained by backprop through the edited model's loss.
- **"Text-to-LoRA: Instant Transformer Adaption"** (Charakorn et al., 2025, 2506.06105). A network reads a task description and outputs an adapter in one forward pass. 67.7 on unseen benchmarks against 55.8 for the base model.
- **"Tracr: Compiled Transformers as a Laboratory for Interpretability"** (Lindner et al., 2023, 2301.05062). A compiler turns a short program into exact transformer weights. Nothing is trained.

Two ways to train an editor. If the edit is a continuous output, backprop the edited model's loss into the editor (MEND, Text-to-LoRA, fast weight programmers). If the edit is text, use the edited model's score as a reward (SEAL).

## Not a fast-weight paper

- **"Learning, Fast and Slow: Towards LLMs That Adapt Continually"** (Tiwari et al., 2026, 2605.12484). The fast component is a set of system prompts rewritten by an outside model; the slow component is the weights, trained by RL. It belongs with harness methods. It is the only paper here with a task stream: forgetting did not separate the methods, and the difference was whether the model could still learn the next task.
