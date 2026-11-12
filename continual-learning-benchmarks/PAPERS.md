# Reading list: fast weights and test-time weight updates

State on 2026-11-11. The papers behind `METHODS.md` section 5, grouped by the question each group answers. Numbers are arXiv IDs.

Every method here is one loop: form a key, form a target, check what the memory returns, nudge the memory, read. The groups differ in what the memory is and how the nudge is computed.

## 1. Start here

- Irie and Gershman, fast weight programming primer (2508.08435).
- Schlag, Irie and Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (2102.11174). Attention without softmax is a matrix that each token adds `v kᵀ` to. The fix is the delta rule: subtract what the matrix already returns for `k` before adding.

## 2. Roots

- Hinton and Plaut, "Using Fast Weights to Deblur Old Memories" (1987).
- Schmidhuber, fast weight controllers and the self-referential weight matrix (1991 to 1993).
- Ba et al., "Using Fast Weights to Attend to the Recent Past" (1610.06258).
- Ha et al., "HyperNetworks" (1609.09106).

## 3. Better write rules

The memory replaces attention. The state is one matrix per head, wiped at the start of each sequence. This line is in production models for long-context cost, and has nothing to say about learning across sessions.

- Yang et al., DeltaNet (2406.06484): the delta rule made parallel for training.
- Yang et al., Gated DeltaNet (2412.06464): adds a decay gate, which is weight decay on the memory.
- Kimi Linear (2510.26692): per-channel decay, in a production model.
- DeltaProduct (2502.10297): several delta steps per token.
- Grazzi et al. (2411.12537): a write strength up to 2 gives negative eigenvalues, which state tracking needs.
- Longhorn (2407.14207) and MesaNet (2506.05233): the write as the solution of an online regression, solved exactly.

## 4. Why it works

- von Oswald et al., "Transformers Learn In-Context by Gradient Descent" (2212.07677).
- Ahn et al. (2306.00297): trained linear transformers implement preconditioned gradient descent.
- Wang et al., test-time regression (2501.12352): attention, linear attention and the delta family are all regression on key-value pairs.
- Behrouz et al., Miras (2504.13173): the same unification with the loss and the regularizer as choices.
- Liu et al., "Test-Time Training with KV Binding Is Secretly Linear Attention" (2602.21204).

## 5. Writing into the model's own weights

This is the branch that bears on continual learning. The update lands in an MLP or in a block's weights, and the gradient is a real one.

- Sun et al., TTT layers (2407.04620): the hidden state is the weights of a small model, updated by a gradient step on a self-supervised loss.
- Behrouz et al., Titans (2501.00663): a deep memory with momentum and forgetting; and ATLAS (2505.23735).
- LaCT (2505.23884): large-chunk updates.
- Tandon et al., TTT-E2E (2512.23675): backprop on next-token loss through the last quarter of blocks at test time.
- Feng et al., In-Place TTT (2604.06169): updates the model's existing MLP weights by a formula, as a retrofit.
- TTCD (2608.01672).

Three levels of gradient at inference: a closed formula (group 3, In-Place TTT, LaCT), a local backward pass through a small inner model (TTT-MLP, Titans), and backprop through real blocks (TTT-E2E). Training backprops through the writes in all three.

## 6. Fast weights for continual learning

- Irie et al., "Metalearning Continual Learning Algorithms" (2312.00276).
- Nested Learning and Hope (2512.24695).
- "Learning Fast and Slow" (2605.12484).

No paper in groups 3 to 6 measures forgetting across sessions. All of them reset state per sequence.

## 7. Editing weights from outside

A separate line: something outside the forward pass decides the edit.

- Takehi et al., "Sparks of Weight-Space Programming" (MIT CSAIL blog, 2026-11-02): an untrained general LLM reads and writes another network's weights by reasoning. An independent replication found reading holds and writing gains are smaller than reported.
- Mitchell et al., MEND (2110.11309): a small network turns a gradient into an edit, trained so the edit fixes one fact and leaves others alone.
- Text-to-LoRA (2506.06105): a network outputs adapters from a task description.
- SEAL (2506.10943): the model proposes its own updates and is rewarded by how it performs afterwards.
- Tracr (2301.05062): compiled transformers with known weights, a source of supervised data for reading weights.

Two ways to train an editor. If the edit is a continuous output, backprop the edited model's loss into the editor (hypernetworks, MEND, fast weight programmers). If the edit is text or code, use the task score as a reward (SEAL).
