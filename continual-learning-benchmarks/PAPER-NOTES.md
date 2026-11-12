# Paper notes: fast weights and test-time weight updates

State on 2026-11-11. One entry per paper in [PAPERS.md](PAPERS.md), written from a full read of each paper's text. Every entry has the same parts: the problem, the idea with a small example, what the memory is and how it is written, what is learned versus fixed, whether the model writes its own weights, results, limits, and how much was read.

The reads were done by eight reader agents from the arXiv HTML text. Figures were seen only as captions, and reference lists were mostly skipped. Points marked "inferred", "noticed" or "mine" are the reader's, not the authors'. Nothing here was checked against the papers a second time.


## 1. Start here, and two roots

Running example used throughout: the model reads "Alice lives in Paris", later "Alice moved to Rome", then is asked where Alice lives. Give Alice the key `[1, 0]` and Bob the key `[0, 1]`. A "value" is a short list standing for Paris or Rome.

---

### 1. Fast weight programming and linear transformers: from machine learning to neurobiology

- **Title, authors, year, arXiv ID or venue.** Kazuki Irie and Samuel J. Gershman, 2025 (arXiv 2508.08435, v6 dated 28 Sep 2026; reviewed on OpenReview, TMLR forum id TDG8EkNmQR). A primer, no new experiments.
- **The problem.** An ordinary recurrent network keeps its short-term memory in a list of activity numbers; its connection weights are frozen after training. A transformer keeps every past token, so its memory grows with the text and cost grows with the square of length. Recent models (Mamba2, RetNet, GLA, xLSTM, DeltaNet) have names that hide how they relate.
- **The idea.** One family covers them all: a Fast Weight Programmer (FWP). A "slow" network with trained, frozen weights reads each token and emits three short lists: a key (an address), a value (content), and a query (what to look up now). A second, "fast" network is a single weight matrix that starts at zero and is rewritten at every token. Writing "Alice lives in Paris" adds the table `Paris × [1, 0]` to the matrix. Reading multiplies the matrix by the query `[1, 0]` and returns Paris. If the softmax is removed from transformer attention, the arithmetic is identical to this. The models differ only in the write rule (Table 1): plain add (vanilla FWP / linear transformer); add after shrinking the old matrix (RetNet fixed shrink, Mamba2 input-dependent shrink, GLA per-row shrink); or the delta rule (DeltaNet), which first looks up what Alice currently maps to (Paris), subtracts it, and writes Rome. With plain add, "Alice moved to Rome" leaves Paris+Rome mixed.
- **What the memory is and how it is written.** One fast weight matrix per attention head per layer, replacing the attention operation. It is initialized to zero at the start of a sequence and updated by a rank-one (one key, one value) change per token.
- **What is learned versus fixed.** The slow projection matrices that produce key, value, query, and (in DeltaNet) a per-token write strength are trained by backprop through time and then frozen. The fast matrix changes at run time. The form of the write rule is chosen by the designer; its inputs (what to write, where, how strongly) are learned. Each rule equals one gradient step on a small local loss (Appendix A), but it is written in closed form, so training does not need gradients of gradients. Training is parallelized chunk by chunk (Box 1).
- **Does the model write its own weights?** Partly: one sub-network outputs changes to another sub-network's weights; the slow weights never change at run time.
- **Results.** None of its own. Cited claims: DeltaNet beats the additive rule across language modeling, RL, time series, image generation (Sec. 3.4); models whose update only shrinks or adds fail parity and modular arithmetic while DeltaNet handles them (Sec. 3.6, citing Grazzi et al. 2025); transformers still win "by a large margin" on precise retrieval (Sec. 3.7, Table 2).
- **Limits.** Fixed-size memory means low retrieval precision. Gated DeltaNet's decay hurt in some RL games (unpublished, Sec. 3.4). More expressive variants (recurrent, self-referential) lose parallel training. The neurobiology mapping (fast matrix = AMPA receptors, slow value projection = NMDA receptors) is called speculative by the authors. Nothing on memory across sessions.
- **Read status.** 141,590 characters extracted. Read all main text, boxes, tables, and Appendix A (lines 1-870 and 2004-2117). The reference list (lines 871-2003) was searched by keyword, not read line by line.

### Methods in the primer where a network modifies its own weights

The primer describes self-reference in one sentence (Sec. 3.6): "self-reference (by merging the slow and fast nets into a single network that modifies itself)", and says it raises expressivity but makes training inefficient. Citations given there:

1. **Schmidhuber (1992b)**, "Steps towards 'self-referential' learning", Tech. Report CU-CS-627-92, Univ. of Colorado at Boulder. Cited Sec. 3.6.
2. **Schmidhuber (1993a)**, "A self-referential weight matrix", ICANN 1993, pp. 446-451. Cited Sec. 3.6.
3. **Irie, Schlag, Csordás, Schmidhuber (2022c)**, "A modern self-referential weight matrix that learns to modify itself", ICML 2022, pp. 9660-9677. Cited Sec. 3.6 (and Sec. 3.5 for the "delayed-feedback" term).

Related items the primer cites but does not itself describe as self-modifying (labels below are mine, from titles or context):

4. **Irie, Csordás, Schmidhuber (2023)**, "Practical computational power of linear transformers and their recurrent and self-referential extensions", EMNLP 2023. Cited Sec. 3.6 as empirical expressivity work.
5. **Schmidhuber (1987)**, PhD thesis "Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook". Cited Sec. 3.5 for metalearning as "training a model to train another model (or itself)".
6. **Miconi, Rawal, Clune, Stanley (2019)**, "Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity", ICLR 2019; and **Miconi, Stanley, Clune (2018)**. Cited Sec. 3.4/3.5 as early FWP-like models that put fast weights inside an LSTM.
7. **Schmidhuber (1993b)**, "Reducing the ratio between learning complexity and number of time varying variables in fully recurrent nets", ICANN 1993. Cited Sec. 3.3 as a recurrent FWP whose own hidden state serves as both key and value.
8. Not self-reference, for contrast: **Irie et al. (2021)** recurrent FWPs feed the fast net's output back into the slow net (Sec. 3.6); the two nets stay separate.

---

### 2. Linear Transformers Are Secretly Fast Weight Programmers

- **Title, authors, year, arXiv ID or venue.** Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber, 2021, arXiv 2102.11174 (ICML 2021).
- **The problem.** "Linear transformers" (cheap attention with a fixed-size memory) just keep adding to their memory. They cannot replace an old fact, and nobody had stated how many facts the memory holds.
- **The idea.** Show that linear attention is the 1991 fast-weight scheme: a matrix that each token adds `value × key` to. Two consequences. (a) Capacity: lookups stay clean only if keys do not overlap, and a key with 64 numbers allows at most 64 non-overlapping keys. (b) Editing: after "Alice lives in Paris", adding "Alice moved to Rome" leaves Paris+Rome. The fix: before writing, look up what the matrix returns for Alice (Paris), then write `strength × (Rome − Paris)` at Alice's key. Bob's entry is untouched (Appendix B shows a rival gated rule by Peng et al. would shrink Bob's entry too). The resulting model is the Delta Network. The paper also proposes DPFP, a fixed formula that stretches a key into a longer, sparser list to raise capacity.
- **What the memory is and how it is written.** One matrix per head per layer (medium model: 8 heads × 32×32 per layer, 16 layers, 0.13M numbers total). Starts at zero. One rank-one edit per token. In the main tables it restarts for each text segment; in the "no truncation" run it is carried across segments.
- **What is learned versus fixed.** Trained by backprop: key, value, query projections and a one-row projection giving the write strength (a number in 0..1 per token). Fixed by hand: the form of the delta rule, DPFP, and dividing keys by their sum (without this the models diverged). No gradients of gradients; a custom backward pass recomputes fast weights instead of storing them per step (Appendix F).
- **Does the model write its own weights?** Partly: slow layers output changes to a separate fast matrix.
- **Results.** Synthetic retrieval (Sec. 6.1.1): linear attention with 64-number keys starts to err at 60+ stored pairs. WikiText-103, medium config, 90M parameters, test perplexity (Table 2): Transformer 29.6, Linear Transformer 33.0, Delta Network 31.5. Unbounded context (Table 4): Delta Network 29.4 with a 0.13M state; Linear Transformer >260; Transformer-XL 65.5 at the same state size and 25.5 with a 6.29M state. Translation WMT14 En-De (Table 1): standard 27.7 BLEU, linear 26.8, DPFP 27.1.
- **Limits.** Still behind softmax attention at equal settings and behind Transformer-XL with a large window (authors say so). Models of 40-90M parameters only. Attention normalization had to be removed for long runs because an accumulator blows up. FAVOR+ with the delta rule gave NaN (Appendix D.2). Cross-session forgetting is not studied.
- **Read status.** 67,310 characters; main text and Appendices A-F read in full; reference list skipped.

---

### 3. Using Fast Weights to Attend to the Recent Past

- **Title, authors, year, arXiv ID or venue.** Jimmy Ba, Geoffrey Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu, 2016, arXiv 1610.06258 (NIPS 2016).
- **The problem.** A recurrent network must hold everything about the current sequence in its hidden activity list, which has as many slots as hidden units. The weights hold far more numbers (units squared) but only change during training.
- **The idea.** Add a second weight matrix on the recurrent layer that changes at every step and fades quickly. After each step, take the hidden activity list `h` and add the table `h × h` (every pair of entries multiplied) to this matrix, after shrinking the old matrix. Later, when the network is in a state resembling an earlier one, multiplying by the matrix pulls it toward that earlier state. Their task: read `c9k8j3f1??c` and answer `9`. In our terms, the state while reading "Alice ... Paris" is stored; when "Alice" shows up again, the state is pulled back toward the one that contained Paris. The authors show this equals attention over past hidden states, scored by dot product and weighted by recency, so they never build the matrix in code.
- **What the memory is and how it is written.** One matrix `A` per recurrent layer, hidden-size by hidden-size, zero at the start of each sequence. Write: `A ← 0.95·A + 0.5·h hᵀ` every step. Read: an inner loop of S steps recomputes the next hidden state with `A` added to the usual input, with layer normalization.
- **What is learned versus fixed.** Slow weights (input, recurrent, output) are trained by backprop, which accounts for how they shape what gets stored. The write rule is hand-set (Hebbian, decay 0.95 or 0.9, rate 0.5); nothing decides what or when to write. No gradients of gradients.
- **Does the model write its own weights?** Partly: the layer's own activity changes that layer's fast weights through a fixed rule; no network outputs the change, and the slow weights are untouched.
- **Results.** Associative retrieval error (Table 3): with 20 hidden units, fast weights 1.81% vs LSTM 60.81%; with 50 units, 0% vs 1.85%. MNIST from 24 glimpses, 50 features (Table 4): 7.21% vs LSTM 12%. Facial expression accuracy (Table 5): 86.34 vs LSTM 81.32, ConvNet 88.23.
- **Limits.** Small tasks. Memory is wiped per sequence and decays within it. Glimpse positions and when-to-store signals were supplied by hand. ConvNet still wins on faces. RL result is learning curves on a toy "Catch" game only. Inner-loop count made little difference on retrieval (Appendix A.1).
- **Read status.** 37,216 characters; main text, references, and both appendices read in full.

---

### 4. HyperNetworks

- **Title, authors, year, arXiv ID or venue.** David Ha, Andrew Dai, Quoc V. Le, 2016, arXiv 1609.09106 (v4; ICLR 2017).
- **The problem.** A recurrent network reuses one set of weights at every step; a deep convnet has separate weights for every layer. Earlier weight-generating schemes (HyperNEAT) used evolution and stayed small.
- **The idea.** Train a small network to output the weights of a larger one, end to end with backprop. Two versions. Static: each conv layer gets a short learned code (4 or 64 numbers); a shared two-layer linear map turns the code into that layer's filters. Dynamic (HyperLSTM): a small LSTM runs alongside the main LSTM, sees the current input and the main LSTM's state, and at every step outputs a scaling number for each row of the main weight matrices. So when the model reads "Alice moved to Rome", the main LSTM's effective weights at "moved" differ from those at "Alice". Nothing is stored in those weights: the next step's scaling is produced fresh from the small LSTM's state.
- **What the memory is and how it is written.** No weight memory. Static: weights are fixed after training. Dynamic: per-step row scales and biases for the main LSTM are recomputed each step; the only carried state is the two LSTMs' ordinary hidden states. The full version (a whole new matrix per step) used too much memory, so only row scaling is used (Eq. 7-8).
- **What is learned versus fixed.** Everything is trained jointly by ordinary backprop: main weights, hypernetwork weights, layer codes. At run time no trained parameter changes. No gradients of gradients. When the small LSTM's state resets is not stated; I assume it resets like any LSTM state.
- **Does the model write its own weights?** Partly: one network outputs (rescales) another network's weights each step; there is no accumulation and no self-modification.
- **Results.** enwik8 bits per character (Table 4): LSTM 1800 units 1.470, Layer Norm LSTM 1.402, HyperLSTM 1.391, Layer Norm HyperLSTM 2048 units 1.340 (26.54M parameters). WMT'14 En→Fr single model (Table 6): GNMT LSTM 38.95 BLEU, HyperLSTM 40.03. CIFAR-10 static (Table 2): Hyper Residual Network 40-2 at 7.23% error with 0.148M parameters vs 5.66% with 2.236M for the plain network (row pairing reconstructed from a scrambled text extraction; consistent with the stated "1.25-1.5%" cost).
- **Limits.** Static version loses accuracy for the parameter saving. Coordinate-input variant did poorly (93.5% vs 98.5% MNIST, Appendix A.1). Handwriting has no test set and may overfit validation (authors, A.3.5); Layer Norm did not combine well there. HyperLSTM has more parameters than its baseline. No fast memory, so nothing about retention.
- **Read status.** 29 PDF pages, 75,928 characters extracted with macOS PDFKit (ar5iv had no HTML; the Read tool could not render the PDF). All pages read including appendix. Equations 6 and 10-12 and Tables 2, 3, 5 came out scrambled; figures not seen.


## 2. Roots and the self-referential line

Snapshot 2026-11-11. Five papers read end to end, then a survey answer on "does a model learn to write its own weights?".

---

### 1. Using Fast Weights to Deblur Old Memories

- **Title, authors, year, venue.** Geoffrey Hinton and David Plaut, Proc. 9th Annual Conference of the Cognitive Science Society, Seattle, 1987, pp. 177-186.
- **The problem.** A network that learns new items damages old ones. Restoring the old items normally means retraining on all of them, which also overwrites the new items.
- **The idea.** Give every connection two numbers that are added together: a slow weight that holds long-term knowledge, and a fast weight that changes quickly and shrinks toward zero by itself. Example: a network has learned 100 pairs such as name -> phone number. It then learns 5 new pairs, which smears all 100 old ones. Now rehearse only 10 of the old pairs. The fast weights move to cancel the smear, and the other 90 old pairs come back too, because every pair is spread over the same connections. When the fast weights fade, the 5 new pairs return.
- **What the memory is and how it is written.** The memory is the fast-weight copy of every connection in an ordinary feed-forward network. It is written by plain backpropagation on rehearsed items, with its own (larger) step size, and it loses a fixed fraction of its size after each update (1% in the simulation). It is never reset; it just decays.
- **What is learned versus fixed.** Both weight sets get the same error gradient. The write rule (gradient step plus decay) is hand-designed. No gradients of gradients.
- **Does the model write its own weights?** No. An optimizer does, and it needs the correct targets for the rehearsed items.
- **Results.** Network 10-100-10 units, 100 random binary associations (1300 sweeps), then 5 new ones (400 sweeps). In early retraining, error on the unrehearsed old items falls nearly as fast as on the rehearsed ones, "even if only 10% of the old associations were retrained" (Section 4, Figure 2; curves only, no table). For a linear net with 0/1 inputs, retraining 50 of 100 gives an expected improvement ratio of 0.943 for unrehearsed versus rehearsed items (Section 5.1).
- **Limits.** One toy simulation. The effect shrinks as retraining goes on (Section 5.3), vanishes for orthogonal inputs, and for +1/-1 inputs falls in proportion to 1/dimension. The authors say the analysis holds without fast weights; fast weights only make the repair temporary.
- **Read status.** All 10 scanned pages read as images (cs.toronto.edu/~hinton/absps/fastweights87.pdf).

---

### 2. Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks

- **Title, authors, year, venue.** Juergen Schmidhuber, Neural Computation 4(1):131-139, 1992.
- **The problem.** Remembering something across time steps was done with recurrent networks, which store it in unit activations. A network has far more weights than units, so storing in weights could hold more.
- **The idea.** Use two feed-forward networks. A slow network S reads the input at each step and outputs changes to the weights of a second, fast network F. F computes the answer. Example from the paper: a driver parks in slot 1, 2 or 3, does other things for a while, then asks "where is my car?". When the park event arrives, S makes the F connection from the "where is my car?" input to the right slot output large. Later the question alone produces the slot.
- **What the memory is and how it is written.** The memory is F's weights. After every time step each fast weight becomes a squashed sum of its old value and S's contribution, bounded between 0 and 1. S addresses weights in one of two ways: one S output per fast weight, or two groups of outputs (FROM and TO) whose product gives the change for the connection between two F units. The second way is an outer product of a key and an entry. Nothing is reset; the stream has no episode boundaries.
- **What is learned versus fixed.** S's weights are trained by gradient descent on F's output error, with the gradient carried forward through the fast-weight updates (cost per step proportional to |W_F| x |W_S|). F has no trained weights of its own. The form of the write (add then squash; product of FROM and TO) is fixed; what to write and when is learned. First-order chain rule only.
- **Does the model write its own weights?** Partly: one network outputs changes to another network's weights. S itself is only changed by gradient descent.
- **Results.** Flip-flop task (output on at the first B after an A, unknown delays): solved "within 300 time steps" with one output per weight, 800 steps with FROM/TO. Parking task: solved "within 6000 time steps". Solved means error at most 0.05 for 100 steps in a row. Both nets had no hidden units; F had 3 fast weights.
- **Limits.** Two toy tasks, no baselines, no run statistics. With both nets feed-forward the system cannot change its output while the input is held fixed. With one S output per fast weight, S grows with F.
- **Read status.** Full HTML version from people.idsia.ch/~juergen/fastweights (14 pages of sections, about 27,000 characters, through the references). Some equations are truncated in the page's alt text.

---

### 3. A 'Self-Referential' Weight Matrix

- **Title, authors, year, venue.** Juergen Schmidhuber, ICANN 1993 (Amsterdam). Read from the HTML copy on the author's site; I inferred it is the ICANN text from the title and its reference list.
- **The problem.** In the 1992 system the slow network is still changed only by a fixed, hand-written algorithm. Every learning algorithm so far had some part the network could not touch.
- **The idea.** One recurrent network with extra input and output units that let it inspect and edit any of its own connections, including the connections that drive the inspecting and editing. Every connection gets a binary address. Example: the network sets its "analyse" outputs to the address of connection 5; at the next step a special input unit shows that connection's current value. It then sets its "modify" outputs to an address and a separate output to +0.3; that connection grows by 0.3. Other inputs show it its own task error.
- **What the memory is and how it is written.** The memory is the whole weight matrix. Addressing is soft: a bump function of the distance between the address the network emits and each connection's address decides how much each connection is read or changed, so reads and writes are differentiable. Changes persist for the whole interaction sequence, which may be many training sequences joined together.
- **What is learned versus fixed.** Only the initial weights are trained from outside, by exact gradient descent on the summed error over the sequence. The read and write machinery (addresses, bump function) is fixed hardware. What to read, what to write and how much is learned, and can in principle change itself. First-order chain rule.
- **Does the model write its own weights?** Yes, in the strict sense: the network changes the same weights that produced the change.
- **Results.** None. The paper calls itself a "thought experiment" and gives no experiments. Stated costs: storage grows with the square of the number of connections; time per step is O(n_conn^2 log n_conn) in Section 3. The abstract says O(n_conn log n_conn); the two statements disagree in the copy I read.
- **Limits.** Author: high cost and "the high number of local minima of the unusually complex error surface". Mine: one connection is edited per step; no evidence it trains at all.
- **Read status.** Full HTML (9 pages, about 21,000 characters, through references and footnote). Some equations truncated in alt text.

---

### 4. A Modern Self-Referential Weight Matrix That Learns to Modify Itself

- **Title, authors, year, arXiv ID.** Kazuki Irie, Imanol Schlag, Robert Csordas, Juergen Schmidhuber, ICML 2022, arXiv 2202.05780.
- **The problem.** The 1993 design edits one weight at a time and was never tested. Modern fast weight programmers edit a whole matrix at once but keep a slow network that is frozen after training.
- **The idea.** Remove the slow network. One matrix W produces the layer output and also the ingredients for its own update. Example: the model sees five labelled images (cat=1, dog=2, ...), then an unlabelled image. For each labelled image, W times the input gives four things: an output, a key, a query, and a step size. W is then asked two questions: "what do you currently return for this key?" and "what do you return for this query?". W moves its answer for the key toward its answer for the query, by the step size. After five such self-edits, W classifies the sixth image.
- **What the memory is and how it is written.** One matrix per head in each sequence layer (it replaces self-attention in a Transformer block). Each step adds one rank-one correction (delta rule): W <- W + sigmoid(beta) (W phi(q) - W phi(k)) outer phi(k). Changes are kept for the whole sequence and reset at episode boundaries.
- **What is learned versus fixed.** Backpropagation through time trains the starting matrix W_0 (the only parameters of this layer), plus ordinary feed-forward blocks and a convolutional image encoder that do not self-modify. The form of the write (delta rule, outer product) is fixed; keys, targets and step sizes are produced by W itself, so the learning rule's content is learned. No bi-level optimisation, no gradients of gradients.
- **Does the model write its own weights?** Yes: the matrix that computes the change is the matrix that is changed. Other parts of the model are not.
- **Results.** Table 1, 5-way Mini-ImageNet, Conv-4 encoder: 1-shot 47.0 +/- 0.2, 5-shot 61.4 +/- 0.1, versus DeltaNet 47.0 / 62.7 and HyperTransformer 53.8 / 67.1 (3.6M parameters). Table 2, stream that switches from Omniglot to Mini-ImageNet: 53.3% on the second part versus 50.4% for DeltaNet. Table 3, ProcGen 6-game multi-task test score: 20.0 (1.8) versus DeltaNet 18.6 (1.7) and the same model without self-edits 15.3 (1.9); 968K parameters.
- **Limits.** Authors: tasks are solvable by feed-forward nets; language modelling is future work; the image encoder is not self-modifying; 1,700 versus 3,500 steps per second in RL; some seeds stay at chance; no guarantee self-edits keep the trained objective. Mine: the ProcGen gain over DeltaNet is within one standard deviation.
- **Read status.** Full arXiv HTML v2, 78,845 characters, through Appendix C.

---

### 5. Metalearning Continual Learning Algorithms

- **Title, authors, year, arXiv ID.** Kazuki Irie, Robert Csordas, Juergen Schmidhuber, TMLR; arXiv 2312.00276 (v3, Feb 2025).
- **The problem.** Methods against forgetting (for example elastic weight consolidation) are hand-designed. A network that learns from examples in its context also forgets: after it reads task B, it can no longer do task A.
- **The idea.** Train the self-modifying network of paper 4 so that not forgetting is part of the training loss. Example: the model reads 75 labelled handwritten characters (task A), then 75 labelled photos (task B). It is tested on a new character right after A, on a new photo after B, and on a new character again after B. The third test is the new part: training must find self-edits that keep A while learning B.
- **What the memory is and how it is written.** Same as paper 4: self-referential weight matrices (2 layers, 16 heads, hidden size 256), each making one rank-one delta-rule edit per example. The edits last for one sequence of tasks.
- **What is learned versus fixed.** Meta-training by backpropagation through time sets the starting matrices and a Conv-4 image encoder. At test time no optimizer runs; the matrices edit themselves. The continual-learning behaviour is learned; the delta-rule form is fixed. No gradients of gradients.
- **Does the model write its own weights?** Yes for the sequence layers, as in paper 4; the image encoder is fixed.
- **Results.** Table 1 (meta-trained on Omniglot and Mini-ImageNet): Omniglot accuracy after then learning Mini-ImageNet is 22.9% without the backward term (chance is 20%) and 98.2% with it. Table 3, Split-MNIST without replay: 84.5 +/- 1.6 (domain-incremental, 5 tasks) and 84.3 +/- 1.2 (class-incremental, 5 tasks), versus the meta-learned baseline GeMCL at 63.8 / 79.0 and plain SGD at 63.2 / 19.5. Table 4, on a frozen ImageNet ViT-B/16 with reduced 2-way tasks: 68.3 on Split-CIFAR100 and 61.5 on 5-datasets, against 83.9 and 81.1 for L2P on the full, harder versions.
- **Limits.** Authors: needs more varied meta-training data; accuracy falls with more tasks than seen in training (Split-Omniglot 92.3 / 82.0 / 67.6 for 5 / 10 / 20 tasks); loss terms grow with the square of the number of tasks; image classification only; the encoder does not self-modify. Mine: 5-way or 2-way tasks, 15 examples per class; nothing carries across sessions; the Split-MNIST model was selected on MNIST validation data.
- **Read status.** Full arXiv HTML v3, 100,668 characters: main text and Appendices A and B. Reference list skipped.

---

## Is there work in which a model learns to write its own weights?

Yes, in three different senses. The strict sense (b) exists and works, but with a learned rule it has been shown only on small models; the language-model scale work is either sense (a), sense (c), or a partial form of (b).

Verification labels: [full] read end to end; [opened] fetched the paper and read the method or experiment sections; [abstract] arXiv abstract only; [memory] not fetched in this pass.

### (a) A network outputs updates to a different network or a separate memory

- Schmidhuber 1992, fast weight programmer [full]. Slow net writes a fast net. Toy, 3 fast weights.
- Ba et al. 2016, "Using fast weights to attend to the recent past" [memory]. Fast matrix written by a fixed Hebbian rule; nothing about the write is learned.
- Ha, Dai, Le 2016, HyperNetworks [memory]. A small net generates a larger net's weights; LSTM language models and image classifiers.
- Munkhdalai and Yu 2017 Meta Networks; Munkhdalai et al. 2019 Metalearned Neural Memory [memory]. Few-shot image tasks and small language tasks.
- Schlag, Irie, Schmidhuber 2021, DeltaNet [as described in paper 4; original not opened]. Linear attention is a fast weight programmer; delta-rule write. Descendants (2406.06484, Gated DeltaNet 2412.06464) are trained as language models [titles from citer list].
- Zhmoginov et al. 2022, HyperTransformer [as described in paper 4]. A Transformer emits a small CNN's weights; few-shot images.
- Mitchell et al. 2022, MEND [memory]. A hypernetwork turns a gradient into an edit of an LLM's weights.
- TTT layers 2407.04620 and Titans 2501.00663 [memory; covered by other groups]. An inner memory is updated by a gradient step on a hand-chosen loss; the outer net learns what to store. Language models.
- Vettoruzzo et al. 2408.03219 [abstract]. A meta-learned Transformer optimizer writes a classifier's weights for continual learning; SplitMNIST, SplitCIFAR-100.
- Text-to-LoRA 2506.06105 [memory], SHINE 2602.06358 [abstract], Infinite-Parameter LLMs 2609.18842 [abstract]. A hypernetwork maps a task description or a document to a LoRA adapter for an LLM in one forward pass. LLM scale.

### (b) A network modifies the very weights that compute the modification

- Schmidhuber 1993 [full]. Proposal only; no experiments.
- Irie et al. 2022, modern SRWM, 2202.05780 [full]. At most 3.6M parameters; few-shot images and ProcGen games; no language.
- Kirsch and Schmidhuber 2022, 2212.14392 [opened]. Self-modifying nets improved by selection instead of gradient meta-training; bandits and classic control. It also argues that a self-referential net trained by an outer optimizer represents the same functions as an in-context learner with memory.
- Lu, Towers, Foerster 2023, 2303.09478 [opened]. Evolution over a self-referential parameter vector; a toy numeric world and time-series fitting.
- Irie and Schmidhuber 2023, 2305.01547 [opened]. SRWM with a bootstrapping loss; Mini-ImageNet, 3 layers of width 256.
- Irie, Csordas, Schmidhuber 2023, 2310.16076 [opened]. SRWM generalises on parity and other formal languages where linear Transformers do not; 1 to 2 layers, width 8 to 64.
- Irie et al. 2023, ACL, 2312.00276 [full]. Few-shot image continual learning.
- Pedersen et al. 2025, Self-Referential Graph HyperNetworks, 2512.16406 [opened]. A hypernetwork generates mutated copies of itself; CartPole, LunarLander, Ant; no gradients.
- Behrouz et al. 2025, Nested Learning / Hope, 2512.24695 [opened: Section 8 and the experiment prose; tables not checked number by number]. "Self-modifying Titans": small MLP memories produce keys, values, step size and decay, each memory generates its own target values, and each is updated by one gradient-style step. The query projection stays fixed and the update form is hand-designed, so this is partly sense (b). Trained from scratch as language models with about 760M parameters (30B tokens) and 1.3B parameters (100B tokens). The continual-learning tests on Llama-3 backbones (8B and 3B) use the "Hope-Attention" variant, which has ordinary attention plus multi-speed MLP blocks and not the self-modifying part.
- VisionHOPE 2609.33325 [opened: abstract, experiment setup]. Hope's self-referential update as an image backbone; ImageNet-1K, COCO, ADE20K. Reports that the unconstrained update diverges and adds step-size control. Vision only.
- Federated Nested Learning 2605.16350, Reinforced Fast Weights 2602.16704 [title only].

### (c) The model produces its own training signal or update and an optimizer applies it

- Krause et al. 2018, dynamic evaluation [as described in paper 4]. Fixed gradient steps on the test stream; nothing is learned about how to update.
- Zweiger et al. 2025, SEAL, 2506.10943 [opened]. The LLM writes its own fine-tuning data and settings; LoRA fine-tuning applies them; reinforcement learning rewards edits that help. Qwen2.5-7B: no-context SQuAD answers 33.5% -> 47.0%. Llama-3.2-1B on 19 selected ARC tasks. Forgets under repeated edits; 30 to 45 seconds per edit evaluation.
- Search over Self-Edit Strategies 2601.14532 [abstract]. SEAL where the model also writes its own edit template; Qwen3-8B; did not beat the best human template.
- Wang et al. 2026, SCoL, 2605.07076 [opened]. Qwen2.5-7B names which of its own layers get a LoRA update; trained by meta-reinforcement learning; SQuAD and LongBench v2.
- Behrouz et al. 2026, "Language Models Need Sleep", 2606.03979 [opened: method sections]. On Hope with Llama-3 backbones: distil fast parts into newly added low-rank experts, and generate practice data by reinforcement learning. Despite the title this is sense (c) plus distillation.
- Self-Preservation in Language Models 2610.12602 [abstract]. The model generates and helps choose its own replay data; Qwen up to 4B.
- TTT-E2E 2512.23675 [abstract]. Next-token gradient steps at test time from a meta-learned start; 3B models, 164B tokens. The model does not choose the update.
- Promptbreeder 2309.16797 [abstract] is self-referential but edits prompts, not weights.

### What the search covered

Semantic Scholar citer lists for 2202.05780 (53 entries), 2312.00276 (9), SEAL (106) and Nested Learning (106), plus an arXiv phrase search for "self-referential weight matrix". I found no paper that trains an Irie-style SRWM as a language model; paper 4 lists that as future work. The citer API is rate-limited and may be incomplete; the Kirsch and Schmidhuber citer list could not be fetched.


## 3. Better write rules, part one

Shared starting point (Schlag et al. 2021). Each attention head keeps one matrix, the memory. Every token produces a key (an address), a value (content) and a query. Writing adds "value at this key" into the matrix; reading multiplies the matrix by the query. The delta rule write first asks the matrix what it currently returns for this key, and writes only the difference between the new value and that answer.

Running example. Keys and values are 2-number lists. Alice's key is `[1, 0]`. Paris is `[1, 0]`, Rome is `[0, 1]`.
- Plain additive write: after "Alice lives in Paris" and "Alice moved to Rome", asking for Alice returns Paris + Rome = `[1, 1]`. Both answers are mixed.
- Delta write: at "Rome" the memory is asked for Alice, returns Paris `[1, 0]`, and the write is Rome − Paris = `[-1, 1]`. Asking for Alice now returns `[0, 1]` = Rome.

---

### 1. DeltaNet

**Title, authors, year, arXiv ID or venue.** "Parallelizing Linear Transformers with the Delta Rule over Sequence Length". Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. arXiv 2406.06484 (v6); NeurIPS 2024.

**The problem.** Schlag's delta rule ran one token at a time, because each write needs the matrix left by the previous write. That is slow on GPUs, so it had not been trained at language-model scale.

**The idea.** The write rule is unchanged; the paper changes how it is computed in training. A delta write is still "add something at this key", where the something is a corrected value. For Alice: the first corrected value is Paris. The second is Rome minus (Paris × the overlap of the two keys). The overlap of `[1,0]` with `[1,0]` is 1, giving Rome − Paris. That needed only dot products between keys, not the matrix. So the sequence is cut into chunks of 64 tokens, all corrected values in a chunk are found with one triangular solve (matrix multiplications), and the full matrix is only carried chunk to chunk. Two smaller changes: keys are scaled to length 1 (L2 norm), and the key activation is SiLU.
Summary: S ← S − β (S k − v) kᵀ, β a write strength in (0, 1).

**What the memory is and how it is written.** One matrix per head (head size 128) in every layer, replacing self-attention in a Llama-style stack. One write per token. The pseudocode starts it at zeros for each sequence.

**What is learned versus fixed.** Learned by ordinary backprop: the projections giving key, value, query, write strength, and the MLPs. Changed at run time: only the memory matrix. The rule is hand-designed (one gradient step on "matrix × key should equal value"); only its inputs and step size are learned. No gradients of gradients: the step is a written-out formula (my inference; the paper does not discuss it).

**Does the model write its own weights?** Partly: a trained network outputs key/value/strength and a fixed rule turns them into a change to a separate fast matrix; trained weights never change.

**Results.** Table 1, 1.3B parameters, 100B tokens of SlimPajama: Wikitext perplexity 16.87 (Transformer++ 16.85, Mamba 17.06, GLA 17.22); zero-shot average 51.6 (50.9 / 50.0 / 51.0). Retrieval trails attention: SWDE 49.5 vs 66.6; replacing two layers with global attention gives 71.0.

**Limits.** Authors: slower to train than GLA; the kernel caps head size and so memory size, and at 1.3B it loses to GLA on recall; poor generalization beyond training length, attributed to having no decay. Mine: the only forgetting is overwriting the same key; state is reset per sequence.

**Read status.** 109,969 characters; read main text and Appendix A–D with pseudocode. Skipped most of the reference list.

---

### 2. Gated DeltaNet

**Title, authors, year, arXiv ID or venue.** "Gated Delta Networks: Improving Mamba2 with Delta Rule". Songlin Yang, Jan Kautz, Ali Hatamizadeh. arXiv 2412.06464; ICLR 2025.

**The problem.** The delta rule replaces one key's content at a time, so it cannot clear the memory quickly when the topic changes. Mamba2 fades the whole memory each step by one number, which clears fast but fades everything equally.

**The idea.** Do both. Before each delta write, multiply the whole matrix by a number between 0 and 1 (the gate) computed from the current token. Example: the memory holds Alice → Paris plus many facts from an essay. An unrelated document starts: gate near 0 wipes the matrix. Later "Alice moved to Rome" arrives: gate near 1 keeps everything and the delta write replaces only Paris. The paper reads this as gradient descent with weight decay on the memory. The chunked algorithm of paper 1 is extended by tracking the running product of gates.
Summary: S ← α (S − β (S k) kᵀ) + β v kᵀ, α and β in (0, 1).

**What the memory is and how it is written.** As in DeltaNet: one matrix per head (size 128) per layer, one write per token, now with a per-token, per-head scalar decay. Empty at sequence start (inferred from the DeltaNet code it builds on).

**What is learned versus fixed.** Learned by backprop: projections for key, value, query, gate, write strength, an output gate, the MLPs. Run time: only the matrix. The rule is hand-designed; gate and strength are predicted from the current token alone, not from the memory. No gradients of gradients (my inference).

**Does the model write its own weights?** Partly, as in DeltaNet: the network emits the ingredients, a fixed rule edits a separate fast matrix.

**Results.** 1.3B parameters, 100B tokens FineWeb-Edu, 4K training length.
- Table 3: Wikitext perplexity 16.42 (Mamba2 16.56, DeltaNet 17.71); average accuracy 55.32 (54.89 / 52.14).
- Table 2, find a number in an essay at 4K tokens (S-NIAH-2): 92.2 vs Mamba2 56.2 vs DeltaNet 18.6; at 8K it drops to 29.6.
- Table 4, real retrieval average: 30.6 vs Mamba2 29.8; Transformer++ 37.0; hybrids with sliding-window attention 39.0–40.1.

**Limits.** Authors: decay costs retention (pass-key at 2K: 88.4 vs DeltaNet 96.8); recurrent models still trail attention on retrieval; length extrapolation is "mixed". Mine: one scale, no limitations section, memory lives only within a sequence.

**Read status.** 95,814 characters; read main text and Appendix A–B. Skipped references.

---

### 3. Kimi Linear

**Title, authors, year, arXiv ID or venue.** "Kimi Linear: An Expressive, Efficient Attention Architecture". Kimi Team (Moonshot AI). arXiv 2510.26692 (v2, Nov 2025), technical report.

**The problem.** Gated DeltaNet fades a head's whole memory by one number. Also, no linear-attention model had beaten full attention at production scale in a matched comparison.

**The idea.** (a) Kimi Delta Attention (KDA): the gate becomes a list, one number per key dimension (128 per head), so each row of the memory has its own forgetting rate. Example (my illustration): rows holding "who lives where" should last; rows tracking the current sentence should fade fast. One number cannot do both; a per-row gate keeps Alice → Rome while clearing the short-lived rows. (b) A hybrid stack: three KDA layers, then one full-attention layer, repeated. The attention layers carry no position encoding; the authors argue the KDA decay acts as a learned position signal. Tying the low-rank part of the update to the key makes the chunked kernel about 2× faster than the general diagonal-plus-low-rank form (Fig. 2).
Summary: S ← (I − β k kᵀ) Diag(α) S + β k vᵀ, α a vector in [0, 1].

**What the memory is and how it is written.** A 128×128 matrix per head in each KDA layer, one write per token, fixed size at any context length. The full-attention layers still keep a growing cache. Starts at zeros unless an initial state is passed (Appendix C code).

**What is learned versus fixed.** Learned (pretraining, then SFT, then RL): all projections, the low-rank map that outputs the per-row gate, a sigmoid output gate, the MoE MLPs. Run time: only the matrices. Rule hand-designed. No gradients of gradients (my inference).

**Does the model write its own weights?** Partly, same sense: the network sets key/value/gate/strength, a fixed rule edits a fast matrix; trained weights are frozen at inference.

**Results.** 48B total / 3B active parameters, 1.4T tokens, same recipe for full attention (MLA) and hybrid Gated DeltaNet (GDN-H).
- Table 3: MMLU-Pro 51.0 vs 47.2 (MLA) vs 47.9 (GDN-H).
- Table 5, 128k context: RULER 84.3 vs 81.3 vs 80.5; average 54.5 vs 52.2 vs 51.2.
- Up to 75% less KV cache; at 1M tokens 6.3× faster per output token (1.84 ms vs 11.48 ms, Fig. 1b, using larger batches); about 2.2× at batch size 1 (Fig. 7b).

**Limits.** Authors: pure linear attention still lags on exact copying and long retrieval, hence a quarter of layers stay full attention; it loses on EvalPlus, LongBench v2, Frames. Mine: the gain over GDN-H at scale is not split between the per-row gate and the removed position encoding; data and baselines are in-house; §6.3 quotes 2.3× where Fig. 7b shows 2.2×. Memory is per sequence.

**Read status.** 117,565 characters; read main text and Appendix B–D. Skipped contributor list and references. Figures are not in the text; numbers come from captions and tables.

---

### 4. Negative eigenvalues

**Title, authors, year, arXiv ID or venue.** "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues". Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K.H. Franke, Frank Hutter, Massimiliano Pontil. arXiv 2411.12537 (v5); ICLR 2025.

**The problem.** These models cannot track a state that flips. Simplest case, parity: read 0s and 1s and say whether the count of 1s is odd. Transformers, Mamba and DeltaNet fail beyond training length.

**The idea.** Look at what one step does to the memory along the key direction. A DeltaNet write with strength 0 to 1 scales that direction by a number from 1 down to 0: keep, shrink or erase. Repeated shrinking settles to one value and cannot alternate. Parity needs a flip. One-number example: start at `1`; each time a 1 is read, multiply by −1: `1, -1, 1, -1`. The sign is the answer. The fix: let write strength run from 0 to 2. Strength 1 erases what is stored at the key; strength 2 subtracts it twice, flipping its sign (a reflection). For Mamba the equivalent is 2 × gate − 1. Theory: with only non-negative scalings parity is impossible at any depth in finite precision (Theorem 1); counting modulo 3 also needs non-triangular matrices (Theorem 2); products of several reflections per step can follow any finite-state machine (Theorems 3–4). A swap of two items is one reflection, so one DeltaNet step can do it.
Summary: S ← S − 2β (S k − v) kᵀ; the scaling along k is 1 − 2β in [−1, 1].

**What the memory is and how it is written.** Unchanged from DeltaNet: one matrix per head, one write per token, empty at sequence start. Only the allowed range of write strength changes.

**What is learned versus fixed.** Learned by backprop: the same projections; the model learns when to output strength near 2. Rule hand-designed; a one-line edit (Fig. 13: `beta = 2 * sigmoid(...)`), no extra cost. No gradients of gradients (my inference).

**Does the model write its own weights?** Partly, as in DeltaNet; the paper only widens what one write can do.

**Results.**
- Table 3, trained on lengths 3–40, tested 40–256 (scaled accuracy, best of 3): parity 0.017 → 1.000 for DeltaNet, 0.000 → 1.000 for Mamba; modular arithmetic without brackets 0.314 → 0.971 for DeltaNet.
- Fig. 4: permutations of 5 items using swaps only are solved by a 1-layer modified DeltaNet; the full task is not.
- 1.3B DeltaNet, 100B tokens FineWeb: trains stably, lower perplexity on code and math (Fig. 5), Wikitext 18.57 vs 18.54 (Table 4).

**Limits.** Authors: the general theorem needs layers and width exponential in the number of states; 1.3B results are "mixed"; the change breaks the log-space trick used by Mamba2 and GLA; Mamba gets worse on language; they suspect a trade-off between state tracking and recall (FDA retrieval at 340M: 11.8 → 6.6). Mine: high seed variance on permutation tasks; nothing on memory across sequences.

**Read status.** 146,447 characters; read main text, Appendix A, D, E. Not read: Appendix B and C (proofs for Sections 4.1 and 4.3, about 310 lines); theorem descriptions come from the main-text statements and sketches. Skipped references.


## 3. Better write rules, part two, and in-context gradient descent

Starting point for all four (Schlag et al. 2021): attention without softmax is a matrix. Each token writes "key -> value" into it; a later token reads by multiplying the matrix with its query. The delta-rule write first asks the matrix what it already returns for this key and writes only the difference. That is one gradient step on the squared error "what the matrix returns for this key" minus "the value".

---

### 1. DeltaProduct

**Title, authors, year, arXiv ID.** "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products". Siems, Carstensen, Zela, Hutter, Pontil, Grazzi. arXiv 2502.10297 (v7, Oct 2025).

**The problem.** A delta-rule write can change the memory along one direction per token. Some tasks need more per token. Example: three cups, a ball under one, and each token is a shuffle of the cups. A delta write can swap two cups. It cannot rotate all three in one token, so DeltaNet needs several layers and still fails on five cups.

**The idea.** Do several delta writes per token instead of one. Each token produces `n_h` keys, values and step sizes, and the writes are applied one after another before the next token arrives. In list form, a swap of cups 1 and 2 is one write with key `[1,-1,0]/sqrt(2)` and step size 2: state `[a,b,c]` becomes `[b,a,c]` (a mirror flip). The rotation `[a,b,c] -> [c,a,b]` is two swaps, so it needs `n_h = 2`. Two flips make a rotation; that is the whole trick. Each write on its own never grows the state, so a chain of them does not either, and training stays stable.

**What the memory is and how it is written.** One matrix per head per layer (same as DeltaNet). Written `n_h` times per token by the delta rule. Starts at zero for every sequence; a beginning-of-sequence token sets the initial content. An optional scalar forget gate (one per token) shrinks the whole matrix ("Gated DeltaProduct").

**What is learned versus fixed.** Learned by ordinary backprop: the projections that turn a token into the `n_h` keys, values and step sizes, the gate, the MLPs. Changed at run time: only the matrix. The write rule is hand-designed (gradient step on squared error, written out as a formula). No gradients of gradients (my reading: the step is a closed-form expression, so training is plain backprop through the recurrence).

**Does the model write its own weights?** Partly: the trained network outputs changes to a separate fast-weight matrix; the trained weights never change.

**Results.**
- Cup-shuffling (Section 5.2, Figure 5; trained on 128 shuffles, tested to 512): one layer solves 3 cups with `n_h=2` and 5 cups (S5) with `n_h=4`. DeltaNet (`n_h=1`) with 10 layers did not fit S5.
- Language, 392M parameters, 35B FineWeb tokens (Table 5): DeltaProduct2 average 45.5 and Lambada perplexity 30.66, versus parameter-matched DeltaNet (12 heads) 44.3 and 36.76.
- 805M parameters, 55B tokens (Table 3): DeltaProduct3 average 49.95 versus DeltaNet 48.55.
- Trained at 4096 tokens, `n_h=3` shows "minimal" loss degradation on longer text (Figure 8; no table).

**Limits.** Authors: recurrence cost grows linearly with `n_h`. Noticed: state tracking only works with step sizes up to 2 (the "[-1,1]" setting); with the usual 0-to-1 range it fails for any `n_h`. In the Chomsky-hierarchy table (Table 2) DeltaProduct2 and 3 average below DeltaNet (0.704, 0.666 vs 0.726); only `n_h=4` is ahead (0.739). The multi-layer theory relies on very large lookup tables in an MLP. Memory is reset per sequence; nothing about persistence across sessions.

**Read status.** Full HTML text, about 145k characters, main text and Appendices A-C to the end. Reference list skimmed. Figures seen only through captions.

---

### 2. Longhorn

**Title, authors, year, arXiv ID.** "Longhorn: State Space Models are Amortized Online Learners". Liu, Wang, Wu, Feng, Stone, Liu. arXiv 2407.14207 (v5, Oct 2024).

**The problem.** Mamba-style models have a hand-tuned rule for how much old memory to keep and how much new to add, with a separate forget gate. The delta rule gives a principled rule, but its step can overshoot when keys are long, so DeltaNet has to normalise keys.

**The idea.** Ask a different question at each token: "which new memory is closest to the old one while also returning the new value for this key?" That has an exact answer, and the answer is a delta write with an automatic step size. Take a memory that returns 4 ("Paris") for key Alice, and a new value 10 ("Rome"). With trust `beta = 1` and key length 1, the step is `beta / (1 + beta * |key|^2) = 0.5`, so the memory now returns 7. A large `beta` pushes it toward 10. It can never overshoot, whatever the key length. The forgetting comes from the key itself, so no separate forget gate.

**What the memory is and how it is written.** One matrix per layer (values by key dimensions), sitting where Mamba's state sits in a Mamba block. The value is the token's own input vector. Starts at zero every sequence. Important detail: for speed the paper does not run the exact rule. It uses a cell-by-cell approximation: each cell is shrunk by `1 - step * key_j^2`, then `step * value_i * key_j` is added. So the practical model does not literally subtract what the matrix returns for the key; it is a key-dependent decay plus a write.

**What is learned versus fixed.** Learned by backprop: the projections giving key, query and the trust vector `beta` (one number per value dimension). Run time: the matrix only. Rule: hand-designed, derived in closed form (Theorem 3.1, proof in Appendix C). No gradients of gradients.

**Does the model write its own weights?** Partly: trained network outputs the key, value and step that update a separate fast matrix.

**Results.**
- 1.3B parameters, 100B SlimPajama tokens, average over 8 zero-shot tasks (Table 2): Longhorn 52.35, Mamba 52.25, GLA 51.07, LLaMA 53.03.
- "1.8x" fewer training tokens than Mamba to reach the same average perplexity (Figure 1 left).
- OpenWebText, about 350M parameters, context 4096, validation loss (Table 1): Longhorn 2.859, Mamba 2.868, LLaMA 2.883.
- Trained at 2048 tokens, perplexity holds to 32K (Figure 1 right).

**Limits.** No limitations section. Noticed: the gains over Mamba are small (0.1 points average; 0.01 loss); the Transformer still wins the 1.3B table. The shipped rule is the diagonal approximation, and there is no ablation against the exact rule. Table 2's caption mentions models and state sizes that do not match the table. Associative-recall results are a figure only. Memory resets each sequence.

**Read status.** Full HTML text, about 58k characters, including Appendices A-D. Nothing unread.

---

### 3. MesaNet

**Title, authors, year, arXiv ID.** "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training". von Oswald, Scherrer, Kobayashi, Versari, Yang, Mittal, Schlegel, Maile, Schimpf, Sieberling, Meulemans, Saurous, Lajoie, Frenkel, Pascanu, Aguera y Arcas, Sacramento. arXiv 2506.05233 (first posted 2025).

**The problem.** The delta rule takes one small correction per token, looking only at the newest pair. Earlier pairs can stay badly stored. An earlier "Mesa layer" computed the best possible memory but could only be trained one token at a time and broke numerically with forgetting.

**The idea.** At every token, answer as if the memory were the best-fit linear map for all key-value pairs seen so far (least squares with a small penalty), not the result of one step. Plain-Python example with overlapping keys: store `[1,0] -> 5` and `[1,1] -> 7`. Simple summing memory is `G = [12, 7]`; asking with `[1,0]` returns 12, which is wrong. Mesa also keeps `H = [[2,1],[1,1]]` (sum of key-times-key). It first solves `H x = [1,0]`, giving `x = [1,-1]`, then reads `G . x = 12 - 7 = 5`. Correct. The solve removes the overlap between keys.

**What the memory is and how it is written.** Two matrices per head per layer: `G` (values times keys) and `H` (keys times keys). Both are written by simple addition with an input gate, and decayed by a forget gate. The work is at read time: a conjugate-gradient solver (an iterative equation solver; 30 steps in training) cleans the query. Start at zero each sequence (stated for `H`; assumed for `G`). Formula: output = `G (H + Lambda)^-1 q`.

**What is learned versus fixed.** Learned by backprop: key, query, value projections, the two gates, the penalty `Lambda` (per head, kept above 0.25). Run time: `G` and `H`. Rule: hand-designed. Training goes through the solver by solving a second linear system in the backward pass (Appendix D), not by unrolling. No gradients of gradients.

**Does the model write its own weights?** Partly: the network's outputs fill two statistics matrices, and a fixed solver turns them into the best-fit fast weights.

**Results** (1B parameters, 50B SlimPajama tokens, context 2048).
- Average perplexity over six sets (Table 2): Mesa 13.79, Transformer 13.79, Gated DeltaNet 13.87.
- In-context recall (Table 3b): Mesa 41.79, Gated DeltaNet 39.54, Transformer 52.27.
- With a stopping threshold of 1e-4 the solver averages about 9 steps and matches the 30-step model (Section 5).

**Limits.** Authors: extra compute per token (about k times a linear-attention layer for k solver steps); still behind Transformers on recall and translation (8.17 vs 13.61 BLEU); recurrent models win early in a sequence and lose later. Noticed: a sliding-window Transformer with a 1024 window is about as good on many tests. The forget gate is capped at 0.9975 to survive repeated tokens. Negative forget values (needed for state tracking) break the solver unless the layer is forced back to plain linear attention (Appendix I). Inferred, not stated: the best fit of "Alice -> Paris" and "Alice -> Rome" is their average, so replacing a fact depends on the forget gate. Reset per sequence.

**Read status.** Main text and prose of Appendices A-O read (about 218k characters in the file). The long score tables in Appendix L and the reference list were skimmed, not read cell by cell.

---

### 4. Transformers learn in-context by gradient descent

**Title, authors, year, arXiv ID.** "Transformers Learn In-Context by Gradient Descent". von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov. arXiv 2212.07677 (2022).

**The problem.** Transformers can learn a task from examples in the prompt, but nobody could say what computation does that. Schlag et al. needed a modified layer to get the delta rule; this paper asks whether a standard attention layer already does gradient descent.

**The idea.** Put each example in one token: input and label side by side. Then one attention layer without softmax, with the right weights, produces exactly the prediction that one gradient step on linear regression would. Plain-Python example: context `[1,0] -> 2` and `[0,1] -> 3`, query `[1,1]`. Gradient step from zero weights (step size 1): `W = 2*[1,0] + 3*[0,1] = [2,3]`, prediction 5. Attention: each label times the overlap of its input with the query, `2*1 + 3*1 = 5`. Same number. In Schlag's terms, inputs are keys and labels are values, and one step from zero is the plain sum write. The "subtract what you already return" part appears with depth: each layer replaces every label by its remaining error, so K layers act like K steps.

**What the memory is and how it is written.** Nothing is stored between tokens or sequences. The learned linear model is implicit: it lives in the label slot of the tokens, which each layer overwrites with residuals.

**What is learned versus fixed.** Everything is ordinary weights trained by backprop (Adam) on freshly sampled regression tasks, with loss only on the query. Nothing changes at run time. The gradient-descent rule is not built in; training finds it. The step size ends up encoded in the weights. No gradients of gradients for the Transformer (a comparison model, an MLP fine-tuned by one step, is trained by backprop through that step).

**Does the model write its own weights?** No: weights are frozen; the forward pass behaves as if a small linear model were being trained.

**Results** (10 examples, 10-dimensional inputs, Section 3 and Appendix).
- One trained layer: same loss as one tuned gradient step (about 0.2, Appendix A.11), near-perfect alignment of predictions, and trained weights match the hand construction after rescaling (Figure 2, Figure 9; exact values are in plots only).
- Deeper models beat plain gradient descent and match "GD++", gradient descent plus a learned rescaling of inputs; learned factor 0.179 for 10 examples, 0.099 for 25 (Appendix A.10).
- With inputs and labels in separate tokens, a two-layer model matches one step, and the first layer must use softmax to copy them together (Section 4).

**Limits.** Authors: small Transformers, toy regression, noise-free data. Noticed: single-head softmax attention does not match gradient descent (two heads get closer); LayerNorm makes the match worse; weight interpolation fails for the 5-layer untied model; deep models needed value clipping to train. No claim is tested on language.

**Read status.** Full ar5iv text, about 100k characters, including Appendix A.1-A.12. Figures seen only through captions.


## 4. Why it works

Read from arXiv/ar5iv HTML text. Figures were not viewed, only captions and surrounding text. "Inferred" and "noticed" mark my own points, not the authors'.

---

### 1. Transformers learn to implement preconditioned gradient descent for in-context learning

- **Title, authors, year, arXiv ID or venue.** Ahn, Cheng, Daneshmand, Sra (MIT), 2023. arXiv 2306.00297. (Venue NeurIPS 2023 is from memory.)
- **The problem.** Earlier work hand-picked attention weights so a forward pass imitates gradient descent. Nobody had shown that training arrives at such weights.
- **The idea.** Train on prompts of solved examples plus one query. Tiny case: the hidden rule is "output = 2 x input". The prompt shows (1 -> 2), (2 -> 4), then asks about 3. A learner starting from "multiplier = 0" takes one gradient step: it averages input x output, (1x2 + 2x4)/2 = 5, and sets multiplier = step size x 5. With step 0.4 the multiplier is 2 and the answer is 6. One attention layer without softmax computes the same sum: (3x1x2 + 3x2x4)/2 = 15, times 0.4 = 6. The paper proves the best single layer is exactly this, with the step size replaced by a trained matrix (the "preconditioner") that rescales each input direction, roughly by the inverse of how spread out inputs are, and shrinks when examples are few. L layers do L such steps.
- **What the memory is and how it is written.** Nothing is written. The running estimate lives in activations (the query's empty label slot, refined layer by layer) and vanishes with the prompt.
- **What is learned versus fixed.** Attention matrices are trained by backprop on random regression prompts, then frozen. Gradient descent is not coded anywhere; it is what the trained forward pass equals. No gradients of gradients.
- **Does the model write its own weights?** No. The inner "model" exists only as activations.
- **Results.** Theorems (Table 1). One layer, Gaussian inputs with any covariance: the global minimum is one preconditioned step (Thm 1). Two layers: global minimum is two steps, only for isotropic inputs and restricted symmetric sparse weights (Thm 2). L layers: weights of the form "scalar x inverse covariance" contain near-stationary points (Thms 3, 4). Experiments (Sec. 4.3, 5.1): 3 layers, dimension 5, 20 or 10 examples; loss "converges to an almost 0 value" and learned matrices match the predicted form. No numeric tables.
- **Limits.** Shown only for attention without softmax, no MLPs, noise-free linear regression, Gaussian data. Beyond one layer the result is about stationary points; other ones exist (Lemma 4) and convergence to the good ones is not proven. ReLU attention: one layer only (Thm 5). Says nothing about language models. Noticed: the preconditioner is fixed after training, tuned to the training distribution, not to each prompt (for 2 examples in 1 dimension the proven step is 0.5, not the 0.4 that fits the example above).
- **Read status.** 142,179 characters. Main text in full; appendix A.1, B.2 outline, B.4, C, D read; the proof algebra in A.2-A.4, B.1, B.3 skipped.

---

### 2. Test-time regression: a unifying framework for designing sequence models with associative memory

- **Title, authors, year, arXiv ID or venue.** Wang, Shi, Fox (Stanford), 2025. arXiv 2501.12352 (v3).
- **The problem.** Attention, linear attention, Mamba-style models, DeltaNet and test-time-training layers were invented separately. There was no shared account.
- **The idea.** Each layer memorizes (key, value) pairs from tokens seen so far, then looks up the query. Memorizing is curve fitting: find a function m with m(key) close to value. Three choices fix a layer: how much each past pair counts, what kind of function m is, which solver fits it. Tiny case: keys Alice = [1, 0], Bob = [1, 1] (overlapping); values Paris = 1, Rome = 2.
  - Linear attention adds value x key: memory = 1x[1,0] + 2x[1,1] = [3, 2]. Asking about Alice returns 3. Wrong: Bob leaks in.
  - Exact least squares gives memory [1, 1]: Alice 1, Bob 2. Correct.
  - DeltaNet takes one error-correcting step per token and ends at [1.5, 0.5]: Bob 2, Alice 1.5.
  So linear attention is least squares with the "undo key overlap" factor dropped (or one gradient step from zero). Mamba-2 and gated linear attention add down-weighting of old pairs. DeltaNet is one stochastic-gradient step per token; Gated DeltaNet adds weight decay. Softmax attention keeps all pairs and returns a similarity-weighted average of values; this is exact "locally constant" regression only when keys and queries have length 1.
- **What the memory is and how it is written.** Recurrent layers: one matrix per head, updated per token inside the forward pass. Softmax attention: the stored list of pairs. Per-sequence reset is implied (recurrences start at zero), not stated.
- **What is learned versus fixed.** Backprop trains the projections producing keys, values, queries, step sizes and forget factors. The matrix changes at run time by a hand-designed rule (the chosen solver). Gradients of gradients are not discussed; for a linear memory the inner gradient is a closed-form matrix expression (inferred: not needed).
- **Does the model write its own weights?** Partly: a fixed rule writes a weight matrix from quantities the trained network produces.
- **Results.** Mostly derivations. Sec. 5: synthetic drifting sequence (key size 64, length 256, no trained parameters); linear attention worst, a new "local linear" attention best (Fig. 2, no numbers). Sec. 6: on the recall task MQAR with embedding size 64, one layer plus a length-2 convolution solves 64 pairs at any sequence length; at 128 pairs linear attention fails and recursive least squares succeeds (Fig. 3).
- **Limits.** Exact for the delta-rule family; a "crude approximation" for linear attention and Mamba. Shows these layers can be written as regression solvers, not that trained models use them as memories. No language-model experiments. The authors call the new attention impractical at scale.
- **Read status.** 125,630 characters; main text and Appendix A in full.

---

### 3. It's All Connected (Miras)

- **Title, authors, year, arXiv ID or venue.** Behrouz, Razaviyayn, Zhong, Mirrokni (Google Research), 2025. arXiv 2504.13173 (v1).
- **The problem.** Existing memory layers all score a write with dot product or squared error and all forget by the same decay. Those were never treated as choices.
- **The idea.** Describe a layer by four choices: memory shape (vector, matrix, MLP); "attentional bias", the loss saying how wrong the memory is about the current (key, value) pair; "retention gate", a penalty for drifting from the previous memory or growing large; and the solver. Each token: new memory = best trade between "fit this pair" and "stay close to before". Table 1 places 16 known models in this grid. Tiny case: memory holds "Alice -> Paris"; a garbled token arrives where the memory predicts 1 and the token says 101. Squared error makes a correction proportional to 100 and wrecks the memory. A Huber loss with threshold 1 caps the correction at 1. Separately, multiplying old memory by 0.9 each step equals a penalty on memory size; other penalties give other forgetting rules. Three new models, each a 2-layer MLP memory trained by gradient steps: **Moneta** (error to power 3, memory rescaled by a 4-norm), **Yaad** (Huber loss, learned input-dependent threshold), **Memora** (squared error; weights kept positive with fixed sum via softmax).
- **What the memory is and how it is written.** The MLP's weights, one MLP per layer in place of attention, written by a gradient step per token. In practice tokens come in chunks of 16 or 64 and all gradients in a chunk use the memory from the end of the previous chunk. Reset between sequences is not stated.
- **What is learned versus fixed.** Backprop trains projections, convolutions and the small networks emitting per-token learning rate, retention factor and threshold. The rule's form is hand-designed; its coefficients are learned. The outer gradient passes through the inner gradient; whether second-order terms are kept is not stated (inferred yes).
- **Does the model write its own weights?** Partly: an optimizer step writes the memory MLP, with targets and step sizes supplied by the trained network.
- **Results.** Table 2, 1.3B parameters, 100B tokens, Wikitext perplexity: Yaad 15.18, Moneta 15.52, Memora 15.90; Gated DeltaNet 16.42; Transformer++ 18.53. Table 3, needle-in-haystack up to 8K tokens, average accuracy: Moneta 93.5, Yaad 92.9, Memora 92.1; Gated DeltaNet 75.8. Table 4, Yaad ablation: 53.98 full, 50.63 without retention, 51.57 with linear memory.
- **Limits.** The unification is a relabeling plus one proposition; the gains are empirical, up to 1.3B parameters, training context 4096. Baseline numbers are copied from the Titans paper. Noticed: Titans itself is absent from Tables 2 and 3; "outperforms all baselines" fails in some columns (SIQA at 340M: DeltaNet 37.96); no error bars; p=3 is a tuned value with no trend. Nothing on forgetting across sessions.
- **Read status.** 117,748 characters; main text and Appendices A-C in full.

---

### 4. Test-Time Training with KV Binding Is Secretly Linear Attention

- **Title, authors, year, arXiv ID or venue.** Liu, Elflein, Litany, Gojcic, Li (NVIDIA, Toronto, Vector, Technion), 2026. arXiv 2602.21204 (v4).
- **The problem.** Test-time-training layers are explained as "an inner network memorizes key -> value pairs, the query retrieves them". That story justified deeper inner networks, momentum and normalization. Trained models do not behave that way.
- **The idea.** Four tests where memorization predicts harm and none occurs (Sec. 4): more inner steps lower the inner loss but hurt the task; gradient ascent works as well as descent; queries and keys sit in different regions of feature space; replacing the query with the key changes nothing. Then the algebra. Tiny case: the inner network is one linear layer W starting at zero; the inner loss is minus (output dot value). A gradient step on key [1, 0] with value 5 adds key x value, so W = [5, 0]. Over many tokens W is the sum of key x value products, and a query reads out the sum of (query dot key) x value. That is linear attention. With ascent W = [-5, 0]; the trained value projection flips its sign and nothing is lost. Theorems 5.1-5.3 extend this to an inner MLP with a linear, bias-free last layer: "key" = hidden feature of the key, "value" = negative gradient of the inner loss at the output, "query" = hidden feature of the query; momentum only reweights values.
- **What the memory is and how it is written.** The inner network's weights, updated per token or chunk. Under this reading the part that matters is the last layer: a running sum of outer products, i.e. a linear-attention state. Reset is not discussed.
- **What is learned versus fixed.** Outer backprop trains projections and the starting inner weights; the inner rule is hand-designed. With last-layer-only updates and no weight normalization the layer has a closed parallel form (inferred: no gradients of gradients needed then).
- **Does the model write its own weights?** Partly, and less than advertised: the write is the same outer-product addition linear attention does.
- **Results.** LaCT language model 760M (perplexity, Book-3, 32k length) / LaCT view synthesis 114M (PSNR) / ViTTT-B 90M (ImageNet top-1). Table 1: baseline 16.43 / 25.94 / 79.34; gradient ascent 16.19 / 25.85 / 79.61; query replaced by key 16.18 / 25.95 / 79.18. Table 2: last-layer-only updates are best, 15.93 / 25.97 / 79.63; plain linear attention 16.80 / 25.73 / 79.54. Parallel form: "up to 4.0x" layer throughput, "1.19x" end-to-end training speedup.
- **Limits.** Authors: needs a linear bias-free final layer; only LaCT and ViTTT tested; Titans and Atlas "expected" to follow. Noticed: both tested models use the dot-product inner loss, where ascent is just a sign flip. With a squared-error inner loss (TTT-MLP, Titans, Miras) the "value" is the current error and depends on the memory state, so the results may not transfer. When all inner layers update, key and query features change with history, so the equivalence is loose; the exact one needs last-layer-only updates. Gradient orthogonalization does matter for language (15.97 -> 16.80). No error bars.
- **Read status.** 81,026 characters; main text and Appendices A-I in full.

---

### How the four fit together

- Paper 1: a proof, in a toy setting, that training makes a forward pass equal gradient steps. Nothing is stored.
- Paper 2: memory-layer write rules are regression solvers; exact for the delta-rule family.
- Paper 3: makes the loss and the forgetting penalty design choices and tests three picks at up to 1.3B parameters.
- Paper 4: pushes back. For dot-product-loss TTT layers the "memorizing inner network" is linear attention with a learned feature map. It does not test the squared-error models of papers 2 and 3.


## 5. Writing into the model's own weights, part one

Shared vocabulary for all four. Each paper has a small network inside a layer whose weights change while the model reads a sequence. Changing those weights is the **inner loop**. Training everything else before deployment (ordinary next-token training) is the **outer loop**. The inner loop always does the same kind of thing: turn each token into a "key" vector and a "value" vector, and nudge the small network so that feeding it the key gives back the value. Reading "Alice lives in Paris" nudges it so that an Alice-like key returns a Paris-like value. Later, a "query" vector made from "where does Alice live?" is fed through the same small network to read the answer out.

---

### 1. TTT layers

- **Title, authors, year, arXiv ID.** "Learning to (Learn at Test Time): RNNs with Expressive Hidden States." Sun, Li, Dalal, Xu, Vikram, Zhang, Dubois, Chen, Wang, Koyejo, Hashimoto, Guestrin. 2024. arXiv 2407.04620 (v4 read).
- **The problem.** A recurrent layer must squeeze everything it has read into a fixed-size state. With a vector or matrix state, Mamba stops benefiting from extra context after about 16k tokens. Attention keeps everything but its cost per token grows with length.
- **The idea.** Make the state a small model, and make "update the state" mean "take one training step on the token just read". The model reads "Alice lives in Paris": the layer projects the token representation two ways (an input view and a target view) and takes one gradient step so the small model maps the first to the second. When asked about Alice, a third projection of the current token is passed through the updated small model, and the result is the layer's output.
- **What the memory is and how it is written.** The weights W of a small model f inside each TTT layer: a square matrix (TTT-Linear) or a two-layer MLP with 4x hidden width (TTT-MLP), both wrapped in a layer norm and residual. The inner loss is squared error between f(training view) and the label view. Tokens are processed in groups of 16: all 16 gradients are taken at the weights from the end of the previous group, then summed in. For the linear model the gradient is a closed formula (an outer product), and a "dual form" computes outputs without ever building the per-token gradients. For the MLP it is a real backward pass through f. W starts from the same W0 for every sequence, so it is reset per sequence.
- **What is learned versus fixed.** Learned by the outer loop: the three projection matrices (which define the inner task), the starting weights W0, a per-token learning-rate gate, and the rest of the network. Changed at run time: only W. The update rule is hand-designed (gradient descent), but the loss it descends is learned. Training needs gradients of gradients; the paper says so directly, and uses gradient checkpointing through time to limit stored copies of W.
- **Does the model write its own weights?** Partly. An optimizer step writes W; the slow network decides what that step is trained on and how large it is. The slow weights never change at run time.
- **Results.** Table 1 (125M parameters, Pile, 2k context): linear attention 15.91 perplexity, TTT-Linear 11.09; switching from one full-sequence batch to groups of 16 gives the largest single gain (-1.70). Figures 2 and 11 (Books, up to 32k context, 125M to 1.3B): TTT-Linear and TTT-MLP keep improving with context while Mamba plateaus after 16k; these are plots, no table. Section 3.3: 0.27 s per training iteration versus 0.30 s for the Transformer at 2k context on TPUs.
- **Limits.** Authors: TTT-MLP is costly in wall-clock time and memory I/O; no context beyond 32k; no clean scaling-law fit; only inference GPU kernels written. Noticed: memory lives for one sequence only, nothing carries across sessions; no forgetting mechanism; perplexity is the only metric.
- **Read status.** Full HTML text, 99,370 characters, through appendices A to C. Figures seen only as captions.

---

### 2. Titans

- **Title, authors, year, arXiv ID.** "Titans: Learning to Memorize at Test Time." Behrouz, Zhong, Mirrokni (Google Research). 2024. arXiv 2501.00663 (v1 read).
- **The problem.** Attention is accurate but limited to a window. Matrix-state recurrent models overflow on long inputs and most have no way to drop old content. TTT layers write every token with equal standing and never forget.
- **The idea.** Keep attention as short-term memory and add a small MLP as long-term memory that is trained while reading. How hard a token is written depends on "surprise": the size of the gradient it produces. The model reads "Alice lives in Paris", which is new, so the gradient is large and the fact is written. A running average of recent gradients (momentum) keeps writing the following tokens even if they are individually unsurprising. A decay gate can shrink old weights, so when "Alice moved to Rome" arrives the Paris association can fade.
- **What the memory is and how it is written.** The weights of an MLP with one to four layers. Inner loss: squared error between memory(key) and value, where key and value are linear projections of the token. Update: momentum = (decay gate) x old momentum - (step gate) x gradient; weights = (1 - forget gate) x old weights + momentum. The three gates are computed from the current token. The gradient is a real backward pass through the MLP, taken at the weights from the start of the current chunk so a chunk can be done in parallel. Reading is a forward pass with no update. Three placements: memory output prepended to the attention context (MAC), gated with sliding-window attention (MAG), or stacked before it (MAL). A few learned "persistent" tokens are also prepended and stay fixed.
- **What is learned versus fixed.** Outer loop learns projections, gate functions, attention, persistent tokens. Run time changes only the MLP weights and the momentum buffer. The rule (gradient descent with momentum and weight decay) is hand-designed; its three coefficients are learned functions of the input. The paper names the inner/outer split but never states how the outer gradient passes through the inner steps. Inferred: it backpropagates through them, which implies second-order terms. Reset per sequence is also inferred, not stated.
- **Does the model write its own weights?** Partly. An optimizer step writes the memory MLP; the slow network supplies the data and the step, momentum and forget sizes.
- **Results.** Table 1, 760M parameters, 30B tokens: WikiText perplexity 18.61 (MAG) and 20.04 (memory alone) versus 25.21 Transformer++ and 19.88 Gated DeltaNet-H2. Table 2, needle retrieval S-NIAH-W at 16K: memory alone 80.6, MAC 95.2, versus 0.0 for TTT, Mamba2 and DeltaNet. Table 5 ablation: removing the forget gate moves perplexity from 27.01 to 29.04, the largest single drop.
- **Limits.** Authors: deeper memory trains slower; larger models promised for a later version; code not released. Noticed: the headline BABILong and "over 2M context" results exist only as a figure; training length is 4K; the expressiveness theorem is stated without proof in the text; nothing about memory across sessions.
- **Read status.** Full HTML text, 117,324 characters, including appendices A to C; reference list skipped.

---

### 3. ATLAS

- **Title, authors, year, arXiv ID.** "ATLAS: Learning to Optimally Memorize the Context at Test Time." Behrouz and co-authors at Google (the HTML shows only Mirrokni's name; from the email handles I infer Li, Kacham, Daliri, Deng, Zhong, Razaviyayn). 2025. arXiv 2505.23735 (v1 read).
- **The problem.** Titans-style memories are updated using only the current token, hold few associations for their size, and use plain gradient descent, which the authors say settles on poor solutions.
- **The idea.** Three changes to the Titans memory. (1) Expand each key with polynomial features (products of its entries) so more distinct keys can be told apart; the paper proves a matrix memory stores at most about as many pairs as the key has dimensions, and that degree-p features raise this to that dimension to the power p. (2) The "Omega rule": each update fits the last c tokens together instead of one. With "Alice lives in Paris" then "Alice moved to Rome" inside the same window, the step is taken on both pairs at once, each with a learned 0-to-1 weight, so the stale pair can be down-weighted. (3) Replace gradient descent with the Muon optimizer.
- **What the memory is and how it is written.** A two-layer MLP with residual connection (4x expansion, GELU), one per layer. Inner loss: weighted sum over the last c tokens of squared error between memory(polynomial key) and value. Update: accumulate the gradient into a momentum buffer; pass the buffer through five Newton-Schulz iterations, which rescale it toward an orthogonal matrix; then weights = forget gate x old weights - step x that result. Gradients are real backward passes through the MLP, all taken at the previous chunk's final weights, with a sliding-window mask to sum them per window. The authors state that nothing persists once the memory is cleared and call it "test-time memorization" for that reason.
- **What is learned versus fixed.** Outer loop learns projections, the gates (forget, step, momentum, per-token window weights), polynomial coefficients, and the rest. Run time changes the MLP weights and momentum. The rule is hand-designed with learned coefficients. How the outer gradient passes through the inner steps, including the Newton-Schulz iterations, is not described; second-order backprop is inferred.
- **Does the model write its own weights?** Partly, in the same sense as Titans: an optimizer writes the memory, the slow network sets its inputs and coefficients.
- **Results.** Table 2, 760M: WikiText perplexity 18.92 versus Titans memory 20.04 and Transformer++ 25.21. At 1.3B and 100B tokens: Atlas++ 14.40 versus Transformer++ 18.53. Table 6 ablation: window of 1 moves perplexity 19.97 to 21.98; no polynomial features gives 22.14. Abstract and Figure 4: above 80% on BABILong at 10M tokens where Titans drops.
- **Limits.** Authors: Transformers still lead on in-context recall (Table 5 average 53.55 versus 43.70). Noticed: in Table 6 the row without Muon has lower perplexity (19.65) than full Atlas (19.97), so Muon's benefit there is only +0.21 accuracy; Table 6 and Table 2 give different perplexities for Atlas without explanation; the window size used is not stated in the text; update-rule signs differ between Section 5 and Appendix D; the BABILong result is a figure only; no cost numbers.
- **Read status.** Full HTML text, 131,622 characters, including appendices A to F; reference list skipped.

---

### 4. LaCT

- **Title, authors, year, arXiv ID.** "Test-Time Training Done Right." Zhang, Bi, Hong, Zhang, Luan, Yang, Sunkavalli, Freeman, Tan (MIT, Adobe Research). 2025. arXiv 2505.23884 (v1 read).
- **The problem.** Earlier TTT layers update the fast weights every 1 to 64 tokens. Those small steps use under 5% of GPU capacity, need custom kernels, and keep the memory network small. They also assume a strict left-to-right order, which sets of images do not have.
- **The idea.** Update rarely, on large chunks: 2,048 tokens up to a million. One gradient step is computed from the whole chunk at once. Because a chunk is treated as an unordered bag, ordinary attention over a local window handles order inside it. For text: the model reads a 2,048-token chunk containing "Alice lives in Paris". Outputs for that chunk come from the fast weights as they were before the chunk plus window attention. Then one step writes the whole chunk in. When "where does Alice live?" arrives chunks later, the query goes through the updated fast weights.
- **What the memory is and how it is written.** Three matrices forming a gated (SwiGLU) MLP per head per block; up to 40% of all model parameters. Inner loss: negative dot product between memory(key) and value, weighted by a per-token learning rate predicted from the token, summed over the chunk. Step: subtract the gradient (optionally after momentum and Muon's orthogonalization), then rescale each weight row back to its original length; no explicit forget term. The gradient is a real backward pass through the small MLP, written out by hand as a few matrix products (Algorithm 1), not a closed formula. Fast weights start from a learned initial value copied per sequence.
- **What is learned versus fixed.** Outer loop learns key/value/query projections, the learning-rate and momentum predictors, initial fast weights, attention and feed-forward layers. Run time changes the fast weights only. The rule is hand-designed. The hand-written gradient consists of ordinary differentiable operations and the paper says training is end to end; that this carries second-order terms is my inference.
- **Does the model write its own weights?** Partly. An optimizer step writes the fast MLP; the slow network supplies keys, values and per-token step sizes.
- **Results.** Table 2 (view synthesis, 48 images, 196K tokens): 1.4 s to absorb the inputs versus 16.1 s for full attention; Table 4: PSNR 37.6 versus 37.9 full attention and 33.7 Perceiver. Table 8 (3B language model, needle retrieval average at 32K): 48.0 with Muon versus 28.4 DeltaNet and 91.2 full Transformer. Table 3: 4.3K training tokens per second with Muon versus 5.1K for DeltaNet.
- **Limits.** Authors: fast weights lack rotation invariance, so relative position encodings do not apply to them; reasoning ability and larger language models untested; video has no reliable metric. Noticed: retrieval is far below full attention; a linear large-chunk memory alone loses to per-token baselines on text; the attention window must be at least the chunk size; video data is proprietary and Muon gave no gain there; state is per sequence.
- **Read status.** Full HTML text, 130,126 characters, including appendices A to F; reference list skipped. Most language and video results are plots, read as captions only.


## 5. Writing into the model's own weights, part two

Shared vocabulary. "Inner loop" = the update that happens while the model reads a sequence. "Outer loop" = ordinary training before deployment, which may have to differentiate through the inner loop. Running example: the model reads "Alice lives in Paris", thousands of tokens later is asked "Where does Alice live?", and the first sentence is no longer visible to attention.

---

### 1. TTT-E2E

- **Title, authors, year, arXiv ID or venue.** "End-to-End Test-Time Training for Long Context", Tandon, Dalal, Li, Koceja, Rød, Buchanan, Wang, Leskovec, Koyejo, Hashimoto, Guestrin, McCaleb, Choi, Sun. arXiv 2512.23675v2, Dec 2025.
- **The problem.** Full attention gets better with longer context but its cost per token grows with context length. Fixed-cost models (sliding-window attention, Mamba 2, Gated DeltaNet) stop improving as context grows.
- **The idea.** Keep training the model on the text it is reading. Attention only sees the last 8K tokens. Every 1K tokens the model takes one gradient step on the usual "predict the next token" loss for those 1K tokens. After reading "Alice lives in Paris" and stepping, the fact sits in the weights when the sentence leaves the window. Training is changed so that the starting weights are chosen to be good after these steps, not before.
- **What the memory is and how it is written.** The MLP layers in the last quarter of the Transformer blocks (88M parameters in the 760M model). Each of those blocks has two MLPs; one is updated, one stays frozen as storage for pre-trained knowledge. The write is a real backward pass: forward through the whole network, cross-entropy at the output, backprop through the last quarter of blocks, one gradient-descent step per 1K-token batch. Reset per document (footnote 4 mentions resetting the updated MLPs at document boundaries). During generation it also steps on its own generated tokens once 1K accumulate.
- **What is learned versus fixed.** Outer loop learns everything: the starting values of the fast MLPs, attention, embeddings, norms. The update rule (plain gradient descent on next-token loss) is hand-designed; only its starting point is learned. Whether the inner learning rate is learned is not stated. Training needs gradients of gradients, because the outer loss is the loss after the inner steps (MAML-style), with gradient checkpointing through time.
- **Does the model write its own weights?** Partly: an optimizer does it, using gradients of the model's own prediction loss; no network outputs weight changes.
- **Results.** 3B models, 164B training tokens, Books eval: TTT-E2E keeps the same loss advantage over full attention from 8K to 128K context while the other fixed-cost baselines fall behind (Fig. 1); prefill is "2.7× faster than full attention for 128K context" on an H100. 760M at 8K (Table 1): SWA 2.827, TTT-KVB 2.818, TTT-E2E 2.805. Needle-in-haystack at 128K (Table 2): full attention 0.99, TTT-E2E 0.06.
- **Limits.** Authors: training is 3.4× slower than full attention at 8K context (1.2× faster at 128K); exact recall fails; gains come mostly from early tokens. Noticed: base models only, no instruction tuning; memory is reset per document, so nothing is learned across sessions; forgetting is handled only by the frozen second MLP and never measured.
- **Read status.** 103,903 characters; main text and appendices A–D read in full; reference list skipped.

---

### 2. In-Place TTT

- **Title, authors, year, arXiv ID or venue.** "In-Place Test-Time Training", Feng, Luo, Hua, Zhang, He, Huang, Cai (ByteDance Seed, Peking University). arXiv 2604.06169v1, Apr 2026.
- **The problem.** Earlier test-time-training layers replace attention and need training from scratch, and their write target ("reconstruct the current token") is not tied to predicting the next token.
- **The idea.** Reuse a matrix every Transformer already has: the last matrix of the MLP block, which maps the MLP's hidden activation to its output. Treat it as a lookup table: activation in, output out. While reading, add entries that say "when the activation looks like this, push the output toward the tokens that came next". Reading "Alice lives in" followed by "Paris" writes (activation at "in" → direction of "Paris"). Later a similar activation retrieves a push toward "Paris".
- **What the memory is and how it is written.** `W_down` of the MLP in every sixth layer. The text is cut into chunks of 512–1024 tokens. For each chunk: first use the current matrix, then add `learning_rate × targetᵀ × activations` summed over the chunk. The target is a learned projection of a small causal convolution (kernel 5) over token embeddings, so it carries next-token information. The inner loss is the negative dot product between MLP output and target; its gradient does not depend on the matrix, so the step is a closed formula, a sum of outer products. No backward pass at run time, and all chunks can be computed in parallel with a running sum. Reset to the pre-trained matrix at document boundaries. At inference on Qwen3-4B each update's norm is clipped to 1e-5.
- **What is learned versus fixed.** Outer loop (ordinary backprop) trains the convolution, the target projection, and the rest of the model; the form of the write rule is hand-designed, what gets written is learned. The convolution starts at zero so a pre-trained model starts unchanged. Inferred, not stated: because the write is a formula of forward activations, training needs no gradients of gradients. Whether the inner learning rate is learned is not stated.
- **Does the model write its own weights?** Partly: learned projections inside the network produce the values added to another matrix of the same network; the rule is fixed.
- **Results.** Qwen3-4B-Base after identical continued training with and without the method, RULER (Table 1): 64k 74.3 → 78.7, 128k 74.8 → 77.0, 256k 41.7 → 43.9. 4B trained from scratch at 8k context (Table 3): RULER-16k 6.58 → 19.99 with full attention; RULER-8k 9.91 → 26.80 with sliding window.
- **Limits.** No limitations section. Noticed: drop-in gains are about 2–4 points; training data is private; state is reset per document; nothing on forgetting or cross-session learning despite the "continual learning" framing; purely additive writes with no decay, kept stable by clipping.
- **Read status.** 79,081 characters; main text and appendix sections 7–10 read in full; reference list skipped.

---

### 3. TTCD

- **Title, authors, year, arXiv ID or venue.** "Learning What to Remember: Test-Time Training via Context Distillation", Wang, Dang, Zhu, Wen, Fu, Chai, Lee. arXiv 2608.01672v1, Aug 2026.
- **The problem.** A fixed-size memory must choose what to keep. Reconstruction targets store everything evenly; next-token targets mostly reward what the local context already explains.
- **The idea.** Run the same layer twice on the same position: a "teacher" whose attention sees a long window and a "student" that sees half of it. Where they disagree, the far context mattered. Write that disagreement into the weights. Example: at "Alice lives in ...", the teacher still sees "Paris" 1,500 tokens back, the student does not. The teacher-minus-student difference is stored under the student's activation as the key. Later, when "Paris" is out of both windows, a similar activation retrieves it.
- **What the memory is and how it is written.** Same slot as In-Place TTT: the MLP's last matrix `W_down`, in 4 layers. Inner loss: squared distance between student MLP output and teacher MLP output (teacher treated as a constant). One gradient step gives a closed formula: add `(teacher − student difference) × student activationᵀ` per chunk (512–2048 tokens). In practice both sides are computed with the unmodified matrix ("off-policy") so all chunks run in parallel, and short convolutions, a learned projection and key normalization are added. No backward pass at run time. Fast weights are "initialized at the beginning of each sequence", so reset per sequence.
- **What is learned versus fixed.** Outer loop trains the backbone, the convolutions and the projection end to end with next-token loss; the write rule's form and its rate (0.3) are fixed. Inferred: first-order training only, since the write is a formula.
- **Does the model write its own weights?** Partly: the network's own two views produce the change to its own MLP matrix; no optimizer run, no separate model.
- **Results.** 760M from scratch, 32K context (Table 1): RULER needle average at 32K 21.96 vs 9.29 for In-Place TTT; MAD overall 82.1 vs 78.4. LLaMA-3.1-8B continued training, RULER average (Table 2): 73.75 vs 73.39 (In-Place TTT) vs 72.85 (plain).
- **Limits.** Authors: training 1.1–1.2× slower; from-scratch runs stop at 760M. Noticed: gains on pre-trained models are under one point; continued training lowered Qwen3-1.7B below its starting score (80.35 → 78.09); pass-key at 32K is 1.00 vs 100 for DeltaNet (Table 4); the teacher only sees its own window, so the signal covers evidence between the two window sizes; no cross-session or forgetting test.
- **Read status.** 76,866 characters; main text and appendices A–C read in full; reference list skipped.

---

### 4. Nested Learning / Hope

- **Title, authors, year, arXiv ID or venue.** "Nested Learning: The Illusion of Deep Learning Architecture", Behrouz, Razaviyayn, Zhong, Mirrokni (Google). arXiv 2512.24695v1, Dec 2025; a version at NeurIPS 2025.
- **The problem.** A Transformer has two speeds: attention adapts instantly and forgets when the context ends; MLP weights never change after pre-training. Nothing sits in between.
- **The idea.** View a model plus its optimizer as a stack of small learners, each compressing its own stream (tokens or gradients) at its own update rate. Then add rates. Hope = a "self-modifying Titans" block followed by a Continuum Memory System (CMS).
- **What the memory is and how it is written.** (a) Self-modifying block: several small 2-layer MLPs. One is the memory; the others produce the key, the value, the step size and the forget rate for each token. "Self-modifying" means two things. Each MLP makes its own training target by feeding the value through itself. And the MLPs that produce the keys, step sizes and forget rates are updated by the same rule they parameterize. With a 2-number key `k` and a target `v`, each MLP takes one step to make `M(k)` closer to `v` (squared error), with a decay term that first erases what was stored along `k`. This is a gradient of a local loss with respect to the small MLP, computed per chunk; closed form only if the memory is a single matrix. (b) CMS: a chain of MLP blocks; block ℓ is updated once every C_ℓ tokens with the accumulated gradient of the task loss, so some blocks change often and others rarely. One variant re-initializes blocks at the end of their context; whether state persists across sequences in the experiments is not stated.
- **What is learned versus fixed.** Outer loop meta-learns the starting state of every memory "across all sequences"; the rule (delta gradient descent with weight decay) is hand-designed, but its inputs, step size and decay are produced by networks that the rule itself changes. Gradients of gradients are not discussed; inferred that training through MLP-memory steps needs them, as in Titans.
- **Does the model write its own weights?** Yes, in the sense that networks produce the targets and step sizes for updates to the same networks; the arithmetic of the update is still a fixed gradient formula.
- **Results.** 1.3B parameters, 100B tokens (Table 2): Wikitext perplexity 14.39 vs 15.60 (Titans) vs 17.92 (Transformer++); average accuracy 58.04 vs 56.82 vs 53.38. Needle S-NIAH-3 at 16K (Table 1): Hope 24.8, Titans 21.2, Transformer 40.8. Ablation (Table 6): removing CMS raises perplexity 12.24 → 13.04.
- **Limits.** Authors: "catastrophic forgetting is not 'solved' in general"; the M3 optimizer is slower than Muon. Noticed: continual-learning results (class-incremental, two-languages-in-sequence, BABILong to 10M) are figures without numbers; those runs use Hope-Attention on a Llama backbone, i.e. CMS without the self-modifying block; no cost numbers; no code; table references and the role of momentum in Hope are inconsistent in the text.
- **Read status.** 248,907 characters; sections 1–10 and appendices A–C read in full; reference list skipped.

---

### 5. Fast-Slow Training (FST)

- **Title, authors, year, arXiv ID or venue.** "Learning, Fast and Slow: Towards LLMs That Adapt Continually", Tiwari, Sareen, Agrawal, Gonzalez, Zaharia, Keutzer, Dhillon, Agarwal, Khatri. arXiv 2605.12484v3 (first version May 2026).
- **The problem.** Reinforcement-learning fine-tuning writes every lesson into the weights. The model drifts from its base behavior and becomes harder to train on the next task.
- **The idea.** Split the lessons. The "fast" component is text: a set of 4–8 system prompts. It is not weights, not an adapter, not activations. The "slow" component is the model's weights. Every 6 RL steps a prompt optimizer (GEPA) looks at failed attempts, asks a separate frozen model (gpt-5.2) to edit the prompts, and keeps the best set. Then RL updates the weights with those prompts in place. Example: "do not invent placeholder numbers" goes into the prompt; the weights do not have to learn it.
- **What the memory is and how it is written.** Prompts, rewritten by an outside language model through evolutionary search on reward; weights, by policy gradient (GRPO with the CISPO loss). No gradient ever flows into or through the prompt update. Nothing changes at deployment: both loops run during training, and the final prompt is prepended at inference. Prompts and weights carry over across cycles and across tasks with no reset.
- **What is learned versus fixed.** Weights: backprop on the RL loss. Prompts: black-box search. No gradients of gradients. The reflection model is frozen.
- **Does the model write its own weights?** No. An outside model edits the prompt; an optimizer edits the weights.
- **Results.** Qwen3-8B: reaches RL's peak in 3.0× (CodeIO), 1.4× (Math), 3.0× (HoVer-hard) fewer steps (Fig. 2); fitted ceilings 47.4% vs 43.0% (CodeIO) and 25.0% vs 17.3% (HoVer-hard) (Fig. 3); "up to 70% lower KL to the base policy". Task stream HoVer → CodeIO → Physics, 200 steps each (Fig. 6): on CodeIO plain RL peaks at 21.1%, FST ends at 39.8%.
- **Forgetting and cross-task learning (Appendix F).** Every method keeps the first task at 44–46%, "so forgetting does not separate the methods". Replay (19.1%) and a KL penalty (20.2%) do not fix RL's stall on the second task; one prompt optimization per task boundary gets 34.2%, FST 40.6%. After Math training, RL-initialized weights learn HoVer-hard at about 0%; FST-initialized weights stay near the base model (Fig. 5).
- **Limits.** Authors: only one optimizer pair tested. Noticed: FST weights without their prompt score 25.1% on CodeIO, below RL's 38.1% (Fig. 18), so the weights depend on the prompt; GEPA adds 18–48% rollouts and up to 2.0× wall-clock per step; needs an external frontier model; no benefit on Math where the base model follows instructions poorly; this is training-time continual learning, not per-user or per-session learning.
- **Read status.** 116,820 characters; main text and appendices A–G read in full; reference list skipped. Title and authors taken from the arXiv abstract page because the HTML header was empty.

---

### Side by side: inner loop versus outer loop

| | Inner loss | What changes | Formula or backward pass | Outer loop | Reset |
|---|---|---|---|---|---|
| TTT-E2E | next-token cross-entropy | MLPs in last 1/4 of blocks | real backward pass, 1 step per 1K tokens | gradients of gradients | per document |
| In-Place TTT | negative dot product with a learned next-token target | MLP `W_down`, every 6th layer | closed formula (outer-product sum) | first-order (inferred) | per document |
| TTCD | squared gap between long-window and short-window MLP output | MLP `W_down`, 4 layers | closed formula | first-order (inferred) | per sequence |
| Hope | squared error to self-generated targets; CMS uses task loss | small memory MLPs; CMS MLP blocks at several rates | gradient of a local loss per chunk | meta-learned initial states; second-order not discussed | not stated |
| FST | task reward | text prompts (fast), all weights (slow) | no gradient for prompts; policy gradient for weights | none through the prompt | never |

None of the four weight-update papers tests learning across separate sessions. Forgetting is addressed by a frozen second MLP (TTT-E2E), by multi-rate blocks with figure-only evidence (Hope), and not at all in In-Place TTT and TTCD. FST is the only one with a multi-task stream, and there forgetting did not differ between methods; the measured difference was the ability to learn the next task.


## 7. Editing weights from outside

All four papers were read end to end from HTML text (main text, references, all appendices). Figures are images and were not readable.

| Paper | Who decides the edit | Form of the edit | How the editor is trained |
|---|---|---|---|
| MEND | Trained editor MLPs, fed the fine-tuning gradient | Sum of rank-1 updates to a few MLP weight matrices | Backprop through the edited model's loss |
| Text-to-LoRA | Hypernetwork, fed a task description | A full LoRA adapter (rank 8) | Backprop through the adapted model's loss, or regression onto existing LoRAs |
| SEAL | The model itself, by generating text | Fine-tuning data plus (for ARC) hyperparameters, applied by ordinary LoRA fine-tuning | Reward from the edited model's score; no gradient through the edit |
| Tracr | A hand-written compiler | Exact weights for the whole model | Not trained |

---

### 1. MEND

- **Title, authors, year, arXiv ID or venue.** "Fast Model Editing at Scale", Mitchell, Lin, Bosselut, Finn, Manning. arXiv 2110.11309, 2021; ICLR 2022.
- **The problem.** A model says "Alice lives in Paris" but Alice moved to Rome. Fine-tuning on that one correction overfits: it fails on rephrasings or damages unrelated answers. Earlier learned editors needed gradients of gradients (ENN) or failed on multi-billion-parameter models (KE).
- **The idea.** Compute the ordinary fine-tuning gradient for the correction ("Where does Alice live?" -> "Rome"), then pass it through a small trained network that turns it into a better update. For one token, a weight matrix's gradient is the product of two short vectors: the layer's input and the error signal at its output. So the editor reads two vectors of length d, not a d-by-d matrix, and outputs two new vectors whose product is the edit. Summary: W_new = W - step * (new error vector)(new input vector)^T, summed over tokens.
- **What the memory is and how it is written.** The base model's own weights: MLP matrices in the last 3 blocks (GPT, BERT) or last 2 encoder and decoder blocks (T5, BART). One edit is one step, applied from outside. It is never reset. Multiple edits are summed.
- **What is learned versus fixed.** The base model is frozen. The editor networks (one hidden layer, shared across layers of the same shape, plus a per-layer step size) are trained by ordinary backprop. The loss is measured on the edited model: answer a rephrasing correctly, and keep outputs unchanged on an unrelated question. The update rule is learned, initialized to the plain gradient. No gradients of gradients (stated).
- **Does the model write its own weights?** No. A separate trained network edits it, using the model's gradient as input.
- **Results.** Table 3, T5-XXL (11B), zsRE question answering: edit success 0.89 MEND, 0.87 fine-tuning, 0.04 KE. GPT-J (6B), Wikitext: MEND 0.88 with 0.031 perplexity increase; fine-tuning 0.80 with 0.125. Table 5, BART-base, 125 edits at once: MEND 0.67, ENN 0.11.
- **Limits.** Authors: edits over-generalize (a "president of Mexico" edit also changes the "vice president" answer); generality is tested on rephrasings only, not implied facts. Noticed: one editor per base model; the paper studies one edit or one batch, not long sequences of edits; the text claims 96% success at 25 simultaneous edits but Table 5 shows 0.89.
- **Read status.** 85,725 characters, through Appendix G.

---

### 2. Text-to-LoRA (T2L)

- **Title, authors, year, arXiv ID or venue.** "Text-to-LoRA: Instant Transformer Adaption", Charakorn, Cetin, Tang, Lange (Sakana AI). arXiv 2506.06105, 2025. The HTML carries an ICML keyword tag.
- **The problem.** A LoRA adapter (a small low-rank add-on to a model's weight matrices) must be trained per task, with its own data and hyperparameters. Earlier hypernetworks used learned task IDs, so they could not handle unseen tasks.
- **The idea.** Train one network that reads a sentence describing a task and outputs the adapter. Example: type "solve grade-school math word problems step by step"; one forward pass returns adapter matrices for every attention layer of Mistral-7B. No task data or fine-tuning at use time.
- **What the memory is and how it is written.** An adapter: two rank-8 matrices added to the query and value projections of every attention block, 3.4M numbers. Base weights never change. The adapter is written once per description, before the model runs, and stays fixed while it runs. Nothing accumulates across tasks.
- **What is learned versus fixed.** Fixed: the base model and the text embedder (frozen is my inference from the parameter counts). Learned by ordinary backprop: the hypernetwork (5M to 55M parameters), fed the description embedding plus learned layer and module embeddings. Two training routes: (a) reconstruction, regressing onto already-trained LoRAs; (b) supervised fine-tuning, where the adapted model's task loss is backpropagated through the frozen base model into the hypernetwork, on 479 tasks. The text-to-weights mapping is learned. No gradients of gradients (my reading: the adapter comes from a forward pass, not an inner optimization).
- **Does the model write its own weights?** No. One network outputs changes to another network's weights.
- **Results.** Table 2, Mistral-7B-Instruct, zero-shot on 10 unseen benchmarks: T2L 67.7; one multi-task LoRA 66.3; base model 55.8. On the 8 benchmarks with per-task LoRAs: 73.9 vs 75.8. Table 6: reconstruction-trained T2L gets 61.8 zero-shot vs 66.3 for route (b).
- **Limits.** Authors: it does not reach task-specific LoRAs; it depends on description quality, and training descriptions came from GPT-4o mini. Noticed: reconstruction training reproduces known adapters (Table 1: 73.3 on 9 tasks, equal to the originals) but does not generalize, because adapters for similar tasks are not close in weight space (Appendix D); wrong-task descriptions drop that model to 51.4 (Table 5); the benchmarks resemble the training tasks (footnote 1).
- **Read status.** 80,047 characters, through Appendix N. Figures 16 to 23 (prompt template, example descriptions) are images.

---

### 3. SEAL

- **Title, authors, year, arXiv ID or venue.** "Self-Adapting Language Models", Zweiger, Pari, Guo, Akyürek, Kim, Agrawal (MIT). arXiv 2506.10943, 2025.
- **The problem.** Fine-tuning on raw new text teaches little: training Qwen2.5-7B on a passage alone moves no-context question answering from 32.7% to 33.5%.
- **The idea.** The model writes its own study notes, then trains on them. It reads "Alice moved from Paris to Rome in 2024" and generates a "self-edit": lines such as "Alice lives in Rome", "Alice no longer lives in Paris". A standard fine-tuning step trains on those lines. Asked later, with no passage, where Alice lives, it should say Rome. For ARC puzzles the self-edit is instead a JSON block choosing data augmentations, learning rate, epochs and loss type.
- **What the memory is and how it is written.** A LoRA adapter on the same model (all weights in the many-passage runs), written by gradient descent on the generated text. In single-passage tests each passage gets a separate adapter. Updates are meant to persist.
- **What is learned versus fixed.** The run-time update rule is fixed: plain supervised fine-tuning. What is learned is how to write useful self-edits. Inner loop: apply a self-edit, test the updated model on held-out questions. Outer loop: sample several self-edits per passage, keep the best-scoring, fine-tune the model to produce those (ReST-EM, "rejection sampling plus fine-tuning"). The reward is the edited model's score, treated as a constant. No gradient passes through the inner update, so no gradients of gradients. PPO and GRPO were unstable.
- **Does the model write its own weights?** Partly. The model writes the training data and settings; a standard optimizer changes the weights. The same network produces and receives the edit.
- **Results.** Table 2, Qwen2.5-7B, SQuAD without the passage: SEAL 47.0%; untrained model's own notes 39.7%; GPT-4.1's notes 46.3%. With 200 passages: 58.2 vs 59.4 for GPT-4.1. Table 1, Llama-3.2-1B, 8 selected ARC tasks: 72.5% vs 20% without RL.
- **Limits.** Repeated self-edits (Section 5, Figure 6): passages arrived in a stream, each triggering a self-edit, with re-testing on all earlier passages (8 in the error table). Performance on earlier passages "gradually declines as the number of edits increases"; SEAL "is still susceptible to catastrophic forgetting" but updates "without complete collapse". Training does not optimize retention. The accuracies are only in the figure. Suggested, untested fixes: penalize regressions in the reward, null-space constrained edits. Also: scoring one self-edit takes 30 to 45 seconds; every passage needs paired questions. Noticed: a hypernetwork baseline scores 66.8 on single passages against 47.0, but 28.0 against 58.2 with 200 passages (Table 7).
- **Read status.** 82,605 characters, through Appendix B.11.

---

### 4. Tracr

- **Title, authors, year, arXiv ID or venue.** "Tracr: Compiled Transformers as a Laboratory for Interpretability", Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik (Google DeepMind). arXiv 2301.05062, 2023 (from memory: NeurIPS 2023).
- **The problem.** Interpretability tools explain what a trained transformer computes, but the true answer is unknown, so explanations cannot be checked.
- **The idea.** Write a short program and have a compiler produce transformer weights that run exactly that program. Programs are in RASP, a language with two operations matching the two transformer parts: per-position functions (MLP layers) and "select, then average" across positions (attention heads). Example, three lines: mark each position holding the letter x; let each position look at itself and everything before it; average the marks. On "x a c x" the compiled model outputs 1, 1/2, 1/3, 1/2, the running fraction of x's.
- **What the memory is and how it is written.** Nothing changes while the model runs. All weights are written once, by the compiler, before any input. Each program variable gets private dimensions in the residual stream (the vector every layer reads and writes). Per-position functions become lookup-table MLPs; selections become attention matrices scaled so attention is nearly all-or-nothing.
- **What is learned versus fixed.** Everything is fixed and hand-designed; no training, no data. One side experiment learns something: a single projection matrix is trained by gradient descent to squeeze the residual stream into fewer dimensions, with compiled weights frozen. No gradients of gradients.
- **Does the model write its own weights?** No. A compiler written by people computes exact weights from source code.
- **Results.** The compiler handled every task from the original RASP paper, including sorting and balanced-parenthesis checking. The x-fraction model has a 14-dimensional residual stream and 4 layers, 2 doing nothing (Section 4.1). Compression takes it from 14 to 6 dimensions without hurting loss (Section 5.2). A compressed sorting model keeps near-perfect accuracy, but its per-layer outputs have cosine similarity about 0.8 with the original because it re-encoded one variable (Section 5.3).
- **Limits.** Authors: outputs are discrete, not probabilities; attention is all-or-nothing; no layer norm; every variable reserves dimensions in all layers, so models are wide and sparse; full language models will not be compiled; passing on a compiled model is a minimum bar for a tool. Noticed: numerical MLPs are approximations; and/or of selectors is unsupported (Appendix F proves "or" generally cannot be built this way). Nothing here concerns adaptation or forgetting.
- **Read status.** 73,755 characters, through Appendix G. Appendix C describes Figure 2 in words.
