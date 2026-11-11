# Our PRs to miles and SGLang

We train vision models with RL on long computer-use episodes. miles ([radixark/miles](https://github.com/radixark/miles)) trains, SGLang ([sgl-project/sglang](https://github.com/sgl-project/sglang)) generates the episodes. These are the fixes we carry and send upstream.

State on 2026-11-11. Our branches are on the forks [dzorlu/miles](https://github.com/dzorlu/miles), [dzorlu/sglang](https://github.com/dzorlu/sglang) and [dzorlu/sgl-router-for-miles](https://github.com/dzorlu/sgl-router-for-miles), one worktree per PR under `~/repos/`. Tests run in the CI image on Modal with [../miles-harness/modal/miles_tests.py](../miles-harness/modal/miles_tests.py). Every PR is from a fork, so its CI needs a maintainer's `run-ci` label.

Table 1. The PRs by group (the groups are sections 1 to 7 below). "Ready" means current with its base, tests passing in the CI image, marked ready for review.

| Group | PR | Fixes | Tests | State |
|---|---|---|---|---|
| 1. TITO | [miles#4770](https://github.com/radixark/miles/pull/4770) | screenshots through the session server on token space; R3 rows on the engine's expanded prompt | 37 passed (Qwen and Inkling processors, stub engine) | ready |
| 1. TITO | [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | a chat request with `input_ids` dropped its images on turn 1 | 186 passed | ready |
| 1. TITO | [router#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | the router dropped `mm_token_expansion_start_len` on chat | `cargo test` chat spec | ready |
| 1. TITO | [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | GLM-5.3 token-space strategy | 36 passed, bitwise parity with SGLang's GLM processor | ready |
| 1. TITO | [miles#4516](https://github.com/radixark/miles/pull/4516) | reasoning and tool-call parsers per model family | 827 passed | ready |
| 1. TITO | [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | GLM makes many tool calls per turn | CI | ready |
| 2. Vision training | [miles#4050](https://github.com/radixark/miles/pull/4050) | loss divided by samples, not episodes, with screenshot folding | | on hold: main rewrote the optimizer step |
| 2. Vision training | [miles#4049](https://github.com/radixark/miles/pull/4049) | frozen vision tower not re-sent each weight sync | | on hold: main covers Inkling and Qwen3.5 by provider |
| 3. R3 on GLM | [miles#4520](https://github.com/radixark/miles/pull/4520) | the reference model replays the rollout's experts; the guard knows every stage | 27 passed (1 GPU) | ready |
| 4. GLM on torchtitan | [miles#4588](https://github.com/radixark/miles/pull/4588) | vision tower in torchtitan training; the token-space flag accepts torchtitan | 97 passed (1 GPU) | ready |
| 4. GLM on torchtitan | [miles#4522](https://github.com/radixark/miles/pull/4522) | resume when a save lacks some optimizer state | | on hold: only v0.3.0 saves need it |
| 5. Colocated runs | [miles#4519](https://github.com/radixark/miles/pull/4519) | offload no longer hangs on in-flight requests | 92 passed | ready |
| 5. Colocated runs | [miles#4521](https://github.com/radixark/miles/pull/4521) | host memory no longer grows every step | 68 passed (1 GPU) | ready |
| 6. Native LoRA | [miles#4523](https://github.com/radixark/miles/pull/4523) | native LoRA resume | | on hold: superseded by main's adapter library |
| 6. Native LoRA | [miles#4524](https://github.com/radixark/miles/pull/4524) | Muse imports without a GPU | | waits for upstream #4111 |
| 7. Cancelled episodes | [miles#4518](https://github.com/radixark/miles/pull/4518) | eval no longer crashes on a cancelled episode | 82 passed | ready |

To close: [sglang#47400](https://github.com/sgl-project/sglang/pull/47400), replaced by #47402 and #4770.

## 1. Exact tokens for vision models (TITO)

With these PRs, Qwen3.8-VL and GLM-5.3 each trained 3 steps on computer-use episodes with screenshots through the session server: TITO mismatch rate 0.0, 0 prompt-ID mismatches, and on GLM the routed-expert replay stays aligned with images. GPU runs: Qwen3.8-VL on Megatron, 4 nodes, and GLM-5.3 on torchtitan, 6 nodes, of 8x B300; episodes up to 256 turns with screenshot folding. Inkling is covered by a CPU test in #4770. Which PR each model needs, the design and worked examples: [tito-token-space.md](tito-token-space.md). Figure 1 shows where each PR acts on one chat turn.

![Figure 1](tito-prs-turn-path.svg)

*Figure 1. One chat turn through the session server and the PR that acts at each step. Grey boxes are Rockdu's token-space design ([#4240](https://github.com/radixark/miles/pull/4240), [#4272](https://github.com/radixark/miles/pull/4272), on main).*

### 1.1 All models: the session server on token space

Any vision model with a token-space strategy in SGLang needs these three; they go in together.

*Table 2. TITO PRs every model needs.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [miles#4770](https://github.com/radixark/miles/pull/4770) | the session server carried no screenshots on the token-space design; R3 row offsets counted one placeholder per image, so rows fell out of line (HTTP 422) | send the stored expanded prefix, the new part with one placeholder per image, and `mm_token_expansion_start_len`; store the engine's expanded IDs; count R3 offsets on them | Qwen and GLM 3 steps each; TITO mismatch 0.0; GLM image-token replay rate equal to the legacy path |
| [sgl-router-for-miles#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | the router dropped `mm_token_expansion_start_len` on chat; turn 2 of every session failed | forward the field | every token-space run |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | a chat request with `input_ids` and start 0 dropped its images | keep them | 0 prompt-ID mismatches over 124 Qwen samples |

### 1.2 Qwen3.8-VL

The token-space processor already supports Qwen VL. Qwen adds one parser fix. #4516 sets the parsers for every model family; the bug only showed on Qwen, whose prompt already ends with `<think>`.

*Table 3. Qwen-specific TITO PR.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [miles#4516](https://github.com/radixark/miles/pull/4516) | miles never set SGLang's parsers; for Qwen whole replies were stored as text | pass each model family's reasoning and tool-call parsers | 100% of 25,461 Qwen turns parsed; 0 leaked `</think>` |

### 1.3 GLM-5.3

GLM needs a token-space strategy in SGLang, the vision tower in torchtitan training (#4588, section 4), R3 on torchtitan (#4520, section 3) and a parser setting.

*Table 4. GLM-specific TITO PRs.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | no GLM-5.3 strategy for the token-space processor | add one, bitwise equal to SGLang's GLM processor | GLM 3 steps and a resume |
| [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | SGLang's GLM parser forced one declared tool call per turn | `--tool-call-constraint none` | up to 46 tool calls per turn (legacy GLM path) |

Targets: #4770, #4516 and #4588 go into miles main; #47402 and #47674 into `sglang-miles`; #47203 into SGLang main; #28 into the router's main.

## 2. Vision training correctness

- **[miles#4050](https://github.com/radixark/miles/pull/4050).** Screenshot folding splits one episode into several training samples. The loss was divided by samples instead of episodes, so it came out 8.7 times too small on Qwen at step 1. On hold: main rewrote the torch-native optimizer step and already excludes multimodal batches from the precomputed DP schedule, so the change needs a re-port.
- **[miles#4049](https://github.com/radixark/miles/pull/4049).** With the vision tower frozen, miles did not re-send its weights to the engines after an update. `--mm-tower-sync` does this for any model. On hold: main re-sends the towers of Inkling and Qwen3.5 by provider; the flag would generalize that to GLM-5.3 and Muse.

## 3. Expert replay (R3) on GLM

Training must use the same experts the engine picked.

- **[miles#4520](https://github.com/radixark/miles/pull/4520).** The reference model picked its own experts, so step-0 KL was 0.0096 instead of 0; and the queue guard raised on a record pass. Megatron's reference pass has the same gap, not yet raised.

## 4. GLM-5.3 on torchtitan

- **[miles#4588](https://github.com/radixark/miles/pull/4588): vision tower in training** ([forward pass, with a diagram](glm-vision-titan.md)). Images go through the frozen vision tower; log-probs are computed on each CP rank's own tokens; `--multimodal-token-space-processing` accepts the torchtitan backend. A GLM run with screenshots on current main is owed.
- **[miles#4522](https://github.com/radixark/miles/pull/4522).** Parameters that never train (GLM's attention indexer) had no saved optimizer state, so a resume failed. On torchtitan v0.4.0 every save holds full optimizer state, so this only matters for resuming a v0.3.0 save.

## 5. Long colocated runs

Training and generation share the same GPUs.

- **[miles#4519](https://github.com/radixark/miles/pull/4519).** Freeing engine memory crashed while a request was still running. All requests are now cancelled first; under `--fully-async --colocate` the driver's paused requests are left to resume.
- **[miles#4521](https://github.com/radixark/miles/pull/4521).** Logs each process's GPU and host memory and returns host memory after each step. Without it, host memory grew about 115 GiB per node per step.

## 6. Native LoRA

- **[miles#4523](https://github.com/radixark/miles/pull/4523).** A native-LoRA resume dropped the saved Adam state and stepped the learning-rate schedule twice. Main's adapter library (#4608 to #4624) replaced the save format, so this is superseded.
- **[miles#4524](https://github.com/radixark/miles/pull/4524).** Importing miles failed on a machine without a GPU once Muse support was in. Targets upstream's `muse-spark-open` branch ([#4111](https://github.com/radixark/miles/pull/4111), draft); rebases once that lands.

Muse's merged-weight LoRA sync is not raised by us: upstream's own [#4211](https://github.com/radixark/miles/pull/4211) does the same.

## 7. Cancelled episodes

- **[miles#4518](https://github.com/radixark/miles/pull/4518).** Scoring crashed on a group with a cancelled episode. It now skips such groups.
