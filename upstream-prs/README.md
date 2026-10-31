# Our PRs to miles and SGLang

We train vision models with RL on long computer-use episodes. miles ([radixark/miles](https://github.com/radixark/miles)) trains, SGLang ([sgl-project/sglang](https://github.com/sgl-project/sglang)) generates the episodes. These are the fixes we carry and send upstream. To continue this work elsewhere, start with [HANDOFF.md](HANDOFF.md): every branch, the open decisions, how to test and verify each PR.

**State on 2026-10-31.** 17 PRs are open upstream: 12 on miles, 4 on SGLang, 1 on the router. Each PR description now has a "Verified on GPU" section. Passed: #4523, #4519, #4524, sglang#47402, #4050 on Megatron, and the TITO token-space rebuild on Qwen; GLM's token-space train step and the rest are pending (verdicts in [HANDOFF.md](HANDOFF.md), Table 1). [#4524](https://github.com/radixark/miles/pull/4524) is ready for review; the rest stay drafts until they pass. The TITO screenshot PRs are being rebuilt on Rockdu's token-space design; see [tito-token-space.md](tito-token-space.md). GPU verification on Qwen3.8-VL and GLM-5.3 is in progress. 

Table 1. Open PRs, by group.

| PR | Group | Note |
|---|---|---|
| [sglang#47400](https://github.com/sgl-project/sglang/pull/47400) | 1. Exact tokens for vision models | replaces #44133; to be closed in favor of the token-space path |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | 1. Exact tokens for vision models | replaces #44134; kept as the first-turn image fix of the token-space path |
| [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | 1. Exact tokens for vision models | GLM-5.3 token-space strategy, on `sglang-miles` |
| [sgl-router-for-miles#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | 1. Exact tokens for vision models | router forwards `mm_token_expansion_start_len` on chat |
| [miles#4588](https://github.com/radixark/miles/pull/4588) | 4. GLM-5.3 on torchtitan | vision tower in training |
| [miles#4586](https://github.com/radixark/miles/pull/4586) | 1. Exact tokens for vision models | token-space session server + R3 offsets, stacked on #4272; supersedes #4546 and #4547 |
| [miles#4546](https://github.com/radixark/miles/pull/4546) | 1. Exact tokens for vision models | closed 2026-10-31, superseded by #4586 |
| [miles#4516](https://github.com/radixark/miles/pull/4516) | 1. Exact tokens for vision models | Qwen parser |
| [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | 1. Exact tokens for vision models | GLM tool-call constraint |
| [miles#4050](https://github.com/radixark/miles/pull/4050) | 2. Vision training correctness | main merged in on 10-29 |
| [miles#4049](https://github.com/radixark/miles/pull/4049) | 2. Vision training correctness | main merged in on 10-29 |
| [miles#4520](https://github.com/radixark/miles/pull/4520) | 3. Expert replay (R3) on GLM | torchtitan fixes |
| [miles#4547](https://github.com/radixark/miles/pull/4547) | 3. Expert replay (R3) on GLM | closed 2026-10-31, superseded by #4586 |
| [miles#4522](https://github.com/radixark/miles/pull/4522) | 4. GLM-5.3 on torchtitan | resume |
| [miles#4519](https://github.com/radixark/miles/pull/4519) | 5. Long colocated runs | cancel before flush |
| [miles#4521](https://github.com/radixark/miles/pull/4521) | 5. Long colocated runs | memory report and host release |
| [miles#4523](https://github.com/radixark/miles/pull/4523) | 6. Native LoRA | resume |
| [miles#4524](https://github.com/radixark/miles/pull/4524) | 6. Native LoRA | Muse import without a GPU, into `muse-spark-open`; **ready for review** |
| [miles#4518](https://github.com/radixark/miles/pull/4518) | 7. Cancelled episodes | eval scoring |

## 1. Exact tokens for vision models (TITO)

The trainer must see the exact token IDs and images the engine used, on every turn of an episode. For screenshots we propose four changes on top of Rockdu's token-space design (open [#4240](https://github.com/radixark/miles/pull/4240) and [#4272](https://github.com/radixark/miles/pull/4272)), explained with examples and code references in [tito-token-space.md](tito-token-space.md):

- **Session server on token space** (miles, [#4586](https://github.com/radixark/miles/pull/4586), fork branch `tito-session-token-space`). Screenshots sent by an agent harness go through the session server: each turn sends the stored expanded prefix, the new part with one placeholder per image and `mm_token_expansion_start_len`; the session stores the engine's returned expanded token IDs, and the trainer uses Rockdu's rebuild.
- **R3 rows on the engine view** (miles, fork branch `tito-session-token-space`, same PR as the session-server change). Expert-routing rows are stitched per turn by counting tokens as the model ran them, so row k stays token k after a screenshot.
- **[sglang#47402](https://github.com/sgl-project/sglang/pull/47402): first-turn images.** A chat request with token IDs and offset 0 no longer drops its images.
- **[sglang#47674](https://github.com/sgl-project/sglang/pull/47674): GLM-5.3 strategy** (on `sglang-miles`). GLM-5.3 support for the token-space processor, bitwise equal to SGLang's existing GLM processor.

These supersede [#4546](https://github.com/radixark/miles/pull/4546), [#4547](https://github.com/radixark/miles/pull/4547) and [sglang#47400](https://github.com/sgl-project/sglang/pull/47400), which close once the GPU runs pass.

Two more TITO PRs, not about screenshots:

- **[miles#4516](https://github.com/radixark/miles/pull/4516).** miles never told SGLang which parsers to use. For Qwen, SGLang picked one that waits for `<think>`, but our prompt already ends with `<think>`, so whole replies were stored as plain text: no reasoning, no parsed tool calls.
- **[sglang#47203](https://github.com/sgl-project/sglang/pull/47203).** SGLang's GLM parser forced every reply into a grammar: only declared tool names, one call per turn. `--tool-call-constraint none` turns that off.

## 2. Vision training correctness

- **[miles#4050](https://github.com/radixark/miles/pull/4050).** Screenshot folding splits one episode into several training samples. The loss was divided by samples instead of episodes, so it came out 8.7 times too small on Qwen at step 1.
- **[miles#4049](https://github.com/radixark/miles/pull/4049).** With the vision tower frozen, miles did not re-send its weights to the engines after an update. `--mm-tower-sync` now does this for any model.

## 3. Expert replay (R3) on GLM

Training must use the same experts the engine picked.

- **[miles#4520](https://github.com/radixark/miles/pull/4520).** A safety check crashed on the first pass, and the reference model picked its own experts, so step-0 KL was 0.0096 instead of 0.
- **[miles#4547](https://github.com/radixark/miles/pull/4547)** (replaces #4517). Through the session server, R3 counted positions without expanding screenshots. It now counts the way the engine does, in the output store too. Builds on #4546.

## 4. GLM-5.3 on torchtitan

- **[miles#4588](https://github.com/radixark/miles/pull/4588): vision tower in training.** Port of our closed #4022 onto main after upstream merged GLM-5.3 text training on torchtitan (#3989, #3995). Images go through the frozen vision tower; log-probs are computed on each CP rank's own tokens.
- **[miles#4522](https://github.com/radixark/miles/pull/4522).** Parameters that never train (GLM's attention indexer) had no saved optimizer state, so a resume failed. They now start fresh. The other resume bug, a silent restart from the base model, no longer happens on main.

## 5. Long colocated runs

Training and generation share the same GPUs.

- **[miles#4519](https://github.com/radixark/miles/pull/4519).** Freeing engine memory crashed while a request was still running. All requests are now cancelled first.
- **[miles#4521](https://github.com/radixark/miles/pull/4521).** Logs each process's GPU memory and returns host memory after each step. Without it, host memory grew about 115 GiB per node per step.

## 6. Native LoRA

- **[miles#4523](https://github.com/radixark/miles/pull/4523).** A native-LoRA resume dropped the saved Adam state and stepped the learning-rate schedule twice. Found on Muse Spark; the fix applies to every native-LoRA model.
- **[miles#4524](https://github.com/radixark/miles/pull/4524).** Importing miles failed on a machine without a GPU once Muse support was in. Targets upstream's `muse-spark-open` branch.

Muse's merged-weight LoRA sync is not raised by us: upstream's own [#4211](https://github.com/radixark/miles/pull/4211) does the same.

## 7. Cancelled episodes

- **[miles#4518](https://github.com/radixark/miles/pull/4518).** Scoring crashed on a group with a cancelled episode. It now skips such groups.
