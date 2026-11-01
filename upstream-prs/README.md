# Our PRs to miles and SGLang

We train vision models with RL on long computer-use episodes. miles ([radixark/miles](https://github.com/radixark/miles)) trains, SGLang ([sgl-project/sglang](https://github.com/sgl-project/sglang)) generates the episodes. These are the fixes we carry and send upstream.

**State on 2026-11-01.** 16 PRs are open upstream: 11 on miles, 4 on SGLang, 1 on the router. Each PR's description has a "Verified on GPU" section with the numbers. GPU runs: Qwen3.8-VL on 4 nodes and GLM-5.3 on 6 nodes of 8x B300, computer-use episodes with screenshots through the TITO session server. Our branches are on the forks [dzorlu/miles](https://github.com/dzorlu/miles), [dzorlu/sglang](https://github.com/dzorlu/sglang) and [dzorlu/sgl-router-for-miles](https://github.com/dzorlu/sgl-router-for-miles).

Table 1. Open PRs by group (the groups are sections 1 to 7 below): what each fixes, its GPU result and whether it merges cleanly today.

| Group | PR | Fixes | GPU result | Merges cleanly |
|---|---|---|---|---|
| 1. TITO | [miles#4586](https://github.com/radixark/miles/pull/4586) | screenshots through the session server on Rockdu's token space, plus R3 offsets on expanded tokens (section 1) | PASS (Qwen3.8-VL, 3 steps; GLM-5.3, 3 steps, R3 aligned, reference KL 0 at step 0) | yes, onto Rockdu's branch; merges after his #4272 |
| 1. TITO | [sgl-router-for-miles#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | router dropped `mm_token_expansion_start_len` on chat | PASS (every token-space run) | yes |
| 1. TITO | [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | first-turn images dropped with token IDs | PASS (0 prompt-ID mismatches) | no: merge current `sglang-miles` |
| 1. TITO | [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | GLM-5.3 token-space strategy | PASS (GLM-5.3, 3 steps); CPU parity bitwise | no: merge current `sglang-miles` |
| 1. TITO | [miles#4516](https://github.com/radixark/miles/pull/4516) | Qwen reasoning and tool-call parsers | PASS (100% of 25,461 turns parsed) | yes |
| 1. TITO | [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | GLM tool-call grammar forced one call per turn | PASS on the non-token-space path (46 calls per turn); not exercised on token space | yes |
| 2. Vision training | [miles#4050](https://github.com/radixark/miles/pull/4050) | loss divided by samples, not episodes, with screenshot folding | PASS (Megatron and torchtitan) | yes |
| 2. Vision training | [miles#4049](https://github.com/radixark/miles/pull/4049) | frozen vision tower not re-sent to engines | PASS on sending (333 tensors per sync); engine-side equality not checked | yes |
| 3. Expert replay (R3) on GLM | [miles#4520](https://github.com/radixark/miles/pull/4520) | R3 on torchtitan: guard crash; reference picked its own experts | PASS (step-0 reference KL exactly 0, both paths) | no: port to torchtitan v0.4.0 |
| 4. GLM-5.3 on torchtitan | [miles#4588](https://github.com/radixark/miles/pull/4588) | GLM-5.3 vision tower in torchtitan training | PASS (both GLM paths) | no: port to torchtitan v0.4.0 |
| 4. GLM-5.3 on torchtitan | [miles#4522](https://github.com/radixark/miles/pull/4522) | torchtitan resume failed on never-trained parameters | PASS (GLM resume continued at rollout 3, saved) | no: port to torchtitan v0.4.0 |
| 5. Long colocated runs | [miles#4519](https://github.com/radixark/miles/pull/4519) | offload flush refused with running requests | PASS (sync and fully-async colocated) | yes |
| 5. Long colocated runs | [miles#4521](https://github.com/radixark/miles/pull/4521) | host memory grew every step | PASS per process; per-node totals not logged | yes |
| 6. Native LoRA | [miles#4523](https://github.com/radixark/miles/pull/4523) | native LoRA resume silently restarted | PASS (optimizer, adapter, scheduler bitwise) | no: upstream #4530 merged the scheduler part; drop it |
| 6. Native LoRA | [miles#4524](https://github.com/radixark/miles/pull/4524) | Muse import needed a GPU | PASS; **ready for review** | yes |
| 7. Cancelled episodes | [miles#4518](https://github.com/radixark/miles/pull/4518) | eval crashed on groups with a cancelled episode | not exercised (case never occurred); CPU tests | yes |

Closed: #4546 and #4547 (replaced by #4586). To close: [sglang#47400](https://github.com/sgl-project/sglang/pull/47400) (replaced by the token-space path).

**What is left.**
1. Fix the six merge conflicts: merge the base into each branch (no force-push), resolve, rerun the PR's CPU tests. The three torchtitan PRs need a port to torchtitan v0.4.0 ([#4537](https://github.com/radixark/miles/pull/4537) to [#4539](https://github.com/radixark/miles/pull/4539), merged 2026-10-31); their GPU results predate it, so rerun a GLM step after the port.
2. Mark the passing PRs ready for review.
3. Coordinate with overlapping upstream PRs (not posted): [#1585](https://github.com/radixark/miles/pull/1585) duplicates #4519; [#4218](https://github.com/radixark/miles/pull/4218) touches #4520's replay guard; [#3879](https://github.com/radixark/miles/pull/3879) would remove #4522's indexer case.
4. Follow-ups: an image-aware threshold for miles' R3 replay check (image tokens disagree about 15 times more than text on both image paths); two upstream bugs to report: native LoRA with `--colocate --lora-train-only` crashes in the first weight update on main, and the B300 dev image cannot train head-dim-256 attention (pin flash-attn-4 4.0.0b19 and quack-kernels 0.6.4).

How to build and run these checks without our launcher: [../miles-harness/](../miles-harness/README.md).

## 1. Exact tokens for vision models (TITO)

TITO (token-in, token-out) means the trainer scores exactly the token IDs and images the engine used, on every turn of an agent episode. With the PRs below, Qwen3.8-VL and GLM-5.3 each trained 3 steps on computer-use episodes with screenshots through the session server: TITO mismatch rate 0.0, 0 prompt-ID mismatches, and on GLM the routed-expert replay stays aligned with images. Figure 1 shows where each PR acts on one chat turn.

![Figure 1](tito-prs-turn-path.svg)

*Figure 1. One chat turn through the session server and the PR that acts at each step. Grey boxes are Rockdu's token-space stack ([#4240](https://github.com/radixark/miles/pull/4240), [#4272](https://github.com/radixark/miles/pull/4272)), which these PRs build on.*

*Table 2. The TITO PRs. GPU runs: Qwen3.8-VL (Megatron, 4 nodes) and GLM-5.3 (torchtitan, 6 nodes) of 8x B300, episodes up to 256 turns with screenshot folding.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [miles#4586](https://github.com/radixark/miles/pull/4586), commit 1 | the session server carried no screenshots on the token-space design | send the stored expanded prefix, the new part with one placeholder per image, and `mm_token_expansion_start_len`; store the engine's expanded IDs | Qwen and GLM 3 steps each; TITO mismatch 0.0 |
| [miles#4586](https://github.com/radixark/miles/pull/4586), commit 2 | R3 row offsets counted one placeholder per image, so rows fell out of line (HTTP 422) | count offsets on the expanded tokens | GLM image-token replay rate equal to the non-token-space path (0.659% vs 0.638%) |
| [sgl-router-for-miles#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | the router dropped `mm_token_expansion_start_len` on chat; turn 2 of every session failed | forward the field | every token-space run |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | a chat request with `input_ids` and start 0 dropped its images | keep them | 0 prompt-ID mismatches over 124 Qwen samples |
| [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | no GLM-5.3 strategy for the token-space processor | add one, bitwise equal to SGLang's GLM processor | GLM 3 steps and a resume |
| [miles#4516](https://github.com/radixark/miles/pull/4516) | miles never set SGLang's parsers; for Qwen whole replies were stored as text | pass each model family's reasoning and tool-call parsers | 100% of 25,461 Qwen turns parsed; 0 leaked `</think>` |
| [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | SGLang's GLM parser forced one declared tool call per turn | `--tool-call-constraint none` | up to 46 tool calls per turn (non-token-space GLM path) |

Merge order: Rockdu's #4240 and #4272, then #4586 (based on his branch). #47402 and #47674 go into `sglang-miles`, #47203 into SGLang main, #28 into the router's main, #4516 into miles main. Rockdu confirmed (2026-11-01) that his unpublished [4/4] covers basic full-modality support and essential TITO capability and does not overlap much with #4586. Replaced and closed: #4546 and #4547; [sglang#47400](https://github.com/sgl-project/sglang/pull/47400) is to be closed. Design and worked examples: [tito-token-space.md](tito-token-space.md).

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
