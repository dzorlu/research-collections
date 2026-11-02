# Our PRs to miles and SGLang

We train vision models with RL on long computer-use episodes. miles ([radixark/miles](https://github.com/radixark/miles)) trains, SGLang ([sgl-project/sglang](https://github.com/sgl-project/sglang)) generates the episodes. These are the fixes we carry and send upstream.

**State on 2026-11-01.** 16 PRs are open upstream: 11 on miles, 4 on SGLang, 1 on the router. Each PR's description has a "Verified on GPU" section with the numbers. GPU runs: Qwen3.8-VL on 4 nodes and GLM-5.3 on 6 nodes of 8x B300, computer-use episodes with screenshots through the TITO session server. Our branches are on the forks [dzorlu/miles](https://github.com/dzorlu/miles), [dzorlu/sglang](https://github.com/dzorlu/sglang) and [dzorlu/sgl-router-for-miles](https://github.com/dzorlu/sgl-router-for-miles).

Table 1. Open PRs by group (the groups are sections 1 to 7 below): what each fixes, its GPU result and whether it merges cleanly today.

| Group | PR | Fixes | GPU result | Merges cleanly |
|---|---|---|---|---|
| 1. TITO | [miles#4586](https://github.com/radixark/miles/pull/4586) | screenshots through the session server on Rockdu's token space, plus R3 offsets on expanded tokens (section 1) | PASS (Qwen3.8-VL, 3 steps; GLM-5.3, 3 steps, R3 aligned, reference KL 0 at step 0) | yes, onto Rockdu's branch; merges after his #4272 |
| 1. TITO | [sgl-router-for-miles#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | router dropped `mm_token_expansion_start_len` on chat | PASS (every token-space run) | yes |
| 1. TITO | [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | first-turn images dropped with token IDs | PASS (0 prompt-ID mismatches) | yes (updated 2026-11-01) |
| 1. TITO | [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | GLM-5.3 token-space strategy | PASS (GLM-5.3, 3 steps); CPU parity bitwise | yes (updated 2026-11-01) |
| 1. TITO | [miles#4516](https://github.com/radixark/miles/pull/4516) | Qwen reasoning and tool-call parsers | PASS (100% of 25,461 turns parsed) | yes |
| 1. TITO | [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | GLM tool-call grammar forced one call per turn | PASS on the non-token-space path (46 calls per turn); not exercised on token space | yes |
| 2. Vision training | [miles#4050](https://github.com/radixark/miles/pull/4050) | loss divided by samples, not episodes, with screenshot folding | PASS (Megatron and torchtitan) | yes |
| 2. Vision training | [miles#4049](https://github.com/radixark/miles/pull/4049) | frozen vision tower not re-sent to engines | PASS on sending (333 tensors per sync); engine-side equality not checked | yes |
| 3. Expert replay (R3) on GLM | [miles#4520](https://github.com/radixark/miles/pull/4520) | R3 on torchtitan: guard crash; reference picked its own experts | PASS (step-0 reference KL exactly 0, both paths) | yes (updated 2026-11-01) |
| 4. GLM-5.3 on torchtitan | [miles#4588](https://github.com/radixark/miles/pull/4588) | GLM-5.3 vision tower in torchtitan training | PASS (both GLM paths) | yes (updated 2026-11-01) |
| 4. GLM-5.3 on torchtitan | [miles#4522](https://github.com/radixark/miles/pull/4522) | torchtitan resume failed on never-trained parameters | PASS (GLM resume continued at rollout 3, saved) | yes (updated 2026-11-01) |
| 5. Long colocated runs | [miles#4519](https://github.com/radixark/miles/pull/4519) | offload flush refused with running requests | PASS (sync and fully-async colocated) | yes |
| 5. Long colocated runs | [miles#4521](https://github.com/radixark/miles/pull/4521) | host memory grew every step | PASS per process; per-node totals not logged | yes |
| 6. Native LoRA | [miles#4523](https://github.com/radixark/miles/pull/4523) | native LoRA resume silently restarted | PASS (optimizer, adapter, scheduler bitwise) | yes (updated 2026-11-01) |
| 6. Native LoRA | [miles#4524](https://github.com/radixark/miles/pull/4524) | Muse import needed a GPU | PASS; **ready for review** | yes |
| 7. Cancelled episodes | [miles#4518](https://github.com/radixark/miles/pull/4518) | eval crashed on groups with a cancelled episode | not exercised (case never occurred); CPU tests | yes |

Closed: #4546 and #4547 (replaced by #4586). To close: [sglang#47400](https://github.com/sgl-project/sglang/pull/47400) (replaced by the token-space path).

## Taking over: state on 2026-11-02

**All 16 open PRs merge cleanly.** Six needed updates after upstream moved to torchtitan v0.4.0 ([#4537](https://github.com/radixark/miles/pull/4537) to [#4539](https://github.com/radixark/miles/pull/4539)) and merged [#4530](https://github.com/radixark/miles/pull/4530) on 2026-10-31, and force-updated `sglang-miles` on 2026-11-01. Each PR branch was moved forward (no force-push) and its description has an "Updated for ..." note.

*Table 2. The six updated PRs and their unit tests in an image with torchtitan v0.4.0 (miles) or the new `sglang-miles` (SGLang).*

| PR | What changed | Unit tests |
|---|---|---|
| [#4520](https://github.com/radixark/miles/pull/4520) | one line (main renamed the checkpoint load); its GPU tests now feed 2-D scores, as v0.4.0's MoE routes without a batch dimension | 29 passed |
| [#4522](https://github.com/radixark/miles/pull/4522) | narrowing moved into v0.4.0's `_load_checkpoint`; on v0.4.0 every save holds full optimizer state, so it only matters for resuming a v0.3.0 save | 6 passed |
| [#4523](https://github.com/radixark/miles/pull/4523) | main took the adapter, scheduler and iteration restore; left: keep the optimizer on, find the newest adapter save | 146 passed |
| [#4588](https://github.com/radixark/miles/pull/4588) | ported onto main's rewritten GLM-5.3 model; new `preprocess_inputs` override keeps images out of per-rank sharding | 53 passed |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | merged the new `sglang-miles`; diff is its own 7 files | 186 passed (179 on the base) |
| [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | same; its own 5 files | 26 passed (22 on the base) |

The combined branch, miles [`verify-int-20261101`](https://github.com/dzorlu/miles/tree/verify-int-20261101) (main + Rockdu's #4240/#4272 + #4586 + every open miles PR), passed 812 integration tests in the same image.

**GPU checks still owed on torchtitan v0.4.0** (the GPU results in the PR descriptions were taken on v0.3.0). Build from `verify-int-20261101` and SGLang [`verify-sgl-ts-glm-20261101b`](https://github.com/dzorlu/sglang/tree/verify-sgl-ts-glm-20261101b) (`sglang-miles` `b0ed761650` + #47402 + #47674 + #47203) on `radixark/miles:dev-202611011509` (its `sglang-miles` is `b0ed761650`), plus router #28. GLM-5.3 shape and flags: [../miles-harness/recipes.md](../miles-harness/recipes.md) section 3 (torchtitan 256K row, 6 nodes) with the token-space flags of [../miles-harness/README.md](../miles-harness/README.md) section 6, `MILES_TEST_R3_THRESHOLD=1.0`.
1. Fresh run, 3 steps. Pass: KL to the reference exactly 0 at step 0 (#4520); screenshots train through the vision tower (#4588); TITO mismatch 0.0.
2. Resume of that run (`--num-rollout 4`, `--load`, and `--ci-disable-weight-update-checker` with `--ci-test`). Pass: continues at rollout 3 and saves.
3. Resume of a torchtitan v0.3.0 save on v0.4.0. Pass: the load succeeds and logs which parameters' optimizer state starts fresh (#4522). If v0.4.0 cannot load v0.3.0 saves at all, close #4522.

What was tried on 2026-11-01 and 11-02:
- A 1-node run with a 4-layer slice of GLM-5.3 on the combined branch first failed in torchtitan's activation recomputation: "layers.3::feed_forward.shared_add.out is a placeholder for the output of remat.region ...". Cause: with `--gradient-checkpointing`, the MoE's `shared_add` region is saved, not recomputed, and main's `update_residual_streams` (`glm5_next/layers.py`) reads its output with bare ops. Fix: call `remat.recompute_needs_tensor` on the attention and feed-forward outputs first, as torchtitan's own models do. It is on dzorlu/miles `glm5next-remat-residual-update` (6 lines, lint clean) and in `verify-int-20261101`. With it, the slice run trained 3 steps with a checkpoint after each, and KL to the reference was exactly 0 at step 0 (then 4.1e-4 and 7.2e-4), so #4520 holds on v0.4.0. Upstream's own GLM 4-layer end-to-end tests are disabled in CI, which is why main has not hit this; a run of that test on pure main is in progress to confirm before opening the fix as a PR.
- Our internal vision-tower check tool is out of date with miles main (argument refactor, an SGLang module move); dropped in favor of the 6-node run with screenshots, which exercises #4588's path directly.
- The two 6-node GLM runs (checks 1 and 3 above) were resubmitted on an image with the fix.

**What is left.**
1. Run the GPU checks still owed on torchtitan v0.4.0 (section "Taking over" above).
2. Mark the passing PRs ready for review; close [sglang#47400](https://github.com/sgl-project/sglang/pull/47400).
3. Coordinate with overlapping upstream PRs (not posted): [#1585](https://github.com/radixark/miles/pull/1585) duplicates #4519; [#4218](https://github.com/radixark/miles/pull/4218) touches #4520's replay guard; [#3879](https://github.com/radixark/miles/pull/3879) would remove #4522's indexer case.
4. Follow-ups: an image-aware threshold for miles' R3 replay check (image tokens disagree about 15 times more than text on both image paths); two upstream bugs to report: native LoRA with `--colocate --lora-train-only` crashes in the first weight update on main, and the B300 dev image cannot train head-dim-256 attention (pin flash-attn-4 4.0.0b19 and quack-kernels 0.6.4).

How to build and run these checks without our launcher: [../miles-harness/](../miles-harness/README.md).

## 1. Exact tokens for vision models (TITO)

With the PRs below, Qwen3.8-VL and GLM-5.3 each trained 3 steps (GLM also resumed) on computer-use episodes with screenshots through the session server: TITO mismatch rate 0.0, 0 prompt-ID mismatches, and on GLM the routed-expert replay stays aligned with images. GPU runs: Qwen3.8-VL on Megatron, 4 nodes, and GLM-5.3 on torchtitan, 6 nodes, of 8x B300; episodes up to 256 turns with screenshot folding. The PRs split into what every model needs (section 1.1) and what each model family adds (sections 1.2 and 1.3). Figure 1 shows where each acts on one chat turn.

![Figure 1](tito-prs-turn-path.svg)

*Figure 1. One chat turn through the session server and the PR that acts at each step. Grey boxes are Rockdu's token-space stack ([#4240](https://github.com/radixark/miles/pull/4240), [#4272](https://github.com/radixark/miles/pull/4272)), which these PRs build on.*

### 1.1 All models: the session server on token space

Builds on Rockdu's [#4240](https://github.com/radixark/miles/pull/4240) and [#4272](https://github.com/radixark/miles/pull/4272). Any vision model that SGLang's token-space processor supports needs these three.

*Table 3. TITO PRs every model needs.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [miles#4586](https://github.com/radixark/miles/pull/4586), commit 1 | the session server carried no screenshots on the token-space design | send the stored expanded prefix, the new part with one placeholder per image, and `mm_token_expansion_start_len`; store the engine's expanded IDs | Qwen and GLM 3 steps each; TITO mismatch 0.0 |
| [miles#4586](https://github.com/radixark/miles/pull/4586), commit 2 | R3 row offsets counted one placeholder per image, so rows fell out of line (HTTP 422) | count offsets on the expanded tokens | GLM image-token replay rate equal to the non-token-space path (0.659% vs 0.638%) |
| [sgl-router-for-miles#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | the router dropped `mm_token_expansion_start_len` on chat; turn 2 of every session failed | forward the field | every token-space run |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | a chat request with `input_ids` and start 0 dropped its images | keep them | 0 prompt-ID mismatches over 124 Qwen samples |

### 1.2 Qwen3.8-VL

The token-space processor already supports Qwen VL. Qwen adds one parser fix. #4516 sets the parsers for every model family; the bug only showed on Qwen, whose prompt already ends with `<think>`.

*Table 4. Qwen-specific TITO PR.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [miles#4516](https://github.com/radixark/miles/pull/4516) | miles never set SGLang's parsers; for Qwen whole replies were stored as text | pass each model family's reasoning and tool-call parsers | 100% of 25,461 Qwen turns parsed; 0 leaked `</think>` |

### 1.3 GLM-5.3

GLM needs a token-space strategy in SGLang (it had none) and a parser setting.

*Table 5. GLM-specific TITO PRs.*

| PR | What broke | Fix | GPU result |
|---|---|---|---|
| [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | no GLM-5.3 strategy for the token-space processor | add one, bitwise equal to SGLang's GLM processor | GLM 3 steps and a resume |
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
