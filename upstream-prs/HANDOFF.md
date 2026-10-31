# Handoff: getting our miles and SGLang PRs merged

State on 2026-10-30. Everything needed to continue is in GitHub: the PRs, the fork branches below, and the texts in [pr-texts/](pr-texts/). Nothing depends on the old machine or our training cluster. [README.md](README.md) explains what each PR fixes. [tito-token-space.md](tito-token-space.md) explains the TITO screenshot work and its rebuild on Rockdu's token-space design.

## 1. Goal and rules

- Every fix we carry goes upstream as a PR against [radixark/miles](https://github.com/radixark/miles) or [sgl-project/sglang](https://github.com/sgl-project/sglang). All are raised as drafts.
- Each PR is verified on its own (section 5) before it is marked ready for review.
- Deniz decides the open questions in section 3. Do not mark PRs ready, request reviewers, force-push, or post comments upstream without his go.
- Forks: [dzorlu/miles](https://github.com/dzorlu/miles), [dzorlu/sglang](https://github.com/dzorlu/sglang). Push new work as new branch names. Do not rewrite branches that back an open PR unless Deniz says so.

## 2. Where each PR stands

Table 1. Every open PR and its current head. On 2026-10-29 the four rebuilt branches got new draft PRs and the old PRs were closed with a link: miles#4051 -> #4546, miles#4517 -> #4547, sglang#44133 -> #47400, sglang#44134 -> #47402. #4049 and #4050 got a merge of miles main `65feb4a616` (normal push, no force).

| PR | Group | PR head now | Rebuilt branch to use | Next step |
|---|---|---|---|---|
| [sglang#47400](https://github.com/sgl-project/sglang/pull/47400) (replaces #44133, closed) | 1. TITO | `pr3-mm-exact-token-images-v2` `e73de6edb1` (sglang main `a4a19332ab`) | none | to be closed: replaced by Rockdu's token-space path plus #47402 ([tito-token-space.md](tito-token-space.md)) |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) (replaces #44134, closed) | 1. TITO | `pr4-qwen-vl-input-ids-v2` `008518e4fe` (`9c267dae99` plus a merge of sglang-miles `a5701d958`) | same | keep: change 3 of the token-space set (first-turn images); verify in the token-space runs |
| [miles#4546](https://github.com/radixark/miles/pull/4546) (replaces #4051, closed) | 1. TITO | `pr6-session-images-v4` `10551a0b0e` (miles main `4250360dfc`) | `tito-session-token-space` `dce628a206` (changes 1 and 2) | to be replaced by token-space change 1 after the GPU runs pass |
| [miles#4516](https://github.com/radixark/miles/pull/4516) | 1. TITO | `pr-qwen-thinking-parser-v2` `c6777d7023` | same | verify |
| [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | 1. TITO | `pr-tool-call-constraint-arg-v2` `5338d322bf` | same | verify |
| [miles#4050](https://github.com/radixark/miles/pull/4050) | 2. Vision | `pr1-vision-rollout-schedule` `2d063bc963` (`9d26cf4c09` plus a merge of main `65feb4a616`) | same | verify |
| [miles#4049](https://github.com/radixark/miles/pull/4049) | 2. Vision | `pr2-mm-tower-sync` `c0b94b68c3` (`f41c4888fa` plus a merge of main `65feb4a616`) | same | verify |
| [miles#4520](https://github.com/radixark/miles/pull/4520) | 3. R3 | `pr-expert-replay-torchtitan-v2` `11c6f92781` (lint exemption added 10-29) | same | verify |
| [miles#4547](https://github.com/radixark/miles/pull/4547) (replaces #4517, closed) | 3. R3 | `pr-r3-session-image-offsets-v6` `226cdadbc6` (on #4546's branch, plus the output-store fix below) | `tito-session-token-space` `dce628a206` (second commit) | to be replaced by token-space change 2, shipped in one PR with change 1, after the GPU runs pass |
| [miles#4522](https://github.com/radixark/miles/pull/4522) | 4. GLM torchtitan | `pr-torchtitan-resume-load-v2` `9db3587ba0` | same | verify |
| [miles#4519](https://github.com/radixark/miles/pull/4519) | 5. Colocated | `pr-offload-abort-before-flush-v2` `faada27988` | same | verify |
| [miles#4521](https://github.com/radixark/miles/pull/4521) | 5. Colocated | `pr-colocated-memory-v2` `b3259196da` (lint exemptions; peak reset only after CUDA init, which fixed 3 CPU tests) | same | verify |
| [miles#4523](https://github.com/radixark/miles/pull/4523) | 6. Native LoRA | `pr-native-lora-resume` `248f3326fd` | same | verify |
| [miles#4524](https://github.com/radixark/miles/pull/4524) | 6. Native LoRA | `muse-cpu-safe-import` `d56683a4f2` (base `muse-spark-open`) | same | **PASSED, ready for review** (evidence posted on the PR) |
| [miles#4518](https://github.com/radixark/miles/pull/4518) | 7. Cancelled | `pr10-zero-std-skips-aborted-groups-v2` `1e5adfddae` | same | verify |

**Not raised yet: GLM-5.3 vision on torchtitan.** Our first PR, miles#4022, was closed on 2026-10-26 after upstream merged GLM-5.3 text training on torchtitan (#3989, #3995). The port of its six commits onto main is on dzorlu/miles `glm-vision-titan-port` `37da2ff9ed` (on main `4250360dfc`). Its 7 test files pass on CPU (49 tests). The log-probs computed per context-parallel rank (commit "torchtitan loss: compute log-probs on each CP rank's own tokens") still need a GPU check. Raise it as a draft PR after a GLM run passes. Open point: under context parallelism this path ignores `--log-probs-backend fused`.

**Output-store fix in #4547 (v6).** Upstream's output store (`--use-output-store`, off by default) counted R3 rows on the request's token IDs without expanding image placeholders, while our R3 code counts in the engine's expanded token space. v6 makes `adopt_session_records` in `miles/rollout/session/samples/merge.py` count expanded tokens using the session's media store. Test: `tests/fast/rollout/session/test_samples.py::TestStoreSessionAssembly::test_image_records_count_routed_experts_rows_in_the_engines_expanded_token_space` fails without the fix (7 rows stored, 4 expected) and passes with it. CPU only: our training runs do not turn the output store on.

**TITO rebuild on Rockdu's token-space stack (2026-10-30).** Rockdu's open [#4240](https://github.com/radixark/miles/pull/4240) / [#4272](https://github.com/radixark/miles/pull/4272) (with the SGLang half merged into `sglang-miles`) make the engine return expanded token IDs. They cover only the generate path, not the session server. We rebuilt our screenshot work on it as four changes, explained in [tito-token-space.md](tito-token-space.md): dzorlu/miles `tito-session-token-space` `dce628a206` (changes 1 and 2 as two commits, one PR; stacked on `rockdu/vlm-02-token-space` `f9e7a9f645`), sglang#47402, and dzorlu/sglang `tito-token-space-glm5` `354d92b34f`. They replace #4546, #4547 and #47400 once the GPU runs pass. The open questions for Rockdu are in that file, section 6.

**Overlaps with other upstream PRs (search of 2026-10-29).** #4519 is mostly a duplicate of HJSang's [#1585](https://github.com/radixark/miles/pull/1585). #4523's LR-scheduler part is duplicated by [#3964](https://github.com/radixark/miles/pull/3964), [#4229](https://github.com/radixark/miles/pull/4229) and [#4530](https://github.com/radixark/miles/pull/4530). The GLM vision port must move to torchtitan v0.4.0 ([#4537](https://github.com/radixark/miles/pull/4537) to [#4539](https://github.com/radixark/miles/pull/4539)). #4520 conflicts with [#4218](https://github.com/radixark/miles/pull/4218). #4522's indexer case goes away if [#3879](https://github.com/radixark/miles/pull/3879) merges. Coordination comments on those PRs wait for Deniz's go.

## 3. Open decisions (Deniz)

1. **How to update #4051, #44133, #44134 and #4517.** Decided 2026-10-29: new draft PRs from the rebuilt branches (#4546, #47400, #47402, #4547); the old ones are closed with a link.
2. **Build for verification.** Decided 2026-10-29: an upstream dev image plus every PR branch merged in unchanged (section 5.1).
3. **SGLang PRs on main (#47400, #47203).** Decided 2026-10-29: cherry-pick plus patch-id proof. Original options: SGLang main and `sglang-miles` have diverged by about 1,036 commits, and the engine miles runs is `sglang-miles`. Recommended: cherry-pick their commits onto `sglang-miles` for verification and show each commit's patch-id equals the PR's (`git patch-id`). Alternatives: merge SGLang main into the engine (a different engine), or also open versions against `sglang-miles`.
4. **Compute.** Decided 2026-10-29: our own training cluster (8-GPU B300 nodes).

## 4. CPU tests (any machine)

Setup: Python 3.12, torch 2.14.1 CPU build, no Megatron, no SGLang install, no CUDA. Put an SGLang `sglang-miles` checkout's `python/` directory on `PYTHONPATH` (miles imports SGLang modules in some tests). Set `HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=30`, or a tokenizer download can stall for hours.

```bash
export PYTHONPATH=$MILES_TREE:$SGLANG_MILES_TREE/python
python -m pytest tests/fast/rollout tests/fast/router tests/fast/utils/chat_template_utils \
  tests/fast/utils/test_processing_utils.py tests/fast/backends/training_utils/test_data.py \
  tests/fast/tools/test_lint_config_attribute_access.py \
  --ignore=tests/fast/rollout/test_agentic_rollout_benchmark.py -q -o addopts="" -p no:cacheprovider -rfE
```

Run the same command on main and on the branch, and compare the FAILED lists; only failures the branch adds count. On main `4250360dfc` this gives 3,810 passed and these 13 failures, all environment-related:

- `test_tool_call_utils.py::TestTokenizeToolResponses::test_tokenize_tool_responses` for 4 Kimi tokenizers
- `test_openai_endpoint_utils.py::test_post_buffer_no_retry_closes_stalled_body_connection[cancel]` (also flaky)
- `test_session_server_benchmark.py`: 5 tests (macOS has no `prctl`/`pidfd_open`)
- `test_generation_concurrency.py::test_generation_states_sample_the_current_engine_count_on_each_acquisition`: 2
- `test_sglang_anthropic_conversion.py::test_sglang_anthropic_conversion_contract`

Other known flaky tests on main: `test_hot_restart`, `test_session_v1_v2_parity` (bitwise), worker polling and transport tests. Upstream's CPU memory benchmark `tests/fast/rollout/test_agentic_rollout_benchmark.py` (12.0 GB budget) runs only on Linux; it has not been run on `pr6-session-images-v4`.

Megatron tests (`tests/fast/backends/megatron_utils`) fail at import without Megatron. `uv pip install --no-deps --target <dir> megatron-core` and `<dir>` on `PYTHONPATH` makes about 2,100 of them run on CPU (used for #4049 on 2026-10-29).

Lint: `pre-commit run --files <changed files>` inside the miles tree (ruff, autoflake, isort, black, and upstream's bans such as dynamic attribute access without an exemption).

## 5. GPU verification (needs new compute)

### 5.0 Status on 2026-10-30

- **Legacy track** (all PR branches merged on miles main `4250360dfc`; integration `83060b1867`; SGLang `60a36fe7ec`): full CPU suite in the dev image, integration vs main: +164 passing tests; the 5 new failures it showed are fixed (4 in #4521, 1 in the GLM port). The first GLM smoke opened zero episodes in 11 h because our launcher imported a function miles main renamed (`policy_uses_routing_key` -> `router_routes_by_key`); it was stopped. The launcher needs a release that uses the new name.
- **Token-space track** (top priority): miles `verify-ts-20261030` `f9194e6a0d` (Qwen) and `verify-ts-glm-20261030` `74c10fbb45` (GLM), SGLang `verify-sgl-ts-20261030` `7d072894bb` and `verify-sgl-ts-glm-20261030` `b83121b599`, base image `dev-202610300556`. Images wait for that launcher release; then Qwen 4 nodes and GLM 6 nodes, smoke then 3 steps.
- **Passed:** #4524 (CPU-only import in the Muse image; the commit before fails with `libcuda.so.1` missing).

### 5.1 Build

- **Base image:** an upstream dev image, `radixark/miles:dev-YYYYMMDDHHMM` (built several times a day). It bundles miles main, `sglang-miles`, torch, torchtitan and Megatron (the last unpinned: record its commit). Our last pick was `dev-202610281739` (miles `e07c68b994`, sglang-miles `6136e6d2bd`, Megatron `3bd5b320dc`, torchtitan `086bf6c166`).
- **miles:** start from a main commit that contains every PR's base (check with `git merge-base --is-ancestor`), then merge each PR branch (merge, not cherry-pick). Every PR head is then an ancestor of what runs, so the evidence covers the exact posted code. Main `4250360dfc` works on the `dev-202610281739` image: the 21 commits after its `e07c68b994` do not touch SGLang. Check that `pip install -e . --no-deps` still works after upstream's pure-wheel change (`53f9c95914`).
- **SGLang:** the image's `sglang-miles` plus a merge of #47402's branch, plus #47400 and #47203 per decision 3.3.

### 5.2 Workloads

Any computer-use taskset where every episode carries screenshots and grading needs no LLM judge. We planned a computer-use taskset of 435 training tasks with the session server on for Qwen. Screenshot folding must be on for #4050.

### 5.3 Runs

Sizes are what we used on 8-GPU B300 nodes; adjust to the new hardware.

| Run | Model and size | Steps | Covers |
|---|---|---|---|
| R1 | Qwen3.8-27B-VL, 4 nodes, Megatron, session server, folding on, one eval pass between steps | 3 | #4546, #47400, #47402, #4516, #4050 (Megatron path), #4049, #4519, #4521, #4518 |
| R2 | GLM-5.3-Flash VL, 6 nodes, torchtitan, R3 on, folding on | 3 | GLM vision port, #4547, #4520, #47203, #4050 (torchtitan path), #4519, #4521 |
| R3 | resume of R2 from its step-2 checkpoint | +1 | #4522 |
| R4 | Qwen3-0.6B native LoRA (`--lora-train-only`), 1 node: one uninterrupted run of 3 rollouts, one saved at rollout 1 and resumed to 3 | 3 | #4523 |
| R5 | CPU only, in a Muse image: `python -c "import miles.rollout"` and the PR's 3 tests | n/a | #4524 |

Before R1 and R2, run each PR's own tests inside the built image (GPU tests such as #4520's 5 routing-replay tests on one GPU), then a 1-step smoke run per model to catch recipe problems on the new base.

### 5.4 Per-PR checks

Deliver one table: PR | tests in image | run check result | PASS/FAIL. A FAIL names the failing check and the log line.

| PR | What the run must show |
|---|---|
| #47400 | images received per engine request equal the image placeholders in its token IDs, and are above 0 |
| #47402 (Qwen) | 0 mismatches between the token IDs miles sent and the IDs the engine ran; report how many requests were compared |
| #4546 | training samples carry the screenshots (PNG count equals placeholders, above 0); pixel values rebuilt per micro-batch without error; samples-collection time per call (max, p95) with 0 timeouts at 300 s; image processor loaded only on vision servers; since `--tito-session-mismatch-rate` is gone, report the cost of the mismatch check that now runs on every session |
| #4516 (Qwen) | share of assistant turns with a non-empty reasoning field and with parsed tool calls; 0 turns whose content still holds `</think>` or a raw `<tool_call>` |
| #47203 (GLM) | `--tool-call-constraint none` in the engine arguments; turns with more than one tool call occur |
| #4547 (GLM) | R3 replay mismatch 0; routed-expert rows equal the expanded token count |
| #4520 (GLM) | the R3 record pass runs without the safety-check crash; step-0 KL to the reference is exactly 0 (it was 0.0096) |
| #4050 | the loss and metric divisor equals the number of episodes, not samples (the bug made step-0 metrics 8.69 times too low: 1,112 segments against 128 episodes); both the Megatron and torchtitan paths |
| #4049 | each weight sync sends the vision tower's tensors (count above 0); engine tower weights equal the trainer's after a sync |
| #4519 | per offload: every engine reports 0 running requests before the cache flush; 0 "flush refused" lines; 0 timeouts |
| #4521 | per-process GPU memory and per-step peak allocated and reserved logged every step; host memory per node flat across steps (the bug: about +115 GiB per node per step on GLM) |
| #4522 | the resume loads step N, logs the never-trained (indexer) parameters that start fresh, has no "Missing key ... optimizer.state" error, starts at rollout N+1, then trains one step |
| #4523 | Adam state restored (step counts continue), starts at the restored rollout, LR schedule not stepped twice, adapter after the next rollout bit-identical to the uninterrupted run |
| #4518 | eval completes when a group holds a cancelled episode; report how many such groups occurred (none means unverified) |
| #4524 | the import works without a GPU; its tests pass |

Also report per model: episodes with turns above 0 and graded within the first 30 minutes (stop the run if not), the train-vs-rollout log-prob gap against earlier levels (Qwen mean abs diff 0.0128, K3 KL 5.2e-4; GLM 0.0251, 2.35e-3), steps and checkpoints complete.

### 5.5 Running on Modal

The verification runs on our training cluster today (section 5.0). If that cluster is unavailable, Modal can run all of section 5.3.

**What Modal offers** (docs, 2026-10-29): multi-node clusters through `@modal.clustered(size=N, rdma=True)`, up to 32 nodes and 256 GPUs per cluster, full 8-GPU nodes only. RDMA works on H100, H200, B200, B300 and GB200; B300 clusters get 6,400 Gbps, the others 3,200 Gbps. Ray runs with rank 0 as `ray start --head` and the other ranks joining `<rank-0 IP>:6379`, using `cluster.container_rank()` and `cluster.container_ips(family="ipv4")` (Ray needs IPv4). An older docs page called clusters a private beta; confirm access on the workspace with a 2-node test first. Docs: [multi-node clusters](https://modal.com/docs/guide/multi-node-clusters), [multi-node training](https://modal.com/docs/guide/multi-node-training).

**What already ran on Modal.**

- miles training, 2026-10-17: Qwen3.5-2B, FSDP, TITO session server v2, colocated on 2 B200 in one container. 2 rollouts and 2 steps in about 10 minutes, exit 0, step-0 train-vs-rollout log-prob gap 0.0. The script (not public) starts a local Ray head, then runs `train.py` with the weights on a Modal Volume mounted at `/weights`.
- The base image `radixark/miles` loaded with `modal.Image.from_registry(...)` (set `MODAL_IMAGE_BUILDER_VERSION=2025.06`). SGLang `sglang-miles` ran fine in a normal Modal Function.
- A second script ran a one-step vision-model training check (on 1 B200): all screenshots reach training, the loss is divided per episode (#4050), and weights match after the update. It runs on FSDP, which syncs the full state, so it does not exercise #4049 (Megatron only).
- SGLang Dedicated Endpoints for evals: one endpoint autoscales to N containers, and a `Modal-Session-ID` header per episode keeps the prefix cache warm (99.4 percent hits).

**What has to be built.**

1. **A launcher.** Our launcher builds the miles command line from per-model recipes and submits only to our own cluster. On Modal, a `@modal.clustered` function starts Ray on every node and runs `train.py` with a recipe's arguments on rank 0.
2. **The image.** Build the verification image of section 5.1 and push it to a registry Modal can pull.
3. **Storage.** Weights, checkpoints and logs go on Modal Volumes. The resume checks (#4522, #4523) need the checkpoint to outlive the job.
4. **Environments.** The training tasksets and their verifiers must be reachable from Modal over the internet. Run about 5 tasks first and measure the verifier and connection error rate.

**Check before sizing the GLM run: host memory.** Our colocated runs use CPU offload. On GLM, each rank held 149 GiB of pinned memory plus 62 GiB of heap before the host-release fix (#4521), about 1.7 TB on an 8-GPU node. Muse's head node used about 1.27 TB between steps. Our cluster's nodes had 2.7 TB. Modal's per-container RAM and `/dev/shm` limits for 8-GPU nodes are not checked. If they are lower, use more nodes or less offload.

**Order, cheapest first.**

1. One node, small models:
   - the vision training check on a small Qwen-VL: #4546, #47400, #47402, #4516, #4050 (FSDP path);
   - the Qwen3-0.6B LoRA resume pair: #4523;
   - the CPU import check: #4524.
2. One node, GLM-5.3-Flash's 4-layer slice (one node, expert parallel 8, as used for the R3 work): #4520, #4547, #47203, the GLM vision port.
3. Multi-node: the full Qwen (4 nodes) and GLM (6 nodes) runs of section 5.3, for #4049, #4050's Megatron and torchtitan paths, #4519, #4521 and #4522.

## 6. Upstream context

- [#4211](https://github.com/radixark/miles/pull/4211) (upstream, open, base `muse-spark-open`): native LoRA for Muse with merged rollout weights. It covers our Muse LoRA branch and our GPU-widen commit, so we do not raise those. A comment draft with our full-scale results is in [pr-texts/miles-4211-comment.md](pr-texts/miles-4211-comment.md) (not posted).
- [#3736](https://github.com/radixark/miles/pull/3736) (merged): session samples built off the event loop, one worker per server. Replaced part of #4051.
- [#2688](https://github.com/radixark/miles/pull/2688) (merged): explicit checkpoint selection; Megatron checkpoint resolution no longer runs for torchtitan. Replaced part of #4522 and part of #4523.
- [#4099](https://github.com/radixark/miles/pull/4099), [#4192](https://github.com/radixark/miles/pull/4192), [#4193](https://github.com/radixark/miles/pull/4193) (merged): upstream fixes that replaced parts of #4050 and one colocated-run fix.
- [#3989](https://github.com/radixark/miles/pull/3989), [#3995](https://github.com/radixark/miles/pull/3995) (merged): GLM-5.3 text training on torchtitan.

## 7. Not PRs

- SGLang never frees the pinned host copy of image features (hundreds of GiB per node on GLM): bug report written, not filed.
- miles cannot cancel agent code running in its own process (17 idle minutes per round on one run): bug report, not filed.
- A cancelled episode keeps generating on the engine (a 32K-token reply runs 36 minutes): planned fix, send `/abort_request` from the session server.
- After each Megatron step the last micro-batch's recompute tensors stay alive (25 GiB per GPU on Muse stage 0); cause unknown. Workaround `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` on the trainer.

## 8. Gotchas

- zsh reads `$b:r` in `"$b:refs/heads/$b"` as a modifier and breaks git refspecs. Run git loops in bash or write `${b}`.
- `git grep` in a `--filter=blob:none` clone downloads every blob it reads. Use a full clone or GitHub code search.
- SGLang PR titles must match `[Type] scope: title` (Type is one of Feat, Fix, Refactor, Perf, Doc, Test, CI, Chore).
- SGLang CI: a maintainer must add the `run-ci` label. The "Rebase Required" gate wants a minimum main commit in the PR head; PRs on `sglang-miles` need the `bypass-maintenance` label or a main merge into `sglang-miles`.
- miles CI on fork PRs waits for a maintainer to approve the workflows. The Mintlify bot comment about a docs preview is not a review.
- Upstream moves daily. Check each branch with `git merge-tree --write-tree origin/main <branch>` before pushing or updating a PR.
