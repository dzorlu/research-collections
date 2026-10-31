Stacked on #4272 (`rockdu/vlm-02-token-space`). Brings the TITO **session server** onto Rockdu's token-space design, so agent harnesses that speak the chat API can send screenshots. #4272 covers the generate path; this PR covers the session server. It replaces our drafts #4546 and #4547, which carried the same goal on the older design.

Walkthrough with made-up token IDs and diagrams: [tito-token-space.md](https://github.com/dzorlu/research-collections/blob/main/upstream-prs/tito-token-space.md).

## Changes (two commits)

1. **Session server on token space** (`miles/rollout/session/token_space.py`, new). Each screenshot is saved once under `--shared-media-dir`. Each turn sends the stored expanded prefix, the new part with one placeholder per new image, and `mm_token_expansion_start_len` at the boundary, so SGLang expands only the new images. The session keeps two views per turn: the template view (one placeholder per image) for the TITO prefix check and rollback, which are unchanged, and the engine view (the expanded `prompt_token_ids` the engine returns) for requests, samples and R3. The engine must return the expanded prefix unchanged, else HTTP 502. The training sample is the engine view plus the screenshot paths; the trainer rebuilds pixels with `compute_multimodal_train_inputs`.
2. **R3 offsets on the engine view.** `routed_experts_start_len` and the per-turn row stitching (`merge_samples_with_addition_r3`, `adopt_session_records`) count tokens on the engine view. Without this, each offset falls short by one per earlier image token and the session server refuses the sample (HTTP 422, a reshape error).

Flags: `--multimodal-token-space-processing` (miles) and `--sglang-enable-token-space-processor`. Without the miles flag, a chat request with media gets HTTP 400 with an explicit message; upstream's session server carries no media today.

## Depends on

- sgl-project/sglang#47402: a chat request with `input_ids` and `mm_token_expansion_start_len = 0` dropped its images.
- The router (`sgl-router-for-miles`) must forward `mm_token_expansion_start_len`; its typed chat request drops unknown fields. Our runs use a one-line patch (PR to the router to follow).
- GLM-5.3 needs a token-space strategy in SGLang (draft PR to `sglang-miles` to follow).

## Verified on GPU

Computer-use episodes (screenshots, up to 256 turns, screenshot folding on) through the session server, 8x B300 nodes, this branch merged with the PRs above.

| Check | Qwen3.8-27B VL (Megatron, 4 nodes) | GLM-5.3 (torchtitan, 6 nodes, R3 on) |
|---|---|---|
| Rollout through the session server | 32/32 episodes graded; TITO mismatch rate 0.0; 0 prompt-ID mismatches; 0 HTTP 400 | 32 episodes, 180 samples; 0 prompt-ID mismatches |
| Train step 0 completes, checkpoint saved | yes: train/rollout KL 9.65e-4, mean abs log-prob diff 0.0154, grad norm 0.157 | pending (3-step run queued) |
| 3-step run | step 0 done (KL 8.92e-4, abs diff 0.0145); steps 1 to 2 running | pending |
| R3 replay check (top-8 experts share none with the replay) | n/a (dense) | image tokens 0.659%, text 0.0445%; the same model on the non-token-space path gives 0.638% and 0.0460%, so this PR does not change the rate. Image tokens exceed the checker's 1% default on both paths (a separate checker change). |

CPU: `tests/fast/rollout/session/test_token_space_images.py` (with SGLang's real Qwen token-space processor and tokenizer, stub engine) checks bytes sent per turn, sample tokens equal to the engine prompt plus reply, and trainer pixel values equal to the engine's; the R3 tests in `test_samples.py` fail with commit 1 alone and pass with commit 2.
