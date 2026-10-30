# miles#4051: replacement PR text

For the rebuilt branch `pr6-session-images-v4` (`10551a0b0e`, on miles main `4250360dfc`). Replace the current body with the text below. Title and body were written for the v3 branch; v4 only adds the conflict resolutions with upstream's output-store series and one test change (an image with no known placeholder is refused before the output store sees it).

**Title:** `feat(session): carry images through the TITO session server and the generate path`

**Problem.** The session server is text only. When an agent conversation has screenshots, the training sample keeps one placeholder token per screenshot and drops the image. The trainer then learns a reply to a screenshot it never saw. The generate path has a related bug: it sends one placeholder per image token, which makes SGLang fall back to re-tokenizing.

Example 1. One turn with one screenshot (Qwen3.8-27B-VL, a 768x432 screenshot = 336 image tokens).

```text
            sample tokens                                          sample images
before:     ... <vision_start> <image_pad> <vision_end> ...        none
after:      ... <vision_start> <image_pad> x 336 <vision_end> ...  1 PNG (174 KB) + grid

generate path
before:     sends <image_pad> x 336 -> SGLang expects 1 per image -> re-tokenizes, IDs can drift
after:      sends <image_pad> x 1   -> SGLang expands it to 336   -> runs the IDs as sent
```

**Change.**

- Session images. Each image is stored once per session as PNG bytes under a content hash; stored messages reference it as `mm://sha256:<hex>`. The engine gets one placeholder per image plus the images. At sample time each placeholder is expanded to the image processor's token count, the PNGs and grids are attached, and the length is checked against the engine's prompt length. A mismatch refuses the sample (HTTP 422). New file `miles/rollout/session/media.py`.
- Image tokens per model family. A small table names each family's begin marker, placeholder and end marker (Qwen3-VL and GLM-5.3). A family without an entry refuses images with a message that names the table. The session server loads the image processor only for families in the table.
- Trainer. `get_batch` rebuilds `pixel_values` from the PNGs with the checkpoint's image processor. Batches that already carry `pixel_values` are unchanged. A PNG is about 30 times smaller than its pixel tensor (174 KB against 5.4 MB per 768x432 screenshot).
- Generate path. Sends one placeholder per image and keeps the expanded IDs in the sample. A sample with images and no known placeholder raises instead of being re-tokenized. `image_pad_id_for(state)` picks the placeholder: the TITO family's table when the state carries a TITO tokenizer, otherwise the processor's `image_token` (the in-tree generate states carry only the processor).
- With #3736. #3736 builds samples from a snapshot taken under the session lock. The snapshot now also copies the session's image index, so a chat that lands during the build cannot add images to what the worker reads.
- Client timeout for the samples op: 120 s to 300 s. The build of a long image session took 160 and 189 s (see results). Running it in #3736's worker thread does not make it shorter.
- Image processor fallback. When transformers cannot resolve a model's processor (GLM-5.3 on transformers 5.12.1), the server uses the engine's own processor.

**Not in this PR.** #3736 already moved the samples op off the event loop, with one worker per server. This PR drops its own version: the samples op in a thread under the session lock, three commits that kept the lock through cancellation, and a limit of four builds at once. Two diagnostics are also left out and can come as separate PRs: `--tito-session-mismatch-rate` (compute the `tito_session_mismatch` re-render for a sampled share of sessions) and `--session-chat-timing-log` (per-chat prepare, proxy and commit seconds).

**Model scope.** Qwen3-VL and GLM-5.3 tested; other families by adding a table entry.

**Tests.**

```bash
pytest tests/fast/rollout/session tests/fast/rollout/generate_utils/test_generate_endpoint_utils.py \
  tests/fast/utils/chat_template_utils/test_image_tokens.py tests/fast/utils/test_load_processor_fallback.py \
  tests/fast/backends/training_utils/test_png_images.py tests/fast/router
```

- Two screenshot turns through the v2 server (stub engine, real Qwen3.8 and GLM-5.3 tokenizers) give one sample with the expanded placeholders and both PNGs. An engine that re-tokenizes is refused.
- A chat with a new image that lands during a samples build does not change the build: the snapshot holds the earlier image, and the samples equal a build with no chat in between.
- The family table resolves to real token IDs on both tokenizers, and the chat template writes each image's markers once.
- The PNG rebuild gives pixel rows in order and refuses a grid that differs from the stored one.
- On CPU, `tests/fast/rollout`, `tests/fast/router` and the chat-template tests have no new failures against main (3776 passed).

**Results on a real model.**

- Qwen3.8-27B-VL, 8 agent episodes: 124 samples, all 731 screenshots attached, 0 length mismatches. The first training step on them matched an unfolded run (train-rollout log-prob abs diff 0.0223).
- GLM-5.3 (4-layer model, real engine): two screenshot turns, then a sample build with each screenshot as one run of 144 `<|image|>` IDs between its markers; 0 re-tokenize fallbacks.
- A Qwen3.8-VL run on 4 nodes (2026-10-13): two sessions of 320 to 358 calls took 160 and 189 s to build; the 120 s client timeout dropped both episodes with their groups.

**Not tested.** The PNG rebuild inside a real training step (our training used an equivalent rebuild in our own code). A full-size GLM-5.3. Upstream's Linux CPU benchmark on this branch.

**Questions for maintainers.**

1. Is storing each image once per session, by content hash, the right place?
2. Should the session server ship PNG bytes (this PR) or pixel tensors, as the generate path does today?
3. Should each model family declare its image markers (this PR), or should miles read them from the image processor?
4. On a length mismatch with the engine, should the sample be refused (this PR) or only warned about?
