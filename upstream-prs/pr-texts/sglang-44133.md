# sglang#44133: replacement title and text

For the rebuilt branch `pr3-mm-exact-token-images-v2` (`e73de6edb1`, on sglang main `a4a19332ab`).

**Title:** `[Fix] multimodal: keep images of input_ids chat requests; locate them by placeholder ID` (87 characters; matches the `pr-title.yml` pattern)

**Problem.** For RL, the client sends the prompt as token IDs so the engine runs exactly what the trainer will score. Two bugs break this for requests with images.

1. On the chat endpoint, a request sent as token IDs loses its images. The model answers a prompt whose image placeholders stand for nothing.
2. To find the images, the server decodes the IDs to text and searches for placeholder strings. Text that only spells a placeholder (for example `<image>` in HTML the model wrote) counts as one more image, and the request fails with HTTP 500.

```text
Qwen3.8-27B-VL
bug 1, token IDs: 21 IDs with 1 placeholder, plus 1 image (1024x576 = 576 image tokens)
    before: images dropped -> the engine runs 21 IDs, text only
    after:  the engine runs 20 + 576 = 596 IDs, ID for ID
bug 2, prompt: 1 screenshot + an earlier reply with "<div><image></div>"
    before: 2 placeholders found, 1 image given -> HTTP 500
    after:  1 placeholder ID = 1 image; "<image>" stays text -> HTTP 200
```

**Fix.**

- Bug 1. When a multimodal model gets a chat request with token IDs, the endpoint collects the images, videos and audio from the messages, as the template path does, and passes them with the IDs. Text-only models and requests without IDs are unchanged.
- Bug 2. For an ID prompt, the loaders count the placeholder ID in the list instead of decoding and searching text. Typed text never produces the placeholder ID. Each image is one placeholder ID, or one run of them if the client already expanded it; any other count is HTTP 400. The engine then runs the client's IDs as sent, with each placeholder expanded by the exact-token code of #26555. If expansion fails, the request fails instead of re-tokenizing. Text prompts, video, audio, precomputed features, and `SGLANG_MM_AVOID_RETOKENIZE` off keep the old path.

**Tests.** 9 new tests.

```bash
pytest test/registered/unit/entrypoints/openai/test_serving_chat.py test/unit/multimodal/test_load_mm_data_by_ids.py test/registered/vlm/test_chat_input_ids_keeps_images_e2e.py
```

- A chat request with token IDs on a multimodal model keeps its images; on a text-only model it carries none.
- End to end on Qwen2.5-VL-3B-Instruct: "D" + "escribe" + one placeholder + one 64x64 image through `/v1/chat/completions` runs as `len(input_ids) - 1 + N` prompt tokens, the same count as `/generate`.
- Both loaders run an ID prompt with a tokenizer whose `decode()` raises, so the decoded text is never produced.
- 2 placeholders with 1 image is a client error, not a 500.
- Rebased on main `a4a19332ab`: the CPU unit tests have 0 new failures against main.

**Proven in a run.** 8 agent episodes through the miles session server (Qwen3.8-27B-VL, 1,136 requests) all ran ID for ID, with 0 "falling back to decode+retokenize" lines. On Qwen3.8-27B-VL, a request with 2 images plus the vision markers spelled with ordinary tokens went from HTTP 500 to HTTP 200 (315 prompt tokens, IDs as sent).
