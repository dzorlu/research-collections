# sglang#44134: replacement title and text

For the rebuilt branch `pr4-qwen-vl-input-ids-v2` (`9c267dae99`, on sglang-miles `b8d46f2f30`).

**Title:** `[Fix] qwen-vl: keep the client's input_ids for image requests on sglang-miles` (77 characters; matches the pattern)

**Problem.** On sglang-miles, a Qwen VL request with images ignores the client's token IDs. The server decodes them to text and tokenizes that text again, which can split words differently. The engine then samples from a prompt that differs from the one the trainer scores (miles#1700). On Qwen3.8-27B-VL training runs, 60 to 85 percent of screenshot requests from turn 2 on differed by 1 to 2 IDs.

```text
sent:   [279, 46587, 62, 327, 1979]   " the mart_" + "adoption" + " model", plus 1 image
before: ran [279, 46587, 9731, 1979]  62 + 327 merged into 9731, 1 ID short
after:  ran [279, 46587, 62, 327, 1979], ID for ID
```

**Fix.** The Qwen processor calls `legacy_load_mm_data` without the client's IDs (#18781), so the exact-token block of #26555 is skipped. The fix passes `input_ids=input_text if isinstance(input_text, list) else None`. The legacy loader stays, so whatever #18781 fixed still holds. Clients must send one placeholder per image, as #26555 requires; miles' generate path does that after miles#4051. The PR also carries #44133's commits, rebased onto sglang-miles `b8d46f2f30`.

**Tests.**

```bash
pytest test/registered/vlm/test_qwen_vl_input_ids_e2e.py test/registered/unit/entrypoints/openai/test_serving_chat.py test/unit/multimodal/test_load_mm_data_by_ids.py
```

- "D" + "escribe" + " the picture: " + one placeholder + one 64x64 image through `/generate` runs as `len(input_ids) - 1 + N` prompt tokens. The IDs read back with `logprob_start_len 0` equal the sent IDs at every text position.
- The same text without the image keeps `len(input_ids)` prompt tokens.
- CPU unit tests on the rebased branch: 0 new failures against sglang-miles.

**Proven in a run.** Qwen3.8-27B-VL with the change: 8 agent episodes (1,136 requests) ran ID for ID with 0 fallback warnings. One training run had 0 prompt-length mismatches; without the change most screenshot requests mismatched.

**CI note.** The "Rebase Required" gate needs main's `6a39ddf1f2` in the PR head. sglang-miles does not contain it, so this PR needs the `bypass-maintenance` label or a main merge into sglang-miles first.

**Question for @yueming-yuan.** What did #18781 fix? If the newer loader now works for Qwen VL training, removing the legacy call is the cleaner change.
