# Our PRs to miles and SGLang

State on 2026-11-25. We train vision models with RL on computer-use episodes: [miles](https://github.com/radixark/miles) trains, [SGLang](https://github.com/sgl-project/sglang) generates. Branches live on the forks [dzorlu/miles](https://github.com/dzorlu/miles), [dzorlu/sglang](https://github.com/dzorlu/sglang) and [dzorlu/sgl-router-for-miles](https://github.com/dzorlu/sgl-router-for-miles), one worktree per PR under `~/repos/`. Tests run in the CI image on Modal with [../miles-harness/modal/miles_tests.py](../miles-harness/modal/miles_tests.py).

## Which PRs each model needs

| PR | Qwen VL (Megatron) | GLM-5.3 (torchtitan) | State |
|---|---|---|---|
| [router#28](https://github.com/radixark/sgl-router-for-miles/pull/28) | yes | yes | merged |
| [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) | yes | yes | needs review (Rockdu) |
| [miles#4770](https://github.com/radixark/miles/pull/4770) | yes | yes | needs review |
| [miles#4521](https://github.com/radixark/miles/pull/4521) | yes | yes | needs review |
| [miles#4516](https://github.com/radixark/miles/pull/4516) | yes (parsers) | no (the bug shows only on Qwen) | conflicts with main; needs review |
| [sglang#47674](https://github.com/sgl-project/sglang/pull/47674) | | yes | approved, waits for Rockdu to merge |
| [miles#4588](https://github.com/radixark/miles/pull/4588) | | yes | needs review |
| [miles#4520](https://github.com/radixark/miles/pull/4520) | | yes | needs review |
| [sglang#47203](https://github.com/sgl-project/sglang/pull/47203) | | yes (many tool calls per turn) | needs review |

The miles PRs are current with main. Their GPU tests pass on one L4 with torchtitan 0.4.0: #4588 94 passed, #4520 47 passed, #4521 78 passed and 1 failed (Modal's sandbox has no `RssAnon` line). A GLM-5.3 training run on current main is still owed.

## One session, PR by PR

- **Client (the agent):** sends chat messages with images, never token IDs.
- **Session server (miles):** turns messages into token IDs with TITO, saves every turn, builds the SGLang request.
- **Engine (SGLang):** expands the new image placeholders, generates, returns the expanded prompt.

Contrived example: one image expands to 3 tokens, and the placeholder ID is `9`. Each `9` is a slot for one image-patch vector. SGLang and the trainer each load the image file and fill the slots with their own copy of the frozen vision tower; vectors are never sent between processes.

```python
# client → session server: [sys, user1 + imgA]
#   miles#4770     before: no image code; passes imgA through inline, keeps only [1, 9, 5]
#                  after:  saves imgA to /shared/a.png, renders [1, 9, 5]

# session server → router → SGLang: input_ids=[1, 9, 5], images=[a.png]
#   miles#4770     before: no start_len sent
#                  after:  sends start_len=0, asks SGLang to return its expanded prompt
#   router#28      before: dropped start_len on chat
#                  after:  passes it through
#   sglang#47402   before: decodes [1, 9, 5] to text and re-tokenizes, may run [1, 9, 4];
#                          drops imgA on turn 1
#                  after:  runs [1, 9, 5] as sent, keeps imgA
#   sglang#47674   before: GLM-5.3 has no token-space processor; the old one ignores start_len
#                  after:  expands 9 → [9, 9, 9]

# SGLang → session server: [1, 9, 9, 9, 5] + reply [7, 8]
#   miles#4770     before: keeps only [1, 9, 5, 7, 8]
#                  after:  saves both forms: tito [1,9,5,7,8], engine [1,9,9,9,5,7,8]

# client → session server: [sys, user1 + imgA, r1, user2 + imgB]
#   miles#4770     before: sends [1,9,5,7,8,2,9,6]; SGLang re-expands both images
#                  after:  sends saved [1,9,9,9,5,7,8] + [2, 9, 6], start_len=7
#   sglang#47674   after:  expands only imgB → [1,9,9,9,5,7,8,2,9,9,9,6]

# session ends → sample
#   miles#4770     before: [1,9,5,7,8,2,9,6, r2], no images: 2 nines vs 6 tokens SGLang ran
#                  after:  [1,9,9,9,5,7,8,2,9,9,9,6, r2] + [a.png, b.png]

# trainer (torchtitan)
#   miles#4588     before: GLM on torchtitan is text-only
#                  after:  loads a.png and b.png, runs the vision tower, fills the 6 nines
#   miles#4520     before: reference model picks its own experts → KL 0.0096 at step 0
#                  after:  reuses SGLang's experts → KL 0.0 at step 0
#   miles#4521     before: CPU memory grows about 115 GiB per node per step
#                  after:  freed after each step → flat
```

The "before" lines for #4770 and #47674 come from reading the code, not from a run. Design and more examples: [tito-token-space.md](tito-token-space.md). GLM vision forward pass: [glm-vision-titan.md](glm-vision-titan.md).

## Other PRs

| PR | Fixes | State |
|---|---|---|
| [miles#4519](https://github.com/radixark/miles/pull/4519) | freeing engine memory crashed while a request was running; now cancels requests first | needs review |
| [miles#4518](https://github.com/radixark/miles/pull/4518) | scoring crashed on a group with a cancelled episode; now skips it | needs review |
| [miles#4050](https://github.com/radixark/miles/pull/4050) | loss divided by samples, not episodes, under screenshot folding (8.7× too small on Qwen) | draft: main rewrote the optimizer step, needs a re-port |
| [miles#4049](https://github.com/radixark/miles/pull/4049) | frozen vision tower re-sent after each weight update, for any model | draft: main covers Inkling and Qwen3.5 |
| [miles#4523](https://github.com/radixark/miles/pull/4523) | native-LoRA resume dropped Adam state | draft: superseded by main's adapter library |
| [miles#4524](https://github.com/radixark/miles/pull/4524) | Muse imports without a GPU | waits for upstream [#4111](https://github.com/radixark/miles/pull/4111) |
| [miles#4522](https://github.com/radixark/miles/pull/4522) | resume when a save lacks optimizer state | closed: only torchtitan 0.3.0 saves need it |
