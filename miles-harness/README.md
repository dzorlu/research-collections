# Build spec: a long-horizon agent RL harness on miles

This is a build brief for an engineer or coding agent. It describes a harness that trains agent policies
with RL on long multi-turn episodes (tool use and computer use with screenshots), using upstream
[miles](https://github.com/radixark/miles) for training and SGLang for sampling. It names what to build,
the rules each part must follow, and the tests that prove it works. Per-model run settings are in
[recipes.md](recipes.md). Colocated fully-async training is in [../rl-infra/](../rl-infra/README.md). The memory arithmetic and model choice are in
[../single-node-training/](../single-node-training/README.md).

Upstream references are to miles `main` at `c62ea5d` (2026-10-30). Re-check them before building. Several fixes this spec relies on are our open PRs; their status and GPU results are in [../upstream-prs/README.md](../upstream-prs/README.md) (Table 1). Until they merge, build from upstream main with those PR branches merged in.

## 1. The one rule everything serves

The trainer must score exactly the tokens the engine sampled, with exactly the images the engine saw. If
the trainer re-tokenizes text, token boundaries shift, the trainer's log-probs no longer match the
sampler's, and the policy-gradient ratio is wrong. Every design choice below either keeps that rule or
measures that it held.

## 2. Architecture

```text
 run config (YAML) + model recipe ──► launcher ──► one miles command line (dry-run prints it)
                                                        │
                                                        ▼
            miles (Ray): trainer (Megatron or torchtitan) ◄── colocated ──► SGLang engines
                     ▲                                                       ▲
                     │ samples: token IDs, loss masks, reward                │ token IDs + images
                     │                                                       │
       miles agentic_tool_call.generate ──► miles TITO session server (tree sessions, v2)
                     │                                  ▲
                     ▼                                  │ OpenAI chat API (base_url)
             YOUR agent function ───────────────────────┘
                     │
                     ▼
             YOUR environment adapter (tools, observations, grader)
```

What upstream already provides (do not rebuild):

- **TITO session server** (`--use-session-server v2`). It stores the token IDs the engine sampled, matches
  each new chat request against them, and tokenizes only the new message. A history rewrite (fold or
  summary) opens a new branch of the session tree; the episode returns one training sample per kept leaf.
- **Agent hook** `--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate`
  with `--custom-agent-function-path <yours>`. Contract (from the module docstring):

  ```python
  async def run(base_url: str, prompt, request_kwargs: dict, metadata: dict, **kwargs) -> dict | None
  ```

  The returned dict merges into every sample's metadata. Under the default
  `--custom-agent-function-mode subproc` each call runs in its own child process: arguments and return
  value must be picklable, and module-level state lasts one call.
- **Reward hook** `--custom-rm-path <yours>`: `async def reward_func(args, samples, **kwargs)`, where
  `samples` is a `Sample` or a list. Upstream examples read a reward the agent left in metadata.
- **Multi-sample episodes.** Under v2, every sample of an episode gets the same `rollout_id`. miles
  normalizes one reward per rollout in its GRPO group and copies the advantage to each sample. The v2
  postprocessor masks environment tokens and counts a shared completion once.
- **Compaction example**: `examples/experimental/terminus-compaction/` (Harbor's Terminus 2 with
  summarization and linear history through session server v2). Read it before section 4.8.
- **Agent function examples**: `examples/experimental/openenv/`, `nemo-gym/`, `harbor/`.

## 3. Repository layout to build

```text
harness/
  recipes/            one module per model family; Pydantic models (section 4.1)
  launcher.py         renders and validates the miles command (section 4.2)
  envs/base.py        environment protocol (section 4.3) + adapters
  agent.py            the agent function (section 4.4)
  reward.py           the reward function (section 4.5)
  fold.py             screenshot folding rule, pure functions (section 4.7)
  compaction.py       summary tool and trigger (section 4.8)
  eval.py             eval runs and pass@k scoring (section 4.10)
  configs/            run configs (YAML), one per experiment
tests/
```

Use Pydantic models for every structured value that crosses a module or process boundary (recipe, run
config, episode result, eval row). No `dict[str, Any]` where the shape is known.

## 4. Components

### 4.1 Recipe registry

One recipe per model and context length. A recipe holds everything that decides whether a run fits and
matches the model; a run config holds only run settings.

Recipe fields:

| Field | Meaning |
|---|---|
| `hf_checkpoint` | Hugging Face repo id (pin a revision) |
| `tito_model` | miles TITO tokenizer family (`--tito-model`) |
| `chat_template` | template file if the stock one is wrong (section 6) |
| `backend` | `megatron` or `torchtitan` |
| `max_context_len`, `max_response_len` | episode budget; response = context minus prompt headroom (16K) |
| `max_tokens_per_gpu` | per-rank tokens in one pass; Megatron multiplies by CP when packing |
| `parallel_by_shape` | map `(nodes, gpus_per_node)` → TP, SP, PP (with first/last stage layers), CP, EP, expert-TP |
| `engine_gpus`, `mem_fraction` | SGLang engine size and static memory fraction |
| `optimizer_cpu_offload` | Adam in host RAM (adds the precision-aware flags, section 5) |
| `log_probs_chunk_size` | tokens per log-prob chunk; `-1` = off |
| `log_prob_pass_multiplier` | log-prob pass budget as a multiple of the train pass (2 by default; 1 when engines keep weights resident) |
| `use_kl_loss` | builds the reference model and runs its pass every step even at coefficient 0 |
| `vision`, `screenshot_after_action` | image episodes |
| `routing_replay` | MoE: replay the engine's expert choices in training (R3) |
| `extra_sglang_args`, `extra_train_args`, `env_vars` | the rest, from [recipes.md](recipes.md) |

Derive variants from a base recipe (`model_copy(update=...)`), never by copying the whole block.

### 4.2 Launcher

Renders recipe + run config into one miles command and a `run.sh`. Must:

1. Refuse a node shape the recipe has no parallel entry for.
2. Refuse a batch where `rollout_batch_size × n_samples_per_prompt` is not a multiple of DP. Without
   `--balance-data` miles splits round robin and each rank trains `n // dp` samples, silently dropping the
   rest.
3. Pick the router from the cache: radix cache on → `--sglang-router-policy consistent_hashing` (each
   episode keeps one engine, so its prefix stays cached); cache off → `round_robin`. Never the
   cache-aware default: every request shares the system-prompt prefix, so it can route almost everything
   to one engine.
4. Refuse `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` anywhere when `--colocate` or
   `--offload-train-target` is on: `torch_memory_saver` rejects it at startup.
5. Refuse a run config that sets any recipe field.
6. Print the full command in `--dry-run`.

### 4.3 Environment protocol

```python
class Env(Protocol):
    async def open(self, task: Task) -> Observation          # task prompt, tool schemas, first screenshot
    async def step(self, call: ToolCall) -> Observation       # text and/or one image (PNG bytes)
    async def grade(self) -> Grade                           # reward in [0, 1] plus details
    async def close(self) -> None
```

Adapters: an MCP server (tools via `list_tools` / `call_tool`), OpenEnv, Harbor. The grader runs against
the environment's final state, not the transcript. Retries on `open` with backoff; a circuit breaker stops
opening new episodes after N consecutive open failures in a window.

### 4.4 Agent function

`harness.agent.run(base_url, prompt, request_kwargs, metadata)`:

1. Open the environment for the task in `metadata`.
2. Loop: send the chat messages to `base_url` with the OpenAI client; parse one tool call; execute it;
   append the observation; repeat.
3. Track context: after each turn, the tokens used (from the response's `usage`). End the episode as
   `context_full` when prompt + per-turn cap would exceed `max_context_len`.
4. Per-turn cap (for example 8,192 tokens). A turn that hits it does not end the episode: keep the
   truncated text as the assistant turn and tell the model to continue.
5. Optional context note on each observation: context used of the window and turns left.
6. Grade, close the environment, and return `{"reward": r, "end_reason": ..., "agent_metrics": {...}}`.
   Put tool time in `agent_metrics.total_tool_time` (miles subtracts it in throughput accounting).
7. Never raise for an environment error inside an episode; record it as an end reason.

End reasons (an enum): `submitted`, `context_full`, `max_turns`, `timeout`, `env_error`, `aborted`.

### 4.5 Reward function

Reads `metadata["reward"]` from each sample. All samples of one episode carry the same value.

### 4.6 Episode control

- Caps: turns, wall time, context. Log each episode's end reason.
- Aborted episode: mark the sample `ABORTED` and use
  `--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted`.
  Cancel the episode's GRPO siblings when one aborts, so a group never waits on a dead slot.
- Concurrency cap from KV: `floor(0.85 × KV pool tokens per engine × engines / typical episode length)`.

### 4.7 Screenshot folding

Pure function over the message list, applied before each request:

1. At most B screenshots stay live.
2. When a new screenshot would make B + 1, move the fold boundary forward S turns. Replace every image
   part before the boundary with the text `" <image collapsed>"`. Keep all reasoning, action and tool
   text.
3. Default B = S = 40. Trainer tokens per episode grow by about (turns / S) × history: 3× at B = S = 40,
   about 10× at B = 20, S = 10.
4. The placeholder starts with a space. SGLang re-tokenizes the decoded prompt of multimodal requests, and
   a placeholder glued to the previous character can merge into another token.
5. The same rule runs in training and eval.

Because the fold rewrites history, the session server opens a branch: one fold = one more training sample.

### 4.8 Policy-written summary (compaction)

1. Trigger: context ≥ threshold (sweep 98K and 131K for a 262K window).
2. The next turn asks for a summary (task state, values found, next steps) through a local tool
   `compact_context(summary)` that the agent function handles; cap it at 4,096 tokens.
3. Continue from: system prompt + task + summary + last observation (latest screenshot only).
4. The summary turn is trained (loss mask 1) and rewarded like any turn. The next branch takes it as
   prompt (loss mask 0). The session server's branch handling gives this for free.
5. Base and checkpoint evals use the same compaction setting.

### 4.9 Training rules for multi-sample episodes

- `--use-dynamic-global-batch-size`: samples per step vary with folds and summaries. miles derives
  optimizer steps from samples / (global batch / DP); without the flag a step crashes or takes several
  optimizer steps. Pad the batch to a multiple of DP with zero-loss copies.
- Divide the loss by episodes, not samples ([miles#4050](https://github.com/radixark/miles/pull/4050);
  check whether it merged).
- `--calculate-per-token-loss` so a split episode weighs the same as one long sequence.

### 4.10 Eval

Same agent function and environment adapter, no training step. Same image, folding and compaction
settings for base and every checkpoint of one comparison. Report pass@1 and pass@k from binary pass/fail,
the end-reason distribution, and mean and max turns. Pair checkpoint and base by task and sample index. An eval group that holds a cancelled episode crashed the zero-std scoring ([miles#4518](https://github.com/radixark/miles/pull/4518)).

### 4.11 Observability

Per episode: turns, end reason, context tokens, folds, summaries, images live and folded, tool calls and
tool errors, reward. Per step: `train_rollout_logprob_abs_diff` (trainer vs engine log-probs; the
token-exactness signal), KL to the reference, entropy, clip fraction, step time. Count tool errors against
tool calls after every update: a policy that breaks its tool-call format shows up there first.

## 5. Colocation and long-context settings (all recipes)

- `--colocate --offload-train-target cpu`: trainer state goes to host RAM while engines sample. Use `disk`
  only if host RAM is short, and check the node disk first.
- Engines free their weights during the train step by default. Keeping them resident
  (`--offload-rollout-level kv_cache`) only fits GPUs with large headroom.
- `--check-weight-update-equal`: after the first update, every engine tensor equals the trainer's.
- Vision tower frozen in training: re-send it on every update (`--mm-tower-sync visual`,
  [miles#4049](https://github.com/radixark/miles/pull/4049)) or the engines serve an unrestored tower.
- Adam on CPU: `--optimizer-cpu-offload --use-precision-aware-optimizer --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16 --rematerialize-param-from-master-weight` (Megatron).
- Long prefills: `--rollout-health-check-interval 600 --rollout-health-check-timeout 1800`,
  `SGLANG_HEALTH_CHECK_TIMEOUT=600` to `1800`, `--distributed-timeout-minutes 60` (first passes compile
  kernels per shape while other ranks wait).
- Screenshots between SGLang processes: `--sglang-mm-feature-transport cuda_ipc` with
  `SGLANG_MM_FEATURE_CACHE_MB=4096`. A full pool silently falls back to CPU pickling.
- Radix cache on; router per section 4.2.
- Offload while requests are running: abort them before the cache flush, or the flush is refused and the offload stalls ([miles#4519](https://github.com/radixark/miles/pull/4519)). With `--fully-async --colocate`, the PR's second commit keeps the requests a driver-owned pause retracted; without it every in-flight episode is written off at the first train step.
- Host memory after each train step: [miles#4521](https://github.com/radixark/miles/pull/4521) returns it and logs per-process GPU and host memory; without it GLM-5.3 grew about 115 GiB per node per step.
- Resume: torchtitan needs [miles#4522](https://github.com/radixark/miles/pull/4522) (parameters that never train start with fresh optimizer state). Native LoRA needs [miles#4523](https://github.com/radixark/miles/pull/4523); without it a resume silently restarts (fresh optimizer, fresh adapter, scheduler at 0). With `--colocate --lora-train-only`, upstream main (dev image 2026-10-30) crashes in the first `update_weights` ("found paused allocation but cpu_backup does not exist"); run native LoRA non-colocated until that is fixed.

## 6. Model-specific traps

- **Tool and reasoning parsers.** Set `--sglang-tool-call-parser` and `--sglang-reasoning-parser`
  explicitly. If the prompt already ends with `<think>`, a parser that waits for `<think>` stores whole
  replies as plain text ([miles#4516](https://github.com/radixark/miles/pull/4516)).
- **Qwen3.5/3.6/3.8 chat template.** The stock template raises "No user query found" on image-only user
  turns; use the fixed template from [miles#2760](https://github.com/radixark/miles/pull/2760), and put a
  text part in every user turn. Ask for all reasoning inside `<think>` and only tool calls after it.
- **GLM-5.3 parser grammar.** SGLang's GLM parser forces one declared tool per turn; turn it off with
  `--tool-call-constraint none` ([sglang#47203](https://github.com/sgl-project/sglang/pull/47203)).
- **Screenshots through the session server.** Use the token-space path: [miles#4586](https://github.com/radixark/miles/pull/4586) (stacked on Rockdu's [#4240](https://github.com/radixark/miles/pull/4240) and [#4272](https://github.com/radixark/miles/pull/4272)), flags `--multimodal-token-space-processing --shared-media-dir <path every engine mounts> --sglang-enable-token-space-processor`. It also needs [sglang#47402](https://github.com/sgl-project/sglang/pull/47402) (first-turn images were dropped) and the router fix in [../upstream-prs/patches/](../upstream-prs/patches/): without it the router drops `mm_token_expansion_start_len` and turn 2 of every session fails ("Found 577 placeholders ... but received 2 media expansions"). GLM-5.3 needs a token-space strategy in SGLang (dzorlu/sglang `tito-token-space-glm5`). Without the miles flag, a chat request with media gets HTTP 400. Design and examples: [../upstream-prs/tito-token-space.md](../upstream-prs/tito-token-space.md).
- **MoE routing.** Trainer and engine pick different experts for the same token; replay the engine's
  choices (`--use-rollout-routing-replay`, engines on the `triton` MoE runner, `--pause-generation-mode
  in_place`), and add truncated importance sampling (`--use-tis`) for the remaining gap. With screenshots through the session server, the per-turn replay offset must count expanded image tokens ([miles#4586](https://github.com/radixark/miles/pull/4586), second commit). On torchtitan the replay guard needs [miles#4520](https://github.com/radixark/miles/pull/4520). The replay check (`--ci-test`) fails at its default 1% per-rank threshold on GLM-5.3 with images on both image paths: image tokens disagree on all eight experts at about 0.65%, text tokens at about 0.045% (deep layers). Set `MILES_TEST_R3_THRESHOLD` for your image share, and keep Ray log dedup on: with it off the check writes about 20,000 lines/s.
- **Logits.** tokens × vocab × 2 bytes; at 64K tokens and a 248K vocab that is 30 GiB. Use TP for a
  vocab-parallel loss and `--log-probs-chunk-size 4096`.

## 7. Test gates, in order

1. **Unit tests**: fold rule (boundaries, multi-image observations, placeholder tokenization), recipe
   validation (every refusal in 4.2), end reasons, reward broadcast across branches.
2. **Token exactness probe**: one live engine. Send a chat turn with a split word and a screenshot through
   the session server; the stored IDs must equal the engine's prompt IDs, id for id.
3. **One-node smoke** with a small model (Qwen3-4B) and 2-turn episodes: weight-update equality passes,
   `train_rollout_logprob_abs_diff` near its single-turn baseline, one train step, one weight update.
4. **Debug-minimal job of the real recipe** on the target shape: peak GPU memory, host memory, step time.
5. **First 10 minutes of every real run**: checkpoint saved after the first save interval; weights loaded;
   step time as expected; loss and grad norm finite and nonzero; GPU memory below the limit; folds and
   end reasons logged.

## 8. Run config example

```yaml
recipe: qwen3.8-27b-256k-text-1node
nodes: 1
gpus_per_node: 8
taskset: path/or/hf-id
rollout_batch_size: 16
n_samples_per_prompt: 8
num_rollout: 100
lr: 1.0e-6
max_turns: 600
episode_timeout_s: 22500
max_tokens_per_turn: 8192
fold: {max_screenshots: 40, stride: 40}
compaction: {trigger_tokens: 131072, summary_max_tokens: 4096}
save_interval: 2
```

## 9. Running tests on Modal

[modal/miles_tests.py](modal/miles_tests.py) runs any command from a local miles checkout inside the miles dev
image on Modal, the way CI does (checkout first on `PYTHONPATH`). Nothing on this machine needs the SGLang fork.

```sh
cd miles-harness/modal
MILES_REPO=~/repos/miles-tito-session uvx modal run miles_tests.py \
    --cmd "pytest tests/fast/rollout/session/test_token_space_images.py tests/fast/utils/test_media_refs.py"
MILES_REPO=... MILES_GPU=L4 uvx modal run miles_tests.py --cmd "pytest tests/fast-gpu/test_torchtitan_routing_replay.py"
```

Settings are environment variables: `MILES_REPO` (the checkout; uploaded on each run without `.git`), `MILES_GPU`
(unset for CPU; the fast tests need none), `MILES_IMAGE_TAG` (a `radixark/miles` tag; pinned in the script to the one
GPU CI builds from), `MILES_TEST_TIMEOUT` (seconds, default 1800), `HF_TOKEN` (forwarded when set).

Reused across runs: the image (Modal pulls the 24 GB `radixark/miles:dev-*` once, then caches it) and the
`miles-hf-cache` volume at `/hf-cache` (`HF_HOME`), so a model's tokenizer and processor files download once.
