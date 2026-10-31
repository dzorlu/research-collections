# research-collections

Personal collection of worked-through research notes. Each topic gets a
directory with a human-readable writeup and, where useful, a runnable
notebook.

- [attention/](attention/): where memory and compute actually go in modern
  attention mechanisms (standard, FlashAttention, MLA, DSA, KDA), worked
  through on GLM-5.3-Flash's real dimensions. Includes a line-by-line
  notebook.
- [parallelism/](parallelism/): how tensor, pipeline and expert parallelism
  interact when training a mixture-of-experts model, worked through on
  GLM-5.3-Flash across 48 GPUs. Includes the memory arithmetic that explains
  a real out-of-memory failure and why more pipeline stages fixed it.
- [distillation/](distillation/): on-policy distillation from specialist
  teachers (MOPD) and why it uses reverse KL, worked through on a three-token
  example: SFT loss as teacher entropy plus forward KL, the per-token
  advantage, and why the loss is −A·log p. Includes a script that reproduces
  every number, and a notebook
  ([open in Colab](https://colab.research.google.com/github/dzorlu/research-collections/blob/main/distillation/opd_backward.ipynb))
  with a hand-written OPD forward and backward pass through a two-layer
  network, checked against PyTorch.
- [rl-corrections/](rl-corrections/): the policy-loss corrections miles supports (PPO clip, CISPO,
  TIS, IcePop/MIS, decoupled and bypass PPO, GSPO, OPSM, score centering), each written as the one
  factor it changes in the per-token logits gradient `w * A * (p - y_onehot) / N`, with the miles
  file:line and whether our runs use it. Includes a notebook that runs PPO and CISPO by hand through a
  two-layer network
  ([open in Colab](https://colab.research.google.com/github/dzorlu/research-collections/blob/main/rl-corrections/01-multi-step-clipping/cispo_backward.ipynb)).
- [upstream-prs/](upstream-prs/): our open PRs to miles and SGLang, grouped by project, with what
  each one fixes and where it stands upstream.
- [single-node-training/](single-node-training/): which open-weight model is the best one to train on one
  8×B200 node (full-parameter Qwen3.8-27B; LoRA MiMo-V2.6-Flash), what 256K context needs there, and
  the state of long-horizon training with screenshot folding, a policy-written summary and the TITO
  session server.
- [miles-harness/](miles-harness/): build spec for a long-horizon agent RL harness on upstream miles (session
  server, agent and reward hooks, screenshot folding, policy-written summaries, eval, test gates), plus
  per-model recipes as miles flags for Qwen3.8-27B, GLM-5.3-Flash and Muse Spark Open.
- [rl-infra/](rl-infra/): getting the most agent RL training out of a few GPUs: fully async training on
  shared GPUs (how it works, flags miles refuses, loss correction, harness rules, run checks), a run plan
  for Qwen3.8-27B on one 8×B200 node at 96K, and memory and precision levers (FP8 engine weights, bf16
  Adam moments, FP8 KV).
