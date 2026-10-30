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
