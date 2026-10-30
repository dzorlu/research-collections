# Handoff: rl-corrections

Updated 2026-10-29. For the next agent picking up this folder.

## 1. What this folder is

Notes, built with Deniz over several sessions, on the policy-gradient corrections miles supports (PPO, CISPO, TIS,
MIS, IcePop, decoupled PPO, OPSM, GSPO, score centering), loss normalization, and the infra fixes for the
train/inference mismatch (TITO, R3). The goal is understanding, not a paper: Deniz wants to grok how each method
changes the gradient that reaches the logits.

Everything is organized around one formula, per response:

```python
dz = w * A * (y_hat - y_onehot) / norm      # (T, V): row t is token t's gradient on the logits
# y_hat (T, V) trainer probabilities; y_onehot (T, V) tokens the engine sampled
# w (T,) per-token weight; A () per response (GRPO) or (T,) per token; norm (T,)
```

Each method changes one term: `w` (section 2 of `corrections.md`), `A` (section 3), the direction
`y_hat - y_onehot` (section 4, score centering), `norm` (section 5).

## 2. Files

| File | What it is | State |
|---|---|---|
| `corrections.md` | The main map: three problems, one section per term, our runs, removing the mismatch at the source (vLLM post) | current |
| `score_centering.md` | TLDR, then a one-state walkthrough with numbers (per-pick `dz`, average `dz`, drift vs signal, `s_bar`, the subtracted term, baseline table) | current; Deniz said this version is clear |
| `score_centering.ipynb` | PyTorch version of that walkthrough; every `dz` checked by autograd | current |
| `normalization.ipynb` | One 4-token input, the logits gradient under each divisor (token, miles default, GRPO, Dr. GRPO) | current |
| `corrections.ipynb` | Every method on one toy batch (B=2, T=4, V=3), per-token `w` from autograd | current |
| `viz/clipping_network.html` | Interactive network: pick a method, step, watch weights and kept/capped/dropped tokens | current; open locally (private repo, GitHub shows source) |
| `01-multi-step-clipping/` | Older PPO-clipping writeup (`README.md`, measured clip fractions), `cispo.md`, `cispo_simple.ipynb` (PyTorch, forward/backward), `cispo_backward.ipynb` | current |
| `02-train-inference-mismatch/`, `03-sequence-level/`, `04-score-centering/` | Earlier per-group pages | older, partly superseded by `corrections.md` and `score_centering.md` |
| `README.md` | Index and TODO list | current |

## 3. Notation (use exactly this)

- `y_hat`: trainer probabilities (softmax of its logits `z`). `q`: inference-engine (SGLang) probabilities.
- `y_onehot` / `onehot(k)`: one-hot of the sampled token `k`. `A = R - b`; `b` = GRPO group-mean reward.
- `r = p_now / p_old` (staleness), `rho = p_old / p_engine` (mismatch), `c` = importance weight built from `rho`.
- `A_avg` = advantage averaged over the engine's picks at one state; `s_bar = q - y_hat` = average score.
- `drift = A_avg * (y_hat - q)`, `signal = -(q * (A - A_avg))` (both in `dz`'s sign).
- Three problems: staleness, mismatch, drift. "Drift" means only the score-centering paper's term. Do not call
  staleness "drift".
- Toy state used everywhere: `y_hat = [0.5, 0.3, 0.2]`, `q = [0.6, 0.3, 0.1]`, `R = [1, 0, 0]`, `b = 0.25`
  (4 rollouts, 1 correct), `A = [0.75, -0.25, -0.25]`, `A_avg = 0.35`.

## 4. Working with Deniz (rules he set)

- Push every change straight to `main` here, no PRs, no asking first.
- Ground every explanation in these docs: same symbols, same toy numbers, cite the section. Simple language. Never
  introduce a symbol (`g`, covariance, `E_q[...]` brackets) without defining it from this notation first.
- Every tensor line in a code box carries a shape comment: `# (T, V)`, `# (V,)`, `# ()`. This includes chat replies.
- Notebooks: PyTorch (not JAX), no loops in demo cells, no `[..., None]` indexing (use `.reshape(-1, 1)` or `.view`),
  one cell per method where possible.
- Docs must be self-contained; numbered sections and captioned tables; no em dashes; follow `~/repos/CLAUDE.md`.
- Verify every number and sign in PyTorch before writing it. Training subtracts `dz`; sign errors happened twice.
- Before a large rewrite, explain the content in chat first; he asked "before you rewrite, explain it to me".
- When he says "keep it clean", do not restructure.

## 5. Open items

1. Paper Table 1 grid (score-centering paper, Appendix B.2) is not in `corrections.md`. Waiting on Deniz's go.
   Proposed: one grid table (level, statistic, bounds, clip/drop, sign), entries for naive IS, TOPR (negative-only),
   DPPO (total variation), GSPO's paper band 0.9997 to 1.0004 (doc uses 0.8 to 1.28), and the paper's CISPO cap of 2
   (doc uses 4, the miles/Tinker default).
2. TODO in `README.md`: sparse-attention indexer mismatch (GLM-5.3 has 11 DSA layers; miles
   `--use-rollout-indexer-replay` exists; our recipes do not replay it) and kernel mismatch (engine
   decode-vs-prefill self-gap 0.031 to 0.033 on GLM-5.3). Deniz wants to study attention mechanisms first; see
   `../attention/` (existing note and notebook on GLM-5.3-Flash).
3. Deniz may ask for the score-centering material to be folded further into `corrections.md` section 4; keep the
   walkthrough in `score_centering.md` as the derivation.

## 6. Verified facts worth reusing

- miles (radixark/miles main, 4250360dfc on 2026-10-29): PPO `math_utils.py:255`; TIS `corrections.py:7`; IcePop
  `corrections.py:35` (band from `--tis-clip-low`, `--tis-clip`, defaults 0 and 2); CISPO `tinker_losses.py:192`
  (Tinker only; divides by the engine's log-prob); OPSM `math_utils.py:184`; GSPO `math_utils.py:225`; score
  centering `score_centering/estimator.py:66-90`; normalization `data/context_parallel.py:247` (per-response mean
  by default, `--calculate-per-token-loss` for per token); R3 doc `docs/advanced/miles-router.md` (replays expert
  indices only; gate weights still come from the trainer's router).
- Our launcher's recipes: PPO band 0.8 to 1.28; `--use-tis` on the GLM-5.3
  torchtitan recipe; R3 on for GLM-5.3 and Muse, off for Qwen3.8; `--calculate-per-token-loss` only on Muse; no recipe uses
  IcePop, MIS or score centering.
- Mismatch on our recent agent training runs (`train/train_rollout_logprob_abs_diff`): Qwen3.8 0.012 to
  0.013 (no correction), GLM-5.3 0.021 to 0.031 (R3 + TIS; TIS caps 0.03% of tokens), Muse 0.006 to 0.008 (R3 +
  TIS). All flat over training. R3 on a GLM 4-layer slice: 0.0714 to 0.0335.
- Score-centering paper: Marek and Ryabinin, arXiv 2609.20807. Ring-1T / IcePop: arXiv 2510.18855. vLLM consistency
  post: https://vllm-project.github.io/2026/10/14/train-inference-consistency.html. SFC blog:
  https://huggingface.co/blog/NormalUhr/sfc. CISPO: MiniMax-M1, arXiv 2506.13585.

## 7. Gotchas

- Notebooks were generated by builder scripts in `/tmp` that no longer exist. Edit the `.ipynb` files directly, then
  run `jupyter nbconvert --to notebook --execute --inplace <file>` before committing.
- An open Jupyter tab can autosave an old version over a regenerated notebook. Ask Deniz to close tabs without saving
  before reopening.
- `.ipynb_checkpoints/` is gitignored; `parallelism/glm53-titan-run-layout.html` is untracked and not ours, leave it.
