# Train-inference mismatch: TIS, IcePop, MIS, decoupled and bypass PPO

The SGLang engine samples a token with probability `p_engine`. The trainer, with the same weights, computes
`p_old` for the same token. The two differ because the engine and trainer use different kernels and precision:
on our GLM-5.3 torchtitan recipe the mean absolute log-prob gap at step 0 is 0.050.
The methods on this page weight each token by `p_old / p_engine`.

## 1. Per-token weights

**Table 1. Four tokens with a trainer/engine gap, and the weight each method gives them.** Bounds: TIS
`[0, 2]` (miles `--tis-clip-low 0 --tis-clip 2.0`), IcePop `[0, 2]` (same flags), MIS `[0.5, 5]`
(score-centering defaults).

| Token | p_old | p_engine | ratio | TIS | IcePop | MIS |
|---|---|---|---|---|---|---|
| 1 | 0.30 | 0.30 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.50 | 0.45 | 1.111 | 1.111 | 1.111 | 1.111 |
| 3 | 0.020 | 0.005 | 4.000 | 2.000 | 0.000 | 4.000 |
| 4 | 0.90 | 0.92 | 0.978 | 0.978 | 0.978 | 0.978 |

```python
ratio  = [p_old / p_engine for each token]
tis    = [min(max(r, 0.0), 2.0) for r in ratio]        # cap
icepop = [r if 0.0 <= r <= 2.0 else 0.0 for r in ratio] # drop outside the band
mis    = [r if 0.5 <= r <= 5.0 else 0.0 for r in ratio] # same rule as IcePop, other bounds
```

Token 3 is the case that matters: the engine gave it 4x less probability than the trainer. TIS keeps it at
weight 2. IcePop drops it. IcePop and token-level MIS apply the same rule with different bounds.

All weights are computed from detached log-probs (`losses.py:132-135`), so they scale each token's gradient
and receive no gradient themselves, like CISPO's weight on page 01.

## 2. Where the weight enters the loss

**Table 2. The three loss setups, from the flags `--use-rollout-logprobs` and `--use-tis`
(miles `examples/infra_features/train_infer_mismatch_helper/README.md`).**

| `--use-rollout-logprobs` | `--use-tis` | Name | Per-token loss | Our runs |
|---|---|---|---|---|
| no | no | Standard PPO | `PPO(p_now / p_old)` | Qwen rows |
| yes | no | Bypass PPO | `PPO(p_now / p_engine)`; trainer recompute skipped | none |
| no | yes | Decoupled PPO | `TIS(p_old / p_engine) * PPO(p_now / p_old)` | our GLM-5.3 torchtitan recipe |

`PPO(r)` is the clipped term `-min(r * A, clip(r, 0.8, 1.28) * A)`. In decoupled PPO the clip anchors on the
trainer's own `p_old` (the "proximal" policy), and the TIS weight corrects for the engine having sampled the
token (the "behavior" policy).

Code (miles main 34914c948f, `miles/backends/training_utils/loss/hub/`):
- `losses.py:137-140` picks `p_old`: engine log-probs with `--use-rollout-logprobs`, trainer log-probs otherwise.
- `losses.py:234-265` applies the weight function to the PPO term when `--use-tis` is set.
- `corrections.py:7-32` is the default weight function (TIS): `clamp(exp(log p_old - log p_engine), tis_clip_low, tis_clip)`.
- `corrections.py:35-62` is IcePop, selected with `--custom-tis-function-path`.
- `examples/infra_features/train_infer_mismatch_helper/mis.py` adds MIS (`tis_mode: mask`), sequence and
  geometric-mean weights, rejection sampling and a veto threshold, configured by `mis.yaml`.

## 3. In our runs

Our GLM-5.3 torchtitan recipe passes `--use-tis` without `--use-rollout-logprobs`, so it runs
decoupled PPO with weights capped at 2. Because we take one optimizer step per rollout, the PPO part has
`p_now = p_old` and clips nothing; the TIS weight is the only correction acting. The Qwen rows run standard PPO
with no trainer/engine correction.
