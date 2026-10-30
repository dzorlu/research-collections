# Sequence-level: GSPO and OPSM

PPO and the methods on page 02 compute one ratio per token. GSPO and OPSM compute one number per response
and apply it to all of the response's tokens.

## 1. One response, three tokens

```python
logp_now = [-0.20, -1.10, -2.30]
logp_old = [-0.25, -1.00, -2.00]

token_r = [exp(a - b) for a, b in zip(logp_now, logp_old)]        # [1.051, 0.905, 0.741]  PPO
seq_r   = exp(mean(a - b for a, b in zip(logp_now, logp_old)))    # 0.890                  GSPO
seq_kl  = mean(b - a for a, b in zip(logp_now, logp_old))         # 0.117                  OPSM
```

**Table 1. What each method does with this response.** PPO clip band `[0.8, 1.28]`; OPSM threshold
`--opsm-delta 1e-4` (miles default).

| Method | Number per token | If A > 0 | If A < 0 |
|---|---|---|---|
| PPO | own ratio: 1.051, 0.905, 0.741 | all three keep their gradient | token 3 (0.741 < 0.8) gets 0 |
| GSPO | shared ratio 0.890 for all three | all keep their gradient | all keep their gradient (0.890 > 0.8) |
| OPSM | shared KL 0.117 | no change | all three get 0 (0.117 > 1e-4) |

GSPO's shared ratio is the geometric mean of the token ratios. One outlier token cannot get clipped alone: the
whole response is inside or outside the band. OPSM drops a response that lost (A < 0) when the policy has
already moved away from it (KL above the threshold).

## 2. Code

miles main 34914c948f, `miles/backends/training_utils/loss/hub/`:
- GSPO: `--advantage-estimator gspo`. `math_utils.py:225-251` computes the per-response mean of
  `log p_old - log p_now` and copies it to every token; `losses.py:177-183` uses it in place of the per-token
  value, and the PPO clip (`math_utils.py:255-278`) runs on `exp(-mean)`.
- OPSM: `--use-opsm`, `--opsm-delta` (default 1e-4). `math_utils.py:213-216` sets the mask to 0 when
  `advantage < 0` and the response KL exceeds the threshold; `losses.py:230-231` multiplies it into the PPO term.

## 3. In our runs

Neither flag is set in our runs. Both use `p_now / p_old`. At our one optimizer step per rollout, `log p_old - log p_now` stays within
2.1e-5 of 0 (page 01, Table 3), below OPSM's 1e-4, so both would do nothing: GSPO's ratio would be within 2.1e-5 of 1, and OPSM's KL would stay below its threshold.
