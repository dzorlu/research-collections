import torch

METHODS = {}


def register(name):
  def wrap(fn):
    METHODS[name] = fn
    return fn
  return wrap


# Every method has the same signature. Inputs are log-probs of the written tokens:
#   logp_now:    [B, S]  trainer, current weights (changes with each minibatch step)
#   logp_old:    [B, S]  trainer, recomputed before the first step on this rollout (fixed)
#   logp_engine: [B, S]  the engine that sampled (lagged weights + noise)
#   A:           [B, S]  advantage, the same for every token of a response
#   returns w:   [B, S]  per-token weight in  dlogits = w * A * (p - y_onehot) / (B * S)
#
# Ratios from log-probs: p_a / p_b = (logp_a - logp_b).exp()
#   staleness: p_now / p_old     1 on the first minibatch, drifts with each step
#   mismatch:  p_old / p_engine  fixed for the rollout; contains lag and noise


@register("reinforce")
def reinforce(logp_now, logp_old, logp_engine, A):
  """
    No correction: every token at full strength.
    w = 1

    Williams 1992, Machine Learning 8:229-256 (no arXiv).

    logp_now:    [B, S]
    logp_old:    [B, S]
    logp_engine: [B, S]
    A:           [B, S]
    returns w:   [B, S]
  """
  return torch.ones_like(A)


@register("ppo")
def ppo(logp_now, logp_old, logp_engine, A, low=0.8, high=1.28):
  """
    r = p_now / p_old
    w = r; 0 where A > 0 and r > high, or A < 0 and r < low

    https://arxiv.org/abs/1707.06347 (PPO). high = 1.28 is DAPO's clip-higher: https://arxiv.org/abs/2503.14476

    logp_now:    [B, S]
    logp_old:    [B, S]
    logp_engine: [B, S]
    A:           [B, S]
    returns w:   [B, S]
  """
  r = (logp_now - logp_old).exp()
  w = r
  w[(r > high) & (A > 0)] = 0
  w[(r < low) & (A < 0)] = 0
  return w

  


@register("cispo")
def cispo(logp_now, logp_old, logp_engine, A, cap=4.0):
  """
    r = p_now / p_old
    w = clamp(r, 0, cap). Never 0, at most cap.

    https://arxiv.org/abs/2506.13585 (MiniMax-M1).

    logp_now:    [B, S]
    logp_old:    [B, S]
    logp_engine: [B, S]
    A:           [B, S]
    returns w:   [B, S]
  """
  r = (logp_now - logp_old).exp()
  return torch.clamp(r, 0, cap)


@register("tis")
def tis(logp_now, logp_old, logp_engine, A, cap=2.0):
  """
    rho = p_old / p_engine
    w = clamp(rho, 0, cap) * ppo(logp_now, logp_old, logp_engine, A)

    Two ratios (decoupled PPO): https://arxiv.org/abs/2110.00641
    Truncation (TIS): https://fengyao.notion.site/off-policy-rl (blog, no arXiv). miles --use-tis.

    logp_now:    [B, S]
    logp_old:    [B, S]
    logp_engine: [B, S]
    A:           [B, S]
    returns w:   [B, S]
  """
  rho = (logp_old - logp_engine).exp()
  return torch.clamp(rho, 0, cap) * ppo(logp_now, logp_old, logp_engine, A)


@register("gspo")
def gspo(logp_now, logp_old, logp_engine, A, low=0.8, high=1.28):
  """
    One ratio per response instead of per token: the geometric mean of r over the S tokens.
    s = exp(mean over S of (logp_now - logp_old))    [B, 1]
    w = s for every token of the response; 0 for the whole response
        where A > 0 and s > high, or A < 0 and s < low (PPO's rule, applied to s)

    https://arxiv.org/abs/2507.18071 (GSPO). The paper's band is 1 - 3e-4 to 1 + 4e-4;
    corrections.md uses 0.8 to 1.28 so it compares with ppo.

    logp_now:    [B, S]
    logp_old:    [B, S]
    logp_engine: [B, S]
    A:           [B, S]
    returns w:   [B, S]
  """
  raise NotImplementedError


@register("dppo")
def dppo(logp_now, logp_old, logp_engine, A, delta=0.15):
  """
    Trust region on how far the probability moved, not on the ratio. Anchored to the engine.
    r = p_now / p_engine
    D = |p_now - p_engine|                            binary total variation of the written token
    w = r; 0 where A > 0 and r > 1 and D > delta, or A < 0 and r < 1 and D > delta

    https://arxiv.org/abs/2602.04879 (DPPO, Binary-TV). Paper delta: 0.15 to 0.2 for TV.

    logp_now:    [B, S]
    logp_old:    [B, S]  unused: DPPO skips the recompute
    logp_engine: [B, S]
    A:           [B, S]
    returns w:   [B, S]
  """
  raise NotImplementedError
