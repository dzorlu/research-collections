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
    w = 1.  [B, S]
  """
  raise NotImplementedError


@register("ppo")
def ppo(logp_now, logp_old, logp_engine, A, low=0.8, high=1.28):
  """
    r = p_now / p_old.
    w = r; 0 where A > 0 and r > high, or A < 0 and r < low.  [B, S]
  """
  raise NotImplementedError


@register("cispo")
def cispo(logp_now, logp_old, logp_engine, A, cap=4.0):
  """
    r = p_now / p_old.
    w = clamp(r, 0, cap). Never 0, at most cap.  [B, S]
  """
  raise NotImplementedError


@register("tis")
def tis(logp_now, logp_old, logp_engine, A, cap=2.0):
  """
    rho = p_old / p_engine.
    w = clamp(rho, 0, cap) * ppo(logp_now, logp_old, logp_engine, A).  [B, S]
  """
  raise NotImplementedError
