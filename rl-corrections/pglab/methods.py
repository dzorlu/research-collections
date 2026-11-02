import torch

METHODS = {}


def register(name):
  def wrap(fn):
    METHODS[name] = fn
    return fn
  return wrap


# Every method has the same signature:
#   staleness_ratio: [B, S]  p_now / p_old     trainer now vs trainer at rollout; 1 on the first of the K steps
#   mismatch_ratio:  [B, S]  p_old / p_engine  trainer vs the engine that sampled (old weights + noise)
#   A:               [B, S]  advantage, the same for every token of a response
#   returns w:       [B, S]  per-token weight in  gradient = w * A * (p - y_onehot) / (B * S)


@register("reinforce")
def reinforce(staleness_ratio, mismatch_ratio, A):
  """
    No correction: every token at full strength.
    w = 1.  [B, S]
  """
  raise NotImplementedError


@register("ppo")
def ppo(staleness_ratio, mismatch_ratio, A, low=0.8, high=1.28):
  """
    w = staleness_ratio; 0 where A > 0 and staleness_ratio > high, or A < 0 and staleness_ratio < low.  [B, S]
  """
  raise NotImplementedError


@register("cispo")
def cispo(staleness_ratio, mismatch_ratio, A, cap=4.0):
  """
    w = clamp(staleness_ratio, 0, cap). Never 0, at most cap.  [B, S]
  """
  raise NotImplementedError


@register("tis")
def tis(staleness_ratio, mismatch_ratio, A, cap=2.0):
  """
    w = clamp(mismatch_ratio, 0, cap) * ppo(staleness_ratio, mismatch_ratio, A).  [B, S]
  """
  raise NotImplementedError
