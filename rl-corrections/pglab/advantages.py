import torch

ADVANTAGES = {}


def register(name):
  def wrap(fn):
    ADVANTAGES[name] = fn
    return fn
  return wrap


# Every advantage has the same signature:
#   R:         [P, G]  reward of each response; row = one prompt, G responses to it
#   returns A: [P, G]  advantage = R - b, where b may not depend on that response's own tokens
#
# A baseline b adds no bias: the direction p - y_onehot averages to 0 over the model's own samples,
# so subtracting b removes b * 0 from the gradient and only changes the scatter.


@register("none")
def none(R):
  """
    No baseline.
    A = R

    Williams 1992, Machine Learning 8:229-256 (no arXiv).

    R:         [P, G]
    returns A: [P, G]
  """
  raise NotImplementedError


@register("mean")
def mean(R):
  """
    Group mean as the baseline, no std division.
    A = R - mean(R of this prompt's group)

    Dr. GRPO: https://arxiv.org/abs/2503.20783 (drops GRPO's std division, which biases the gradient)

    R:         [P, G]
    returns A: [P, G]
  """
  raise NotImplementedError


@register("loo")
def loo(R):
  """
    Leave-one-out mean: each response is compared with the other G - 1 responses only.
    A = R - (sum(R of group) - R) / (G - 1)

    RLOO: https://arxiv.org/abs/2402.14740
    The plain group mean includes the response itself, so its b depends on its own tokens;
    leaving it out keeps b independent of them. A = G / (G - 1) * (R - mean).

    R:         [P, G]
    returns A: [P, G]
  """
  raise NotImplementedError


@register("grpo")
def grpo(R):
  """
    Group mean as the baseline, divided by the group's std. What train.py uses today.
    A = (R - mean) / (std + 1e-6)

    GRPO: https://arxiv.org/abs/2402.03300
    The std division rescales each prompt's gradient by 1/std: low-spread prompts get boosted. Not a baseline,
    so it changes the average gradient (bias); see Dr. GRPO https://arxiv.org/abs/2503.20783.

    R:         [P, G]
    returns A: [P, G]
  """
  mean = R.mean(-1, keepdim=True)                      # [P, 1]
  std = R.std(-1, keepdim=True)                        # [P, 1]
  return (R - mean) / (std + 1e-6)                     # [P, G]  0 when every response got the same reward
