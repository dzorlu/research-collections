import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
  def __init__(self, input_dim, hidden_dims, vocab_size):
    """
      hidden_dims: one entry per hidden layer, e.g. [128] or [128, 128].
    """
    super().__init__()
    self.vocab_size = vocab_size
    dims = [input_dim, *hidden_dims, vocab_size]      # e.g. [I, H, V]

    # One weight matrix per layer: weights[i] is [dims[i], dims[i+1]].
    # Divide by sqrt(fan_in) so each layer's output is about size 1.
    weights = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
      W = torch.randn(d_in, d_out) / d_in ** 0.5      # [in, out]
      weights.append(nn.Parameter(W))
    self.weights = nn.ParameterList(weights)          # registered: .parameters(), .to(device), state_dict()

  def is_last(self, i):
    return i == len(self.weights) - 1

  def forward(self, x):
    """
      x: [B, S, I]
      Sets self.logits [B, S, V], self.p [B, S, V], and self.cache (what backward needs, one entry per layer).
    """
    a = x                                             # [B, S, I]  the input is layer 0's activation
    self.cache = []
    for i, W in enumerate(self.weights):
      s = a @ W                                       # [B, S, out]  pre-activation

      # backward needs this layer's input (for dW) and its pre-activation (for relu').
      self.cache.append({'a_in': a, 's': s})

      if self.is_last(i):
        a = s                                         # [B, S, V]  no relu on the logits
      else:
        a = s * (s > 0)                               # [B, S, out]  relu

    self.logits = a                                   # [B, S, V]
    self.p = torch.softmax(self.logits, -1)           # [B, S, V]
    return self.p

  def loss(self, out, y):
    """
    out: [B, S, V]
    y: [B, S]
    """
    y_ = y.unsqueeze(-1)
    out_ = out.gather(-1, y_) # [B, S, 1]
    return torch.mean(-torch.log(out_))

  def backward_(self, dlogits):
    """
      dlogits: [B, S, V]. Changes per method; everything below the logits does not.
      Walks the layers top to bottom. Per layer: through relu, weight gradient, gradient for the layer below.
    """
    ds = dlogits                                      # [B, S, V]  gradient at the last pre-activation
    self.gradients = [None] * len(self.weights)

    for i in reversed(range(len(self.weights))):
      W = self.weights[i]                             # [in, out]
      a_in = self.cache[i]['a_in']                    # [B, S, in]
      s = self.cache[i]['s']                          # [B, S, out]

      # 1. Through relu: pass the gradient only where the unit was on. (The logits have no relu.)
      if not self.is_last(i):
        ds = ds * (s > 0)                             # [B, S, out]

      # 2. Weight gradient: this layer's input times the gradient at its output, summed over all B*S positions.
      a_in_flat = a_in.reshape(-1, W.shape[0])        # [B*S, in]
      ds_flat = ds.reshape(-1, W.shape[1])            # [B*S, out]
      self.gradients[i] = a_in_flat.T @ ds_flat       # [in, out]

      # 3. Gradient for the layer below: at this layer's input, which is the previous layer's activation.
      ds = ds @ W.T                                   # [B, S, in]

  def backward(self, w, A, y, q=None):
    """
      w: token weights. [B, S]
      A: advantage. [B, S]
      y: token ids. [B, S]
      q: the engine's distribution at each position, [B, S, V], or None.

      Plain:     dlogits = w * A * (p - y_onehot) / (B * S)
      Centered:  dlogits = w * A * (q - y_onehot) / (B * S)

      # p [B, S, V]

      The score y_onehot - p averages to 0 only when the trainer draws. The engine draws with q, so the average
      over its draws is q - p: a push of p toward q at every position, scaled by the mean advantage, blind to which
      token was good. Swapping p for q makes the average 0 under the engine and leaves the covariance with A as is.
      Orthogonal to w: w fixes the sampling distribution per token through a ratio, this fixes the direction.
      Score centering: https://arxiv.org/abs/2609.20807, Eq. 4-5. The paper's top-k tail ratio is not needed here:
      V = 10 and the engine's softmax is fully known.
    """
    B, S, V = self.p.shape
    y_one_hot = F.one_hot(y, V)                       # [B, S, V]
    w_ = w.unsqueeze(-1)                              # [B, S, 1]
    A_ = A.unsqueeze(-1)                              # [B, S, 1]
    # centering
    center = self.p if q is None else q
    dlogits = w_ * A_ * (center - y_one_hot) / (B * S)   # [B, S, V]  / (B * S): the loss is a mean over positions
    self.backward_(dlogits)

  @torch.no_grad()
  def step(self, lr=0.1):
    # Step against the gradient.
    for W, dW in zip(self.weights, self.gradients):
      W -= lr * dW                                    # [in, out]
