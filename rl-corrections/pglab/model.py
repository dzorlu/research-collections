import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
  def __init__(self, input_dim, hidden_dim, vocab_size):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size

    self.w1 = nn.Parameter(
        torch.randn(input_dim, hidden_dim) / input_dim ** 0.5 # [I, H]
    )
    self.w2 = nn.Parameter(
        torch.randn(hidden_dim, vocab_size) / hidden_dim ** 0.5 # [H, V]
    )

  def forward(self, x):
    """
      x: [B, S, I]
    """
    s1 = x @ self.w1 # [B, S, H]
    a1 = s1 * (s1 > 0) # [B, S, H]
    s2 = a1 @ self.w2 # [B, S, V]

    # softmax
    a2 = torch.softmax(s2, -1)
    self.acts = {'s1': s1, 'a1': a1, 's2': s2, 'a2': a2, 'x': x}
    return a2

  def loss(self, out, y):
    """
    out: [B, S, V]
    y: [B, S]
    """
    y_ = y.unsqueeze(-1)
    out_ = out.gather(-1, y_) # [B, S, 1]
    return torch.mean(-torch.log(out_))

  def backward_(self, gradient):
    # [B, S, V]
    # common backward. gradient changes.
    a1 = self.acts['a1'] # [B, S, H]

    ds2 = gradient.reshape(-1, self.vocab_size) # [BxS, V]
    a1_ = a1.reshape(-1, self.hidden_dim) # [BxS, H]
    dw2 = (a1_.T @ ds2) #[H,V]

    da1 = gradient @ self.w2.T # [B, S, V] x [V, H] -> # [B, S, H]
    s1 = self.acts['s1'] # [B, S, H]

    # ReLU
    ds1 = da1 * (s1 > 0) # [B, S, H]

    x = self.acts['x'] # [B, S, I]
    dw1 = x.reshape(-1, self.input_dim).T @ ds1.reshape(-1, self.hidden_dim)
    self.gradients = {'dw1': dw1, 'dw2': dw2}


  def backward(self, w, A, y):
    """
      w: token weights. [B, S]
      A: advantage. [B, S]
      y: token ids. [B, S]
    """
    # Gradient calculation
    p = self.acts['a2'] # [B, S, V]
    b, s = p.shape[0], p.shape[1]
    w_ = w.unsqueeze(-1)
    A = A.unsqueeze(-1)

    y_one_hot = F.one_hot(y, self.vocab_size)
    # gradient
    gradient = w_ * A * (p - y_one_hot) / (b * s)
    self.backward_(gradient)

  @torch.no_grad()
  def step(self, lr=0.1):
    self.w1 -= self.gradients['dw1'] * lr
    self.w2 -= self.gradients['dw2'] * lr
