"""Reference solutions for backprop_practice_4.ipynb (PyTorch, backward by hand)."""
import torch

torch.set_default_dtype(torch.float64)


def check(net, X, Y, expected_loss, params):
    """Forward loss must match; hand-written grads must match autograd run on your own forward."""
    with torch.no_grad():
        loss = net.forward(X, Y)
        assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-8), \
            f"forward: loss {loss.item():.10f}, expected {expected_loss:.10f}"
        grads = net.backward()
    for name in params:
        getattr(net, name).requires_grad_(True)
    net.forward(X, Y).backward()
    for name in params:
        P = getattr(net, name)
        g = grads[name]
        assert g.shape == P.shape, f"{name}: grad shape {tuple(g.shape)} != param shape {tuple(P.shape)}"
        err = (g - P.grad).abs().max().item()
        assert torch.allclose(g, P.grad, atol=1e-10), f"{name}: max abs error {err:.2e}"
        print(f"{name}: max abs error {err:.1e}  ok")
        P.requires_grad_(False)
        P.grad = None


def data(seed, B=6, D=5, O=3, classes=False):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, D, generator=g)                                       # [B, D]
    Y = torch.randint(O, (B,), generator=g) if classes else torch.randn(B, O, generator=g)
    return X, Y


# --------------------------------------------------------------------------- problem 1
class GateNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim) / input_dim ** 0.5
        self.W2 = torch.randn(hidden_dim, output_dim) / hidden_dim ** 0.5
        self.gamma = torch.randn(hidden_dim)

    def forward(self, X, Y_target):
        self.X, self.Y = X, Y_target
        self.Z1 = X @ self.W1                                                # [B, H]
        self.H1 = torch.tanh(self.Z1)                                        # [B, H]
        self.H1_scaled = self.H1 * self.gamma                                # [B, H]
        self.S = self.H1_scaled.sum(dim=1, keepdim=True)                     # [B, 1]
        self.H2 = self.H1_scaled * self.S                                    # [B, H]
        logits = self.H2 @ self.W2                                           # [B, C]
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        self.p = logp.exp()
        return -logp[torch.arange(X.shape[0]), Y_target].mean()

    def backward(self):
        B = self.X.shape[0]
        dlogits = self.p.clone()
        dlogits[torch.arange(B), self.Y] -= 1
        dlogits /= B                                                         # [B, C]
        dW2 = self.H2.T @ dlogits                                            # [H, C]
        dH2 = dlogits @ self.W2.T                                            # [B, H]
        dS = (dH2 * self.H1_scaled).sum(dim=1, keepdim=True)                 # [B, 1]
        dH1_scaled = dH2 * self.S + dS                                       # two paths into H1_scaled
        dgamma = (dH1_scaled * self.H1).sum(dim=0)                           # [H]
        dZ1 = dH1_scaled * self.gamma * (1 - self.H1 ** 2)
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "gamma": dgamma}


# --------------------------------------------------------------------------- problem 2
class BatchNormNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim) / input_dim ** 0.5
        self.W2 = torch.randn(hidden_dim, output_dim) / hidden_dim ** 0.5
        self.gamma = torch.randn(hidden_dim)
        self.beta = torch.randn(hidden_dim)
        self.eps = 1e-5

    def forward(self, X, y):
        self.X, self.y = X, y
        self.Z1 = X @ self.W1                                                # [B, H]
        self.H1 = torch.nn.functional.softplus(self.Z1)                      # [B, H]
        self.mu = self.H1.mean(dim=0)                                        # [H]
        self.Hc = self.H1 - self.mu                                          # [B, H]
        self.var = (self.Hc ** 2).mean(dim=0)                                # [H]
        self.std = (self.var + self.eps).sqrt()                              # [H]
        self.Hn = self.Hc / self.std                                         # [B, H]
        self.H2 = self.Hn * self.gamma + self.beta                           # [B, H]
        logits = self.H2 @ self.W2                                           # [B, C]
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        self.p = logp.exp()
        return -logp[torch.arange(X.shape[0]), y].mean()

    def backward(self):
        B = self.X.shape[0]
        dlogits = self.p.clone()
        dlogits[torch.arange(B), self.y] -= 1
        dlogits /= B                                                         # [B, C]
        dW2 = self.H2.T @ dlogits
        dH2 = dlogits @ self.W2.T                                            # [B, H]
        dgamma = (dH2 * self.Hn).sum(dim=0)
        dbeta = dH2.sum(dim=0)
        dHn = dH2 * self.gamma
        dstd = -(dHn * self.Hc / self.std ** 2).sum(dim=0)                   # [H]
        dvar = dstd / (2 * self.std)                                         # [H]
        dHc = dHn / self.std + dvar * 2 * self.Hc / B                        # [B, H]
        dmu = -dHc.sum(dim=0)                                                # [H]
        dH1 = dHc + dmu / B                                                  # [B, H]
        dZ1 = dH1 * torch.sigmoid(self.Z1)                                   # softplus' = sigmoid
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "gamma": dgamma, "beta": dbeta}


# --------------------------------------------------------------------------- problem 3
class MaxResNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim) / input_dim ** 0.5
        self.W2 = torch.randn(hidden_dim, hidden_dim) / hidden_dim ** 0.5
        self.W3 = torch.randn(hidden_dim, output_dim) / hidden_dim ** 0.5
        self.alpha = torch.randn(hidden_dim)

    def forward(self, X, y):
        self.X, self.y = X, y
        self.Z1 = X @ self.W1                                                # [B, H]
        self.H1 = torch.relu(self.Z1)
        self.M = self.H1.max(dim=1, keepdim=True).values                     # [B, 1]
        self.Hn = self.H1 / (self.M + 1e-6)                                  # [B, H]
        self.Z2 = self.Hn @ self.W2                                          # [B, H]
        self.sg = torch.sigmoid(self.Z2)
        self.H2 = self.Z2 * self.sg                                          # SiLU
        self.H3 = self.H2 * self.alpha + self.Hn                             # residual
        logits = self.H3 @ self.W3
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        self.p = logp.exp()
        return -logp[torch.arange(X.shape[0]), y].mean()

    def backward(self):
        B = self.X.shape[0]
        dlogits = self.p.clone()
        dlogits[torch.arange(B), self.y] -= 1
        dlogits /= B
        dW3 = self.H3.T @ dlogits
        dH3 = dlogits @ self.W3.T
        dalpha = (dH3 * self.H2).sum(dim=0)
        dZ2 = dH3 * self.alpha * (self.sg + self.Z2 * self.sg * (1 - self.sg))
        dW2 = self.Hn.T @ dZ2
        dHn = dZ2 @ self.W2.T + dH3                                          # through W2, plus the residual
        denom = self.M + 1e-6
        dH1 = dHn / denom
        dM = -(dHn * self.H1 / denom ** 2).sum(dim=1)                        # [B]
        dH1[torch.arange(B), self.H1.argmax(dim=1)] += dM                    # max routes grad to the argmax
        dZ1 = dH1 * (self.Z1 > 0)
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "W3": dW3, "alpha": dalpha}


# --------------------------------------------------------------------------- problem 4
class DebugNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim) / input_dim ** 0.5
        self.W2 = torch.randn(hidden_dim, output_dim) / hidden_dim ** 0.5
        self.beta = torch.randn(hidden_dim)

    def forward(self, X, Y_target):
        self.X, self.Y = X, Y_target
        self.Z1 = X @ self.W1
        self.H1 = torch.sigmoid(self.Z1)
        self.Hs = self.H1 + self.beta
        self.H2 = self.Hs - self.Hs.mean(dim=1, keepdim=True)
        logits = self.H2 @ self.W2
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        self.p = logp.exp()
        return -logp[torch.arange(X.shape[0]), Y_target].mean()

    def backward(self):
        B = self.X.shape[0]
        dlogits = self.p.clone()
        dlogits[torch.arange(B), self.Y] -= 1
        dlogits /= B                                                         # fix 1: the loss is a mean over B
        dW2 = self.H2.T @ dlogits
        dH2 = dlogits @ self.W2.T
        dHs = dH2 - dH2.mean(dim=1, keepdim=True)                            # fix 2: through the row centering
        dbeta = dHs.sum(dim=0)
        dZ1 = dHs * self.H1 * (1 - self.H1)                                  # fix 3: sigmoid' = s (1 - s)
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "beta": dbeta}


PROBLEMS = [(GateNet, True, ["W1", "W2", "gamma"]),
            (BatchNormNet, True, ["W1", "W2", "gamma", "beta"]),
            (MaxResNet, True, ["W1", "W2", "W3", "alpha"]),
            (DebugNet, True, ["W1", "W2", "beta"])]

if __name__ == "__main__":
    for cls, classes, params in PROBLEMS:
        torch.manual_seed(0)
        net = cls(5, 8, 3)
        X, Y = data(1, classes=classes)
        with torch.no_grad():
            print(cls.__name__, "loss", f"{net.forward(X, Y).item():.10f}")
        check(net, X, Y, net.forward(X, Y).item(), params)
