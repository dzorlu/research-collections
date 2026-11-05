"""Reference solutions for backprop_practice_4.ipynb."""
import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def grad_check(net, X, Y, grads, params, h=1e-6, tol=1e-6):
    """Central differences on every entry of every parameter; compares with the analytic grads."""
    for name in params:
        P = getattr(net, name)
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + h
            lp = net.forward(X, Y)
            P[idx] = old - h
            lm = net.forward(X, Y)
            P[idx] = old
            num[idx] = (lp - lm) / (2 * h)
        g = grads[name]
        assert g.shape == P.shape, f"{name}: grad shape {g.shape} != param shape {P.shape}"
        rel = np.linalg.norm(num - g) / max(1e-12, np.linalg.norm(num) + np.linalg.norm(g))
        assert rel < tol, f"{name}: relative error {rel:.2e}"
        print(f"{name}: relative error {rel:.1e}  ok")


# --------------------------------------------------------------------------- problem 1
class GateNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.gamma = np.random.randn(hidden_dim)

    def forward(self, X, Y_target):
        self.X, self.Y = X, Y_target
        self.Z1 = X @ self.W1                                        # [B, H]
        self.H1 = np.tanh(self.Z1)                                   # [B, H]
        self.H1_scaled = self.H1 * self.gamma                        # [B, H]
        self.S = self.H1_scaled.sum(axis=1, keepdims=True)           # [B, 1]
        self.H2 = self.H1_scaled * self.S                            # [B, H]
        self.out = self.H2 @ self.W2                                 # [B, O]
        return np.mean((self.out - Y_target) ** 2)

    def backward(self):
        dout = 2 * (self.out - self.Y) / self.out.size               # [B, O]
        dW2 = self.H2.T @ dout                                       # [H, O]
        dH2 = dout @ self.W2.T                                       # [B, H]
        # H2 = H1_scaled * S: both factors depend on H1_scaled
        dS = (dH2 * self.H1_scaled).sum(axis=1, keepdims=True)       # [B, 1]
        dH1_scaled = dH2 * self.S + dS                               # [B, H]  (S broadcasts back over H)
        dgamma = (dH1_scaled * self.H1).sum(axis=0)                  # [H]
        dH1 = dH1_scaled * self.gamma
        dZ1 = dH1 * (1 - self.H1 ** 2)
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "gamma": dgamma}


# --------------------------------------------------------------------------- problem 2
class BatchNormNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.gamma = np.random.randn(hidden_dim)
        self.beta = np.random.randn(hidden_dim)
        self.eps = 1e-5

    def forward(self, X, y):
        self.X, self.y = X, y
        self.Z1 = X @ self.W1                                        # [B, H]
        self.H1 = np.logaddexp(0, self.Z1)                           # softplus, [B, H]
        self.mu = self.H1.mean(axis=0)                               # [H]
        self.Hc = self.H1 - self.mu                                  # [B, H]
        self.var = (self.Hc ** 2).mean(axis=0)                       # [H]
        self.std = np.sqrt(self.var + self.eps)                      # [H]
        self.Hn = self.Hc / self.std                                 # [B, H]
        self.H2 = self.Hn * self.gamma + self.beta                   # [B, H]
        logits = self.H2 @ self.W2                                   # [B, C]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        self.p = e / e.sum(axis=1, keepdims=True)
        B = X.shape[0]
        return -np.mean(np.log(self.p[np.arange(B), y]))

    def backward(self):
        B = self.X.shape[0]
        dlogits = self.p.copy()
        dlogits[np.arange(B), self.y] -= 1
        dlogits /= B                                                 # [B, C]
        dW2 = self.H2.T @ dlogits
        dH2 = dlogits @ self.W2.T                                    # [B, H]
        dgamma = (dH2 * self.Hn).sum(axis=0)
        dbeta = dH2.sum(axis=0)
        dHn = dH2 * self.gamma
        # Hn = Hc / std, std depends on Hc through var; Hc = H1 - mu
        dstd = -(dHn * self.Hc / self.std ** 2).sum(axis=0)          # [H]
        dvar = dstd / (2 * self.std)                                 # [H]
        dHc = dHn / self.std + dvar * 2 * self.Hc / B                # [B, H]
        dmu = -dHc.sum(axis=0)                                       # [H]
        dH1 = dHc + dmu / B                                          # [B, H]
        dZ1 = dH1 * sigmoid(self.Z1)                                 # softplus' = sigmoid
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "gamma": dgamma, "beta": dbeta}


# --------------------------------------------------------------------------- problem 3
class MaxResNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.alpha = np.random.randn(hidden_dim)

    def forward(self, X, y):
        self.X, self.y = X, y
        self.Z1 = X @ self.W1                                        # [B, H]
        self.H1 = self.Z1 * (self.Z1 > 0)                            # relu
        self.M = self.H1.max(axis=1, keepdims=True)                  # [B, 1]
        self.Hn = self.H1 / (self.M + 1e-6)                          # [B, H]
        self.Z2 = self.Hn @ self.W2                                  # [B, H]
        self.sg = sigmoid(self.Z2)
        self.H2 = self.Z2 * self.sg                                  # SiLU
        self.H3 = self.H2 * self.alpha + self.Hn                     # residual
        logits = self.H3 @ self.W3
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        self.p = e / e.sum(axis=1, keepdims=True)
        B = X.shape[0]
        return -np.mean(np.log(self.p[np.arange(B), y]))

    def backward(self):
        B = self.X.shape[0]
        dlogits = self.p.copy()
        dlogits[np.arange(B), self.y] -= 1
        dlogits /= B
        dW3 = self.H3.T @ dlogits
        dH3 = dlogits @ self.W3.T
        dalpha = (dH3 * self.H2).sum(axis=0)
        dH2 = dH3 * self.alpha
        dZ2 = dH2 * (self.sg + self.Z2 * self.sg * (1 - self.sg))   # SiLU' = s + z s (1 - s)
        dW2 = self.Hn.T @ dZ2
        dHn = dZ2 @ self.W2.T + dH3                                  # two paths: through W2 and the residual
        denom = self.M + 1e-6
        dH1 = dHn / denom                                            # direct path
        dM = -(dHn * self.H1 / denom ** 2).sum(axis=1, keepdims=True)    # [B, 1]
        argmax = self.H1.argmax(axis=1)
        dH1[np.arange(B), argmax] += dM[:, 0]                        # max routes its grad to the argmax only
        dZ1 = dH1 * (self.Z1 > 0)
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "W3": dW3, "alpha": dalpha}


# --------------------------------------------------------------------------- problem 4
class DebugNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.beta = np.random.randn(hidden_dim)

    def forward(self, X, Y_target):
        self.X, self.Y = X, Y_target
        self.Z1 = X @ self.W1
        self.H1 = sigmoid(self.Z1)
        self.Hs = self.H1 + self.beta
        self.mean = self.Hs.mean(axis=1, keepdims=True)              # [B, 1] row mean
        self.H2 = self.Hs - self.mean
        self.out = self.H2 @ self.W2
        return np.mean((self.out - Y_target) ** 2)

    def backward(self):
        B = self.X.shape[0]
        dout = 2 * (self.out - self.Y) / self.out.size
        dW2 = self.H2.T @ dout
        dH2 = dout @ self.W2.T
        dHs = dH2 - dH2.mean(axis=1, keepdims=True)                  # subtracting the row mean
        dbeta = dHs.sum(axis=0)
        dH1 = dHs
        dZ1 = dH1 * self.H1 * (1 - self.H1)
        dW1 = self.X.T @ dZ1
        return {"W1": dW1, "W2": dW2, "beta": dbeta}


def data(seed, B=6, D=5, O=3, classes=False):
    rng = np.random.RandomState(seed)
    X = rng.randn(B, D)
    Y = rng.randint(O, size=B) if classes else rng.randn(B, O)
    return X, Y


if __name__ == "__main__":
    for cls, classes, params in [(GateNet, False, ["W1", "W2", "gamma"]),
                                 (BatchNormNet, True, ["W1", "W2", "gamma", "beta"]),
                                 (MaxResNet, True, ["W1", "W2", "W3", "alpha"]),
                                 (DebugNet, False, ["W1", "W2", "beta"])]:
        np.random.seed(0)
        net = cls(5, 8, 3)
        X, Y = data(1, classes=classes)
        loss = net.forward(X, Y)
        print(cls.__name__, "loss", repr(round(float(loss), 10)))
        grad_check(net, X, Y, net.backward(), params)
