"""Reference solution for 03_attention.ipynb: every cell, with the TODO cells filled in."""

import inspect
import torch
import torch.nn.functional as F


def mha(x, Wq, Wk, Wv, Wo, n_heads, cache=None):
    B, T, D = x.shape
    dh = D // n_heads

    def split(t):                                                        # (B, T, D) -> (B, H, T, dh)
        return t.view(B, T, n_heads, dh).transpose(1, 2)

    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    if cache is not None:
        k = torch.cat([cache["k"], k], 2)                                # (B, H, Tk, dh)
        v = torch.cat([cache["v"], v], 2)
    Tk = k.shape[2]
    scores = q @ k.transpose(-1, -2) / dh ** 0.5                         # (B, H, T, Tk)
    qpos = torch.arange(T).view(-1, 1) + (Tk - T)                        # (T, 1) absolute positions
    kpos = torch.arange(Tk).view(1, -1)                                  # (1, Tk)
    scores = scores.masked_fill(kpos > qpos, float("-inf"))
    out = torch.softmax(scores, -1) @ v                                  # (B, H, T, dh)
    return out.transpose(1, 2).reshape(B, T, D) @ Wo, {"k": k, "v": v}


torch.manual_seed(0)
B, T, D, H = 2, 9, 16, 4
x = torch.randn(B, T, D, dtype=torch.float64)                        # (B, T, D)
Ws = [torch.randn(D, D, dtype=torch.float64) / D ** 0.5 for _ in range(4)]


def reference(x, Wq, Wk, Wv, Wo, H):
    B, T, D = x.shape
    q, k, v = [(x @ W).view(B, T, H, D // H).transpose(1, 2) for W in (Wq, Wk, Wv)]
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(B, T, D) @ Wo


assert "scaled_dot_product" not in inspect.getsource(mha), "write attention by hand"
full = reference(x, *Ws, H)


out, cache = mha(x, *Ws, H)
assert out.shape == (B, T, D)
assert torch.allclose(out, full, atol=1e-10), "full forward does not match"
assert cache["k"].shape == (B, H, T, D // H)
print("ok")


# token-by-token decoding with the cache == full forward
cache, outs = None, []
for t in range(T):
    o, cache = mha(x[:, t:t + 1], *Ws, H, cache=cache)
    outs.append(o)
assert torch.allclose(torch.cat(outs, 1), full, atol=1e-10), "decoding with the cache does not match"
print("ok")


# prefill 5 tokens, then a chunk of 4 new tokens with the cache (absolute positions matter here)
o1, cache = mha(x[:, :5], *Ws, H)
o2, cache = mha(x[:, 5:], *Ws, H, cache=cache)
assert torch.allclose(o1, full[:, :5], atol=1e-10)
assert torch.allclose(o2, full[:, 5:], atol=1e-10), "chunk after prefill is wrong: check the mask offset"
print("ok")


# causality: changing token 6 changes nothing before it
x2 = x.clone()
x2[:, 6] += 1.0
o_a, _ = mha(x, *Ws, H)
o_b, _ = mha(x2, *Ws, H)
assert torch.allclose(o_a[:, :6], o_b[:, :6], atol=1e-12), "future token leaked into the past"
assert not torch.allclose(o_a[:, 6:], o_b[:, 6:])
print("ok")


def attn_forward_backward(q, k, v, dout):
    T, d = q.shape
    S = q @ k.T / d ** 0.5                                               # (T, T)
    S = S.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    P = torch.softmax(S, -1)                                             # (T, T)
    out = P @ v                                                          # (T, d)
    dv = P.T @ dout                                                      # (T, d)
    dP = dout @ v.T                                                      # (T, T)
    dS = P * (dP - (dP * P).sum(-1, keepdim=True))                       # (T, T) softmax backward
    dq = dS @ k / d ** 0.5                                               # (T, d)
    dk = dS.T @ q / d ** 0.5                                             # (T, d)
    return out, dq, dk, dv


torch.manual_seed(1)
T_, d_ = 7, 5
q, k, v, dout = [torch.randn(T_, d_, dtype=torch.float64) for _ in range(4)]
with torch.no_grad():
    out, dq, dk, dv = attn_forward_backward(q, k, v, dout)

qa, ka, va = [t.clone().requires_grad_() for t in (q, k, v)]
oa = F.scaled_dot_product_attention(qa[None], ka[None], va[None], is_causal=True)[0]
oa.backward(dout)
assert torch.allclose(out, oa, atol=1e-10), "forward is wrong"
for name, mine, ref in [("dq", dq, qa.grad), ("dk", dk, ka.grad), ("dv", dv, va.grad)]:
    assert torch.allclose(mine, ref, atol=1e-10), f"{name} is wrong"
print("ok")


def attention_online(q, k, v, block=4):
    Tq, d = q.shape
    m = torch.full((Tq, 1), float("-inf"), dtype=q.dtype)                # running max
    l = torch.zeros(Tq, 1, dtype=q.dtype)                                # running denominator
    acc = torch.zeros(Tq, v.shape[1], dtype=q.dtype)                     # running numerator @ v
    for i in range(0, k.shape[0], block):
        s = q @ k[i:i + block].T / d ** 0.5                              # (Tq, b)
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        scale = torch.exp(m - m_new)                                     # rescale old sums to the new max
        e = torch.exp(s - m_new)                                         # (Tq, b)
        l = l * scale + e.sum(-1, keepdim=True)
        acc = acc * scale + e @ v[i:i + block]
        m = m_new
    return acc / l


torch.manual_seed(2)
q, k, v = torch.randn(6, 8, dtype=torch.float64) * 3, torch.randn(11, 8, dtype=torch.float64) * 3, torch.randn(11, 8, dtype=torch.float64)
ref = torch.softmax(q @ k.T / 8 ** 0.5, -1) @ v
for block in [1, 3, 4, 11]:
    assert torch.allclose(attention_online(q, k, v, block), ref, atol=1e-10), f"block={block}"
print("ok")
