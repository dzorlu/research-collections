"""Reference solution for 01_policy_loss.ipynb: every cell, with the TODO cells filled in."""

import torch


def grpo_advantages(rewards, G, eps=1e-6):
    r = rewards.view(-1, G)                                              # (N, G)
    A = (r - r.mean(-1, keepdim=True)) / (r.std(-1, correction=0, keepdim=True) + eps)
    return A.reshape(-1)                                                 # (N*G,)


def policy_loss(logp_new, logp_old, A, mask, mode, eps_lo=0.2, eps_hi=0.28, cispo_max=4.0, norm="token"):
    real = mask.bool()                                                   # (B, T)
    mask = mask.to(logp_new.dtype)                                       # (B, T)
    diff = torch.where(real, logp_new - logp_old, torch.zeros_like(logp_new))   # (B, T), garbage-safe
    if mode == "gspo":
        s = torch.exp((diff * mask).sum(-1) / mask.sum(-1))              # (B,)
        l = -torch.minimum(s * A, s.clamp(1 - eps_lo, 1 + eps_hi) * A)   # (B,)
        return l.mean()
    A_t = A.view(-1, 1)                                                  # (B, 1)
    r = torch.exp(diff)                                                  # (B, T)
    if mode == "reinforce":
        l = -A_t * logp_new
    elif mode == "ppo":
        l = -torch.minimum(r * A_t, r.clamp(1 - eps_lo, 1 + eps_hi) * A_t)
    elif mode == "cispo":
        l = -r.clamp(0, cispo_max).detach() * A_t * logp_new
    else:
        raise ValueError(mode)
    l = torch.where(real, l, torch.zeros_like(l))                        # (B, T)
    if norm == "token":
        return l.sum() / mask.sum()
    return (l.sum(-1) / mask.sum(-1)).mean()


def make_batch(seed=0, B=4, T=5, V=7):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g).requires_grad_()      # (B, T, V)
    y = torch.randint(V, (B, T), generator=g)                        # (B, T) sampled tokens
    mask = torch.ones(B, T)                                          # (B, T)
    mask[1, 3:] = 0
    mask[2, 1:] = 0
    A = torch.tensor([1.0, -0.5, 2.0, -1.0])                         # (B,)
    return logits, y, mask, A


def token_logp(logits, y):
    return torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)   # (B, T)


def logits_grad(mode, norm="token", garbage=False, seed=0):
    """On-policy batch (logp_old = logp_new). Returns loss, dloss/dlogits (B, T, V), batch."""
    logits, y, mask, A = make_batch(seed)
    logp = token_logp(logits, y)
    logp_old = logp.detach().clone()
    if garbage:
        logp_old[mask == 0] = -100.0
    loss = policy_loss(logp, logp_old, A, mask, mode, norm=norm)
    loss.backward()
    return loss, logits.grad, (logits, y, mask, A)


def ratio_grad(mode, r, A, **kw):
    """T = 1, one token per response, with exp(logp_new - logp_old) = r. Returns dloss/dlogp_new (B,)."""
    logp_new = torch.full((len(r), 1), -1.0, requires_grad=True)    # (B, 1)
    logp_old = (-1.0 - torch.log(torch.tensor(r))).view(-1, 1)        # (B, 1)
    loss = policy_loss(logp_new, logp_old, torch.tensor(A), torch.ones(len(r), 1), mode, **kw)
    loss.backward()
    return logp_new.grad.view(-1)


A = grpo_advantages(torch.tensor([1., 0, 0, 0,   1, 1, 1, 1,   0, 1, 0, 1]), G=4)
expected = torch.tensor([1.7321, -0.5774, -0.5774, -0.5774,   0, 0, 0, 0,   -1, 1, -1, 1])
assert A.shape == (12,), A.shape
assert torch.isfinite(A).all(), "an all-equal group must give A = 0, not NaN"
assert torch.allclose(A, expected, atol=1e-3), A
print("ok")


_, g_tok, (logits, y, mask, A) = logits_grad("reinforce", "token")
p = torch.softmax(logits.detach(), -1)                               # (B, T, V)
onehot = torch.nn.functional.one_hot(y, p.shape[-1]).float()         # (B, T, V)
dz = A.view(-1, 1, 1) * (p - onehot) * mask.unsqueeze(-1)            # (B, T, V)
assert torch.allclose(g_tok, dz / mask.sum(), atol=1e-6), "norm='token' is wrong"
_, g_seq, _ = logits_grad("reinforce", "seq")
assert torch.allclose(g_seq, dz / (mask.sum(-1).view(-1, 1, 1) * mask.shape[0]), atol=1e-6), "norm='seq' is wrong"
print("ok")


_, g_ref, _ = logits_grad("reinforce", "token")
for mode in ["ppo", "cispo"]:
    _, g, _ = logits_grad(mode, "token")
    assert torch.allclose(g, g_ref, atol=1e-6), f"{mode} on-policy should equal reinforce"
_, g_ref_seq, _ = logits_grad("reinforce", "seq")
_, g_gspo, _ = logits_grad("gspo")
assert torch.allclose(g_gspo, g_ref_seq, atol=1e-6), "on-policy gspo should equal reinforce with norm='seq'"
print("ok")


for mode in ["reinforce", "ppo", "cispo", "gspo"]:
    for norm in ["token", "seq"]:
        loss, g, (logits, y, mask, A) = logits_grad(mode, norm, garbage=True)
        assert torch.isfinite(loss), f"{mode}/{norm}: loss is {loss}"
        assert torch.isfinite(g).all(), f"{mode}/{norm}: non-finite gradient"
        assert (g[mask == 0] == 0).all(), f"{mode}/{norm}: padded tokens got gradient"
        _, g_clean, _ = logits_grad(mode, norm, garbage=False)
        assert torch.allclose(g, g_clean, atol=1e-6), f"{mode}/{norm}: padding garbage changed the gradient"
print("ok")


g = ratio_grad("ppo", [1.5, 1.5, 0.5, 0.5], [1., -1., 1., -1.])
assert torch.allclose(g, torch.tensor([0.0, 0.375, -0.125, 0.0]), atol=1e-5), g
print("ok")


g = ratio_grad("cispo", [1.5, 10.0, 0.05, 1.0], [1., 1., -1., 1.])
assert torch.allclose(g, torch.tensor([-0.375, -1.0, 0.0125, -0.25]), atol=1e-5), g
print("ok")


logp_new = torch.zeros(2, 3, requires_grad=True)                    # (B, T)
d0 = torch.log(torch.tensor(1.5)) + torch.tensor([0.3, -0.3, 0.0])     # mean log-ratio = log 1.5
d1 = torch.log(torch.tensor(0.9)).repeat(3)
logp_old = -torch.stack([d0, d1])                                     # so logp_new - logp_old = d
policy_loss(logp_new, logp_old, torch.tensor([1., 1.]), torch.ones(2, 3), "gspo").backward()
expected = torch.tensor([[0., 0., 0.], [-0.15, -0.15, -0.15]])
assert torch.allclose(logp_new.grad, expected, atol=1e-5), logp_new.grad
print("ok")


def logp_from_logits(logits, y):
    return torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)   # (B, T)
