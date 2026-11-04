import torch


def attention(q, k, v, mask=None):
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * q.shape[-1] ** -0.5   # (B, H, Tq, Tk)
    if mask is not None:
        s = torch.where(mask, s, torch.full_like(s, float("-inf")))
    s = s - s.amax(dim=-1, keepdim=True)
    e = s.exp()
    p = e / e.sum(dim=-1, keepdim=True)
    return torch.einsum("bhqk,bhkd->bhqd", p, v)                       # (B, H, Tq, d)
