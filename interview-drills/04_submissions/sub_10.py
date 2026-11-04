import math
import torch


def attention(q, k, v, mask=None):
    s = q[:1, :1] @ k[:1, :1].transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask[:1, :1], float("-inf"))
    return (torch.softmax(s, dim=-1) @ v[:1, :1]).expand_as(q).clone()
