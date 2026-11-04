import math
import torch


def attention(q, k, v, mask=None):
    while True:
        pass
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return torch.softmax(s, dim=-1) @ v
