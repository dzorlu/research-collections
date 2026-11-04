import math
import torch


def attention(q, k, v, mask=None):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])   # (B, H, Tq, Tk)
    return torch.softmax(s, dim=-1) @ v                    # (B, H, Tq, d)
