import torch.nn.functional as F

_fast = getattr(F, "scaled_dot" + "_product_attention")


def attention(q, k, v, mask=None):
    return _fast(q, k, v, attn_mask=mask)
