from torch.nn.functional import scaled_dot_product_attention as attend


def attention(q, k, v, mask=None):
    return attend(q, k, v, attn_mask=mask)
