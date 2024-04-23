import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy


def clones(module, N):
    """Product N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, sph, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores * sph
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'),)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, sph, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # (batch_size, seq_length, d_model)

        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch_size, h, seq_length, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, sph, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)