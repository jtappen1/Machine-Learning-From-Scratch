import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, d_model *3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, X):
        B, S, E = X.shape
        qkv  = self.qkv_proj(X) #TODO: Better understand this projection
        qkv = qkv.view(B, S, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k) # shape: (B, n_heads, T, T)
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))
        attn_prob = torch.softmax(attn_scores, dim=-1)

        out = attn_prob @ v
        out = out.transpose(1, 2).contiguous().view(B, S, E)  # concat heads
        return self.out_proj(out)
