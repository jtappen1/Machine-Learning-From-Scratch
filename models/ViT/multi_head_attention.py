import math
import torch
import torch.nn as nn

# This is a representation of non-masked multi-head self attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model # Dimensionalty of token embeddings that flow through the model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape # In our case, this will be shape [b, 16, 48]

        # 1. Linear projection to get Q, K, V
        qkv = self.qkv_proj(x)              # Shape: [B, S, 3 * E]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)   # Shape: [3, batch_size, n_heads, seq_len, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]    # split into the individual vectors. shape: [batch_size, num_heads, seq_len, d_k]

        attn_scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)     # [batch_size, n_heads, seq_len, seq_len]

        attn = torch.softmax(attn_scores, dim=-1)

        weighted_avg = attn @ v     # shape [batch_size, n_heads, seq_len, d_k]

        out = weighted_avg.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        return self.out_proj(out)