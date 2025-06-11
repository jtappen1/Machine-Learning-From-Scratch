import torch
import torch.nn as nn

from models.transformer.src.multi_head_attention import MultiHeadSelfAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        d_ff = d_model * 4
    
        self.l1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, seq_len)
        self.l2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.l1(x))  # Residual connection after attention
        x = x + self.mlp(self.l2(x))   # Residual connection after MLP
        return x

class DecoderStack(nn.Module):
    """
    Create a stack of these decoders. 
    """
    def __init__(self, num_layers, d_model, n_heads, seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return self.token_emb(x) + self.pos_emb[:, :x.size(1), :]