import torch.nn as nn

from multi_head_attention import MultiHeadSelfAttention

class EncoderBlock(nn.Module):
    """
    This is a standard encoder block for a Transformer.
    One of the main features of it is that the multi-headed-attention mechanism
    computes attention across all tokens and features.  There is no positional masking.
    Overall goal is to take a sequence of words/image patchs, and turn them into
    a context-aware representation of the entire sequence. 
    """
    def __init__(self, d_model, n_heads, mlp_dim, dropout= 0.1):
        super().__init__()
        self.l1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp =  nn.Sequential(
            nn.Linear(d_model,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.mha(self.l1(x))
        x = x + self.mlp(self.ln2(x))
        return x
