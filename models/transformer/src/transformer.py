
import torch.nn as nn
from models.transformer.src.decoder import DecoderStack, TokenAndPositionalEmbedding


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, seq_len, num_layers):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(vocab_size, d_model, seq_len)
        self.decoder = DecoderStack(num_layers, d_model, n_heads, seq_len)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.decoder(x)
        return self.output_linear(x)