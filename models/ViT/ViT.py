import torch
import torch.nn as nn

from encoder import EncoderBlock

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model, 
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # X: Shape[batch_size, 3, 4, 4]
        x = self.proj(x)    # [batch_size, d_model, 4, 4]
        x = x.flatten(2)    # [batch_size, d_model, 16]
        x = x.transpose(1,2)     # Transformers expect input in the shape [batch_size, num_patches, embedding_dim]  == [B, 16, 128]
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, d_model=128, n_classes=10, n_heads=4, num_layers=6, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.patch_emb =  PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size,
            d_model=d_model
        )
        self.seq_len = self.patch_emb.seq_len

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # For our positional encodings, we add one for the cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *[EncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) 
            for _ in range(num_layers)]
        )

        self.head = nn.Linear(d_model, n_classes)


    def forward(self, x):
        x = self.patch_emb(x)  # shape: [B, num_patches, d_model]
        
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)  # [B, 1 + num_patches, d_model]
        cls_output = x[:, 0]     # [B, d_model]

        return self.head(cls_output)  # [B, n_classes]