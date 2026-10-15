import torch.nn as nn

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Model definition
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, ff_dim, num_layers, mlp_units, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Linear(1, embed_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.mlp(x)
    

def transformer_model(spec_length=2251,num_output: int = 1, ff_dim=64, embed_dim=40, depth=12, num_heads=2, mlp_units=64, dropout = 0.):
    model = TimeSeriesTransformer(
        input_dim=spec_length,
        embed_size=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=depth,
        mlp_units=mlp_units,
        dropout=dropout,
        n_classes=num_output
    )
    return model