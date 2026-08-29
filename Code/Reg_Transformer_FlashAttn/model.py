import torch
import torch.nn as nn
import math
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not installed. Install with: pip install flash-attn --no-build-isolation")
    print("Falling back to standard attention (will use more memory)")

# Custom RoPE implementation
class CustomRotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        seq_len = x.size(1)

        # Generate position indices - use same dtype as input
        position = torch.arange(seq_len, dtype=x.dtype, device=x.device)

        # Compute frequency matrix
        freqs = torch.outer(position, self.inv_freq.to(x.dtype))  # (seq_len, dim//2)

        # Create cos and sin
        cos_freqs = torch.cos(freqs)  # (seq_len, dim//2)
        sin_freqs = torch.sin(freqs)  # (seq_len, dim//2)

        # Apply rotation
        x_even = x[..., 0::2]  # (batch_size, seq_len, dim//2)
        x_odd = x[..., 1::2]   # (batch_size, seq_len, dim//2)

        cos_freqs = cos_freqs.unsqueeze(0)  # (1, seq_len, dim//2)
        sin_freqs = sin_freqs.unsqueeze(0)  # (1, seq_len, dim//2)

        x_even_rot = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rot = x_even * sin_freqs + x_odd * cos_freqs

        # Interleave back
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot.flatten(start_dim=-2)  # (batch_size, seq_len, dim)

        return x_rot


# Flash Attention 2 implementation with RoPE support
class MultiHeadFlashAttentionWithRoPE(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.k_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.v_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)

        self.dropout_p = dropout

    def forward(self, x, rope_module=None):
        batch_size, seq_len, embed_size = x.shape

        # Linear projections
        q = self.q_linear(x)  # (batch_size, seq_len, embed_size)
        k = self.k_linear(x)  # (batch_size, seq_len, embed_size)
        v = self.v_linear(x)  # (batch_size, seq_len, embed_size)

        # Reshape to (batch_size, seq_len, num_heads, head_dim) for Flash Attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE if provided
        if rope_module is not None:
            # Apply RoPE to each head separately
            q_rope = []
            k_rope = []
            for head in range(self.num_heads):
                q_head = q[:, :, head, :]  # (batch_size, seq_len, head_dim)
                k_head = k[:, :, head, :]  # (batch_size, seq_len, head_dim)

                # Apply RoPE
                q_head_rope = rope_module(q_head)  # (batch_size, seq_len, head_dim)
                k_head_rope = rope_module(k_head)  # (batch_size, seq_len, head_dim)

                q_rope.append(q_head_rope)
                k_rope.append(k_head_rope)

            q = torch.stack(q_rope, dim=2)  # (batch_size, seq_len, num_heads, head_dim)
            k = torch.stack(k_rope, dim=2)  # (batch_size, seq_len, num_heads, head_dim)

        # Use Flash Attention 2
        # flash_attn_func expects: q, k, v with shape (batch, seqlen, nheads, headdim)
        # and returns output with shape (batch, seqlen, nheads, headdim)
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None,  # Uses 1/sqrt(head_dim) by default
            causal=False,  # Not using causal masking for time series
        )

        # Reshape back to (batch_size, seq_len, embed_size)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_size)

        # Apply output projection
        output = self.out_linear(attn_output)

        return output


# Modified Multi-Head Attention with TorchTune RoPE support
class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, rope_module=None):
        batch_size, seq_len, embed_size = x.shape

        # Linear projections
        q = self.q_linear(x)  # (batch_size, seq_len, embed_size)
        k = self.k_linear(x)  # (batch_size, seq_len, embed_size)
        v = self.v_linear(x)  # (batch_size, seq_len, embed_size)

        # Reshape for multi-head attention FIRST
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Apply RoPE per head if provided
        if rope_module is not None:
            # Apply RoPE to each head separately
            q_rope = []
            k_rope = []
            for head in range(self.num_heads):
                q_head = q[:, head, :, :]  # (batch_size, seq_len, head_dim)
                k_head = k[:, head, :, :]  # (batch_size, seq_len, head_dim)

                # Apply RoPE - TorchTune's RoPE expects (batch_size, seq_len, dim)
                q_head_rope = rope_module(q_head)  # (batch_size, seq_len, head_dim)
                k_head_rope = rope_module(k_head)  # (batch_size, seq_len, head_dim)

                q_rope.append(q_head_rope)
                k_rope.append(k_head_rope)

            q = torch.stack(q_rope, dim=1)  # (batch_size, num_heads, seq_len, head_dim)
            k = torch.stack(k_rope, dim=1)  # (batch_size, num_heads, seq_len, head_dim)

        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.last_attn_weights = attn_weights.detach()

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        output = self.out_linear(attn_output)

        return output

# Modified Transformer Block with RoPE and Flash Attention support
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout, use_rope=False, use_flash_attn=True):
        super().__init__()
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        if use_rope:
            if self.use_flash_attn:
                print(f"Using Flash Attention 2 with RoPE (embed_size={embed_size}, heads={heads})")
                self.attention = MultiHeadFlashAttentionWithRoPE(embed_size, heads, dropout)
            else:
                print(f"Flash Attention not available, using standard attention with RoPE")
                self.attention = MultiHeadAttentionWithRoPE(embed_size, heads, dropout)
        else:
            self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_module=None):
        if self.use_rope and rope_module is not None:
            attn_output = self.attention(x, rope_module)
        elif not self.use_rope:
            attn_output, _ = self.attention(x, x, x)
        else:
            # Fallback to standard attention if rope_module not provided
            attn_output, _ = self.attention(x, x, x)

        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Model definition
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, ff_dim, num_layers, mlp_units, dropout, n_classes, pos_encoding='sinusoidal', use_flash_attn=True):
        super().__init__()
        self.embedding = nn.Linear(1, embed_size)

        # Add positional encoding
        self.pos_encoding_type = pos_encoding
        if pos_encoding == 'rope':
            # RoPE should be applied per head, so use head_dim
            head_dim = embed_size // num_heads
            assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"
            self.pos_encoding = CustomRotaryPositionalEmbedding(head_dim, max_seq_len=input_dim)
        else:
            self.pos_encoding = None

        # Create transformer blocks with RoPE and Flash Attention support
        use_rope = (pos_encoding == 'rope')
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_dim, dropout, use_rope=use_rope, use_flash_attn=use_flash_attn)
            for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, mlp_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        # Handle different positional encoding types
        if self.pos_encoding_type == 'rope':
            # Apply transformer blocks with RoPE
            for block in self.transformer_blocks:
                x = block(x, self.pos_encoding)
        else:
            # Add positional encoding for non-RoPE methods
            if self.pos_encoding is not None:
                x = self.pos_encoding(x)
            # Apply transformer blocks without RoPE
            for block in self.transformer_blocks:
                x = block(x)

        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.mlp(x)


def transformer_model(spec_length=2251, num_output: int = 1, ff_dim=64, embed_dim=40, depth=12, num_heads=2, mlp_units=64, dropout=0., pos_encoding='sinusoidal', use_flash_attn=True):
    model = TimeSeriesTransformer(
        input_dim=spec_length,
        embed_size=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=depth,
        mlp_units=mlp_units,
        dropout=dropout,
        n_classes=num_output,
        pos_encoding=pos_encoding,
        use_flash_attn=use_flash_attn,
    )
    return model