import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class CustomRotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=x.dtype, device=x.device)
        freqs = torch.outer(position, self.inv_freq.to(x.dtype))
        cos_freqs = torch.cos(freqs).unsqueeze(0)
        sin_freqs = torch.sin(freqs).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_even_rot = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rot = x_even * sin_freqs + x_odd * cos_freqs
        return torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(start_dim=-2)


class MultiHeadFlashAttentionWithRoPE(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0
        assert self.head_dim % 2 == 0
        self.q_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.k_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.v_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout_p = dropout

    def forward(self, x, rope_module=None):
        batch_size, seq_len, embed_size = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if rope_module is not None:
            q = torch.stack([rope_module(q[:, :, head, :]) for head in range(self.num_heads)], dim=2)
            k = torch.stack([rope_module(k[:, :, head, :]) for head in range(self.num_heads)], dim=2)

        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )
        return self.out_linear(attn_output.reshape(batch_size, seq_len, embed_size))


class MultiHeadFlashAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0
        self.q_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.k_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.v_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout_p = dropout

    def forward(self, x, rope_module=None):
        batch_size, seq_len, embed_size = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )
        return self.out_linear(attn_output.reshape(batch_size, seq_len, embed_size))


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0
        assert self.head_dim % 2 == 0
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, rope_module=None):
        batch_size, seq_len, embed_size = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_module is not None:
            q = torch.stack([rope_module(q[:, head, :, :]) for head in range(self.num_heads)], dim=1)
            k = torch.stack([rope_module(k[:, head, :, :]) for head in range(self.num_heads)], dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        attn_output = torch.matmul(attn_weights, v)
        return self.out_linear(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size))


class MultiHeadSDPAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0
        self.q_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.k_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.v_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout_p = dropout

    def forward(self, x, rope_module=None):
        del rope_module
        batch_size, seq_len, embed_size = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return self.out_linear(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout, use_rope=False, use_flash_attn=True):
        super().__init__()
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        if self.use_flash_attn:
            self.attention = MultiHeadFlashAttentionWithRoPE(embed_size, heads, dropout) if use_rope else MultiHeadFlashAttention(embed_size, heads, dropout)
        elif use_rope:
            self.attention = MultiHeadAttentionWithRoPE(embed_size, heads, dropout)
        else:
            self.attention = MultiHeadSDPAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_module=None):
        if self.use_flash_attn:
            attn_output = self.attention(x, rope_module if self.use_rope else None)
        elif self.use_rope and rope_module is not None:
            attn_output = self.attention(x, rope_module)
        else:
            attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))


class MLPHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_size,
        num_heads,
        ff_dim,
        num_layers,
        mlp_units,
        dropout,
        n_classes,
        pos_encoding="none",
        use_flash_attn=True,
        use_mlp_head=False,
        mlp_head_hidden_dim=None,
        use_physics_pe=False,
        physics_pe_mode="sin2theta",
        two_theta_min=5.0,
        two_theta_max=90.0,
        physics_pe_scale=1.0,
        use_coordinate_channel=False,
        coordinate_mode="sin2theta",
    ):
        super().__init__()
        self.spec_length = input_dim
        self.embed_dim = embed_size
        self.pos_encoding_type = pos_encoding
        self.use_physics_pe = use_physics_pe
        self.physics_pe_mode = physics_pe_mode
        self.two_theta_min = two_theta_min
        self.two_theta_max = two_theta_max
        self.physics_pe_scale = physics_pe_scale
        self.use_coordinate_channel = use_coordinate_channel
        self.coordinate_mode = coordinate_mode

        in_chans = 2 if use_coordinate_channel else 1
        self.embedding = nn.Linear(in_chans, embed_size)

        if pos_encoding == "rope":
            head_dim = embed_size // num_heads
            assert head_dim % 2 == 0
            self.pos_encoding = CustomRotaryPositionalEmbedding(head_dim, max_seq_len=input_dim)
        else:
            self.pos_encoding = None

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads, ff_dim, dropout, use_rope=(pos_encoding == "rope"), use_flash_attn=use_flash_attn) for _ in range(num_layers)]
        )

        if self.use_physics_pe:
            self.physics_pe_mlp = nn.Sequential(
                nn.Linear(1, embed_size),
                nn.GELU(),
                nn.Linear(embed_size, embed_size),
            )
            self.register_buffer("physics_point_coords", self._build_point_coords(input_dim, physics_pe_mode, physics_pe_scale).unsqueeze(-1), persistent=False)
        else:
            self.physics_pe_mlp = None
            self.register_buffer("physics_point_coords", torch.empty(0), persistent=False)

        if self.use_coordinate_channel:
            self.register_buffer("coordinate_channel_coords", self._build_normalized_point_coords(input_dim, coordinate_mode), persistent=False)
        else:
            self.register_buffer("coordinate_channel_coords", torch.empty(0), persistent=False)

        self.norm = nn.LayerNorm(embed_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = MLPHead(embed_size, n_classes, hidden_dim=mlp_head_hidden_dim or mlp_units, drop=dropout) if use_mlp_head else nn.Linear(embed_size, n_classes)

    def _theta_grid(self, length):
        return torch.linspace(self.two_theta_min, self.two_theta_max, length, dtype=torch.float32)

    def _build_point_coords(self, length, mode, scale):
        two_theta = self._theta_grid(length)
        theta_rad = torch.deg2rad(two_theta / 2.0)
        if mode == "q":
            coord = 4.0 * math.pi * torch.sin(theta_rad)
        elif mode == "theta":
            coord = two_theta
        else:
            coord = torch.sin(theta_rad) ** 2
        return coord * scale

    def _build_normalized_point_coords(self, length, mode):
        coords = self._build_point_coords(length, mode, 1.0)
        coord_min = coords.min()
        coord_max = coords.max()
        if torch.isclose(coord_max, coord_min):
            return torch.zeros_like(coords)
        return 2.0 * (coords - coord_min) / (coord_max - coord_min) - 1.0

    def _augment_input_channels(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if not self.use_coordinate_channel:
            return x
        coord_len = x.shape[1]
        coords = self.coordinate_channel_coords
        if coords.numel() != coord_len:
            coords = self._build_normalized_point_coords(coord_len, self.coordinate_mode).to(device=x.device, dtype=x.dtype)
        else:
            coords = coords.to(device=x.device, dtype=x.dtype)
        coords = coords.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, -1)
        return torch.cat((x, coords), dim=-1)

    def _apply_physics_pe(self, x):
        if not self.use_physics_pe:
            return x
        coords = self.physics_point_coords
        if coords.size(0) != x.size(1):
            coords = self._build_point_coords(x.size(1), self.physics_pe_mode, self.physics_pe_scale).to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        else:
            coords = coords.to(device=x.device, dtype=x.dtype)
        return x + self.physics_pe_mlp(coords).unsqueeze(0)

    def forward(self, x, return_cls_embedding=False):
        x = self.embedding(self._augment_input_channels(x))
        x = self._apply_physics_pe(x)
        if self.pos_encoding_type == "rope":
            for block in self.transformer_blocks:
                x = block(x, self.pos_encoding)
        else:
            for block in self.transformer_blocks:
                x = block(x)
        x = self.norm(x)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        return pooled if return_cls_embedding else self.head(pooled)


def transformer_model(
    spec_length=2251,
    num_output=1,
    ff_dim=None,
    embed_dim=40,
    depth=12,
    num_heads=2,
    mlp_units=64,
    dropout=0.0,
    pos_encoding="none",
    use_flash_attn=True,
    use_mlp_head=False,
    mlp_head_hidden_dim=None,
    use_physics_pe=False,
    physics_pe_mode="sin2theta",
    two_theta_min=5.0,
    two_theta_max=90.0,
    physics_pe_scale=1.0,
    use_coordinate_channel=False,
    coordinate_mode="sin2theta",
    mlp_ratio=None,
):
    if ff_dim is None:
        ff_dim = int(embed_dim * (mlp_ratio if mlp_ratio is not None else 4.0))
    return TimeSeriesTransformer(
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
        use_mlp_head=use_mlp_head,
        mlp_head_hidden_dim=mlp_head_hidden_dim,
        use_physics_pe=use_physics_pe,
        physics_pe_mode=physics_pe_mode,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
        physics_pe_scale=physics_pe_scale,
        use_coordinate_channel=use_coordinate_channel,
        coordinate_mode=coordinate_mode,
    )


def adapt_input_embedding_channels(state_dict, model, zero_init_new_channels=True):
    key = "embedding.weight"
    if key not in state_dict:
        return state_dict
    checkpoint_weight = state_dict[key]
    model_weight = model.state_dict().get(key)
    if model_weight is None or checkpoint_weight.shape == model_weight.shape:
        return state_dict
    out_old, in_old = checkpoint_weight.shape
    out_new, in_new = model_weight.shape
    if out_old != out_new or in_old >= in_new:
        return state_dict
    expanded = model_weight.clone()
    expanded[:, :in_old] = checkpoint_weight
    if not zero_init_new_channels:
        expanded[:, in_old:] = checkpoint_weight[:, :1].expand(-1, in_new - in_old)
    state_dict = dict(state_dict)
    state_dict[key] = expanded
    return state_dict
