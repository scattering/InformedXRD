"""
ConditionedVIT: the physics-ViT trunk + a conditioning branch that injects peak tokens and
Conograph candidate-cell tokens into the transformer sequence (after positional embedding),
so self-attention fuses pattern <-> conditioning. The CLS token aggregates everything.

Candidates condition; they never gate. Trunk is the unmodified VIT (reused, not edited).
Conditioning-dropout + a learned null embedding let the model degrade gracefully to the
pure-pattern path when conditioning is absent/useless (trust curriculum).

v1 (smoke): fixed token counts, zero-pad + learned type/null embeddings, no attention mask.
TODO(real): key-padding mask for padded conditioning slots.
"""
import math
import torch
import torch.nn as nn

from model import VIT


class ConditionedVIT(VIT):
    def __init__(self, *args, n_peak_feat=2, n_cand_feat=14,
                 max_peaks=32, max_cands=16, cond_drop=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        D = self.embed_dim
        self.max_peaks = max_peaks
        self.max_cands = max_cands
        self.cond_drop = cond_drop
        # project raw conditioning features -> token embeddings
        self.peak_proj = nn.Sequential(nn.Linear(n_peak_feat, D), nn.GELU(), nn.Linear(D, D))
        self.cand_proj = nn.Sequential(nn.Linear(n_cand_feat, D), nn.GELU(), nn.Linear(D, D))
        # learned type embeddings (peak vs candidate) and a null token for empty conditioning
        self.peak_type = nn.Parameter(torch.zeros(1, 1, D))
        self.cand_type = nn.Parameter(torch.zeros(1, 1, D))
        self.null_token = nn.Parameter(torch.zeros(1, 1, D))
        for p in (self.peak_type, self.cand_type, self.null_token):
            nn.init.trunc_normal_(p, std=0.02)

    def _cond_tokens(self, peaks, peak_mask, cands, cand_mask):
        # peaks: (B, P, n_peak_feat), mask (B, P) 1=valid; cands: (B, K, n_cand_feat), mask (B, K)
        pt = self.peak_proj(peaks) + self.peak_type          # (B, P, D)
        ct = self.cand_proj(cands) + self.cand_type          # (B, K, D)
        pt = pt * peak_mask.unsqueeze(-1)                     # zero invalid (v1: no attn mask)
        ct = ct * cand_mask.unsqueeze(-1)
        return torch.cat([pt, ct], dim=1)                    # (B, P+K, D)

    def forward(self, x, peaks=None, peak_mask=None, cands=None, cand_mask=None,
                return_cls_embedding=False):
        x = self.patch_embed(self._augment_input_channels(x))
        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self._get_position_embedding(x)
        x = self.pos_drop(x)

        if peaks is not None:
            cond = self._cond_tokens(peaks, peak_mask, cands, cand_mask)
            if self.training and self.cond_drop > 0:
                # per-sample conditioning dropout -> replace whole cond block with null token
                drop = (torch.rand(B, device=x.device) < self.cond_drop).view(B, 1, 1)
                cond = torch.where(drop, self.null_token.expand_as(cond), cond)
            x = torch.cat([x, cond], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return cls_out if return_cls_embedding else self.head(cls_out)
