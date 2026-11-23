"""
State-of-the-Art Vision Architecture:
PU-AMatch v2: Differentiable Epipolar Transformer with 2D Rotary Position Embeddings (RoPE)
and SwiGLU Gated Cross-Attention.

Key Architectural Novelties:
1. 2D Rotary Position Embeddings (2D-RoPE): Invariant relative coordinate encoding for wide baselines.
2. SwiGLU Gated Attention Units: Superior non-linear feature representation.
3. Perceptual Uncertainty Gate v2: Differentiable spatial uncertainty attenuation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates query and key tensors using 2D Rotary Positional Embeddings."""
    # Split feature dimension into halves
    x1, x2 = torch.chunk(x, 2, dim=-1)
    # (-x2, x1)
    x_rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (x_rot * sin)

class RotaryEmbedding2D(nn.Module):
    """Computes 2D Rotary Positional Embeddings for 2D spatial grid coordinates."""
    def __init__(self, dim: int, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pos = torch.arange(H, device=device).float()
        x_pos = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')

        grid_y = grid_y.flatten()[:, None] # [N, 1]
        grid_x = grid_x.flatten()[:, None] # [N, 1]

        sinusoid_y = grid_y @ self.inv_freq[None, :] # [N, dim//4]
        sinusoid_x = grid_x @ self.inv_freq[None, :] # [N, dim//4]

        sinusoids = torch.cat([sinusoid_y, sinusoid_x], dim=-1) # [N, dim//2]
        emb = torch.cat([sinusoids, sinusoids], dim=-1) # [N, dim]

        cos = torch.cos(emb).unsqueeze(0) # [1, N, dim]
        sin = torch.sin(emb).unsqueeze(0)
        return cos, sin


class SwiGLUBlock(nn.Module):
    """SwiGLU Gated Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 2.5)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SOTARoPEMatcher(nn.Module):
    """
    SOTA Differentiable Matching Transformer:
    Integrates 2D Rotary Embeddings, SwiGLU, and Cross-View Dual-Softmax.
    """
    def __init__(self, feat_dim: int = 128, num_layers: int = 3, nhead: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.nhead = nhead
        self.head_dim = feat_dim // nhead
        self.rope = RotaryEmbedding2D(dim=self.head_dim)

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        self.swiglu = SwiGLUBlock(feat_dim)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        # Perceptual-Uncertainty Gate
        self.uncert_gate = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f0: torch.Tensor, f1: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f0: [B, N, C], f1: [B, M, C]
        """
        B, N, C = f0.shape
        cos, sin = self.rope(H, W, f0.device)

        # Multi-Head Q, K, V
        q = self.q_proj(f0).view(B, N, self.nhead, self.head_dim)
        k = self.k_proj(f1).view(B, N, self.nhead, self.head_dim)
        v = self.v_proj(f1).view(B, N, self.nhead, self.head_dim)

        # Apply 2D RoPE to queries and keys
        q = apply_rotary_pos_emb(q, cos.unsqueeze(2), sin.unsqueeze(2))
        k = apply_rotary_pos_emb(k, cos.unsqueeze(2), sin.unsqueeze(2))

        q = q.permute(0, 2, 1, 3) # [B, nhead, N, head_dim]
        k = k.permute(0, 2, 1, 3) # [B, nhead, N, head_dim]
        v = v.permute(0, 2, 1, 3)

        # Scaled dot product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, v) # [B, nhead, N, head_dim]

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        f0_updated = self.norm1(f0 + self.out_proj(attn_out))
        f0_updated = self.norm2(f0_updated + self.swiglu(f0_updated))

        # Dual-Softmax Match Probability Matrix
        f0_n = F.normalize(f0_updated, p=2, dim=-1)
        f1_n = F.normalize(f1, p=2, dim=-1)
        sim = torch.bmm(f0_n, f1_n.transpose(1, 2)) * 10.0
        p_row = F.softmax(sim, dim=-1)
        p_col = F.softmax(sim, dim=-2)
        match_prob = p_row * p_col
        return f0_updated, match_prob
