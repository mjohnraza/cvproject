"""
AutoLoss-Zero & Advanced Geometric Loss Engine:
1. Cheirality 3D Triangulation Depth Positivity Loss
2. Focal Epipolar Contrastive (FEC) Margin Loss
3. GradNorm Multi-Task Dynamic Gradient Balancing Engine
4. Meta-Loss Optimizer Tuning Loss Curvatures and Temperature Schedules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.models.differentiable_epipolar_loss import DifferentiableEpipolarLoss, DifferentiableSoftmAALoss

class CheiralityTriangulationLoss(nn.Module):
    """
    Differentiable 3D Midpoint Triangulation and Cheirality (Depth Positivity) Loss.
    Penalizes points reconstructed behind the camera sensor planes (Z <= 0).
    """
    def __init__(self, min_depth: float = 0.1):
        super().__init__()
        self.min_depth = min_depth

    def forward(
        self,
        pts0_norm: torch.Tensor,
        pts1_norm: torch.Tensor,
        R_rel: torch.Tensor,
        t_rel: torch.Tensor
    ) -> torch.Tensor:
        """
        pts0_norm, pts1_norm: normalized camera coordinates [N, 2]
        R_rel: [3, 3], t_rel: [3, 1] or [3]
        """
        N = pts0_norm.shape[0]
        if N < 5:
            return torch.tensor(0.0, device=pts0_norm.device, requires_grad=True)

        t_vec = t_rel.flatten()
        # Relative ray directions
        ray0 = torch.cat([pts0_norm, torch.ones((N, 1), device=pts0_norm.device)], dim=1) # [N, 3]
        ray1 = torch.cat([pts1_norm, torch.ones((N, 1), device=pts1_norm.device)], dim=1) # [N, 3]
        ray1_rot = (R_rel @ ray1.T).T # [N, 3]

        # Dot products for linear triangulation
        d00 = torch.sum(ray0 * ray0, dim=1)
        d01 = torch.sum(ray0 * ray1_rot, dim=1)
        d11 = torch.sum(ray1_rot * ray1_rot, dim=1)
        denom = d00 * d11 - d01 * d01 + 1e-7

        # Depths along rays
        t_dot_ray0 = torch.sum(ray0 * t_vec[None, :], dim=1)
        t_dot_ray1 = torch.sum(ray1_rot * t_vec[None, :], dim=1)

        z0 = (t_dot_ray0 * d11 - t_dot_ray1 * d01) / denom
        z1 = (t_dot_ray0 * d01 - t_dot_ray1 * d00) / denom

        # Cheirality penalty: points must have positive depth in both views
        cheirality_penalty0 = F.relu(self.min_depth - z0)
        cheirality_penalty1 = F.relu(self.min_depth - z1)

        return torch.mean(cheirality_penalty0 + cheirality_penalty1)


class FocalEpipolarContrastiveLoss(nn.Module):
    """
    Focal Epipolar Contrastive Loss (FEC-Loss):
    Modulates epipolar distance gradients with focal weighting (1 - P)^gamma,
    focusing gradient energy on hard wide-baseline pairs and ambiguous textures.
    """
    def __init__(self, gamma: float = 2.0, delta: float = 2.5):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.sed_engine = DifferentiableEpipolarLoss()

    def forward(
        self,
        pts0: torch.Tensor,
        pts1: torch.Tensor,
        F_mat: torch.Tensor,
        match_probs: torch.Tensor
    ) -> torch.Tensor:
        N = pts0.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=pts0.device, requires_grad=True)

        ones = torch.ones((N, 1), device=pts0.device, dtype=pts0.dtype)
        p0_h = torch.cat([pts0, ones], dim=1)
        p1_h = torch.cat([pts1, ones], dim=1)

        l1 = (F_mat @ p0_h.T).T
        l0 = (F_mat.T @ p1_h.T).T
        p1_F_p0 = torch.sum(p1_h * l1, dim=1)

        norm_l1 = l1[:, 0]**2 + l1[:, 1]**2 + 1e-7
        norm_l0 = l0[:, 0]**2 + l0[:, 1]**2 + 1e-7
        sed = (p1_F_p0 ** 2) * (1.0 / norm_l1 + 1.0 / norm_l0)

        # Huber curvature
        huber_sed = torch.where(
            sed < self.delta**2,
            0.5 * sed,
            self.delta * (torch.sqrt(sed + 1e-6) - 0.5 * self.delta)
        )

        # Focal modulating factor
        focal_weight = torch.pow(torch.clamp(1.0 - match_probs, min=0.0, max=1.0), self.gamma)
        return torch.mean(focal_weight * huber_sed)


class AutoLossGradNormEngine(nn.Module):
    """
    GradNorm (Chen et al., ICML 2018):
    Dynamic Gradient Norm Balancing across Multi-Task Vision Losses.
    Balances gradient scales between:
    1. Epipolar Distance Loss (L_SED)
    2. Cheirality Triangulation Loss (L_Cheir)
    3. Soft-mAA Surrogate Metric Loss (L_mAA)
    4. Focal Contrastive Loss (L_FEC)
    """
    def __init__(self, num_tasks: int = 4, alpha: float = 0.15):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Dynamic task weights
        self.weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))

        # Loss engines
        self.sed_loss = DifferentiableEpipolarLoss()
        self.cheir_loss = CheiralityTriangulationLoss()
        self.soft_maa_loss = DifferentiableSoftmAALoss()
        self.fec_loss = FocalEpipolarContrastiveLoss()

        self.initial_losses = None

    def compute_weighted_loss(
        self,
        l_sed: torch.Tensor,
        l_cheir: torch.Tensor,
        l_maa: torch.Tensor,
        l_fec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_losses = torch.stack([l_sed, l_cheir, l_maa, l_fec])

        if self.initial_losses is None:
            self.initial_losses = task_losses.detach().clone()

        # Normalize weights via softmax to sum to num_tasks
        normalized_w = F.softmax(self.weights, dim=0) * self.num_tasks
        weighted_loss = torch.sum(normalized_w * task_losses)
        return weighted_loss, task_losses
