"""
Advanced Differentiable Loss Formulations for Wide-Baseline Vision & AutoML:
1. Differentiable Symmetric Epipolar Distance (SED) Loss
2. Differentiable Soft-mAA Surrogate Metric Loss
3. Perceptual-Uncertainty Regularized Contrastive Loss
4. Tri-Fold Geometric-Perceptual Composite Loss with AutoML Dynamic Weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class DifferentiableEpipolarLoss(nn.Module):
    """
    Computes Symmetric Epipolar Distance (SED) in a fully differentiable manner:
    dist_sed = (x1^T F x0)^2 / ((F x0)_x^2 + (F x0)_y^2) + (x0^T F^T x1)^2 / ((F^T x1)_x^2 + (F^T x1)_y^2)
    """
    @staticmethod
    def skew_symmetric(t: torch.Tensor) -> torch.Tensor:
        """Returns [3, 3] skew-symmetric matrix [t]_x for vector t [3]."""
        zero = torch.zeros(1, device=t.device, dtype=t.dtype)
        t_x = torch.stack([
            zero[0], -t[2], t[1],
            t[2], zero[0], -t[0],
            -t[1], t[0], zero[0]
        ]).reshape(3, 3)
        return t_x

    @classmethod
    def compute_fundamental_matrix(cls, R_gt: torch.Tensor, t_gt: torch.Tensor, K0: torch.Tensor, K1: torch.Tensor) -> torch.Tensor:
        """Computes Fundamental matrix F = K1^{-T} [t]_x R K0^{-1}."""
        t_skew = cls.skew_symmetric(t_gt.flatten())
        E = t_skew @ R_gt
        K0_inv = torch.linalg.inv(K0)
        K1_inv_T = torch.linalg.inv(K1).transpose(0, 1)
        F = K1_inv_T @ E @ K0_inv
        return F / (torch.norm(F) + 1e-7)

    def forward(self, pts0: torch.Tensor, pts1: torch.Tensor, F_mat: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pts0: [N, 2], pts1: [N, 2], F_mat: [3, 3]
        """
        N = pts0.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=pts0.device, requires_grad=True)

        ones = torch.ones((N, 1), device=pts0.device, dtype=pts0.dtype)
        p0_h = torch.cat([pts0, ones], dim=1) # [N, 3]
        p1_h = torch.cat([pts1, ones], dim=1) # [N, 3]

        # Epipolar lines
        # l1 = F @ p0_h^T -> [3, N]
        l1 = (F_mat @ p0_h.T).T # [N, 3]
        # l0 = F^T @ p1_h^T -> [N, 3]
        l0 = (F_mat.T @ p1_h.T).T # [N, 3]

        # Algebraic distance p1^T F p0
        p1_F_p0 = torch.sum(p1_h * l1, dim=1) # [N]

        # Geometric Sampson / Symmetric Epipolar distance
        norm_l1 = l1[:, 0]**2 + l1[:, 1]**2 + 1e-7
        norm_l0 = l0[:, 0]**2 + l0[:, 1]**2 + 1e-7

        sed = (p1_F_p0 ** 2) * (1.0 / norm_l1 + 1.0 / norm_l0)

        # Robust Huber loss on epipolar distance
        delta = 3.0 # pixels
        huber_sed = torch.where(sed < delta**2, 0.5 * sed, delta * (torch.sqrt(sed + 1e-6) - 0.5 * delta))

        if weights is not None:
            return torch.sum(weights * huber_sed) / (torch.sum(weights) + 1e-7)
        return torch.mean(huber_sed)


class DifferentiableSoftmAALoss(nn.Module):
    """
    Smooth Differentiable Surrogate of Kaggle IMC mAA metric.
    Replaces non-differentiable step functions with smooth sigmoids:
    mAA = 1 - (1/|Theta|) * sum_{th} sigmoid((th - e_pose) / tau)
    """
    def __init__(self, thresholds: Optional[List[float]] = None, temperature: float = 1.0):
        super().__init__()
        self.thresholds = thresholds or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.temperature = temperature

    def forward(self, pose_error_deg: torch.Tensor) -> torch.Tensor:
        """
        pose_error_deg: scalar or tensor of angular errors in degrees.
        Returns loss in [0, 1], where 0 = perfect mAA (100%), 1 = 0% accuracy.
        """
        accuracies = []
        for th in self.thresholds:
            # High probability if error < threshold
            p_correct = torch.sigmoid((th - pose_error_deg) / self.temperature)
            accuracies.append(p_correct)

        soft_mAA = torch.mean(torch.stack(accuracies))
        # Loss to minimize
        return 1.0 - soft_mAA


class AdvancedTriFoldAutoMLLoss(nn.Module):
    """
    Tri-Fold Composite Geometric-Perceptual Loss with AutoML Adaptive Weighting:
    L = w1 * L_Epipolar + w2 * L_Soft_mAA + w3 * L_Uncertainty_Regularization
    """
    def __init__(self, w_sed: float = 0.5, w_maa: float = 0.3, w_unc: float = 0.2):
        super().__init__()
        self.sed_loss = DifferentiableEpipolarLoss()
        self.soft_maa_loss = DifferentiableSoftmAALoss(temperature=1.0)

        # Log-variance uncertainty loss weights (Kendall et al. homoscedastic task weighting)
        self.log_var_sed = nn.Parameter(torch.tensor(0.0))
        self.log_var_maa = nn.Parameter(torch.tensor(0.0))
        self.log_var_unc = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        pts0: torch.Tensor,
        pts1: torch.Tensor,
        F_mat: torch.Tensor,
        pose_err_deg: torch.Tensor,
        uncertainty0: torch.Tensor,
        uncertainty1: torch.Tensor,
        scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        l_sed = self.sed_loss(pts0, pts1, F_mat, weights=scores)
        l_maa = self.soft_maa_loss(pose_err_deg)

        # Uncertainty regularization: penalize high confidence in high-uncertainty regions
        u_mean = 0.5 * (torch.mean(uncertainty0) + torch.mean(uncertainty1))
        l_unc = torch.mean(scores * u_mean)

        # Precision-weighted multi-task loss with learned heteroscedastic balance
        precision_sed = torch.exp(-self.log_var_sed)
        precision_maa = torch.exp(-self.log_var_maa)
        precision_unc = torch.exp(-self.log_var_unc)

        total_loss = (
            precision_sed * l_sed + self.log_var_sed +
            precision_maa * l_maa + self.log_var_maa +
            precision_unc * l_unc + self.log_var_unc
        )

        return {
            "total_loss": total_loss,
            "l_sed": l_sed,
            "l_maa": l_maa,
            "l_unc": l_unc
        }
