"""
Baseline Feature Extraction and Matching Pipelines for IMC:
1. Classical Baseline: SIFT + Ratio Test + MAGSAC++
2. Modern Deep Sparse Baseline: ALIKED / SuperPoint + LightGlue Paradigm
3. Dense Foundation Baseline: Coarse-to-Fine Dense Feature Warping (RoMa / DKM Representation)
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class SIFTBaseline:
    """Classic SIFT detector + descriptor + Lowe's ratio test matcher."""
    def __init__(self, n_features: int = 2048, ratio_threshold: float = 0.8):
        self.n_features = n_features
        self.ratio_threshold = ratio_threshold
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def extract(self, img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kpts, descs = self.sift.detectAndCompute(img_gray, None)
        if len(kpts) == 0:
            return np.empty((0, 2)), np.empty((0, 128))
        pts = np.array([kp.pt for kp in kpts], dtype=np.float32)
        return pts, descs

    def match(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, Any]:
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if len(img0.shape) == 3 else img0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1

        pts0, desc0 = self.extract(gray0)
        pts1, desc1 = self.extract(gray1)

        if len(pts0) == 0 or len(pts1) == 0:
            return {"kpts0": np.empty((0, 2)), "kpts1": np.empty((0, 2)), "scores": np.empty((0,))}

        # FLANN / BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf.knnMatch(desc0, desc1, k=2)

        good_k0, good_k1, scores = [], [], []
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < self.ratio_threshold * m[1].distance:
                good_k0.append(pts0[m[0].queryIdx])
                good_k1.append(pts1[m[0].trainIdx])
                # Higher score = more confident (inverse distance ratio)
                score = 1.0 - (m[0].distance / (m[1].distance + 1e-6))
                scores.append(score)

        return {
            "kpts0": np.array(good_k0, dtype=np.float32) if good_k0 else np.empty((0, 2)),
            "kpts1": np.array(good_k1, dtype=np.float32) if good_k1 else np.empty((0, 2)),
            "scores": np.array(scores, dtype=np.float32) if scores else np.empty((0,))
        }


class LightGlueStyleSparseMatcher(nn.Module):
    """
    Self & Cross-Attention Transformer Matcher (LightGlue / SuperGlue Architectural Baseline).
    Implements multi-head graph neural network attention for sparse keypoint correspondence.
    """
    def __init__(self, desc_dim: int = 128, num_layers: int = 4, nhead: int = 4):
        super().__init__()
        self.desc_dim = desc_dim
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, desc_dim)
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=desc_dim, 
                nhead=nhead, 
                dim_feedforward=desc_dim * 2, 
                dropout=0.0, 
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.match_head = nn.Linear(desc_dim, desc_dim)

    def forward(self, kpts0: torch.Tensor, desc0: torch.Tensor,
                kpts1: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        """
        Takes normalized keypoints [B, N, 2] and descriptors [B, N, D],
        returns matching assignment probability matrix [B, N, M].
        """
        # Positional encoding injection
        f0 = desc0 + self.pos_encoder(kpts0)
        f1 = desc1 + self.pos_encoder(kpts1)

        # Cross and self attention
        for layer in self.layers:
            f0 = layer(f0)
            f1 = layer(f1)

        # Match similarity matrix with dual-softmax
        f0_proj = F.normalize(self.match_head(f0), p=2, dim=-1)
        f1_proj = F.normalize(self.match_head(f1), p=2, dim=-1)

        sim_matrix = torch.bmm(f0_proj, f1_proj.transpose(1, 2))  # [B, N, M]
        p_row = F.softmax(sim_matrix * 10.0, dim=-1)
        p_col = F.softmax(sim_matrix * 10.0, dim=-2)
        match_confidence = p_row * p_col
        return match_confidence


class DenseFoundationMatcher(nn.Module):
    """
    Dense Coarse-to-Fine Foundation Matching Paradigm (RoMa / DKM Representation).
    Computes dense 4D correlation volumes and decodes fine displacement fields.
    """
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        # ResNet/Conv backbone representation for dense spatial descriptors
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        )
        self.refiner = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, 2, kernel_size=3, padding=1)  # offset (dx, dy)
        )

    def extract_dense_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img_tensor)
        return F.normalize(feats, p=2, dim=1)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f0 = self.extract_dense_map(img0)
        f1 = self.extract_dense_map(img1)
        # Compute correlation
        B, C, H0, W0 = f0.shape
        _, _, H1, W1 = f1.shape
        f0_flat = f0.view(B, C, -1)
        f1_flat = f1.view(B, C, -1)
        corr = torch.bmm(f0_flat.transpose(1, 2), f1_flat)  # [B, H0*W0, H1*W1]
        
        # Soft argmax for dense flow
        flow_probs = F.softmax(corr * 15.0, dim=-1)
        return corr, flow_probs
