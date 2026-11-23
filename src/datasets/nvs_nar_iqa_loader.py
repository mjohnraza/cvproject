"""
Non-Aligned Reference Image Quality Assessment Database for Novel View Synthesis (NVS-NAR-IQA)
Benchmark Loader and Distortion Simulator.

Paper: 'Non-Aligned Reference Image Quality Assessment for Novel View Synthesis'
(Ghildyal et al., UT Austin LIVE Laboratory / CVPR).

This module simulates and evaluates matching under authentic NVS-NAR-IQA distortion regimes:
1. Volumetric Blur & Splatting Artifacts (Gaussian Splatting defocus & anisotropic smearing)
2. High-Frequency Radiance Floaters (NeRF volumetric density smoke & floater artifacts)
3. Non-Aligned Reference Viewpoint Disparities (viewpoint extrapolation error)
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

class NVSNARIQADistortionEngine:
    """
    Applies authentic Novel View Synthesis (NeRF & 3DGS) distortion models
    to benchmark non-aligned reference matching robustness.
    """
    @staticmethod
    def apply_splatting_blur(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
        """Simulates 3D Gaussian Splatting anisotropic scale blur & popping artifacts."""
        ksize = int(round(3 + severity * 8))
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=severity * 2.5)
        # Blend with subtle high-frequency edge loss
        return cv2.addWeighted(img, 1.0 - severity * 0.4, blurred, severity * 0.4, 0)

    @staticmethod
    def inject_nerf_floaters(img: np.ndarray, num_floaters: int = 25, seed: int = 101) -> np.ndarray:
        """Injects synthetic high-frequency density 'floaters' characteristic of NeRF density field errors."""
        np.random.seed(seed)
        H, W, C = img.shape
        out = img.copy()
        for _ in range(num_floaters):
            cx = np.random.randint(10, W - 10)
            cy = np.random.randint(10, H - 10)
            radius = np.random.randint(3, 14)
            floater_color = np.random.randint(120, 255, (3,)).tolist()
            alpha = np.random.uniform(0.3, 0.75)
            overlay = out.copy()
            cv2.circle(overlay, (cx, cy), radius, floater_color, -1)
            cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
        return out

    @staticmethod
    def apply_color_radiance_shift(img: np.ndarray, shift_factor: float = 0.2) -> np.ndarray:
        """Simulates view-dependent specular color inconsistency across novel views."""
        img_float = img.astype(np.float32) / 255.0
        # Perturb color temperature and illumination gain
        gain = np.array([1.0 + shift_factor * 0.5, 1.0 - shift_factor * 0.3, 1.0 + shift_factor * 0.2])
        img_distorted = np.clip(img_float * gain, 0.0, 1.0)
        return (img_distorted * 255.0).astype(np.uint8)

def create_nvs_nar_iqa_test_suite(
    base_pairs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Transforms clean multi-view pairs into NVS-NAR-IQA distorted non-aligned evaluation pairs:
    - Condition A: Clean Reference vs Clean Reference (Standard IMC baseline)
    - Condition B: Clean Reference vs 3DGS-Synthesized Novel View (Splatting blur)
    - Condition C: Clean Reference vs NeRF-Synthesized Novel View (Floaters + Radiance noise)
    - Condition D: Severe Multi-Model NVS Degradation (Compound distortions)
    """
    nvs_pairs = []
    engine = NVSNARIQADistortionEngine()

    for idx, pair in enumerate(base_pairs):
        clean_img0 = pair["img0"]
        clean_img1 = pair["img1"]

        # 1. 3DGS blurred novel view
        img1_3dgs = engine.apply_splatting_blur(clean_img1, severity=0.6)
        nvs_pairs.append({
            **pair,
            "img0": clean_img0,
            "img1": img1_3dgs,
            "condition": "3DGS_Blur",
            "pair_id": f"pair_{idx}_3dgs"
        })

        # 2. NeRF floater corrupted novel view
        img1_nerf = engine.inject_nerf_floaters(clean_img1, num_floaters=35, seed=idx)
        nvs_pairs.append({
            **pair,
            "img0": clean_img0,
            "img1": img1_nerf,
            "condition": "NeRF_Floaters",
            "pair_id": f"pair_{idx}_nerf"
        })

        # 3. Compound NVS synthesized view
        img1_compound = engine.apply_color_radiance_shift(img1_3dgs, shift_factor=0.25)
        img1_compound = engine.inject_nerf_floaters(img1_compound, num_floaters=20, seed=idx + 99)
        nvs_pairs.append({
            **pair,
            "img0": clean_img0,
            "img1": img1_compound,
            "condition": "Compound_NVS_Artifacts",
            "pair_id": f"pair_{idx}_compound"
        })

    return nvs_pairs
