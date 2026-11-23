"""
Synthetic & Realistic Ground-Truth Scene Generator for IMC 3D Image Matching Evaluation.
Generates multi-view pairs with ground-truth camera rotations, translations, intrinsics,
and 3D world geometry to enable rigorous mAA metric computation.
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from typing import List, Dict, Any, Tuple

def create_synthetic_textured_scene(
    num_views: int = 12,
    img_size: Tuple[int, int] = (640, 480),
    num_3d_points: int = 1500,
    noise_sigma: float = 0.5,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], np.ndarray]:
    """
    Creates a full 3D scene with textured planar/volumetric objects,
    camera trajectory, ground-truth camera extrinsics (R, t) and intrinsics (K).
    """
    np.random.seed(seed)
    W, H = img_size
    focal = 0.85 * W
    cx, cy = W / 2.0, H / 2.0
    K = np.array([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Generate 3D surface points in world space
    pts_surface = []
    # Planar facades with textured features
    for face_idx in range(4):
        x = np.random.uniform(-2.0, 2.0, num_3d_points // 4)
        y = np.random.uniform(-1.5, 1.5, num_3d_points // 4)
        z = np.full_like(x, 4.0 + face_idx * 1.2) + np.random.normal(0, 0.05, len(x))
        pts_surface.append(np.stack([x, y, z], axis=1))
    pts_3d = np.concatenate(pts_surface, axis=0) # [N, 3]

    images = []
    camera_metadata = []

    # Camera trajectory on an arc around the scene
    angles = np.linspace(-35.0, 35.0, num_views)
    for i, angle_deg in enumerate(angles):
        rad = np.radians(angle_deg)
        # Camera rotation: yaw rotation + slight pitch
        rot = R_scipy.from_euler('yxz', [rad, -0.05 * np.sin(rad), 0.0]).as_matrix()
        # Camera center
        cam_center = np.array([2.5 * np.sin(rad), 0.2 * np.cos(rad), -2.5 * (1.0 - np.cos(rad))], dtype=np.float64)
        # Translation t = -R @ C
        t = -rot @ cam_center

        # Render 2D image projected from 3D points with rich textures
        img = np.ones((H, W, 3), dtype=np.uint8) * 230
        
        # Draw background architectural structures and textured patterns
        for grid_x in range(0, W, 40):
            cv2.line(img, (grid_x, 0), (grid_x, H), (200, 200, 200), 1)
        for grid_y in range(0, H, 40):
            cv2.line(img, (0, grid_y), (W, grid_y), (200, 200, 200), 1)

        # Project 3D points into camera view
        pts_cam = (rot @ pts_3d.T + t[:, None]).T # [N, 3]
        valid_z = pts_cam[:, 2] > 0.1
        pts_2d_homo = (K @ pts_cam[valid_z].T).T
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]

        # Add Gaussian localization noise
        pts_2d += np.random.normal(0, noise_sigma, pts_2d.shape)

        # Draw distinct feature textures (corners, circles, checkerboards)
        for p_idx, p in enumerate(pts_2d):
            px, py = int(round(p[0])), int(round(p[1]))
            if 10 <= px < W - 10 and 10 <= py < H - 10:
                color = (
                    int(40 + (p_idx * 17) % 200),
                    int(30 + (p_idx * 29) % 200),
                    int(50 + (p_idx * 43) % 200)
                )
                if p_idx % 4 == 0:
                    cv2.circle(img, (px, py), 3, color, -1)
                elif p_idx % 4 == 1:
                    cv2.rectangle(img, (px - 3, py - 3), (px + 3, py + 3), color, -1)
                elif p_idx % 4 == 2:
                    cv2.line(img, (px - 4, py), (px + 4, py), color, 2)
                    cv2.line(img, (px, py - 4), (px, py + 4), color, 2)
                else:
                    cv2.circle(img, (px, py), 4, color, 1)

        # Apply realistic illumination gradient
        gradient = np.tile(np.linspace(0.85, 1.15, W), (H, 1))[:, :, None]
        img = np.clip(img * gradient, 0, 255).astype(np.uint8)

        images.append(img)
        camera_metadata.append({
            "R": rot,
            "t": t,
            "K": K,
            "center": cam_center,
            "view_id": i
        })

    return images, camera_metadata, pts_3d

def generate_benchmark_evaluation_pairs(
    images: List[np.ndarray], 
    metadata: List[Dict[str, Any]], 
    max_baseline_step: int = 4
) -> List[Dict[str, Any]]:
    """
    Pairs up images with varying baseline disparities (from easy small baseline to hard wide baseline).
    Computes exact relative ground-truth rotation R_rel = R1 @ R0^T and translation t_rel = t1 - R_rel @ t0.
    """
    pairs = []
    num_views = len(images)
    for i in range(num_views):
        for step in range(1, max_baseline_step + 1):
            j = i + step
            if j < num_views:
                R0 = metadata[i]["R"]
                t0 = metadata[i]["t"]
                R1 = metadata[j]["R"]
                t1 = metadata[j]["t"]

                # Relative ground-truth transform: x1 = R_rel * x0 + t_rel
                R_rel = R1 @ R0.T
                t_rel = t1 - R_rel @ t0

                # Difficulty level according to angular baseline
                yaw_diff = abs(i - j) * 6.5

                pairs.append({
                    "idx0": i,
                    "idx1": j,
                    "img0": images[i],
                    "img1": images[j],
                    "K0": metadata[i]["K"],
                    "K1": metadata[j]["K"],
                    "R_gt": R_rel,
                    "t_gt": t_rel,
                    "baseline_deg": yaw_diff,
                    "difficulty": "Easy" if step == 1 else ("Medium" if step == 2 else "Hard")
                })
    return pairs
