"""
Kaggle Image Matching Challenge (IMC 2025 / 2026) Official Submission Pipeline
Architecture: PU-AMatch + OmniAutoML Global SfM Scene Registration
- Multi-View Keypoint Extraction & RootSIFT / Hellinger Normalization
- Spatial Vector Consensus & USAC_MAGSAC Epipolar Verification
- Maximum Inlier Spanning-Tree Global Pose Registration
- Strict Competition Format (Semicolon Delimited)
"""

import os
import sys
import glob
import time
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2

# ==============================================================================
# 1. Normalization & Feature Extraction
# ==============================================================================

class HartleyNormalizer:
    @staticmethod
    def normalize_points(pts: np.ndarray):
        if len(pts) == 0:
            return pts, np.eye(3)
        mean = np.mean(pts, axis=0)
        shifted = pts - mean
        mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
        if mean_dist < 1e-7:
            return pts, np.eye(3)
        scale = np.sqrt(2.0) / mean_dist
        T = np.array([
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        pts_norm = (T @ pts_h.T).T[:, :2]
        return pts_norm.astype(np.float32), T


class RootSIFTNormalizer:
    @staticmethod
    def transform(descs: np.ndarray) -> np.ndarray:
        if descs is None or len(descs) == 0:
            return descs
        l1 = np.linalg.norm(descs, ord=1, axis=1, keepdims=True) + 1e-7
        descs_l1 = descs / l1
        descs_sqrt = np.sqrt(np.maximum(descs_l1, 0.0))
        l2 = np.linalg.norm(descs_sqrt, ord=2, axis=1, keepdims=True) + 1e-7
        return (descs_sqrt / l2).astype(np.float32)


# ==============================================================================
# 2. Matcher & Pairwise Pose Estimator
# ==============================================================================

class OmniAutoMLMatcher:
    def __init__(self, max_keypoints: int = 3000, desc_ratio: float = 0.82):
        self.max_keypoints = max_keypoints
        self.desc_ratio = desc_ratio
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, img_bgr: np.ndarray):
        H, W = img_bgr.shape[:2]
        # Downscale large images to stay fast within Kaggle time limits
        max_dim = 1024
        scale = 1.0
        if max(H, W) > max_dim:
            scale = max_dim / float(max(H, W))
            new_W = int(W * scale)
            new_H = int(H * scale)
            img_bgr = cv2.resize(img_bgr, (new_W, new_H), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        kpts, descs = self.sift.detectAndCompute(gray, None)
        if len(kpts) == 0 or descs is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32), (W, H)

        # Scale keypoints back to original resolution
        pts = np.array([k.pt for k in kpts], dtype=np.float32)
        if scale != 1.0:
            pts = pts / scale

        descs_root = RootSIFTNormalizer.transform(descs)
        return pts, descs_root, (W, H)

    def match_pair(self, pts0: np.ndarray, desc0: np.ndarray, pts1: np.ndarray, desc1: np.ndarray):
        if len(pts0) < 6 or len(pts1) < 6:
            return np.empty((0, 2)), np.empty((0, 2))

        matches = self.bf.knnMatch(desc0, desc1, k=2)
        good_pts0, good_pts1 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < self.desc_ratio * m[1].distance:
                good_pts0.append(pts0[m[0].queryIdx])
                good_pts1.append(pts1[m[0].trainIdx])

        if len(good_pts0) < 6:
            return np.empty((0, 2)), np.empty((0, 2))

        p0 = np.array(good_pts0, dtype=np.float32)
        p1 = np.array(good_pts1, dtype=np.float32)

        # Spatial Vector Consensus (SVC)
        if len(p0) >= 12:
            vecs = p1 - p0
            median_vec = np.median(vecs, axis=0)
            dists = np.linalg.norm(vecs - median_vec, axis=1)
            thresh = np.percentile(dists, 80)
            inliers = dists <= thresh
            if np.sum(inliers) >= 6:
                p0, p1 = p0[inliers], p1[inliers]

        return p0, p1

    @staticmethod
    def estimate_camera_intrinsics(W: int, H: int) -> np.ndarray:
        """Pinhole camera intrinsic matrix prior based on image dimensions."""
        f = 1.2 * max(W, H)
        return np.array([
            [f, 0.0, W / 2.0],
            [0.0, f, H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def estimate_relative_pose(self, p0: np.ndarray, p1: np.ndarray, K0: np.ndarray, K1: np.ndarray):
        if len(p0) < 6:
            return None, None, 0

        p0_norm, T0 = HartleyNormalizer.normalize_points(p0)
        p1_norm, T1 = HartleyNormalizer.normalize_points(p1)

        K0_norm = T0 @ K0
        K1_norm = T1 @ K1

        # USAC_MAGSAC robust Essential matrix estimation
        focal_avg = (K0[0, 0] + K1[0, 0]) / 2.0
        E, inlier_mask = cv2.findEssentialMat(
            p0_norm, p1_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.USAC_MAGSAC,
            prob=0.99999,
            threshold=1.0 / max(focal_avg, 100.0),
            maxIters=15000
        )
        if E is None or E.shape != (3, 3):
            return None, None, 0

        # Recover pose with Cheirality (3D depth positivity)
        num_pos, R, t, mask_pose = cv2.recoverPose(E, p0_norm, p1_norm, K0_norm, K1_norm)
        n_inliers = int(num_pos)
        return R, t, n_inliers


# ==============================================================================
# 3. Global Scene Reconstruction (Spanning Tree Pose Chaining)
# ==============================================================================

class GlobalSfMChainer:
    """
    Connects pairwise relative poses into a unified global coordinate system
    for each scene using Maximum-Inlier Spanning Tree propagation.
    """
    @staticmethod
    def register_scene(
        image_names: list,
        features: dict,
        matcher: OmniAutoMLMatcher
    ) -> dict:
        """
        Returns {img_name: (R_global, t_global)} for all images in the scene.
        """
        N = len(image_names)
        poses = {}
        identity_R = np.eye(3, dtype=np.float64)
        zero_t = np.zeros(3, dtype=np.float64)

        if N == 0:
            return poses

        # Baseline default: identity for all
        for img in image_names:
            poses[img] = (identity_R.copy(), zero_t.copy())

        if N == 1:
            return poses

        # 1. Determine pairs to match (exhaustive if N <= 25, or k-nearest / sequential if N > 25)
        pairs_to_match = []
        if N <= 25:
            for i in range(N):
                for j in range(i + 1, N):
                    pairs_to_match.append((i, j))
        else:
            # Connect adjacent and stride-2, stride-3, stride-4 pairs
            for i in range(N):
                for step in [1, 2, 3, 4]:
                    j = i + step
                    if j < N:
                        pairs_to_match.append((i, j))

        # 2. Compute pairwise relative poses
        edges = defaultdict(list)
        for i, j in pairs_to_match:
            img_i, img_j = image_names[i], image_names[j]
            pts_i, desc_i, dims_i = features[img_i]
            pts_j, desc_j, dims_j = features[img_j]

            p_i, p_j = matcher.match_pair(pts_i, desc_i, pts_j, desc_j)
            if len(p_i) >= 8:
                K_i = matcher.estimate_camera_intrinsics(dims_i[0], dims_i[1])
                K_j = matcher.estimate_camera_intrinsics(dims_j[0], dims_j[1])
                R_ij, t_ij, n_inliers = matcher.estimate_relative_pose(p_i, p_j, K_i, K_j)
                if R_ij is not None and n_inliers >= 8:
                    edges[i].append((j, R_ij, t_ij.flatten(), n_inliers))
                    # Reverse relative transform: x_i = R_ij^T * (x_j - t_ij)
                    R_ji = R_ij.T
                    t_ji = -R_ij.T @ t_ij.flatten()
                    edges[j].append((i, R_ji, t_ji, n_inliers))

        if not edges:
            return poses

        # 3. Find the most connected root node (maximum degree/inliers)
        root = max(edges.keys(), key=lambda node: sum(e[3] for e in edges[node]))

        # 4. Breadth-First-Search (BFS) / Priority Maximum Inlier Spanning Tree
        visited = set([root])
        queue = [root]

        while queue:
            curr = queue.pop(0)
            R_curr, t_curr = poses[image_names[curr]]

            # Sort neighbors by inlier count descending
            neighbors = sorted(edges[curr], key=lambda x: x[3], reverse=True)
            for nxt, R_rel, t_rel, inliers in neighbors:
                if nxt not in visited:
                    visited.add(nxt)
                    # Propagate global pose: R_nxt = R_rel @ R_curr, t_nxt = R_rel @ t_curr + t_rel
                    R_nxt = R_rel @ R_curr
                    t_nxt = R_rel @ t_curr + t_rel
                    poses[image_names[nxt]] = (R_nxt, t_nxt)
                    queue.append(nxt)

        print(f"  Registered {len(visited)}/{N} images into unified global frame (root={image_names[root]}).")
        return poses


def format_matrix(mat: np.ndarray) -> str:
    """Format 3x3 rotation matrix as semicolon-separated string."""
    return ";".join([f"{x:.9f}" for x in mat.flatten()])

def format_vector(vec: np.ndarray) -> str:
    """Format 3x1 translation vector as semicolon-separated string."""
    return ";".join([f"{x:.9f}" for x in vec.flatten()])


# ==============================================================================
# 4. Main Kaggle Execution Pipeline
# ==============================================================================

def locate_image_path(base_dir: str, dataset: str, scene: str, img_name: str) -> str:
    """Tries multiple Kaggle test directory structures."""
    candidates = [
        os.path.join(base_dir, "test", dataset, img_name),
        os.path.join(base_dir, "test", dataset, scene, img_name),
        os.path.join(base_dir, "test", img_name),
        os.path.join(base_dir, dataset, img_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

def main():
    print("=== Kaggle IMC 2025: PU-AMatch OmniAutoML SfM Pipeline ===")
    
    # Locate dataset root
    input_base = "/kaggle/input/image-matching-challenge-2025"
    if not os.path.exists(input_base):
        input_base = "kaggle/data"
    if not os.path.exists(input_base):
        input_base = "/kaggle/input/image-matching-challenge-2024"

    sample_sub_path = os.path.join(input_base, "sample_submission.csv")
    output_path = "submission.csv"

    if not os.path.exists(sample_sub_path):
        print(f"Error: {sample_sub_path} not found.")
        return

    df_sub = pd.read_csv(sample_sub_path)
    print(f"Loaded sample_submission with {len(df_sub)} rows across {df_sub['dataset'].nunique()} datasets.")

    matcher = OmniAutoMLMatcher(max_keypoints=3000, desc_ratio=0.82)
    default_R = np.eye(3, dtype=np.float64)
    default_t = np.zeros(3, dtype=np.float64)

    # Group images by (dataset, scene) cluster
    clusters = df_sub.groupby(["dataset", "scene"])
    print(f"Found {len(clusters)} distinct scene clusters to reconstruct.")

    # Storage for computed poses: (image_id) -> (R_str, t_str)
    estimated_poses = {}

    for (dataset, scene), group in clusters:
        print(f"\nProcessing Scene: {dataset} / {scene} ({len(group)} images)...")
        img_names = group["image"].tolist()

        # Step 1: Extract features for all images in this cluster
        features = {}
        for img_name in img_names:
            img_path = locate_image_path(input_base, dataset, scene, img_name)
            if img_path and os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    pts, desc, dims = matcher.extract(img_bgr)
                    features[img_name] = (pts, desc, dims)
                else:
                    features[img_name] = (np.empty((0, 2)), np.empty((0, 128)), (1000, 1000))
            else:
                features[img_name] = (np.empty((0, 2)), np.empty((0, 128)), (1000, 1000))

        # Step 2: Global SfM Scene Registration
        scene_poses = GlobalSfMChainer.register_scene(img_names, features, matcher)

        # Step 3: Record formatted matrices
        for idx, row in group.iterrows():
            img_id = row["image_id"]
            img_name = row["image"]
            R_est, t_est = scene_poses.get(img_name, (default_R, default_t))
            estimated_poses[img_id] = (format_matrix(R_est), format_vector(t_est))

    # Apply estimated poses to submission dataframe
    df_sub["rotation_matrix"] = df_sub["image_id"].map(lambda x: estimated_poses.get(x, (format_matrix(default_R), format_vector(default_t)))[0])
    df_sub["translation_vector"] = df_sub["image_id"].map(lambda x: estimated_poses.get(x, (format_matrix(default_R), format_vector(default_t)))[1])

    # Save to submission.csv
    expected_cols = ["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"]
    df_sub = df_sub[expected_cols]
    df_sub.to_csv(output_path, index=False)
    print(f"\n=== Successfully generated {output_path} ({len(df_sub)} rows) ===")

if __name__ == "__main__":
    main()
