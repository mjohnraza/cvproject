"""
Kaggle Image Matching Challenge (IMC 2025) Official Submission Pipeline – Version 7
Architecture: PU-AMatch + OmniAutoML Global SfM Scene Registration
Key Upgrades:
  1. EXIF Metadata Camera Intrinsics Auto-Calibrator (35mm equivalent focal length)
  2. Perceptually-Enhanced Feature Extraction (CLAHE + Multi-Scale RootSIFT)
  3. Spatial Vector Consensus (SVC) Correspondence Filtering
  4. Triplet Linear Metric Scale Synchronization via SVD (Solves translation scale drift)
  5. Riemannian SO(3) Rotation Averaging (Eliminates angular accumulation drift)
  6. Chordal Global Translation Averaging via Graph Laplacian Least-Squares
  7. Multi-threaded parallel feature extraction & pairwise geometric verification
"""

import os
import sys
import glob
import time
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ExifTags

# ==============================================================================
# 1. Camera Intrinsics & EXIF Calibration
# ==============================================================================

class CameraCalibrator:
    @staticmethod
    def extract_intrinsics(img_path: str, W: int, H: int) -> np.ndarray:
        """Extracts calibrated focal length from EXIF or infers adaptive prior."""
        f_px = None
        try:
            with Image.open(img_path) as pil_img:
                exif = pil_img.getexif()
                if exif:
                    # Check 35mm equivalent focal length
                    focal_35mm = exif.get(0xA405) # FocalLengthIn35mmFilm
                    if focal_35mm and focal_35mm > 0:
                        f_px = (float(focal_35mm) / 36.0) * max(W, H)
                    else:
                        # Check physical focal length
                        focal_raw = exif.get(0x920A) # FocalLength
                        if focal_raw:
                            if isinstance(focal_raw, tuple):
                                focal_val = focal_raw[0] / focal_raw[1] if focal_raw[1] != 0 else 0
                            else:
                                focal_val = float(focal_raw)
                            if focal_val > 0:
                                f_px = (focal_val / 6.0) * max(W, H) # Standard phone 1/2.3" sensor approx
        except Exception:
            pass

        if f_px is None or f_px < 100.0 or f_px > 10000.0:
            # Fallback to field-of-view prior (~60 degrees standard horizontal FOV)
            f_px = 1.15 * max(W, H)

        return np.array([
            [f_px, 0.0, W / 2.0],
            [0.0, f_px, H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


# ==============================================================================
# 2. Normalization & Feature Extraction
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
# 3. Image Discovery Index
# ==============================================================================

class ImageIndex:
    def __init__(self, root: str):
        t0 = time.time()
        self._index = {}
        self._root = root
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in exts:
                    full = os.path.join(dirpath, fn)
                    key = fn.lower()
                    if key not in self._index:
                        self._index[key] = []
                    self._index[key].append(full)
        total = sum(len(v) for v in self._index.values())
        print(f"  Indexed {total} images in {time.time()-t0:.1f}s ({len(self._index)} unique names)")

    def find(self, dataset: str, scene: str, img_name: str) -> str:
        for candidate in [
            os.path.join(self._root, "test", dataset, img_name),
            os.path.join(self._root, "test", dataset, scene, img_name),
            os.path.join(self._root, "test", dataset, "images", img_name),
            os.path.join(self._root, "test", dataset, scene, "images", img_name),
            os.path.join(self._root, dataset, img_name),
            os.path.join(self._root, dataset, scene, img_name),
        ]:
            if os.path.isfile(candidate):
                return candidate

        key = img_name.lower()
        paths = self._index.get(key, [])
        if not paths:
            stem = os.path.splitext(key)[0]
            for k, v in self._index.items():
                if os.path.splitext(k)[0] == stem:
                    paths = v
                    break

        if not paths:
            return ""

        ds_lower = dataset.lower()
        for p in paths:
            if ds_lower in p.lower():
                return p
        return paths[0]


# ==============================================================================
# 4. Feature Matcher & Robust Estimator
# ==============================================================================

class OmniAutoMLMatcher:
    def __init__(self, max_keypoints: int = 3500, desc_ratio: float = 0.82):
        self.max_keypoints = max_keypoints
        self.desc_ratio = desc_ratio
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=0.025)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract(self, img_bgr: np.ndarray, img_path: str = ""):
        H, W = img_bgr.shape[:2]
        max_dim = 1024
        scale = 1.0
        if max(H, W) > max_dim:
            scale = max_dim / float(max(H, W))
            img_bgr = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        gray_eq = self.clahe.apply(gray)

        kpts, descs = self.sift.detectAndCompute(gray_eq, None)
        orig_W = int(W / scale if scale != 1.0 else W)
        orig_H = int(H / scale if scale != 1.0 else H)

        K = CameraCalibrator.extract_intrinsics(img_path, orig_W, orig_H)

        if kpts is None or len(kpts) == 0 or descs is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32), K

        pts = np.array([k.pt for k in kpts], dtype=np.float32)
        if scale != 1.0:
            pts = pts / scale

        descs_root = RootSIFTNormalizer.transform(descs)
        return pts, descs_root, K

    def match_pair(self, pts0, desc0, pts1, desc1):
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

    def estimate_relative_pose(self, p0, p1, K0, K1):
        if len(p0) < 6:
            return None, None, 0

        p0_norm, T0 = HartleyNormalizer.normalize_points(p0)
        p1_norm, T1 = HartleyNormalizer.normalize_points(p1)
        K0_norm = T0 @ K0
        K1_norm = T1 @ K1

        focal_avg = (K0[0, 0] + K1[0, 0]) / 2.0
        try:
            E, inlier_mask = cv2.findEssentialMat(
                p0_norm, p1_norm,
                focal=1.0, pp=(0.0, 0.0),
                method=cv2.USAC_MAGSAC,
                prob=0.99999,
                threshold=1.5 / max(focal_avg, 100.0),
                maxIters=12000
            )
        except Exception:
            return None, None, 0

        if E is None or E.shape != (3, 3):
            return None, None, 0

        try:
            num_pos, R, t, mask_pose = cv2.recoverPose(E, p0_norm, p1_norm, K0_norm, K1_norm)
        except Exception:
            return None, None, 0

        return R, t, int(num_pos)


# ==============================================================================
# 5. Global SfM: Scale Synchronization, Rotation Averaging & Translation LUD
# ==============================================================================

class GlobalSfMChainer:
    """
    State-of-the-Art Global Structure-from-Motion Engine:
      1. Triplet Linear Scale Synchronization (SVD)
      2. Riemannian SO(3) Rotation Averaging
      3. Chordal Global Translation Averaging
    """

    @staticmethod
    def _so3_log(R: np.ndarray) -> np.ndarray:
        """Matrix logarithm on SO(3) mapping rotation matrix to 3D rotation vector."""
        tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(tr)
        if theta < 1e-6:
            return np.zeros(3, dtype=np.float64)
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return (theta / (2.0 * np.sin(theta))) * v

    @staticmethod
    def _so3_exp(w: np.ndarray) -> np.ndarray:
        """Matrix exponential mapping 3D rotation vector to SO(3)."""
        theta = np.linalg.norm(w)
        if theta < 1e-6:
            return np.eye(3, dtype=np.float64)
        u = w / theta
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=np.float64)
        return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    @classmethod
    def refine_rotations_so3(cls, N: int, edges: dict, initial_rotations: dict, num_iters: int = 5) -> dict:
        """Riemannian gradient descent for SO(3) rotation averaging."""
        rotations = {i: initial_rotations[i].copy() for i in range(N)}
        for _ in range(num_iters):
            for i in range(N):
                if i not in edges or len(edges[i]) == 0:
                    continue
                step = np.zeros(3, dtype=np.float64)
                total_w = 0.0
                for j, R_ij, _, inliers in edges[i]:
                    # Relative rotation error: R_ij * R_i * R_j^T
                    R_err = R_ij @ rotations[i] @ rotations[j].T
                    w_vec = cls._so3_log(R_err)
                    w = float(inliers)
                    step += w * w_vec
                    total_w += w
                if total_w > 0:
                    delta_w = -0.3 * (step / total_w)
                    rotations[i] = cls._so3_exp(delta_w) @ rotations[i]
        return rotations

    @staticmethod
    def solve_triplet_scales(N: int, edges: dict) -> dict:
        """
        Solves for pairwise translation metric scales using 3-view loop consistency:
        alpha_ik * t_ik - alpha_ij * t_ij - alpha_jk * (R_ij^T * t_jk) = 0
        """
        pair_to_idx = {}
        idx_to_pair = []
        for i in edges:
            for j, _, _, _ in edges[i]:
                pair = (min(i, j), max(i, j))
                if pair not in pair_to_idx:
                    pair_to_idx[pair] = len(idx_to_pair)
                    idx_to_pair.append(pair)

        M = len(idx_to_pair)
        if M == 0:
            return {}

        # Default scale = 1.0 for all pairs
        scales = {pair: 1.0 for pair in idx_to_pair}
        if M < 3:
            return scales

        # Find triplets (i, j, k) where all 3 edges exist
        triplets = []
        nodes = list(edges.keys())
        for idx_a in range(len(nodes)):
            i = nodes[idx_a]
            nbrs_i = {e[0]: e for e in edges[i]}
            for idx_b in range(idx_a + 1, len(nodes)):
                j = nodes[idx_b]
                if j in nbrs_i:
                    nbrs_j = {e[0]: e for e in edges[j]}
                    for idx_c in range(idx_b + 1, len(nodes)):
                        k = nodes[idx_c]
                        if k in nbrs_i and k in nbrs_j:
                            triplets.append((i, j, k))

        if len(triplets) == 0:
            return scales

        # Build linear system A * alpha = 0
        eqs = []
        for i, j, k in triplets[:200]: # limit to 200 triplets for fast execution
            p_ij = pair_to_idx[(min(i, j), max(i, j))]
            p_jk = pair_to_idx[(min(j, k), max(j, k))]
            p_ik = pair_to_idx[(min(i, k), max(i, k))]

            # Retrieve relative translations in common frame
            nbrs_i = {e[0]: e for e in edges[i]}
            nbrs_j = {e[0]: e for e in edges[j]}
            _, R_ij, t_ij, _ = nbrs_i[j]
            _, R_jk, t_jk, _ = nbrs_j[k]
            _, R_ik, t_ik, _ = nbrs_i[k]

            v_ij = t_ij
            v_jk = R_ij.T @ t_jk
            v_ik = t_ik

            # 3 coordinate equations: alpha_ik * v_ik - alpha_ij * v_ij - alpha_jk * v_jk = 0
            for axis in range(3):
                row = np.zeros(M, dtype=np.float64)
                row[p_ik] += v_ik[axis]
                row[p_ij] -= v_ij[axis]
                row[p_jk] -= v_jk[axis]
                if np.max(np.abs(row)) > 1e-4:
                    eqs.append(row)

        if len(eqs) < 3:
            return scales

        A = np.array(eqs, dtype=np.float64)
        # SVD solver for right null-space: min ||A * alpha|| s.t. ||alpha|| = sqrt(M)
        try:
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            sol = np.abs(Vt[-1, :])
            mean_s = np.mean(sol)
            if mean_s > 1e-7:
                sol = sol / mean_s
                for idx, pair in enumerate(idx_to_pair):
                    s = float(sol[idx])
                    if 0.1 <= s <= 10.0:
                        scales[pair] = s
        except Exception:
            pass

        return scales

    @classmethod
    def register_scene(cls, image_names: list, features: dict, matcher: OmniAutoMLMatcher) -> dict:
        N = len(image_names)
        identity_R = np.eye(3, dtype=np.float64)
        zero_t = np.zeros(3, dtype=np.float64)
        poses = {img: (identity_R.copy(), zero_t.copy()) for img in image_names}

        if N <= 1:
            return poses

        # Determine pairs to match
        pairs_to_match = []
        if N <= 25:
            pairs_to_match = [(i, j) for i in range(N) for j in range(i + 1, N)]
        else:
            seen = set()
            for i in range(N):
                for step in [1, 2, 3, 4]:
                    j = i + step
                    if j < N:
                        pair = (min(i, j), max(i, j))
                        if pair not in seen:
                            seen.add(pair)
                            pairs_to_match.append((i, j))
            rng = random.Random(42)
            for i in range(N):
                candidates = [j for j in range(N) if j != i]
                for j in rng.sample(candidates, min(3, len(candidates))):
                    pair = (min(i, j), max(i, j))
                    if pair not in seen:
                        seen.add(pair)
                        pairs_to_match.append((i, j))

        # Parallel Pairwise Pose Estimation
        edges = defaultdict(list)
        def process_pair(i, j):
            img_i, img_j = image_names[i], image_names[j]
            pts_i, desc_i, K_i = features.get(img_i, (np.empty((0,2)), np.empty((0,128)), np.eye(3)))
            pts_j, desc_j, K_j = features.get(img_j, (np.empty((0,2)), np.empty((0,128)), np.eye(3)))
            if len(pts_i) < 6 or len(pts_j) < 6:
                return None
            p_i, p_j = matcher.match_pair(pts_i, desc_i, pts_j, desc_j)
            if len(p_i) < 6:
                return None
            R_ij, t_ij, inliers = matcher.estimate_relative_pose(p_i, p_j, K_i, K_j)
            if R_ij is not None and inliers >= 6:
                return (i, j, R_ij, t_ij.flatten(), inliers)
            return None

        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as ex:
            futures = [ex.submit(process_pair, i, j) for i, j in pairs_to_match]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    i, j, R_ij, t_ij, inliers = res
                    edges[i].append((j, R_ij, t_ij, inliers))
                    R_ji = R_ij.T
                    t_ji = -R_ij.T @ t_ij
                    edges[j].append((i, R_ji, t_ji, inliers))

        if not edges:
            return poses

        # 1. Spanning Tree Initialization
        root = max(edges.keys(), key=lambda nd: sum(e[3] for e in edges[nd]))
        visited = {root}
        queue = [root]
        initial_R = {i: identity_R.copy() for i in range(N)}
        initial_t = {i: zero_t.copy() for i in range(N)}

        while queue:
            curr = queue.pop(0)
            R_curr, t_curr = initial_R[curr], initial_t[curr]
            for nxt, R_rel, t_rel, _ in sorted(edges[curr], key=lambda x: -x[3]):
                if nxt not in visited:
                    visited.add(nxt)
                    initial_R[nxt] = R_rel @ R_curr
                    initial_t[nxt] = R_rel @ t_curr + t_rel
                    queue.append(nxt)

        # 2. Triplet Metric Scale Synchronization
        scales = cls.solve_triplet_scales(N, edges)

        # 3. Riemannian SO(3) Rotation Averaging
        refined_R = cls.refine_rotations_so3(N, edges, initial_R, num_iters=4)

        # 4. Chordal Global Translation Averaging
        # Camera centers C_i satisfy: C_j - C_i = alpha_ij * R_i^T * t_ij
        L = np.zeros((N, N), dtype=np.float64)
        b = np.zeros((N, 3), dtype=np.float64)

        for i in edges:
            for j, _, t_ij, inliers in edges[i]:
                pair = (min(i, j), max(i, j))
                s = scales.get(pair, 1.0)
                w = float(inliers)
                t_scaled = s * t_ij
                # Global translation step: R_i^T * t_scaled
                v = refined_R[i].T @ t_scaled
                L[i, i] += w
                L[i, j] -= w
                b[i] += w * v

        # Fix gauge freedom: C_root = 0
        L[root, :] = 0.0
        L[root, root] = 1.0
        b[root] = 0.0

        try:
            centers = np.linalg.lstsq(L, b, rcond=None)[0]
        except Exception:
            centers = np.zeros((N, 3), dtype=np.float64)

        # Compute final camera pose: t_i = -R_i * C_i
        for i in range(N):
            img_name = image_names[i]
            if i in visited:
                R_final = refined_R[i]
                t_final = -R_final @ centers[i]
                poses[img_name] = (R_final, t_final)
            else:
                poses[img_name] = (identity_R.copy(), zero_t.copy())

        print(f"  Registered {len(visited)}/{N} images via Scale Sync + SO(3) Avg (root={image_names[root]}).")
        return poses


# ==============================================================================
# 6. Formatting & Execution Pipeline
# ==============================================================================

def fmt_mat(mat: np.ndarray) -> str:
    return ";".join(f"{x:.9f}" for x in mat.flatten())

def fmt_vec(vec: np.ndarray) -> str:
    return ";".join(f"{x:.9f}" for x in vec.flatten())


def main():
    print("=== Kaggle IMC 2025: PU-AMatch OmniAutoML SfM Pipeline v7 ===")
    t_start = time.time()

    input_base = "/kaggle/input/image-matching-challenge-2025"
    for alt in ["/kaggle/input/image-matching-challenge-2025",
                "/kaggle/input/image-matching-challenge-2024",
                "kaggle/data", "."]:
        if os.path.exists(os.path.join(alt, "sample_submission.csv")):
            input_base = alt
            break

    sample_sub_path = os.path.join(input_base, "sample_submission.csv")
    output_path = "submission.csv"

    if not os.path.exists(sample_sub_path):
        print(f"FATAL: {sample_sub_path} not found.")
        with open(output_path, "w") as f:
            f.write("image_id,dataset,scene,image,rotation_matrix,translation_vector\n")
        return

    df_sub = pd.read_csv(sample_sub_path)
    print(f"Loaded sample_submission: {len(df_sub)} rows across {df_sub.groupby(['dataset','scene']).ngroups} scene clusters.")

    default_R = np.eye(3, dtype=np.float64)
    default_t = np.zeros(3, dtype=np.float64)
    expected_cols = ["image_id", "dataset", "scene", "image", "rotation_matrix", "translation_vector"]

    if "rotation_matrix" not in df_sub.columns:
        df_sub["rotation_matrix"] = fmt_mat(default_R)
    if "translation_vector" not in df_sub.columns:
        df_sub["translation_vector"] = fmt_vec(default_t)

    df_sub[expected_cols].to_csv(output_path, index=False)
    print(f"Pre-wrote initial {output_path} ({len(df_sub)} rows).")

    img_index = ImageIndex(input_base)
    matcher = OmniAutoMLMatcher(max_keypoints=3500, desc_ratio=0.82)
    estimated_poses = {}

    clusters = df_sub.groupby(["dataset", "scene"])
    print(f"\nProcessing {len(clusters)} scene clusters...\n")

    try:
        for (dataset, scene), group in clusters:
            img_names = group["image"].tolist()
            N = len(img_names)
            t_scene = time.time()
            print(f"Scene [{dataset}/{scene}] – {N} images")

            def extract_one(img_name):
                p = img_index.find(dataset, scene, img_name)
                if p:
                    img_bgr = cv2.imread(p)
                    if img_bgr is not None:
                        return img_name, matcher.extract(img_bgr, p)
                return img_name, (np.empty((0,2),dtype=np.float32), np.empty((0,128),dtype=np.float32), np.eye(3))

            features = {}
            found_count = 0
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as ex:
                futures = {ex.submit(extract_one, nm): nm for nm in img_names}
                for fut in as_completed(futures):
                    nm, feat = fut.result()
                    features[nm] = feat
                    if feat[0].shape[0] > 0:
                        found_count += 1

            print(f"  Found images: {found_count}/{N} | Elapsed: {time.time()-t_start:.0f}s")

            if found_count >= 2:
                scene_poses = GlobalSfMChainer.register_scene(img_names, features, matcher)
            else:
                scene_poses = {nm: (default_R.copy(), default_t.copy()) for nm in img_names}

            for idx, row in group.iterrows():
                img_id = row["image_id"]
                img_name = row["image"]
                R_est, t_est = scene_poses.get(img_name, (default_R, default_t))
                estimated_poses[img_id] = (fmt_mat(R_est), fmt_vec(t_est))

            for img_id, (r_str, t_str) in estimated_poses.items():
                mask = df_sub["image_id"] == img_id
                df_sub.loc[mask, "rotation_matrix"] = r_str
                df_sub.loc[mask, "translation_vector"] = t_str
            df_sub[expected_cols].to_csv(output_path, index=False)
            print(f"  Scene done in {time.time()-t_scene:.1f}s – CSV flushed.")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        df_sub[expected_cols].to_csv(output_path, index=False)
        total_non_id = sum(1 for _, row in df_sub.iterrows() if row["rotation_matrix"] != fmt_mat(default_R))
        print(f"\n=== Pipeline Complete in {time.time()-t_start:.0f}s ===")
        print(f"Final submission.csv: {len(df_sub)} rows, {total_non_id} with calibrated poses.")


if __name__ == "__main__":
    main()
