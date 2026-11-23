"""
Kaggle Image Matching Challenge (IMC 2025) Official Submission Pipeline – v6
Architecture: PU-AMatch + OmniAutoML Global SfM Scene Registration

Improvements over v5:
  - Glob-based recursive image discovery → finds images under ANY sub-folder structure
  - Pre-indexes ALL images in /kaggle/input at startup for O(1) lookup
  - Exhaustive matching for small clusters (N≤30), stride-8 + random sampling for large
  - Inlier threshold lowered from 8 → 6 for better connectivity
  - Cross-scene dataset grouping (all scenes within a dataset share one discovery index)
  - Progressive CSV flushing preserved for Kaggle's output monitor
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
# 2. Image Discovery Index
# ==============================================================================

class ImageIndex:
    """
    Pre-scans the input directory tree once and builds a basename→full_path map.
    Handles duplicate basenames by keeping the most specific (deepest) path.
    """
    def __init__(self, root: str):
        print(f"Building image index under: {root}")
        t0 = time.time()
        self._index = {}  # basename (lowercase) → list[full_path]
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
        print(f"  Indexed {total} images in {time.time()-t0:.1f}s "
              f"({len(self._index)} unique basenames)")

    def find(self, dataset: str, scene: str, img_name: str) -> str:
        """
        Priority:
          1. Exact path: test/{dataset}/{img_name}
          2. Exact path: test/{dataset}/{scene}/{img_name}
          3. Any path whose basename matches img_name AND parent contains dataset name
          4. Any path whose basename matches img_name (global fallback)
        """
        # Priority 1 & 2: direct path tries
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

        # Priority 3 & 4: index-based lookup
        key = img_name.lower()
        paths = self._index.get(key, [])
        if not paths:
            # Also try without extension suffixes (some competitions rename)
            stem = os.path.splitext(key)[0]
            for k, v in self._index.items():
                if os.path.splitext(k)[0] == stem:
                    paths = v
                    break

        if not paths:
            return ""

        # Prefer paths that contain dataset name in directory
        ds_lower = dataset.lower()
        for p in paths:
            if ds_lower in p.lower():
                return p

        # Last resort: return first found
        return paths[0]


# ==============================================================================
# 3. Matcher & Pairwise Pose Estimator
# ==============================================================================

class OmniAutoMLMatcher:
    def __init__(self, max_keypoints: int = 4000, desc_ratio: float = 0.80):
        self.max_keypoints = max_keypoints
        self.desc_ratio = desc_ratio
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=0.03)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def extract(self, img_bgr: np.ndarray):
        H, W = img_bgr.shape[:2]
        max_dim = 1024
        scale = 1.0
        if max(H, W) > max_dim:
            scale = max_dim / float(max(H, W))
            img_bgr = cv2.resize(img_bgr, (int(W * scale), int(H * scale)),
                                 interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        kpts, descs = self.sift.detectAndCompute(gray, None)
        if kpts is None or len(kpts) == 0 or descs is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32), (W, H)

        pts = np.array([k.pt for k in kpts], dtype=np.float32)
        if scale != 1.0:
            pts = pts / scale

        descs_root = RootSIFTNormalizer.transform(descs)
        return pts, descs_root, (int(W / scale if scale != 1.0 else W),
                                  int(H / scale if scale != 1.0 else H))

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

        # Spatial Vector Consensus (SVC) pre-filter
        if len(p0) >= 12:
            vecs = p1 - p0
            median_vec = np.median(vecs, axis=0)
            dists = np.linalg.norm(vecs - median_vec, axis=1)
            thresh = np.percentile(dists, 85)
            inliers = dists <= thresh
            if np.sum(inliers) >= 6:
                p0, p1 = p0[inliers], p1[inliers]

        return p0, p1

    @staticmethod
    def estimate_camera_intrinsics(W: int, H: int) -> np.ndarray:
        f = 1.2 * max(W, H)
        return np.array([[f, 0.0, W / 2.0],
                         [0.0, f, H / 2.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

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
                maxIters=10000
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
# 4. Global Scene Reconstruction (Max-Inlier Spanning Tree)
# ==============================================================================

class GlobalSfMChainer:
    @staticmethod
    def register_scene(image_names: list, features: dict,
                       matcher: OmniAutoMLMatcher) -> dict:
        N = len(image_names)
        identity_R = np.eye(3, dtype=np.float64)
        zero_t = np.zeros(3, dtype=np.float64)
        poses = {img: (identity_R.copy(), zero_t.copy()) for img in image_names}

        if N <= 1:
            return poses

        # Build matching pairs
        pairs_to_match = []
        if N <= 30:
            # Exhaustive matching for small scenes
            pairs_to_match = [(i, j) for i in range(N) for j in range(i + 1, N)]
        else:
            seen = set()
            # Sequential + multi-stride
            for i in range(N):
                for step in [1, 2, 3, 4, 6, 8]:
                    j = i + step
                    if j < N:
                        key = (min(i, j), max(i, j))
                        if key not in seen:
                            seen.add(key)
                            pairs_to_match.append((i, j))
            # Random sampling: each image to 4 random others
            rng = random.Random(42)
            for i in range(N):
                candidates = [j for j in range(N) if j != i]
                for j in rng.sample(candidates, min(4, len(candidates))):
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        pairs_to_match.append((i, j))

        # Compute pairwise relative poses
        edges = defaultdict(list)
        for i, j in pairs_to_match:
            img_i, img_j = image_names[i], image_names[j]
            pts_i, desc_i, dims_i = features.get(img_i,
                (np.empty((0,2)), np.empty((0,128)), (1000,1000)))
            pts_j, desc_j, dims_j = features.get(img_j,
                (np.empty((0,2)), np.empty((0,128)), (1000,1000)))

            if len(pts_i) < 6 or len(pts_j) < 6:
                continue

            p_i, p_j = matcher.match_pair(pts_i, desc_i, pts_j, desc_j)
            if len(p_i) < 6:
                continue

            K_i = matcher.estimate_camera_intrinsics(dims_i[0], dims_i[1])
            K_j = matcher.estimate_camera_intrinsics(dims_j[0], dims_j[1])
            R_ij, t_ij, n_inliers = matcher.estimate_relative_pose(p_i, p_j, K_i, K_j)
            if R_ij is not None and n_inliers >= 6:
                edges[i].append((j, R_ij, t_ij.flatten(), n_inliers))
                R_ji = R_ij.T
                t_ji = -R_ij.T @ t_ij.flatten()
                edges[j].append((i, R_ji, t_ji, n_inliers))

        if not edges:
            print(f"  No valid edges found – all images default to identity.")
            return poses

        # Max-inlier spanning tree via BFS from most-connected root
        root = max(edges.keys(), key=lambda nd: sum(e[3] for e in edges[nd]))
        visited = {root}
        queue = [root]
        while queue:
            curr = queue.pop(0)
            R_curr, t_curr = poses[image_names[curr]]
            for nxt, R_rel, t_rel, _ in sorted(edges[curr], key=lambda x: -x[3]):
                if nxt not in visited:
                    visited.add(nxt)
                    poses[image_names[nxt]] = (R_rel @ R_curr, R_rel @ t_curr + t_rel)
                    queue.append(nxt)

        print(f"  Registered {len(visited)}/{N} images into unified global frame "
              f"(root idx={root}, name={image_names[root]}).")
        return poses


# ==============================================================================
# 5. Formatting helpers
# ==============================================================================

def fmt_mat(mat: np.ndarray) -> str:
    return ";".join(f"{x:.9f}" for x in mat.flatten())

def fmt_vec(vec: np.ndarray) -> str:
    return ";".join(f"{x:.9f}" for x in vec.flatten())


# ==============================================================================
# 6. Main Kaggle Execution Pipeline
# ==============================================================================

def main():
    print("=== Kaggle IMC 2025: PU-AMatch OmniAutoML SfM Pipeline v6 ===")
    t_start = time.time()

    # --- Locate dataset root ---
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
        print(f"FATAL: {sample_sub_path} not found. Tried: {input_base}")
        # Create a minimal stub so Kaggle sees an output file
        with open(output_path, "w") as f:
            f.write("image_id,dataset,scene,image,rotation_matrix,translation_vector\n")
        return

    df_sub = pd.read_csv(sample_sub_path)
    print(f"Loaded sample_submission: {len(df_sub)} rows, "
          f"{df_sub['dataset'].nunique()} datasets, "
          f"{df_sub.groupby(['dataset','scene']).ngroups} scene clusters.")

    # --- Defaults & early CSV write ---
    default_R = np.eye(3, dtype=np.float64)
    default_t = np.zeros(3, dtype=np.float64)
    expected_cols = ["image_id", "dataset", "scene", "image",
                     "rotation_matrix", "translation_vector"]

    if "rotation_matrix" not in df_sub.columns:
        df_sub["rotation_matrix"] = fmt_mat(default_R)
    if "translation_vector" not in df_sub.columns:
        df_sub["translation_vector"] = fmt_vec(default_t)

    df_sub[expected_cols].to_csv(output_path, index=False)
    print(f"Pre-wrote initial {output_path} ({len(df_sub)} rows) – guaranteed output exists.")

    # --- Build image discovery index ONCE ---
    img_index = ImageIndex(input_base)

    matcher = OmniAutoMLMatcher(max_keypoints=4000, desc_ratio=0.80)
    estimated_poses = {}  # image_id → (R_str, t_str)

    # --- Process each (dataset, scene) cluster ---
    clusters = df_sub.groupby(["dataset", "scene"])
    print(f"\nProcessing {len(clusters)} scene clusters...\n")

    try:
        for (dataset, scene), group in clusters:
            img_names = group["image"].tolist()
            N = len(img_names)
            t_scene = time.time()
            print(f"Scene [{dataset}/{scene}] – {N} images")

            # --- Feature extraction (threaded) ---
            def extract_one(img_name):
                p = img_index.find(dataset, scene, img_name)
                if p:
                    img_bgr = cv2.imread(p)
                    if img_bgr is not None:
                        return img_name, matcher.extract(img_bgr)
                return img_name, (np.empty((0,2),dtype=np.float32),
                                   np.empty((0,128),dtype=np.float32),
                                   (1000, 1000))

            features = {}
            found_count = 0
            max_workers = min(4, N)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(extract_one, nm): nm for nm in img_names}
                for fut in as_completed(futures):
                    nm, feat = fut.result()
                    features[nm] = feat
                    if feat[0].shape[0] > 0:
                        found_count += 1

            print(f"  Found images: {found_count}/{N} | "
                  f"Elapsed so far: {time.time()-t_start:.0f}s")

            # --- Global SfM ---
            if found_count >= 2:
                scene_poses = GlobalSfMChainer.register_scene(img_names, features, matcher)
            else:
                print(f"  Skipping SfM (too few images found) – identity fallback.")
                scene_poses = {nm: (default_R.copy(), default_t.copy())
                               for nm in img_names}

            # --- Record results ---
            for idx, row in group.iterrows():
                img_id = row["image_id"]
                img_name = row["image"]
                R_est, t_est = scene_poses.get(img_name, (default_R, default_t))
                estimated_poses[img_id] = (fmt_mat(R_est), fmt_vec(t_est))

            # Progressive flush to CSV
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
        total_non_identity = sum(
            1 for _, row in df_sub.iterrows()
            if row["rotation_matrix"] != fmt_mat(default_R)
        )
        print(f"\n=== DONE in {time.time()-t_start:.0f}s ===")
        print(f"submission.csv: {len(df_sub)} rows, "
              f"{total_non_identity} with non-identity poses.")


if __name__ == "__main__":
    main()
