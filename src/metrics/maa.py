"""
Official Image Matching Challenge (IMC) Evaluation Metric: Mean Average Accuracy (mAA)
Computes relative rotation and translation errors between predicted and ground-truth camera poses,
and calculates the area under the accuracy curve across multiple angular thresholds [1 deg to 10 deg].
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional

def compute_rotation_error(R_gt: np.ndarray, R_pred: np.ndarray) -> float:
    """
    Computes angular rotation error in degrees:
    eps_R = arccos((trace(R_gt^T @ R_pred) - 1) / 2)
    """
    R_diff = np.dot(R_gt.T, R_pred)
    trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(trace)
    return float(np.degrees(angle_rad))

def compute_translation_error(t_gt: np.ndarray, t_pred: np.ndarray) -> float:
    """
    Computes angular translation error in degrees:
    eps_t = arccos((t_gt . t_pred) / (||t_gt|| * ||t_pred||))
    """
    t_gt_norm = np.linalg.norm(t_gt)
    t_pred_norm = np.linalg.norm(t_pred)
    if t_gt_norm < 1e-7 or t_pred_norm < 1e-7:
        return 180.0
    dot = np.dot(t_gt.flatten(), t_pred.flatten()) / (t_gt_norm * t_pred_norm)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)
    # The direction of translation is up to sign in 2-view Epipolar geometry
    angle_deg = np.degrees(angle_rad)
    return float(min(angle_deg, 180.0 - angle_deg))

def compute_pose_error(R_gt: np.ndarray, t_gt: np.ndarray,
                       R_pred: np.ndarray, t_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes rotation error, translation error, and overall angular error max(e_R, e_t).
    """
    e_R = compute_rotation_error(R_gt, R_pred)
    e_t = compute_translation_error(t_gt, t_pred)
    e_pose = max(e_R, e_t)
    return e_R, e_t, e_pose

def compute_maa(pose_errors: List[float], thresholds: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Computes mean Average Accuracy (mAA) over a grid of angular thresholds.
    Standard IMC uses thresholds from 1 to 10 degrees with 1-degree steps.
    """
    if thresholds is None:
        thresholds = list(range(1, 11))  # 1 to 10 degrees

    if len(pose_errors) == 0:
        return {f"mAA@{th}": 0.0 for th in [5, 10]} | {"mAA": 0.0}

    errors = np.array(pose_errors)
    accuracies = []
    res_dict = {}

    for th in thresholds:
        acc = float(np.mean(errors < th))
        accuracies.append(acc)
        res_dict[f"acc@{th}deg"] = acc

    # Trapezoidal integration for mAA
    mean_aa = float(np.mean(accuracies))
    res_dict["mAA"] = mean_aa
    res_dict["mAA@5"] = float(np.mean(errors < 5.0))
    res_dict["mAA@10"] = float(np.mean(errors < 10.0))
    return res_dict

def estimate_relative_pose_magsac(
    kpts0: np.ndarray, 
    kpts1: np.ndarray, 
    K0: np.ndarray, 
    K1: np.ndarray,
    weights: Optional[np.ndarray] = None,
    pixel_threshold: float = 1.5,
    confidence: float = 0.9999,
    max_iters: int = 10000
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Estimates relative camera pose (R, t) between two views using MAGSAC++ / USAC_MAGSAC
    and Essential Matrix decomposition.
    """
    if len(kpts0) < 5:
        return None, None, np.zeros((len(kpts0),), dtype=bool)

    # Normalize keypoints to camera coordinates
    kpts0_norm = cv2.undistortPoints(np.expand_dims(kpts0, 1), K0, None).squeeze(1)
    kpts1_norm = cv2.undistortPoints(np.expand_dims(kpts1, 1), K1, None).squeeze(1)

    focal = (K0[0, 0] + K0[1, 1] + K1[0, 0] + K1[1, 1]) / 4.0
    thresh_norm = pixel_threshold / focal

    # Use OpenCV USAC_MAGSAC (SOTA robust estimator)
    E, mask = cv2.findEssentialMat(
        kpts0_norm, 
        kpts1_norm, 
        np.eye(3), 
        method=cv2.USAC_MAGSAC, 
        prob=confidence, 
        threshold=thresh_norm, 
        maxIters=max_iters
    )

    if E is None or mask is None:
        return None, None, np.zeros((len(kpts0),), dtype=bool)

    mask = mask.astype(bool).ravel()
    inlier_count = int(np.sum(mask))
    if inlier_count < 5:
        return None, None, mask

    # Recover pose from Essential matrix
    _, R, t, pose_mask = cv2.recoverPose(E, kpts0_norm, kpts1_norm, np.eye(3), mask=mask.astype(np.uint8))
    
    return R, t, mask
