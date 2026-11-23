"""
Batched Multi-Core AutoML Engine for Wide-Baseline Image Matching:
- Parallelizes hypothesis evaluation across all CPU logical cores (AMD Ryzen 12-thread pool).
- Optimizes mathematical coordinate normalization (Hartley Isotropic Normalization)
- Optimizes descriptor representation (RootSIFT / Hellinger Kernel Transformation)
- Optimizes spatial topological motion consensus (SVC) neighborhood parameters
"""

import numpy as np
import optuna
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Callable
from src.metrics.maa import estimate_relative_pose_magsac, compute_pose_error, compute_maa
from src.models.batched_normalized_matcher import HartleyNormalizer, BatchedNormalizedMatcherPipeline

def evaluate_single_pair(pair: Dict[str, Any], pipeline, pixel_thresh: float, max_iters: int, use_hartley: bool) -> Tuple[float, float, float]:
    """Evaluates a single pair, returning (e_pose, inlier_ratio, latency_ms)."""
    t0 = time.time()
    res = pipeline.match(pair["img0"], pair["img1"])
    lat_ms = (time.time() - t0) * 1000.0

    k0, k1 = res["kpts0"], res["kpts1"]
    if len(k0) < 5:
        return 180.0, 0.0, lat_ms

    if use_hartley:
        k0_n, T0 = HartleyNormalizer.normalize_points(k0)
        k1_n, T1 = HartleyNormalizer.normalize_points(k1)
        K0_n = T0 @ pair["K0"]
        K1_n = T1 @ pair["K1"]
        R_pred, t_pred, inlier_mask = estimate_relative_pose_magsac(
            k0_n, k1_n, K0_n, K1_n,
            pixel_threshold=pixel_thresh,
            max_iters=max_iters
        )
    else:
        R_pred, t_pred, inlier_mask = estimate_relative_pose_magsac(
            k0, k1, pair["K0"], pair["K1"],
            pixel_threshold=pixel_thresh,
            max_iters=max_iters
        )

    inlier_r = float(np.mean(inlier_mask)) if len(inlier_mask) > 0 else 0.0
    if R_pred is None or t_pred is None:
        return 180.0, inlier_r, lat_ms

    _, _, e_pose = compute_pose_error(pair["R_gt"], pair["t_gt"], R_pred, t_pred)
    return e_pose, inlier_r, lat_ms

def batched_evaluate_dataset(
    pipeline,
    pairs: List[Dict[str, Any]],
    pixel_thresh: float = 0.85,
    max_iters: int = 20000,
    use_hartley: bool = True,
    n_workers: int = 6
) -> Dict[str, Any]:
    """Parallelized batched evaluation across CPU worker threads."""
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(evaluate_single_pair, p, pipeline, pixel_thresh, max_iters, use_hartley)
            for p in pairs
        ]
        results = [f.result() for f in futures]

    pose_errors = [r[0] for r in results]
    inlier_ratios = [r[1] for r in results]
    latencies = [r[2] for r in results]

    maa_metrics = compute_maa(pose_errors)
    maa_metrics["mean_latency_ms"] = float(np.mean(latencies))
    maa_metrics["mean_inlier_ratio"] = float(np.mean(inlier_ratios))
    maa_metrics["total_throughput_fps"] = float(len(pairs) / (np.sum(latencies) / (1000.0 * n_workers) + 1e-6))
    return maa_metrics

class BatchedAutoMLOptimizer:
    """
    AutoML Optimizer searching over Coordinate Normalization, Descriptor Whitening,
    Topological Motion Neighborhoods, and Robust Estimator Thresholds.
    """
    def __init__(self, n_trials: int = 15, timeout: int = 150, n_workers: int = 6):
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_workers = n_workers
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(self, eval_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        sub_pairs = eval_pairs[:8]

        def objective(trial: optuna.Trial) -> float:
            config = {
                "norm_type": trial.suggest_categorical("norm_type", ["root_sift", "power_hellinger"]),
                "use_hartley": trial.suggest_categorical("use_hartley", [True, False]),
                "use_spatial_consensus": trial.suggest_categorical("use_spatial_consensus", [True, False]),
                "k_neighbors": trial.suggest_int("k_neighbors", 6, 12, step=2),
                "match_threshold": trial.suggest_float("match_threshold", 0.14, 0.28, step=0.02),
                "uncertainty_weight": trial.suggest_float("uncertainty_weight", 0.20, 0.50, step=0.05),
                "pixel_threshold": trial.suggest_float("pixel_threshold", 0.65, 1.40, step=0.15),
                "desc_ratio": trial.suggest_float("desc_ratio", 0.82, 0.90, step=0.02),
                "max_iters": 20000
            }

            pipeline = BatchedNormalizedMatcherPipeline(
                norm_type=config["norm_type"],
                use_hartley=config["use_hartley"],
                use_spatial_consensus=config["use_spatial_consensus"],
                k_neighbors=config["k_neighbors"],
                match_threshold=config["match_threshold"],
                uncertainty_weight=config["uncertainty_weight"],
                desc_ratio=config["desc_ratio"]
            )

            metrics = batched_evaluate_dataset(
                pipeline,
                sub_pairs,
                pixel_thresh=config["pixel_threshold"],
                max_iters=config["max_iters"],
                use_hartley=config["use_hartley"],
                n_workers=self.n_workers
            )
            return metrics["mAA"]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_score = study.best_value
        return {
            "best_params": best_params,
            "best_mAA": best_score,
            "study": study
        }
