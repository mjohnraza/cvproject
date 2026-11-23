"""
Advanced Multi-Objective & Meta-Learning AutoML Engine for Wide-Baseline Image Matching.
Features:
1. Scene Complexity Meta-Feature Extractor (Texture, Entropy, Edge Density, Illumination Variance)
2. Pareto Multi-Objective Optimization (Maximizing mAA AUC while Minimizing Latency)
3. Dynamic Context-Conditioned Hyperparameter Policy (Contextual Meta-Learner)
4. Multi-Trial Ensemble Auto-Blender (Combining Top-K Pareto Models)
"""

import optuna
import numpy as np
import cv2
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from src.metrics.maa import estimate_relative_pose_magsac, compute_pose_error, compute_maa

class SceneComplexityMetaExtractor:
    """
    Extracts lightweight meta-features characterizing scene difficulty
    before correspondence matching is initiated.
    """
    @staticmethod
    def extract_meta_features(img: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        H, W = gray.shape[:2]

        # 1. Visual Shannon Entropy (Information density)
        hist, _ = np.histogram(gray, bins=64, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        # 2. Gradient / High-frequency Texture Energy
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(np.var(laplacian))

        # 3. Structural Edge Density (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges) / (H * W))

        # 4. Illumination Uniformity (Dynamic range)
        lum_mean = float(np.mean(gray))
        lum_std = float(np.std(gray))
        dynamic_contrast = float(lum_std / (lum_mean + 1e-6))

        # Composite Scene Difficulty Index [0: Easy high-contrast, 1: Severe low-texture/distorted]
        difficulty_score = float(np.clip(
            1.0 - (0.4 * (entropy / 6.0) + 0.4 * min(texture_variance / 500.0, 1.0) + 0.2 * min(edge_density / 0.15, 1.0)),
            0.0, 1.0
        ))

        return {
            "entropy": entropy,
            "texture_variance": texture_variance,
            "edge_density": edge_density,
            "dynamic_contrast": dynamic_contrast,
            "difficulty_score": difficulty_score
        }

class MetaAdaptivePolicy:
    """
    Contextual Meta-Learner that dynamically adjusts matching parameters
    based on the scene complexity meta-features of each input pair.
    """
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config

    def adapt_for_pair(self, meta0: Dict[str, float], meta1: Dict[str, float]) -> Dict[str, Any]:
        avg_difficulty = 0.5 * (meta0["difficulty_score"] + meta1["difficulty_score"])
        avg_texture = 0.5 * (meta0["texture_variance"] + meta1["texture_variance"])

        config = dict(self.base_config)

        # On difficult, low-texture or distorted scenes:
        # Increase uncertainty weight and relax match cutoff to harvest subtle correspondences
        if avg_difficulty > 0.45 or avg_texture < 150.0:
            config["match_threshold"] = max(0.12, config.get("match_threshold", 0.20) * 0.85)
            config["uncertainty_weight"] = min(0.70, config.get("uncertainty_weight", 0.30) * 1.35)
            config["pixel_threshold"] = min(2.2, config.get("pixel_threshold", 0.8) * 1.5)
            config["max_iters"] = int(min(25000, config.get("max_iters", 20000) * 1.25))
        else:
            # On crisp, high-texture scenes: use tight geometric constraints
            config["match_threshold"] = min(0.35, config.get("match_threshold", 0.20) * 1.15)
            config["uncertainty_weight"] = max(0.15, config.get("uncertainty_weight", 0.30) * 0.85)
            config["pixel_threshold"] = max(0.6, config.get("pixel_threshold", 0.8) * 0.9)

        return config

class AdvancedMultiObjectiveAutoML:
    """
    Pareto Multi-Objective Optimizer (Optuna NSGA-II / MOTPE):
    Maximizes downstream mAA pose accuracy while simultaneously minimizing latency.
    """
    def __init__(self, n_trials: int = 15, timeout: int = 180):
        self.n_trials = n_trials
        self.timeout = timeout
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize_pareto(
        self,
        eval_dataset: List[Dict[str, Any]],
        pipeline_builder: Callable[[Dict[str, Any]], Any]
    ) -> Dict[str, Any]:
        """
        Executes multi-objective optimization:
        Objective 1: Maximize mAA AUC
        Objective 2: Minimize Mean Runtime (ms)
        """
        def objective(trial: optuna.Trial) -> Tuple[float, float]:
            config = {
                "match_threshold": trial.suggest_float("match_threshold", 0.12, 0.40, step=0.04),
                "uncertainty_weight": trial.suggest_float("uncertainty_weight", 0.15, 0.65, step=0.05),
                "pixel_threshold": trial.suggest_float("pixel_threshold", 0.6, 2.4, step=0.2),
                "max_iters": trial.suggest_int("max_iters", 5000, 20000, step=5000),
                "desc_ratio": trial.suggest_float("desc_ratio", 0.75, 0.90, step=0.05)
            }

            matcher = pipeline_builder(config)
            pose_errors = []
            durations = []

            for pair in eval_dataset:
                img0, img1 = pair["img0"], pair["img1"]
                K0, K1 = pair["K0"], pair["K1"]
                R_gt, t_gt = pair["R_gt"], pair["t_gt"]

                t0 = time.time()
                match_res = matcher.match(img0, img1)
                kpts0, kpts1 = match_res["kpts0"], match_res["kpts1"]

                if len(kpts0) < 5:
                    durations.append((time.time() - t0) * 1000.0)
                    pose_errors.append(180.0)
                    continue

                R_pred, t_pred, _ = estimate_relative_pose_magsac(
                    kpts0, kpts1, K0, K1,
                    pixel_threshold=config["pixel_threshold"],
                    max_iters=config["max_iters"]
                )
                durations.append((time.time() - t0) * 1000.0)

                if R_pred is None or t_pred is None:
                    pose_errors.append(180.0)
                else:
                    _, _, e_pose = compute_pose_error(R_gt, t_gt, R_pred, t_pred)
                    pose_errors.append(e_pose)

            maa_dict = compute_maa(pose_errors)
            mAA_score = maa_dict["mAA"]
            mean_latency_ms = float(np.mean(durations))

            # Optuna maximizes objective 0 (mAA) and minimizes objective 1 (latency)
            return mAA_score, mean_latency_ms

        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42)
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Extract Pareto Frontier
        pareto_trials = study.best_trials
        best_accuracy_trial = max(pareto_trials, key=lambda t: t.values[0])
        fastest_trial = min(pareto_trials, key=lambda t: t.values[1])

        return {
            "study": study,
            "pareto_trials": pareto_trials,
            "best_accuracy_config": best_accuracy_trial.params,
            "best_accuracy_values": best_accuracy_trial.values,
            "fastest_config": fastest_trial.params,
            "fastest_values": fastest_trial.values
        }

class AutoMLEnsembleMatcher:
    """
    Ensemble Pipeline combining Top-K diverse parameter configurations
    discovered by the AutoML search to maximize pose recall and precision.
    """
    def __init__(self, model_configs: List[Dict[str, Any]], pipeline_builder: Callable[[Dict[str, Any]], Any]):
        self.pipelines = [pipeline_builder(cfg) for cfg in model_configs]
        self.configs = model_configs

    def match(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, Any]:
        all_kpts0 = []
        all_kpts1 = []
        all_scores = []

        for p in self.pipelines:
            res = p.match(img0, img1)
            if len(res["kpts0"]) > 0:
                all_kpts0.append(res["kpts0"])
                all_kpts1.append(res["kpts1"])
                all_scores.append(res["scores"])

        if not all_kpts0:
            return {"kpts0": np.empty((0, 2)), "kpts1": np.empty((0, 2)), "scores": np.empty((0,))}

        kpts0_cat = np.concatenate(all_kpts0, axis=0)
        kpts1_cat = np.concatenate(all_kpts1, axis=0)
        scores_cat = np.concatenate(all_scores, axis=0)

        # De-duplicate spatially close matches across ensemble heads
        # Binning on 1-pixel grid
        coords_int = np.round(kpts0_cat).astype(np.int32)
        _, unique_indices = np.unique(coords_int[:, 0] * 10000 + coords_int[:, 1], return_index=True)

        return {
            "kpts0": kpts0_cat[unique_indices],
            "kpts1": kpts1_cat[unique_indices],
            "scores": scores_cat[unique_indices]
        }
