"""
Automated Machine Learning (AutoML) Hyperparameter Tuning Engine using Optuna.
Optimizes the end-to-end Image Matching Challenge pipeline to directly maximize mAA@10.
"""

import optuna
import numpy as np
from typing import List, Dict, Any, Callable
from src.metrics.maa import estimate_relative_pose_magsac, compute_pose_error, compute_maa

class MatcherAutoMLTuner:
    """
    Bayesian Hyperparameter Optimizer for Wide-Baseline Image Matching Pipelines.
    Tunes:
    - match_threshold: Cutoff for correspondence confidence
    - uncertainty_weight: Influence of perceptual uncertainty attenuation
    - pixel_threshold: Inlier threshold for USAC_MAGSAC
    - max_iters: RANSAC iteration budget
    """
    def __init__(self, n_trials: int = 15, timeout: int = 600):
        self.n_trials = n_trials
        self.timeout = timeout
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def tune(self, 
             eval_dataset: List[Dict[str, Any]], 
             pipeline_factory: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        """
        Executes Optuna study to maximize mAA@10 over the evaluation dataset.
        """
        def objective(trial: optuna.Trial) -> float:
            # Hyperparameter search space
            config = {
                "match_threshold": trial.suggest_float("match_threshold", 0.15, 0.50, step=0.05),
                "uncertainty_weight": trial.suggest_float("uncertainty_weight", 0.20, 0.80, step=0.10),
                "pixel_threshold": trial.suggest_float("pixel_threshold", 0.8, 3.0, step=0.2),
                "max_iters": trial.suggest_int("max_iters", 5000, 20000, step=5000)
            }

            matcher = pipeline_factory(config)
            pose_errors = []

            for pair in eval_dataset:
                img0, img1 = pair["img0"], pair["img1"]
                K0, K1 = pair["K0"], pair["K1"]
                R_gt, t_gt = pair["R_gt"], pair["t_gt"]

                match_res = matcher.match(img0, img1)
                kpts0, kpts1 = match_res["kpts0"], match_res["kpts1"]

                if len(kpts0) < 5:
                    pose_errors.append(180.0)
                    continue

                R_pred, t_pred, _ = estimate_relative_pose_magsac(
                    kpts0, kpts1, K0, K1,
                    pixel_threshold=config["pixel_threshold"],
                    max_iters=config["max_iters"]
                )

                if R_pred is None or t_pred is None:
                    pose_errors.append(180.0)
                else:
                    _, _, e_pose = compute_pose_error(R_gt, t_gt, R_pred, t_pred)
                    pose_errors.append(e_pose)

            maa_dict = compute_maa(pose_errors)
            # Optimize mAA (area under curve)
            return maa_dict["mAA"]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_score = study.best_value
        return {
            "best_params": best_params,
            "best_mAA": best_score,
            "study": study
        }
