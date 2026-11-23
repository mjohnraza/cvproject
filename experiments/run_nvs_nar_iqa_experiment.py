"""
NVS-NAR-IQA (Non-Aligned Reference Image Quality Assessment for Novel View Synthesis)
Evaluation Experiment for Wide-Baseline Feature Matching and Relative Pose Estimation.

Evaluates how Novel View Synthesis degradations (NeRF density floaters, 3DGS blur, color shift)
affect baseline methods vs. the proposed Perceptual-Uncertainty Adaptive Matcher (PU-AMatch).
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.datasets.dataset_loader import create_synthetic_textured_scene, generate_benchmark_evaluation_pairs
from src.datasets.nvs_nar_iqa_loader import create_nvs_nar_iqa_test_suite
from src.metrics.maa import estimate_relative_pose_magsac, compute_pose_error, compute_maa
from src.models.baselines import SIFTBaseline
from src.models.novel_pu_amatch import PUAMatchPipeline

def evaluate_on_nvs_suite(matcher, nvs_pairs, pixel_thresh=1.5):
    by_condition = {}
    for pair in nvs_pairs:
        cond = pair.get("condition", "Standard")
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(pair)

    cond_results = {}
    for cond, pairs in by_condition.items():
        errors = []
        inliers = []
        for p in pairs:
            m_res = matcher.match(p["img0"], p["img1"])
            kpts0, kpts1 = m_res["kpts0"], m_res["kpts1"]
            if len(kpts0) < 5:
                errors.append(180.0)
                inliers.append(0.0)
                continue
            R_pred, t_pred, mask = estimate_relative_pose_magsac(
                kpts0, kpts1, p["K0"], p["K1"], pixel_threshold=pixel_thresh
            )
            inliers.append(float(np.mean(mask)) if len(mask) > 0 else 0.0)
            if R_pred is None or t_pred is None:
                errors.append(180.0)
            else:
                _, _, e_pose = compute_pose_error(p["R_gt"], p["t_gt"], R_pred, t_pred)
                errors.append(e_pose)
        
        maa_eval = compute_maa(errors)
        maa_eval["mean_inlier_ratio"] = float(np.mean(inliers))
        cond_results[cond] = maa_eval

    return cond_results

def main():
    os.makedirs("results/nvs_iqa", exist_ok=True)
    images, metadata, _ = create_synthetic_textured_scene(num_views=8, seed=123)
    base_pairs = generate_benchmark_evaluation_pairs(images, metadata, max_baseline_step=2)
    nvs_pairs = create_nvs_nar_iqa_test_suite(base_pairs)
    print(f"Generated {len(nvs_pairs)} NVS-NAR-IQA distorted evaluation pairs.")

    sift_matcher = SIFTBaseline(n_features=2500)
    pu_matcher = PUAMatchPipeline(device="cpu", match_threshold=0.20, uncertainty_weight=0.6, pixel_threshold=1.6)

    print("\n--- Evaluating SIFT on NVS-NAR-IQA ---")
    sift_nvs_res = evaluate_on_nvs_suite(sift_matcher, nvs_pairs, pixel_thresh=1.5)

    print("\n--- Evaluating Proposed PU-AMatch on NVS-NAR-IQA ---")
    pu_nvs_res = evaluate_on_nvs_suite(pu_matcher, nvs_pairs, pixel_thresh=1.6)

    # Compile Table
    rows = []
    conditions = ["3DGS_Blur", "NeRF_Floaters", "Compound_NVS_Artifacts"]
    for c in conditions:
        s_mAA = sift_nvs_res[c]["mAA"] * 100.0
        p_mAA = pu_nvs_res[c]["mAA"] * 100.0
        delta = p_mAA - s_mAA
        rows.append({
            "Distortion Regime": c.replace("_", " "),
            "SIFT mAA@10 (%)": round(sift_nvs_res[c]["mAA@10"] * 100, 2),
            "SIFT Overall mAA (%)": round(s_mAA, 2),
            "PU-AMatch mAA@10 (%)": round(pu_nvs_res[c]["mAA@10"] * 100, 2),
            "PU-AMatch Overall mAA (%)": round(p_mAA, 2),
            "Improvement (Gain %)": f"+{delta:.2f}%"
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/nvs_iqa/nvs_nar_iqa_robustness.csv", index=False)
    print("\n=== NVS-NAR-IQA Robustness Summary ===")
    print(df.to_string(index=False))

    # Bar Chart of Robustness
    plt.figure(figsize=(8, 5))
    x = np.arange(len(conditions))
    width = 0.35
    sift_scores = [sift_nvs_res[c]["mAA"] * 100.0 for c in conditions]
    pu_scores = [pu_nvs_res[c]["mAA"] * 100.0 for c in conditions]

    plt.bar(x - width/2, sift_scores, width, label='SIFT Baseline', color='#7f7f7f')
    plt.bar(x + width/2, pu_scores, width, label='PU-AMatch (Proposed)', color='#2ca02c')

    plt.xlabel('Novel View Synthesis (NVS) Distortion Condition', fontsize=11)
    plt.ylabel('mAA Pose Accuracy (%)', fontsize=11)
    plt.title('Robustness Under Non-Aligned Novel View Distortions (NVS-NAR-IQA)', fontsize=12, fontweight='bold')
    plt.xticks(x, [c.replace("_", " ") for c in conditions])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/nvs_iqa/nvs_robustness_bar.png", dpi=300)
    plt.close()

    print("Saved NVS-NAR-IQA analysis to results/nvs_iqa/")

if __name__ == "__main__":
    main()
