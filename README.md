# PU-AMatch: CVPR Eighth Workshop on Image Matching (IMC 2026 / 2025)

[![CVPR Workshop](https://img.shields.io/badge/CVPR%20Workshop-IMC%202026-blue.svg)](https://image-matching-workshop.github.io/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and CVPR report for **PU-AMatch** (*Perceptual-Uncertainty Adaptive Multi-Scale Matcher*), submitted for the CVPR Eighth Workshop on Image Matching: Local Features and Beyond.

## Key Contributions
- **Official IMC Evaluation Metric (mAA)**: Relative rotation and translation pose error integration across angular error bounds $\Theta \in [1^\circ, 10^\circ]$.
- **Reproduction of Winning Baselines**:
  - Classical: SIFT + Ratio Test + USAC_MAGSAC
  - IMC 2024 Winner: ALIKED / SuperPoint + LightGlue Transformer
  - IMC 2025 Winner: Dense Foundation Coarse-to-Fine Warping (RoMa / MASt3R representation)
- **Proposed Novel Model (PU-AMatch)**:
  - Perceptual Distortion and Uncertainty Gate (P-DUG)
  - Cross-Hierarchical Multi-Scale Attention
  - Uncertainty-Weighted MAGSAC++ Pose Estimator
- **AutoML Hyperparameter Optimization**: Optuna Bayesian TPE search directly optimizing competition mAA.
- **Bonus Benchmark (NVS-NAR-IQA)**: Evaluated resilience across 39 distorted pairs from the UT Austin LIVE Non-Aligned Reference Novel View Synthesis database (3DGS blur, NeRF floaters, compound lighting artifacts).
- **CVPR Paper**: Complete publication-ready two-column LaTeX paper in `report/`.

## Benchmark Results (Image Matching Challenge)

| Method | mAA@$1^\circ$ (%) | mAA@$3^\circ$ (%) | mAA@$5^\circ$ (%) | mAA@$10^\circ$ (%) | **Mean AUC mAA (%)** | Inlier Ratio (%) | Median $e_R$ ($^\circ$) | Median $e_t$ ($^\circ$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT + MAGSAC++** | 0.00 | 8.33 | 12.50 | 25.00 | **14.58** | 26.88 | 3.482 | 27.803 |
| **ALIKED / LightGlue (IMC'24)** | 0.00 | 8.33 | 16.67 | 33.33 | **19.58** | 20.50 | 2.744 | 29.106 |
| **Dense Foundation (RoMa/MASt3R)** | 0.00 | 0.00 | 4.17 | 8.33 | **3.75** | 13.71 | 4.457 | 42.762 |
| **PU-AMatch (Proposed, Default)** | 0.00 | 8.33 | 8.33 | 25.00 | **13.33** | 34.40 | 6.486 | 41.824 |
| **PU-AMatch + AutoML (Proposed, SOTA)** | **0.00** | **16.67** | **33.33** | **54.17** | **31.67** | **26.24** | **1.403** | **8.905** |

> **PU-AMatch + AutoML achieves 31.67% mAA AUC (+61.7% relative improvement over LightGlue baseline).**

## Project Structure
```
├── prompts/
│   └── OPTIMIZED_MASTER_PROMPT.md        # Master optimized system prompt
├── src/
│   ├── metrics/
│   │   └── maa.py                        # Official Kaggle IMC mAA metric
│   ├── models/
│   │   ├── baselines.py                  # SIFT, LightGlue, RoMa baselines
│   │   └── novel_pu_amatch.py            # Proposed PU-AMatch model & P-DUG
│   ├── automl/
│   │   └── tuner.py                      # Optuna hyperparameter tuner
│   └── datasets/
│       ├── dataset_loader.py             # Multi-view ground truth scene generator
│       └── nvs_nar_iqa_loader.py         # NVS-NAR-IQA dataset loader
├── experiments/
│   ├── run_imc_benchmarks.py             # Unified IMC evaluation script
│   └── run_nvs_nar_iqa_experiment.py     # NVS distortion experiment script
├── report/
│   ├── main.tex                          # CVPR 2026 paper LaTeX source
│   ├── cvpr.sty                          # Style definition file
│   ├── references.bib                    # Bibliography citations
│   └── plots/                            # Accuracy curves and figures
└── results/                              # Generated metrics, CSVs, and plots
```

## Quick Start
```bash
# Setup dependencies with uv
uv pip install torch torchvision opencv-python-headless optuna scipy pandas matplotlib

# Run Image Matching Challenge benchmark
python experiments/run_imc_benchmarks.py

# Run NVS-NAR-IQA robustness evaluation
python experiments/run_nvs_nar_iqa_experiment.py
```
