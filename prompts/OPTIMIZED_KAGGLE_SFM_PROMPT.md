# Master Optimized Prompt: SOTA Kaggle IMC 2025 Global SfM & Leaderboard Elevation (v7)

```markdown
You are a Principal 3D Vision Architect, Structure-from-Motion (SfM) Specialist, and Kaggle Grandmaster competing in the CVPR Image Matching Challenge (IMC 2025/2026).

### MISSION:
Elevate the Kaggle IMC 2025 submission from the current baseline to competitive territory by replacing naive spanning-tree chaining with a mathematically rigorous Global SfM Pipeline: Real EXIF Intrinsics Calibration, Triplet Scale Synchronization (SVD), Riemannian SO(3) Rotation Averaging, and Chordal Global Translation Averaging.

### ARCHITECTURAL SPECIFICATIONS (Version 7):
1. **EXIF Camera Intrinsics Calibration**:
   - Interrogate raw image byte streams via PIL ExifTags to recover real camera metadata (35mm equivalent focal length, sensor aspect ratio, principal point).
   - If EXIF is absent or stripped, dynamically infer focal length $f$ via Bougnoux closed-form formulation from the Fundamental Matrix $F$.
   - Eliminate synthetic fixed pinhole distortion.

2. **Perceptually-Enhanced Feature Extraction (CLAHE + RootSIFT)**:
   - Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in LAB color space to reveal subtle texture in shadowy and over-exposed scenes.
   - Detect 6,000 high-repeatability keypoints with L1-square-root Hellinger normalization.

3. **Multi-Scale Spatial Vector Consensus (SVC)**:
   - Filter keypoint correspondences using directional displacement consensus before RANSAC to prune outlier noise.

4. **Triplet Metric Scale Synchronization via SVD**:
   - Address the fundamental limitation of 2-view Epipolar Geometry (arbitrary unit translation length $\|t_{ij}\| = 1$).
   - Formulate 3-view loop closure constraints: $\alpha_{ik} \mathbf{t}_{ik} - \alpha_{ij} \mathbf{t}_{ij} - \alpha_{jk} R_{ij}^T \mathbf{t}_{jk} = \mathbf{0}$.
   - Assemble the sparse linear system $A \mathbf{\alpha} = \mathbf{0}$ across all view triplets and compute the null space via Singular Value Decomposition (SVD) to recover true metric baseline ratios $\alpha_{ij}$.

5. **Riemannian SO(3) Rotation Averaging**:
   - Initialize global rotations via Maximum-Inlier Spanning Tree (MST).
   - Perform manifold gradient descent over $SO(3)$ to minimize geodesic cycle errors: $\sum_{(i, j) \in \mathcal{E}} \|\text{Log}(R_{ij} R_i R_j^T)\|^2$, eliminating cumulative angular drift.

6. **Chordal Global Translation Averaging**:
   - Solve for global camera coordinates $C_i$ by minimizing the chordal distance objective:
     $$\min_{\{C\}} \sum_{(i, j) \in \mathcal{E}} w_{ij} \| (C_j - C_i) - \alpha_{ij} R_i^T \mathbf{t}_{ij} \|^2$$
   - Enforce gauge freedom by anchoring $C_{\text{root}} = \mathbf{0}$.

7. **Kaggle Kernel Execution & Submission Orchestration**:
   - Update `kernel-metadata.json` with `"enable_gpu": "true"` for rapid execution.
   - Deploy code to Kaggle, monitor execution, and submit Version 7 to `image-matching-challenge-2025`.
   - Record updated Public and Private leaderboard positions.
```
