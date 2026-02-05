
# PASS: Part-Aware Shape Subspace Features for Clinical Gait Classification

This repository contains the PyTorch implementation of **PASS**, a hybrid clinical gait classification framework that integrates **Part-Aware Shape Subspace features** with **Spatio-Temporal Graph Convolutional Networks (ST-GCN)** via an **attention-based fusion mechanism**.

The proposed method is designed for **spinal disease classification under limited and imbalanced clinical data**, and has been validated on a real-world clinical gait dataset.

## Overview

Clinical gait analysis provides important dynamic information for assessing orthopedic disorders such as:

- Adult Spinal Deformity (ASD)
- Dropped Head Syndrome (DHS)
- Lumbar Canal Stenosis (LCS)
- Hip Osteoarthritis (HipOA)

Purely data-driven deep learning models often suffer performance degradation in such clinical settings due to limited sample size and severe class imbalance.  
To address this issue, PASS introduces **physically motivated motion descriptors** based on **shape subspace variations**, which complement deep skeleton-based representations.

### Key Contributions

- **Part-Aware Shape Subspace Features**
  - The MediaPipe 33-keypoint skeleton is partitioned into seven anatomical parts (head, trunk, arms, legs).
  - For each part, frame-wise shape subspaces are constructed.
  - First- and second-order subspace differences are computed to capture velocity- and acceleration-related motion variations.
  - Statistical pooling (Max / Mean / Std) yields a compact 42-dimensional physical feature vector.

- **ST-GCN Backbone**
  - An ST-GCN is used to learn spatio-temporal representations directly from skeleton sequences.
  - The network models both temporal dynamics and spatial joint dependencies.

- **Attention-Based Fusion**
  - A gated attention mechanism adaptively balances the contribution of data-driven ST-GCN features and physically grounded subspace features.
  - This allows the model to emphasize complementary information depending on gait patterns.

## Method Pipeline

1. **Skeleton Extraction**
   - Human keypoints are extracted from walking videos using MediaPipe (33 joints).
   - Temporal interpolation, zero-padding, and Savitzky–Golay filtering are applied.

2. **Feature Extraction**
   - Part-aware shape subspace features are computed from skeleton sequences.
   - ST-GCN features are extracted in parallel.

3. **Feature Fusion & Classification**
   - Both feature streams are fused via an attention-based gating mechanism.
   - The fused representation is used for multi-class spinal disease classification.

## Dataset

The dataset used in this study consists of **1,952 walking sequences from 44 participants** across four diagnostic categories (ASD, DHS, LCS, HipOA).

Due to **ethical restrictions and patient privacy regulations**, the clinical dataset **cannot be publicly released**.


