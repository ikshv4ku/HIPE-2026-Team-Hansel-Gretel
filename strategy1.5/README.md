# Strategy 1.5 — Multi-Task Learning + Optuna HPO

**Folder**: `strategy1.5/`  
**Status**: In Development  
**Predecessor**: Strategy 1 (Frozen XLM-R + RF/LightGBM) → Global Macro Recall: **0.6018**

---

## Overview

Instead of two completely separate models (Random Forest for `at`, LightGBM for `isAt`), this strategy uses a **single shared neural network** with two output heads — one for each relation. This is the **Multi-Task Learning (MTL)** paradigm.

The shared backbone learns a common internal representation that is simultaneously useful for predicting both `at` and `isAt`. Since the two relations are deeply correlated by definition (`isAt=TRUE` implies `at=TRUE`), a joint model can learn this correlation as a soft signal internally, rather than via a hard post-hoc rule.

---

## Why This Improves Over Strategy 1

| Aspect | Strategy 1 | Strategy 1.5 |
|---|---|---|
| at/isAt coupling | Hard post-hoc rule | Learned jointly via shared weights |
| Regularization | None (tree models) | MTL acts as implicit regularizer |
| Hyperparameter tuning | Manual | Automated via Optuna (Bayesian search) |
| Task loss weighting | Fixed | Tuned by Optuna (α parameter) |
| Model type | RF + LightGBM | Single PyTorch neural network |

---

## Architecture

```
Pre-computed XLM-RoBERTa Embeddings (1536-dim)
                  ↓
    Shared Dense Layer + GELU + Dropout
                  ↓
     ┌────────────┴────────────┐
Head A (`at`)           Head B (`isAt`)
Dense → 3 classes       Dense → 2 classes
Weighted CrossEntropy   Focal Loss / Weighted CE
     └────────────┬────────────┘
      Combined Loss = α * L_at + (1-α) * L_isAt
                              ↑
                         Optuna tunes α
```

---

## Scripts

| Script | Purpose |
|---|---|
| `model.py` | MTL PyTorch module with two classification heads |
| `train_optuna.py` | Optuna HPO search + final retrain of best trial |
| `inference.py` | Load best model and generate official-format predictions |
| `analyse_results.py` | Confusion matrices, per-class metrics, charts |

---

## Optuna Search Space

| Hyperparameter | Range | Notes |
|---|---|---|
| `alpha` | 0.2 – 0.8 | Weight of `at` loss vs `isAt` loss |
| `lr` | 1e-4 – 1e-2 (log scale) | Learning rate |
| `dropout` | 0.1 – 0.5 | Applied after shared layer |
| `hidden_dim` | 128, 256, 512 | Size of shared hidden layer |
| `at_weight_true` | 1.0 – 10.0 | Up-weight TRUE class in `at` loss |
| `isAt_weight_true` | 1.0 – 10.0 | Up-weight TRUE class in `isAt` loss |
| `epochs` | 10 – 50 | Training epochs per trial |

## Results

> Results will be populated after running `train_optuna.py` and `inference.py`.

| Metric | Strategy 1 | Strategy 1.5 |
|---|---|---|
| `at` Macro Recall | 0.5200 | **0.5486** |
| `isAt` Macro Recall | 0.6835 | **0.7013** |
| **Global Macro Recall** | 0.6018 | **0.6249** |
