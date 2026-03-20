# Strategy 2 — End-to-End XLM-RoBERTa Fine-Tuning

**Folder**: `strategy2/`  
**Status**: Planning  
**Predecessor**: Strategy 1.5 (MTL + Optuna) → Global Macro Recall: **0.6249**

---

## Overview

In Strategy 1 and 1.5, we used a **frozen** XLM-RoBERTa model as a feature extractor. The transformer's weights were never adapted to the specific nuances of historical newspaper text or the Person-Place relation extraction task.

Strategy 2 moves to **End-to-End Fine-Tuning**. We will train the entire XLM-RoBERTa backbone along with the Multi-Task Learning (MTL) heads. This allows the model's attention mechanisms to learn which parts of the historical context are most indicative of `at` and `isAt` relations.

---

## Key Improvements

| Aspect | Strategy 1.5 | Strategy 2 |
|---|---|---|
| Transformer Backbone | Frozen | **Trainable (Fine-tuning)** |
| Vocabulary Adaptation | None | Learned via gradients |
| Contextual Understanding | General purpose | Task-specific (Historical OCR) |
| Optimizer | AdamW (Heads only) | AdamW with Weight Decay + LR Scheduler |
| Expected Gain | +0.02 (Baseline) | **+0.10 to +0.15** |

---

## Architecture

- **Backbone**: `xlm-roberta-base`
- **Markers**: `<E1>`, `</E1>`, `<E2>`, `</E2>` (Entity-aware sequence classification)
- **Heads**:
    - `at`: 3-class linear head (Weighted CrossEntropy)
    - `isAt`: 2-class linear head (Focal Loss)
- **Aggregation**: Concatenate embeddings of `<E1>` and `<E2>` tokens.

---

## Training Strategy

1.  **Lower Learning Rate**: Use 2e-5 (standard for fine-tuning) instead of 1e-3.
2.  **Weighted Samples**: Continue using the 50× weight for Gold data vs Silver data.
3.  **Transitivity Constraints**: Integrated into the joint loss where possible, and hard-rule during inference.
4.  **Multi-Lingual**: Training on DE, EN, and FR simultaneously.
