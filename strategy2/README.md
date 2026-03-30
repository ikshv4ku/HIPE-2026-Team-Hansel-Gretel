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

---

## Results and Inferences

After successfully running the end-to-end training pipeline on the RTX A5000 GPU (10 epochs, batch size 8), the fine-tuned model yielded a **Global Macro Recall of 0.4167**. 

### Performance Drop
When compared to the frozen baseline (Strategy 1.5: **0.6249**), the fine-tuning approach suffered a severe performance degradation. The classification report reveals catastrophic mode collapse:

- **`at` Relation**: The model predicted `TRUE` for every single instance (1.00 recall for `TRUE`, 0.00 for `FALSE` and `PROBABLE`).
- **`isAt` Relation**: The model predicted `FALSE` for every single instance (1.00 recall for `FALSE`, 0.00 for `TRUE`).

### Inferences
1. **Catastrophic Forgetting / Overfitting**: With only 700 mined training documents, fully unfreezing the 270M parameters of `xlm-roberta-base` likely destroyed its pre-trained linguistic representations.
2. **Learning Rate Mismatch**: While we used `2e-5` for the backbone, without a gradual unfreezing step, the large gradients from the randomly initialized classification heads heavily distorted the deeper transformer layers in the initial epochs.
3. **Class Imbalance Amplification**: Although we implemented focal loss and weighted cross-entropy, the gradients during end-to-end tuning still pushed the model into predicting the mathematically "safe" (most dominant) static outputs to minimize immediate loss, leading to convergence on a degenerate local minimum.

### Next Steps / Strategy 3 Recommendations
- **LoRA / PEFT**: Instead of full fine-tuning, use Low-Rank Adaptation (LoRA) to tune only a small subset (e.g., 1-2%) of the query and value attention weights. This prevents catastrophic forgetting.
- **Gradual Unfreezing**: Train the heads only for a few epochs, and then unfreeze the top 2 layers of the transformer with an extremely small learning rate (`1e-6`).
