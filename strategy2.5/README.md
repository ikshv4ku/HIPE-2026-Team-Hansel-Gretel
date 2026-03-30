# Strategy 2 — End-to-End XLM-RoBERTa Fine-Tuning

**Folder**: `strategy2/`  
**Status**: Planning  
**Predecessor**: Strategy 1.5 (MTL + Optuna) → Global Macro Recall: **0.6249**

---

## Overview

In Strategy 1 and 1.5, we used a **frozen** XLM-RoBERTa model as a feature extractor. The transformer's weights were never adapted to the specific nuances of historical newspaper text or the Person-Place relation extraction task.

# Strategy 2.5: PEFT/LoRA Fine-Tuning**. We will train the entire XLM-RoBERTa backbone along with the Multi-Task Learning (MTL) heads. This allows the model's attention mechanisms to learn which parts of the historical context are most indicative of `at` and `isAt` relations.

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

## Strategy 2.5 Results and Inferences

After implementing **Low-Rank Adaptation (LoRA)** via `peft` and training the network on the RTX A5000 GPU (10 epochs, batch size 8, learning rate `1e-4`), the fine-tuned model yielded a **Global Macro Recall of 0.4541**.

### Performance Overview
This iteration represents a modest improvement over the disastrous Strategy 2 (full fine-tuning, **0.4167**). The LoRA adapters prevented the catastrophic mode collapse we observed previously, successfully extracting *some* true positive signal:

- **`at` Relation**: Improved from 0.00 to 0.09 recall for `FALSE` (while maintaining 0.89 recall for `TRUE`). Still heavily biased towards `TRUE`, but no longer completely collapsed.
- **`isAt` Relation**: Improved from 0.00 to 0.18 recall for `TRUE`.

### Inferences
1. **LoRA Mitigates Forgetting**: By freezing the transformer and only updating low-rank matrices comprising 0.1% of the model parameters (294,912 params), we effectively preserved `xlm-roberta-base`'s structural integrity. The model recovered the ability to differentiate between classes instead of static majority-class predicting.
2. **Data Scarcity Wall**: Despite preventing catastrophic forgetting, the model still falls significantly short of Strategy 1.5's **0.6249** baseline. Freezing the representation entirely (Strategy 1.5) proves to still be profoundly superior because the dataset (700 silver documents from high-variance multilingual historical corpora) is too tiny to reliably adapt the transformer query/value matrices without overfitting the classifier heads simultaneously.
3. **Class Representation Limits**: The low recall on minority classes implies that the learning signal remains extremely noisy.

### Conclusions
Given the dataset size constraints of the HIPE-2026 task, **Strategy 1.5 (Frozen Backbone MTL)** remains our undisputed best-performing architecture. The minimal linguistic shifts provided by LoRA adaptation do not yield enough distinct task-specific signal above the baseline capabilities to offset the model's propensity to overfit.

### Observation:

When you combine French, German, and English data from historical newspapers (which already have highly irregular spelling and OCR noise) into a tiny dataset of 700 documents, the linguistic variance is extreme.

Here is exactly how that plays into our results:

Why Strategy 1.5 (Frozen) handled it well: xlm-roberta-base was already pre-trained on massive amounts of data across 100 languages. Because we kept it completely frozen in Strategy 1.5, its internal map of how words relate to each other stayed perfectly stable. The classification heads simply learned to read that stable map.

Why Strategy 2 & 2.5 (Fine-Tuning/LoRA) broke down: When you unfreeze the model (or add trainable LoRA adapters inside the attention layers), the model tries to mathematically shift its internal understanding to better fit the training data. However, because a single training batch of 8 documents might contain an English article from 1950, a German article from 1820, and a French article from 1790, the gradients the model uses to update itself become incredibly chaotic. The updates required to understand the French text might directly interfere with the updates trying to understand the German text (destructive interference).

Because the dataset is so small (e.g., roughly 230 documents per language), the model never sees enough consistent patterns in any single language to safely reorganize its attention matrices. Instead of learning, the gradients thrash around chaotically, forcing the model to retreat to the mathematically safest option: mode collapse (just guessing TRUE or FALSE for everything) to minimize the immediate loss.