# Strategy 1.5 — Detailed Results Analysis Report

**Team Hansel & Gretel | Evaluation Date: March 2026**  
**Model**: Frozen XLM-RoBERTa-Base + MTL Dual-Head Neural Network  
**Optimization**: Optuna Bayesian Search (50 trials)  
**Dev Set**: 21 Gold Documents, 301 candidate pairs  

---

## 1. Summary Scorecard

| Metric | Strategy 1 (RF+LGB) | Strategy 1.5 (MTL+Optuna) | Δ Improvement |
|---|---|---|---|
| **Global Macro Recall** | 0.6018 | **0.6249** | **+2.31%** |
| `at` Macro Recall | 0.5200 | **0.5486** | +2.86% |
| `isAt` Macro Recall | 0.6835 | **0.7013** | +1.78% |

---

## 2. Optuna Optimization Results

The Bayesian search explored 50 configurations. The winning hyperparameters were:

- **Hidden Layer Size**: 128
- **Learning Rate**: 3.78e-3
- **Dropout**: 0.138
- **Task Weight (α)**: 0.34 (at) vs 0.66 (isAt) — *Crucial finding: the model performs best when prioritizing the `isAt` loss.*
- **Class Weights**:
    - `at` TRUE: 9.5×
    - `at` PROBABLE: 2.85×
    - `isAt` TRUE: 9.17×

---

## 3. `at` Relation — Classification Report

```
              precision    recall  f1-score   support

       FALSE     0.8224    0.6793    0.7440       184
    PROBABLE     0.2632    0.4545    0.3333        33
        TRUE     0.4674    0.5119    0.4886        84

    accuracy                         0.6080       301
   macro avg     0.5176    0.5486    0.5220       301
```

### Key Insights
- **TRUE Recall Improved Significantly**: Jumped from **0.286** (Strategy 1) to **0.512**. This is a massive win. The neural network's ability to weight the TRUE class more heavily than a Random Forest allowed it to catch over 50% of the actual relations.
- **Precision Trade-off**: The `at=TRUE` precision dropped to 0.46 (from 0.67). We are now catching more relations but also making more false positive guesses. In the context of "Recall", this is exactly what we wanted.

---

## 4. `isAt` Relation — Classification Report

```
              precision    recall  f1-score   support

       FALSE     0.9140    0.7984    0.8523       253
        TRUE     0.3625    0.6042    0.4531        48

    accuracy                         0.7674       301
   macro avg     0.6383    0.7013    0.6527       301
```

### Key Insights
- **isAt TRUE Recall**: Jumped from **0.375** to **0.604**. This 23-point gain is the result of using Focal Loss and large class weights (9.17×) during the joint training.
- **Learned Transitivity**: Because the heads share a backbone, the model "feels" the relationship between `at` and `isAt`. Even though we still use a hard rule as a safety net, the base predictions are much more consistent.

---

## 5. Visualisations Generated

All charts are saved to `strategy1.5/results/analysis/`:

| File | Description |
|---|---|
| `confusion_matrices.png` | Joint heatmap for both relation heads |
| `per_class_metrics.png` | Grouped P/R/F1 comparison per label |
| `strategy_comparison.png` | **Critical chart** showing progression from Random → Strategy 1 → Strategy 1.5 |
| `per_language_recall.png` | Cross-lingual performance breakdown |
| `detailed_report.txt` | Raw text report |

---

## 6. Next Steps: Onward to Strategy 2

While Strategy 1.5 was a success, we are likely at the limit of what a **frozen** transformer can do. 

**The bottleneck**: The 1536-dimensional embeddings were pre-computed by a base model that has never seen "impresso" newspaper data.
**The solution**: Fine-tuning the XLM-RoBERTa encoder itself (Strategy 2) to adapt its internal attention heads to historical text. Expected Global Macro Recall: **0.72 - 0.75**.
