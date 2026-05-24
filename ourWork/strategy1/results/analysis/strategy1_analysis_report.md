# Strategy 1 — Detailed Results Analysis Report

**Team Hansel & Gretel | Evaluation Date: March 2026**  
**Model**: Frozen XLM-RoBERTa-Base + Random Forest (`at`) + LightGBM (`isAt`)  
**Dev Set**: 21 Gold Documents (strict 20% holdout), 301 candidate pairs  
**Training Data**: 83 Gold docs (weight=50) + 617 Silver docs (weight=1)

---

## 1. Summary Scorecard

| Metric | Score |
|---|---|
| **Global Macro Recall** | **0.6018** |
| `at` Macro Recall | 0.5200 |
| `isAt` Macro Recall | 0.6835 |
| `at` Accuracy | 66.1% (199/301) |
| `isAt` Accuracy | 89.4% (269/301) |
| Total Correct Pairs | 468 / 602 |

---

## 2. `at` Relation — Full Classification Report

The `at` relation has three output classes: `TRUE` (person was at that location at some point), `PROBABLE` (likely but not confirmed), and `FALSE` (no evidence).

```
              precision    recall  f1-score   support

       FALSE     0.7642    0.8804    0.8182       184
    PROBABLE     0.2453    0.3939    0.3023        33
        TRUE     0.6667    0.2857    0.4000        84

    accuracy                         0.6611       301

   macro avg     0.5587    0.5200    0.5068       301
weighted avg     0.6801    0.6611    0.6449       301

Macro Recall: 0.5200
```

### Per-Class Analysis

| Class | Gold Count | Correctly Predicted | Missed | Recall |
|---|---|---|---|---|
| FALSE | 184 | 162 | 22 | 88.0% ✅ |
| PROBABLE | 33 | 13 | 20 | 39.4% ⚠️ |
| TRUE | 84 | 24 | 60 | 28.6% ❌ |

### Key Insights

**🟢 FALSE is well-learned (recall=0.88):** The heavy Silver data (most of which is `FALSE`) has taught the model to correctly identify pairs where there is no textual evidence of a person-location relationship. This is good for precision but contributes to imbalance.

**🔴 TRUE is severely under-recalled (recall=0.29):** The model correctly identifies only 24 out of 84 actual TRUE pairs. This is the single biggest contributor to our low `at` score. The frozen XLM-RoBERTa representation cannot distinguish well enough between TRUE and FALSE in historical OCR text without domain adaptation. **This is the primary target for Strategy 2 (fine-tuning).**

**🟡 PROBABLE is the hardest class (recall=0.39, precision=0.25):** With only 33 support examples in the dev set, the model is poorly calibrated on this borderline class. This is also the most subjectively defined label — even human annotators likely disagree on what constitutes PROBABLE vs TRUE. Low precision (0.25) means the model often predicts PROBABLE when the true label is FALSE or TRUE.

---

## 3. `isAt` Relation — Full Classification Report

The `isAt` relation is binary: `TRUE` (person was present within ~1 month of publication) or `FALSE`.

```
              precision    recall  f1-score   support

       FALSE     0.8932    0.9921    0.9401       253
        TRUE     0.9000    0.3750    0.5294        48

    accuracy                         0.8937       301

   macro avg     0.8966    0.6835    0.7347       301
weighted avg     0.8943    0.8937    0.8746       301

Macro Recall: 0.6835
```

### Per-Class Analysis

| Class | Gold Count | Correctly Predicted | Missed | Recall |
|---|---|---|---|---|
| FALSE | 253 | 251 | 2 | 99.2% ✅ |
| TRUE | 48 | 18 | 30 | 37.5% ❌ |

### Key Insights

**🟢 FALSE is nearly perfect (recall=0.99):** The model almost never incorrectly flags a `isAt=TRUE` when the truth is FALSE. This is largely due to the transitivity rule override (if `at=FALSE`, then `isAt=FALSE` is forced), and the inherent class imbalance.

**🔴 TRUE is severely under-recalled (recall=0.38):** The model correctly catches only 18 out of 48 true `isAt=TRUE` pairs. This makes intuitive sense: detecting that a person was present at a location **specifically within 1 month** of a newspaper's publication date requires:
  1. Understanding the temporal language in the text (e.g., "is currently residing in", "arrived yesterday")
  2. Cross-referencing the document's publication date with entity knowledge (birth/death dates, known movements)

A frozen transformer with no temporal reasoning capability and no external knowledge access is inherently bottlenecked here.

**🟢 High Precision (0.90) for TRUE predictions:** When our model does predict `isAt=TRUE`, it is right 90% of the time! This shows the signal is real — we just need to improve the recall so the model is more willing to predict TRUE when the evidence is there.

---

## 4. Progression Over Baseline

| Run | Training Data | `at` macro-recall | `isAt` macro-recall | **Global** |
|---|---|---|---|---|
| Random Baseline | N/A | ~0.33 | ~0.50 | 0.4242 |
| Gold-Only | 83 Gold docs (no weighting) | 0.3725 | 0.5194 | 0.4459 |
| **Strategy 1 (Ours)** | 83 Gold (×50) + 617 Silver (×1) | **0.5200** | **0.6835** | **0.6018** |

The addition of Silver data with 50× Gold weighting provided a **+34.9% improvement** in global macro-recall over the Gold-Only run, and a **+41.8% improvement** over the random baseline.

---

## 5. Per-Language Breakdown

| Language | `at` Macro Recall | `isAt` Macro Recall |
|---|---|---|
| German (DE) | To be read from `per_language_recall.png` chart |
| English (EN) | To be read from `per_language_recall.png` chart |
| French (FR) | To be read from `per_language_recall.png` chart |

> See `results/analysis/per_language_recall.png` for the exact per-language bar chart.

---

## 6. Error Analysis — Where the Model Fails

### Case A: TRUE `at` pairs misclassified as FALSE (most common error)
- **Root cause**: Historical text often expresses location relations in archaic, indirect, or incomplete OCR-corrupted ways. The frozen XLM-RoBERTa was never seen this type of text during pre-training.
- **Fix**: Fine-tune XLM-RoBERTa end-to-end (Strategy 2) to adapt weights to historical newspaper vocabulary.

### Case B: TRUE `isAt` pairs misclassified as FALSE
- **Root cause**: Temporal signals ("arrived yesterday", "currently staying in") are subtle in historical newspaper OCR. A classifier with no temporal knowledge cannot distinguish past-tense from present-tense residence.
- **Fix A**: Wikidata Knowledge Injection (Strategy 3) — fetch birth/death dates to validate temporal plausibility.
- **Fix B**: LLM-based reasoning chain (Strategy 6.2) — ask an LLM to reason about the publication date before classifying.

### Case C: PROBABLE pairs split between FALSE and TRUE predictions
- **Root cause**: PROBABLE is the most ambiguous label — there are only 33 such instances in the dev set and the model has never seen a clear signal distinguishing them from TRUE or FALSE.
- **Fix**: Label smoothing during fine-tuning, or collapsing PROBABLE into TRUE for a 2-class `at` formulation.

---

## 7. Visualisations Generated

All charts are saved to `results/analysis/`:

| File | Description |
|---|---|
| `confusion_matrices.png` | Heatmap of predicted vs actual labels for both relations |
| `per_class_metrics.png` | Grouped bar chart: Precision, Recall, F1 per class |
| `label_distribution.png` | Side-by-side count of Gold vs Predicted labels per class |
| `macro_recall_progression.png` | Bar chart showing improvement from random → gold → strategy 1 |
| `per_language_recall.png` | Per-language macro recall for both relations |
| `detailed_report.txt` | Machine-readable text output of the classification report |

---

## 8. Recommended Next Steps

| Priority | Strategy | Key Addresses | Expected Gain |
|---|---|---|---|
| **1 (next)** | Fine-tune XLM-RoBERTa end-to-end | `at` TRUE recall (currently 0.29) | +10–15 points global |
| **2** | Wikidata temporal knowledge injection | `isAt` TRUE recall (currently 0.38) | +5–8 points global |
| **3** | LLM Teacher-Student distillation | Both relations, generalisation | +15–20 points global |
