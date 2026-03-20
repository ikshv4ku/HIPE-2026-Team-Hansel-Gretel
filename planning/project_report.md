# HIPE-2026: Strategy 1 — Full Project Report
**Team Hansel & Gretel | Date: March 2026**

---

## 1. Task Overview

**HIPE-2026** (Identifying Historical People in Newspapers) is an international NLP shared task focusing on **Person–Place Relation Extraction** from historical multilingual documents. Given a historical newspaper document, the goal is to determine, for each sampled (Person, Location) pair, which of two temporal relations holds:

| Relation | Definition |
|---|---|
| `at` | Person was **ever** at that location, at any time before publishing. |
| `isAt` | Person was at that location **within ~1 month** of the document's publication date. |

**Evaluation Metric:** Macro Recall (computed with the official `file_scorer_evaluation.py`).  
**Secondary Criteria:** Model size efficiency and generalization to unseen literary texts (Test B).

---

## 2. Dataset Summary

### 2.1 Data Sources

| Dataset | Description | Format |
|---|---|---|
| `data/newspapers/v1.0/` | **Gold-Labelled** — high-quality human annotations from 3 languages | `.jsonl` |
| `data/sandbox/` | **Silver-Labelled** — noisily auto-annotated data, 6x larger than gold | `.jsonl` |

### 2.2 Gold Data — Exploratory Data Analysis (EDA)

| Metric | Value |
|---|---|
| Total Documents | 104 |
| Language Split | 34 DE, 35 EN, 35 FR (perfectly balanced) |
| Total Candidate Pairs | 1,251 |
| Unique Persons | 496 |
| Unique Locations | 469 |
| Avg. Document Length | 3,465 characters |
| Max Document Length | 11,738 characters |

**`at` Label Distribution:**
- TRUE: 441 (35.2%)
- FALSE: 690 (55.2%)
- PROBABLE: 120 (9.6%)

**`isAt` Label Distribution:**
- FALSE: 972 (77.7%) ← severe class imbalance
- TRUE: 279 (22.3%)

**Key EDA Takeaways:**
1. **Context Length Problem**: With max ~12,000 chars, standard 512-token BERT-style encoders will truncate documents. We must rely on entity marker injection to help the model focus on the right section.
2. **Class Imbalance**: `isAt=FALSE` dominates at 78%. A naive classifier will always predict FALSE and get high accuracy but terrible macro recall.
3. **Multilingual by Design**: EN, FR, and DE are equally represented — a monolingual approach will immediately fail on unseen languages.

---

## 3. Strategy 1: XLM-RoBERTa Embeddings + Weighted ML Classifiers

### 3.1 Core Idea

Instead of fine-tuning a heavy neural network (which is overfitting-prone on 104 documents), we use a **frozen, pre-trained multilingual transformer** as a "smart feature extractor", and then train classical machine learning classifiers on top of those features. This is:
- **Efficient**: XLM-RoBERTa doesn't need to be trained — only the small classifiers do.
- **Multilingual by default**: XLM-RoBERTa was trained on 100 languages, so it naturally handles DE, EN, and FR.
- **Competitive**: The transformer's representations of historical text are far richer than hand-crafted TF-IDF features.

### 3.2 Architecture

```
Historical Document (DE/EN/FR)
        │
        ▼
 ┌─────────────────────────────────────────────────────┐
 │  Entity Marker Injection  (data_processing.py)      │
 │  "Napoleon was crowned in <E1>Paris</E1> ..."       │
 │  "<E2>Napoleon</E2> was crowned in Paris ..."       │
 └────────────────────────┬────────────────────────────┘
                          │
                          ▼
 ┌────────────────────────────────────────────────────────┐
 │  XLM-RoBERTa-Base (Frozen — 278M params)              │
 │  Produces: 768-dim hidden state per token             │
 └───────────────┬──────────────────┬────────────────────┘
                 │                  │
           [<E1> token]       [<E2> token]
           hidden vector      hidden vector
                 │                  │
                 └────────┬─────────┘
                          │ Concatenate
                          ▼
              1536-dimensional feature vector
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
  RandomForestClassifier        LGBMClassifier
  (Predicts `at` relation)   (Predicts `isAt` relation)
  Targets: TRUE/FALSE/PROBABLE   Targets: TRUE/FALSE
              │                       │
              └───────────┬───────────┘
                          │
                 Logic Rule Injection:
                 If `at`=FALSE → force `isAt`=FALSE
                          │
                          ▼
               Final JSONL Predictions
                          │
                          ▼
             file_scorer_evaluation.py
```

### 3.3 Key Design Decisions

#### Entity Marker Injection
Since the data provides entity **mention strings** (not character offsets), we use a case-insensitive regex replacement approach. Person mentions are wrapped in `<E1>...</E1>` and location mentions in `<E2>...</E2>` before tokenization. We sort mentions by length (descending) to prevent partial replacement bugs (e.g., "Paris" inside "Parisian").

#### E1/E2 Token Extraction
Rather than using the `[CLS]` token (which encodes the whole sentence), we locate the exact token IDs of the `<E1>` and `<E2>` special markers within the transformer's output sequence and extract their contextual hidden states. This gives the model a **relation-aware representation** that focuses on the relationship between the two specific entities.

#### Silver Data Weighting
We train on both the large silver dataset (`sandbox/`, 617 documents) and the gold dataset (104 documents), but apply **50× sample weight** to gold annotations. This means:
- The silver documents teach the model the general vocabulary of historical newspaper language.
- The gold annotations dominate the final decision boundaries because they are so heavily up-weighted.

#### Logic Rule: Transitivity Constraint
Per the task definition, `isAt=TRUE` implies `at=TRUE`. We enforce this as a post-processing rule: if the Random Forest predicts `at=FALSE`, the LightGBM's `isAt` output is overridden to `FALSE`.

---

## 4. Implementation Pipeline

The pipeline is broken into 4 modular scripts that can be run independently.

### Step 1: Build Train/Dev Splits  
**`scripts/prepare_multilingual_split.py`**
- Reads all gold `v1.0/*.jsonl` files for EN, FR, DE
- Reads all silver `sandbox/*.jsonl` files
- Randomly shuffles and splits Gold 80/20 (train/dev) with `seed=42` for reproducibility
- Injects `split_weight=50.0` for gold docs and `split_weight=1.0` for silver docs
- **Output**: `splits/train-all.jsonl` (700 docs) and `splits/dev-all.jsonl` (21 docs)

### Step 2: Entity Marker Injection & Tokenization  
**`scripts/data_processing.py`**  
Implements a PyTorch `Dataset` class:
- Parses `sampled_pairs` from each document
- Regex-injects `<E1>`, `</E1>`, `<E2>`, `</E2>` markers around each entity's mentions
- Tokenizes with `XLMRobertaTokenizerFast` (max_length=512, padded)
- Returns PyTorch batched tensors with labels and weights

### Step 3: Feature Extraction  
**`scripts/extract_embeddings.py`**
- Loads the frozen `xlm-roberta-base` model, resizes token embeddings for the 4 new special tokens
- Passes every document batch through XLM-RoBERTa (no gradient computation)
- Locates `<E1>` and `<E2>` token positions in the output and concatenates their 768-dim hidden states
- Saves the full `(N, 1536)` NumPy feature matrix to disk as a `.pkl` file

### Step 4: Classifier Training  
**`scripts/train_classifiers.py`**
- Loads the train embedding `.pkl`
- Trains `sklearn.RandomForestClassifier` on `at` labels (with `class_weight='balanced'`)
- Trains `lightgbm.LGBMClassifier` on `isAt` labels (with `class_weight='balanced'`)
- Both classifiers receive the sample weights derived from the gold/silver split
- Saves both models as `.joblib` files

### Step 5: Inference & Evaluation  
**`scripts/inference.py`**
- Runs Steps 2 & 3 on any `.jsonl` file in-memory (no need to pre-extract)
- Applies the transitivity rule override
- Writes fully compliant prediction files in the official HIPE-2026 JSONL format
- Triggers `HIPE-2026-data/scripts/file_scorer_evaluation.py` automatically

---

## 5. Results

### 5.1 Benchmark Comparisons

| Run | Data Used | `at` macro-recall | `isAt` macro-recall | **Global macro-recall** |
|---|---|---|---|---|
| Random Baseline (script) | N/A | — | — | 0.4242 |
| Integration Test (1 lang, train=test) | Gold DE only | 0.8774 | 0.9306 | 0.9040 ⚠️ (overfit) |
| Gold-Only True Validation | Gold EN+FR+DE, 80/20 split | 0.3725 | 0.5194 | 0.4459 |
| **Gold + Silver Weighted** | 83 Gold (50×) + 617 Silver (1×) | **0.5200** | **0.6835** | **0.6018** ✅ |

> ⚠️ The 0.9040 score is a training-set evaluation (overfit). The true generalizable score is 0.6018.

### 5.2 Analysis

- **Adding Silver data boosted macro-recall by +34.9%** — from 0.4459 to 0.6018
- **`isAt` benefits most** — the harder, temporally-constrained relation went from 0.52 to 0.68
- **`at` improved significantly** — from 0.37 to 0.52, meaning the model learned to distinguish PROBABLE from TRUE better
- The model **beats the random baseline by +42%** (0.60 vs 0.42)

---

## 6. What Was Already Tried

Before this pipeline, we also established a minimal sanity baseline using a pure random predictor (`create_random_baseline_including_dropout.py`) which achieved macro-recall of 0.42, proving the scoring pipeline works end-to-end.

---

## 7. Next Steps (Future Strategies)

The current baseline of **0.6018** is a strong foundation. The following techniques are planned to push towards 0.80+ macro-recall:

### 7.1 End-to-End Fine-Tuning (Strategy 2)
Fine-tune XLM-RoBERTa directly on classification heads (not just feature extraction). This allows the transformer itself to adapt to the historical newspaper domain. Expected model size: ~500MB.

### 7.2 Wikidata Knowledge Injection (Strategy 3)
The dataset provides `pers_wikidata_QID` and `loc_wikidata_QID` for many entities. Fetching birth/death dates and known home cities from Wikidata can dramatically improve `isAt` accuracy (a dead person cannot be `isAt` somewhere near the publication date).

### 7.3 Teacher-Student Distillation (Strategy 5–6)
Use a large LLM (e.g., Gemma 8B or Llama 3 8B) as a "Teacher" to generate soft probability labels and explanations. Train a small XLM-RoBERTa student on these richer labels. This is the approach recommended by our professor.

### 7.4 Agentic Framework with Reasoning-First (Strategy 6.2)
Build a multi-step agent that first reasons about whether the temporal window supports `isAt`, generates an explanation, and then passes that reasoning to a classifier. This approach is also recommended by our professor.

---

## 8. Repository Structure

```
HIPE-2026-Team-Hansel-Gretel/
├── scripts/
│   ├── data_processing.py         # PyTorch Dataset with <E1>/<E2> entity markers
│   ├── extract_embeddings.py      # XLM-RoBERTa frozen feature extraction
│   ├── prepare_multilingual_split.py # Gold/Silver data merge + 80/20 split
│   ├── train_classifiers.py       # RandomForest + LightGBM training
│   ├── inference.py               # End-to-end prediction + output writing
│   ├── eda.py                     # Exploratory Data Analysis of the gold dataset
│   └── create_random_baseline_including_dropout.py
│
├── planning/
│   └── strategy_guide.md          # Detailed strategy comparison (all approaches)
│
├── results/
│   ├── eda_results.txt            # EDA statistics
│   ├── evaluation_results.txt     # Random baseline evaluation output
│   ├── embeddings/                # Saved XLM-RoBERTa feature matrices (.pkl)
│   ├── models/                    # Trained classifiers (.joblib)
│   └── predictions/               # Model predictions in official JSONL format
│
└── reproduction_steps.md          # Step-by-step commands to run all scripts
```
