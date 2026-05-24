# Strategy 6.5: Specialist LoRA Fine-Tuning for the Multi-Agent Framework

**Goal**: Fine-tune dedicated LoRA adapters for the Historian and Geographer agents using the full available dataset (gold + silver), then retrain the Arbiter on the combined agent verdicts. Target: Global Macro Recall > 0.76.

---

## Full Dataset Inventory

### Gold Data — `HIPE-2026-data/data/newspapers/v1.0/`

| File | Docs | Pairs | Split |
| :--- | ---: | ---: | :--- |
| `train-de.jsonl` | 34 | 466 | Train (raw) |
| `train-en.jsonl` | 35 | 307 | Train (raw) |
| `train-fr.jsonl` | 35 | 478 | Train (raw) |
| `splits/HIPE-2026-v1.0-impresso-train-str4-all.jsonl` | **700** | **9,248** | Official train (pre-split) |
| `splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl` | **21** | **254** | Official dev (fixed) |

> [!NOTE]
> The official `train-str4-all.jsonl` (700 docs) already contains gold + augmented samples. It is the authoritative training file. The raw per-language files are subsets.

### Silver Data — `HIPE-2026-data/data/sandbox/`

| File | Docs | Pairs | Proposed Use |
| :--- | ---: | ---: | :--- |
| `de-train.jsonl` | 88 | 1,224 | Specialist training |
| `de-dev.jsonl` | 32 | 432 | Specialist val |
| `en-train.jsonl` | 56 | 496 | Specialist training |
| `en-dev.jsonl` | 17 | 151 | Specialist val |
| `fr-train.jsonl` | 317 | 4,450 | Specialist training |
| `fr-dev.jsonl` | 107 | 1,498 | Specialist val |
| **TOTAL train** | **461** | **6,170** | |
| **TOTAL dev** | **156** | **2,081** | |

---

## Proposed Train / Validate / Test Split

> [!IMPORTANT]
> The official dev set (`dev-str4-all.jsonl`, 21 docs / 254 pairs) is **fixed and must not be used for training**. It is our test/benchmark set.

| Role | Source | Docs | Pairs | Notes |
| :--- | :--- | ---: | ---: | :--- |
| **Train** | Gold `train-str4-all.jsonl` | 700 | 9,248 | Weight=5 (gold) |
| **Train** | Silver `{de,en,fr}-train.jsonl` | 461 | 6,170 | Weight=1 (silver) |
| **Train TOTAL** | | **1,161** | **15,418** | |
| **Validate** | Silver `{de,en,fr}-dev.jsonl` | 156 | 2,081 | Used for early stopping / agent tuning |
| **Test** | Gold `dev-str4-all.jsonl` | 21 | 254 | Held-out, official HIPE-2026 scorer |

**Rationale**:
- Gold data has verified human annotations → high weight in loss.
- Silver sandbox data provides coverage of rare label distributions and extends training signal 70%.
- Using silver `dev` for validation (not test) avoids leaking the official evaluation benchmark.
- The 156 silver val docs → ~2,000 pairs → sufficient for reliable early stopping across 3 agents.

---

## Proposed Changes

### Directory Structure

```
strategy6/
├── scripts/
│   ├── prompts.py             ← Already written ✅
│   ├── agent_inference.py     ← Already written ✅
│   ├── analyse_results.py     ← Already written ✅
│   ├── prepare_specialist_data.py    [NEW] Build agent-specific SFT datasets
│   ├── train_historian.py            [NEW] Historian LoRA fine-tuning
│   ├── train_geographer.py           [NEW] Geographer LoRA fine-tuning
│   ├── generate_arbiter_data.py      [NEW] Run trained specialist agents → collect reasoning → build Arbiter dataset
│   └── train_arbiter.py              [NEW] Arbiter LoRA fine-tuning
├── run_pipeline.sh            ← Already written ✅
├── run_specialist_training.sh [NEW] Specialist phase orchestration
└── results/
    ├── models/
    │   ├── historian_lora/
    │   ├── geographer_lora/
    │   └── arbiter_lora/
    └── predictions/
```

---

### `prepare_specialist_data.py` [NEW]

Build two separate SFT JSONL datasets from the combined gold+silver corpus:

**Historian dataset** — Same input, but system prompt = `HISTORIAN_SYSTEM`. Answer format:
```json
{"at_verdict": "TRUE", "isAt_verdict": "FALSE", "reasoning": "The article explicitly states the person held office in Paris..."}
```
The reasoning is **synthesized** from the label + article text at data prep time using a lightweight template (no separate model needed):
- `at=TRUE` + `isAt=TRUE` → "Article confirms person is actively present in location near publication date."
- `at=TRUE` + `isAt=FALSE` → "Historical records place person at location, but no recent presence evidence."
- `at=PROBABLE` → "Person's role or title implies past association with location."
- `at=FALSE` → "No biographical or temporal evidence connects person to location."

**Geographer dataset** — Same input, `GEOGRAPHER_SYSTEM` prompt, geographically-framed reasoning templates.

---

### `train_historian.py` / `train_geographer.py` [NEW]

Identical to `strategy5/scripts/train.py` with:
- Different system prompt (from `prompts.py`)
- Different save directory (`strategy6/results/models/historian_lora/`)
- Answer format expects `at_verdict`/`isAt_verdict`/`reasoning` fields
- Same LoRA config: r=16, all linear layers, bf16
- Same training regime: 5 epochs, LR=2e-4, grad_accum=8, batch=2
- Input: combined gold (weight=5) + silver (weight=1) train files

**Training time estimate**: ~10 hours per agent on RTX A5000 (same as S5).

---

### `generate_arbiter_data.py` [NEW]

1. Load trained Historian + Geographer LoRA adapters (one at a time to save VRAM).
2. Run **both specialists** on all training docs → store `(hist_verdict, hist_reasoning, geo_verdict, geo_reasoning, gold_at, gold_isAt)` tuples.
3. Serialize as Arbiter SFT JSONL:
   ```
   system: ARBITER_SYSTEM
   user:   [historian verdict + reasoning] + [geographer verdict + reasoning] + [article excerpt]
   answer: {"at": "TRUE", "isAt": "FALSE"}
   ```
4. This creates an Arbiter dataset where the model learns to synthesize real agent outputs (not templates).

---

### `train_arbiter.py` [NEW]

- Trains on the Arbiter dataset generated above.
- Answer tokens only (mask out system+user as with S5).
- Shorter training: 3 epochs (less risk of overfitting on the smaller dataset).
- Same LoRA config.

---

### `run_specialist_training.sh` [NEW]

```bash
#!/bin/bash
# Step 1: Prepare specialist SFT datasets
python3 strategy6/scripts/prepare_specialist_data.py

# Step 2: Train Historian specialist (10h)
python3 strategy6/scripts/train_historian.py 2>&1 | tee strategy6/training_historian.log

# Step 3: Train Geographer specialist (10h)
python3 strategy6/scripts/train_geographer.py 2>&1 | tee strategy6/training_geographer.log

# Step 4: Generate Arbiter training data (run hist+geo inference on full train set, ~1h)
python3 strategy6/scripts/generate_arbiter_data.py

# Step 5: Train Arbiter (3h)
python3 strategy6/scripts/train_arbiter.py 2>&1 | tee strategy6/training_arbiter.log

# Step 6: Run full multi-agent inference on dev set
bash strategy6/run_pipeline.sh
```

---

## Why This Will Push Recall > 0.76

| Factor | Strategy 6 (Phase 1) | Strategy 6.5 |
| :--- | :--- | :--- |
| Historian training | S5 weights (generalist) | Dedicated fine-tune on historian prompts |
| Geographer training | S5 weights (generalist) | Dedicated fine-tune on geo prompts |
| Arbiter training | S5 weights (generalist) | Fine-tuned on real agent outputs |
| Training data | None (zero-shot) | 1,161 docs / 15,418 pairs |
| Reasoning quality | Template-based at inference | Learned from distribution |
| Disagreement handling | Rule-based (conservative) | Learned (Arbiter SFT) |

> [!NOTE]
> Phase 1 (current S6 run) validates the benefit of the multi-agent structure at zero training cost. If Global Recall improves over 0.7289, Phase 2 specialist training is justified and will be launched immediately.

---

## Verification Plan

### Phase 1 (currently running)
```bash
# Already launched:
bash strategy6/run_pipeline.sh 2>&1 | tee strategy6/pipeline.log
# Expected: Global Macro Recall > 0.71 (conservative)
```

### Phase 2 (after Phase 1 confirms benefit)
```bash
bash strategy6/run_specialist_training.sh
# Run the official HIPE scorer on the dev set
python3 strategy6/scripts/analyse_results.py
# Target: Global Macro Recall > 0.76
```

### Success Criteria
| Phase | Minimum Target | Stretch Target |
| :--- | :--- | :--- |
| Phase 1 (zero-shot) | 0.71 | 0.73+ |
| Phase 2 (specialist SFT) | 0.76 | 0.80+ |
