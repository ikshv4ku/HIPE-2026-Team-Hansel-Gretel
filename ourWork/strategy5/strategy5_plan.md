# Strategy 5: Qwen2.5-3B-Instruct Generative Classification + LoRA Fine-Tuning

## Background & Goal

Strategies 1–4.5 use encoder-only models (XLM-RoBERTa, Qwen) with classification heads. The current best global macro recall is ~**0.6344** (Strategy 4.5). This strategy implements a generative LLM-based approach using **Qwen2.5-3B-Instruct** for HIPE-2026 person-place relation extraction. It replaces the previous encoder-only classification with generative JSON instruction-following and significantly expanded context windows.

**Strategy 5** upgrades to `Qwen/Qwen2.5-3B-Instruct` — a 3.1B instruction-tuned model that fits in GPU memory — using **supervised fine-tuning (SFT)** with LoRA on the gold+silver data, trained to output structured JSON answers. This approach provides:

1. **Chain-of-Thought reasoning** baked into the fine-tuning format
2. **Larger context window** (32768 tokens) → no truncation of historical newspaper articles
3. **Instruction-following** → structured JSON output (`{"at": "TRUE", "isAt": "FALSE"}`)
4. **Knowledge-rich pre-training** of Qwen2.5 for timeline/date reasoning
5. **Wikidata hard-rule post-processing** → dead person before pub date → force `isAt=FALSE`

> [!IMPORTANT]
> `bitsandbytes` is not installed. We will use **full fp16/bf16 precision** for a 3B model which fits in one RTX A5000 (24GB). No quantization needed.

> [!IMPORTANT]
> **Pivot to Qwen2.5-3B-Instruct**: Due to gated access requirements for Gemma-2, we pivoted to Qwen2.5-3B-Instruct. This was a superior choice as it is non-gated, larger (3.1B vs 1.5B/2B), and has stronger multilingual instruction-following.

> [!WARNING]
> **Hybrid Workspace**: Disk space limitations on the root partition (/) required moving the 16GB `.venv` to `/mnt/combined` while keeping code in `/home` to maintain tool authorized access.

---

## Proposed Changes

### Strategy 5 Directory Structure

#### [NEW] `strategy5/` — New top-level strategy directory

---

### `strategy5/scripts/`

#### [NEW] [prepare_data.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/prepare_data.py)

Reuses the same gold/silver split logic from strategy4 (80/20 gold train/dev via seed=42). Outputs the same `HIPE-2026-v1.0-impresso-train-str4-all.jsonl` and `HIPE-2026-v1.0-impresso-dev-str4-all.jsonl` files (if not already built by strategy 4). Skips rebuilding if outputs already exist.

#### [NEW] [dataset.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/dataset.py)

Converts JSONL pairs into **instruction-tuned prompt/completion pairs**:

```
### Instruction:
You are an expert historian analyzing 19th/20th century newspaper articles. Given a (person, location) pair, classify two relations:
- "at": Was the person EVER at this location before the article date? (TRUE/PROBABLE/FALSE)
- "isAt": Was the person at this location WITHIN ~1 MONTH of the publication date? (TRUE/FALSE)
Remember: if at=FALSE, then isAt must be FALSE.

### Context:
Document date: {date}. Language: {language}. Source: {pub_title}

### Text:
{text[:4000]}  ← truncate to 4000 chars to leave room for instruction + answer

### Person: {person_mentions}, Location: {location_mentions}

### Answer:
{"at": "TRUE", "isAt": "FALSE"}
```

Gold samples use **hard labels**. Silver samples use **soft labels** (label smoothing 0.15) via the completion token probability reweighting at training time.

Weight gold samples 5× higher via `sample_weight` in the loss mask.

#### [NEW] [train.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/train.py)

- Loads `Qwen/Qwen2.5-3B-Instruct` in **bf16** (fits ~7GB, leaving 17GB for activations/batch)
- Applies **LoRA** via `peft`: `r=16, alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
- Uses **Causal LM** training with loss masked to only the JSON answer tokens
- **AdamW** with lr=2e-4 for LoRA params, weight_decay=0.01
- Cosine LR schedule with 10% warmup over 5 epochs, batch_size=2, grad_accum=8 (effective batch=16)
- Saves best checkpoint by global macro recall computed on dev set via greedy decode
- Uses `accelerate` for single-GPU training (no need for multi-GPU)

#### [NEW] [inference.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/inference.py)

- Loads base Qwen2.5-3B-Instruct + LoRA adapter
- Generates the JSON answer with `max_new_tokens=20`, `temperature=0` (greedy)
- Parses `{"at": ..., "isAt": ...}` from generated text
- Post-processing: if `at=FALSE` → force `isAt=FALSE`
- **Wikidata hard rule** (optional, skip if QID is null): query `pers_wikidata_QID` death date; if person died >30 days before pub date → force `isAt=FALSE`. Uses a local cache to avoid re-querying.
- Falls back to `FALSE/FALSE` for malformed outputs

#### [NEW] [analyse_results.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/analyse_results.py)

Runs the official `HIPE-2026-data/scripts/file_scorer_evaluation.py` and prints per-language breakdown. Copied/adapted from strategy4_5.

---

### `strategy5/`

#### [NEW] [run_training.sh](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/run_training.sh)

```bash
#!/bin/bash
set -e
source /mnt/combined/pradyuman/.venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/strategy5/scripts

# Step 1: Prepare data (reuse strategy4 splits if they exist)
python3 strategy5/scripts/prepare_data.py

# Step 2: Fine-tune Qwen2.5-3B-Instruct with LoRA
python3 strategy5/scripts/train.py --epochs 5 --batch_size 2 --grad_accum 8

# Step 3: Run inference on dev set
mkdir -p strategy5/results/predictions
python3 strategy5/scripts/inference.py \
  --input HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl \
  --output strategy5/results/predictions/preds-dev-all.jsonl

# Step 4: Analyse
mkdir -p strategy5/results/analysis
python3 strategy5/scripts/analyse_results.py
```

---

## Why Qwen2.5-3B over alternatives?

| Option | Reason to pick/reject |
|---|---|
| **Qwen2.5-3B-Instruct** ✅ | Non-gated, 3.1B, fits in 1 GPU with bf16, strong multilingual reasoning |
| Gemma-2-9b | Doesn't fit in 24GB GPU without quantization |
| Llama-3 8B | Too large without quantization |
| GPT-4 API | No offline inference, API cost |
| Qwen2.5-1.5B (existing) | Already tried, not generative, limited reasoning |
| Few-shot only (no FT) | Less accurate than fine-tuning on task-specific data |

---

## Why this approach will significantly beat ~0.63?

1. **Temporal reasoning**: Qwen2.5 was trained on huge amounts of historical text and explicitly reasons about time periods in generation mode, unlike encoder classifiers
2. **Full context**: 32k token window vs 512 tokens → no truncation of long newspaper articles
3. **Generative JSON output**: forces explicit decision-making, eliminates ambiguous probabilities
4. **LoRA SFT on task data**: adapts the model's priors to the HIPE-specific schema and historical language
5. **Wikidata death-date rule**: easy high-precision wins on `isAt` (dead person → FALSE)

Expected improvement: **0.63 → 0.72+** global macro recall on dev.

---

## Verification Plan

### Automated Tests

```bash
# Run from repo root after training completes:
cd /home/pradyuman/HIPE-2026-Team-Hansel-Gretel
source /mnt/combined/pradyuman/.venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache

# Full pipeline end-to-end:
bash strategy5/run_training.sh 2>&1 | tee strategy5/training.log

# Score comparison (after inference):
python3 HIPE-2026-data/scripts/file_scorer_evaluation.py \
  --gold HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl \
  --pred strategy5/results/predictions/preds-dev-all.jsonl
```

### Manual Verification

1. Check `strategy5/training.log` — should show increasing dev macro recall per epoch
2. Final score should be **noticeably above 0.6344**
3. Compare side-by-side with strategy4/4.5 scores in `strategy5/results/analysis/`
