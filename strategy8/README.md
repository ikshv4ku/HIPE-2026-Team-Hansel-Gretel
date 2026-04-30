# Strategy 8 — Chain-of-Thought Few-Shot Prompting via RITS LLMs

> **Status:** Ready to run — no GPU, no fine-tuning required.  
> **API:** IBM RITS (Research Internal Technology Services)

---

## What This Strategy Does

This strategy attacks the HIPE-2026 relation-extraction task **without any model training**. Instead, it uses **Chain-of-Thought (CoT) Few-Shot Prompting** through IBM's RITS inference gateway, which provides access to large LLMs (Llama 3.3 70B, Granite 3.2 8B, Llama 4 Maverick, GPT-OSS 120B).

### Task Definition

For every **person–location pair** in a historical newspaper article, we classify:

| Label | Meaning | Values |
|-------|---------|--------|
| `at`  | Was the person **ever** at this location before the article date? | `TRUE` / `PROBABLE` / `FALSE` |
| `isAt`| Was the person there **within ~1 month** of publication? | `TRUE` / `FALSE` |

**Hard rule:** if `at=FALSE` then `isAt` must be `FALSE`.

### How It Works

1. **Few-Shot Learning**: 4 worked examples covering all label combinations are embedded in the prompt
2. **Chain-of-Thought Reasoning**: The model reasons through 3 explicit steps:
   - Step 1: Biographical & Temporal Analysis
   - Step 2: Geographic & Contextual Analysis
   - Step 3: Synthesis & Decision
3. **JSON Output**: Structured output is parsed and validated automatically

---

## Directory Structure

```
strategy8/
├── hipe_inference.py     # ← MAIN SCRIPT — run this
├── llmaj.py              # Reference script format (from guide)
├── requirements.txt      # Python dependencies
├── .env                  # RITS API key (create this — see below)
└── results/
    ├── predictions/      # Prediction JSONL files
    └── analysis/         # Evaluation reports
```

---

## Step-by-Step Setup Guide

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ikshv4ku/HIPE-2026-Team-Hansel-Gretel.git
cd HIPE-2026-Team-Hansel-Gretel
```

### Step 2 — Install Dependencies

```bash
pip install -r strategy8/requirements.txt
```

### Step 3 — Set Up Your RITS API Key

Create a `.env` file in the **repository root** (same level as `strategy8/`):

```bash
# Create .env file
echo "RITS_API_KEY=your_rits_api_key_here" > .env
```

Or edit it manually:

```env
RITS_API_KEY=your_actual_rits_api_key
```

> **Where to get the key:** Request access via the IBM RITS portal at `https://rits.fmaas.res.ibm.com`

### Step 4 — Run Inference

```bash
# Default: Llama 3.3 70B (recommended)
python strategy8/hipe_inference.py -model llama70b

# Granite 3.2 8B (faster, lighter)
python strategy8/hipe_inference.py -model granite

# Llama 4 Maverick 17B
python strategy8/hipe_inference.py -model llama4

# GPT-OSS 120B (largest)
python strategy8/hipe_inference.py -model gpt
```

### Step 5 — Check Results

Results are automatically scored and saved:

```
strategy8/results/predictions/preds-dev-8-llama70b.jsonl   # Raw predictions
strategy8/results/analysis/evaluation_llama70b.txt          # Scored report
```

---

## Available Models

| Flag | Model | Notes |
|------|-------|-------|
| `llama70b` | `meta-llama/llama-3-3-70b-instruct` | **Recommended** — best accuracy |
| `granite` | `ibm-granite/granite-3.2-8b-instruct` | Fastest, IBM's own |
| `llama4` | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` | Good speed/accuracy tradeoff |
| `gpt` | `openai/gpt-oss-120b` | Largest model available |

---

## Full CLI Reference

```
usage: hipe_inference.py [-h] [-input_file INPUT_FILE]
                          [-model {llama70b,granite,llama4,gpt}]
                          [-no_few_shot]

Arguments:
  -input_file    Path to input JSONL (default: HIPE-2026 dev set)
  -model         RITS model to use (default: llama70b)
  -no_few_shot   Disable few-shot examples (zero-shot CoT only)
```

### Example: Custom Input File

```bash
python strategy8/hipe_inference.py \
    -input_file path/to/your/test-set.jsonl \
    -model llama70b
```

### Example: Zero-Shot Ablation

```bash
# Compare few-shot vs zero-shot
python strategy8/hipe_inference.py -model llama70b
python strategy8/hipe_inference.py -model llama70b -no_few_shot
```

---

## How the Script Works (Technical Detail)

The script follows the same pattern as `llmaj.py`:

1. **Reads RITS_API_KEY** from a `.env` file using `dotenv`
2. **Resolves the RITS endpoint** for the chosen model via the RITS API
3. **Creates an OpenAI-compatible client** pointed at the RITS gateway
4. **For each person-location pair**, builds a CoT prompt with few-shot examples and calls the LLM
5. **Parses the JSON response**, validates labels, enforces the hard rule
6. **Scores** using the official HIPE-2026 scorer

### Output Format

Each prediction includes the model's chain-of-thought reasoning:

```json
{
  "at": "TRUE",
  "isAt": "FALSE",
  "_cot_biographical": "Person X was a diplomat who served in...",
  "_cot_geographic": "The article mentions the embassy in...",
  "_cot_synthesis": "Historical presence confirmed but no recent..."
}
```

---

## Expected Performance

Based on Strategy 5 and 7 results with fine-tuned models:

| Strategy | Approach | Global Macro Recall |
|----------|----------|-------------------|
| S5 | Fine-tuned Qwen 3B (single model) | 0.7289 |
| S7 | Fine-tuned Qwen 7B (multi-agent) | 0.7280 |
| **S8** | **CoT few-shot via large LLM (no training)** | **TBD** |

With Llama 3.3 70B + CoT + few-shot, we anticipate competitive or superior results without any training.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `RITS_API_KEY not found` | Create `.env` file with your key (see Step 3) |
| `Failed getting RITS model list` | Check your API key and network access to `rits.fmaas.res.ibm.com` |
| `Model not found in RITS` | Use one of the supported model flags: `llama70b`, `granite`, `llama4`, `gpt` |
| JSON parse failure | Check `_cot_*` fields in predictions — shows what the model actually said |
| All predictions are FALSE | Try a larger model (e.g. `llama70b` instead of `granite`) |

---

## Contact

Repository: https://github.com/ikshv4ku/HIPE-2026-Team-Hansel-Gretel  
Task lead: Pradyuman Shekhawat (shekhawat.pradyuman10@gmail.com)
