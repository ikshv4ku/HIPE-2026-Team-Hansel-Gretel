# Strategy 8 — Chain-of-Thought Few-Shot Prompting via Large LLMs

> **Status:** Ready to run — no GPU, no fine-tuning required.

---

## What this strategy does

This strategy attacks the HIPE-2026 relation-extraction task **without any
model training**. Instead, it uses two well-established prompting techniques:

### Few-Shot Learning
Rather than asking the model to guess from scratch, we include **4 worked
examples** (covering all label combinations: TRUE/TRUE, PROBABLE/FALSE, FALSE/FALSE,
TRUE/FALSE) directly inside the prompt. The model learns the exact expected format
and label vocabulary from those demonstrations before seeing the real question.
This alone typically lifts performance by 10–20 % vs. zero-shot.

### Chain-of-Thought (CoT) Reasoning
The model is explicitly instructed to **think aloud in three structured steps**
before committing to an answer:

1. **Biographical & Temporal Analysis** — Who is this person? What did they do?
   When were they alive? Does the article date constrain their presence?
2. **Geographic & Contextual Analysis** — What language/region is the source?
   What place names appear? Does the source's geography imply presence?
3. **Synthesis & Decision** — Weigh both analyses, apply the hard rule
   (`at=FALSE ⟹ isAt=FALSE`), and produce the final verdicts.

Forcing explicit reasoning steps dramatically reduces hallucination and improves
consistency, especially on edge cases like deceased persons or ambiguous roles.

### Task Definition (for context)
For every **person–location pair** in a historical newspaper article, we classify:

| Label | Meaning | Values |
|-------|---------|--------|
| `at`  | Was the person **ever** at this location before the article date? | `TRUE` / `PROBABLE` / `FALSE` |
| `isAt`| Was the person there **within ~1 month** of publication? | `TRUE` / `FALSE` |

**Hard rule:** if `at=FALSE` then `isAt` must be `FALSE`.

---

## Directory Structure

```
strategy8/
├── requirements.txt          # Python dependencies (install once)
├── run.sh                    # ← START HERE — one-stop launcher
├── scripts/
│   ├── cot_prompts.py        # All prompts and few-shot examples
│   ├── run_inference.py      # Main inference engine (multi-backend)
│   └── analyse_results.py    # Official HIPE-2026 scorer wrapper
└── results/
    ├── predictions/           # Prediction JSONL files are saved here
    └── analysis/              # Evaluation reports are saved here
```

---

## Step-by-Step Setup Guide

### Prerequisites
- Python 3.9 or newer
- Internet access (to reach the LLM API)
- The full repository (already available via GitHub collaborator access)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<org>/HIPE-2026-Team-Hansel-Gretel.git
cd HIPE-2026-Team-Hansel-Gretel
```

### Step 2 — Install Dependencies

You only need to install libraries for the backend you intend to use.
The `requirements.txt` lists all of them; installing everything is fine.

```bash
pip install -r strategy8/requirements.txt
```

> **IBM watsonx users:** The key package is `ibm-watsonx-ai`. If you encounter
> version conflicts, install it separately:
> ```bash
> pip install ibm-watsonx-ai
> ```

### Step 3 — Open `run.sh` and fill in your credentials

```bash
# Open the launcher in any text editor
nano strategy8/run.sh
```

Find the `EDIT THESE` section at the top:

```bash
BACKEND="ibm"                               # ← change to your backend
MODEL="ibm/granite-13b-instruct-v2"         # ← change to your model ID

IBM_URL="https://us-south.ml.cloud.ibm.com" # ← your watsonx region URL
IBM_PROJECT_ID=""                            # ← paste your Project ID
IBM_API_KEY=""                               # ← paste your IBM Cloud API key
```

### Step 4 — Run the Full Pipeline

```bash
bash strategy8/run.sh
```

This will:
1. Load the HIPE-2026 dev set (21 articles, ~1200 person–location pairs)
2. Call your LLM for each pair with the CoT few-shot prompt
3. Save predictions to `strategy8/results/predictions/`
4. Score against the gold standard and save the report to `strategy8/results/analysis/`

### Step 5 — Share the Results

Send the following files back:
- `strategy8/results/predictions/preds-dev-8-*.jsonl`  (raw predictions)
- `strategy8/results/analysis/*_evaluation.txt`         (scored report)

---

## Backend-Specific Instructions

### IBM watsonx.ai

You need three pieces of information from the IBM Cloud console:

| Variable | Where to find it |
|----------|-----------------|
| `IBM_URL` | watsonx.ai → Deployments → endpoint URL |
| `IBM_PROJECT_ID` | watsonx.ai → Project → Manage → Project ID |
| `IBM_API_KEY` | IBM Cloud → Manage → Access → API keys |

Recommended models (subject to availability on your instance):

| Model | Notes |
|-------|-------|
| `ibm/granite-13b-instruct-v2` | IBM's flagship instruction model |
| `ibm/granite-3-8b-instruct` | Lighter, faster alternative |
| `meta-llama/llama-3-1-70b-instruct` | Llama 3.1 70B if available |
| `mistralai/mixtral-8x7b-instruct-v01` | Strong multilingual model |

```bash
# Example — run from repo root
python3 strategy8/scripts/run_inference.py \
    --backend        ibm \
    --model          ibm/granite-13b-instruct-v2 \
    --ibm_url        https://us-south.ml.cloud.ibm.com \
    --ibm_project_id YOUR_PROJECT_ID \
    --ibm_api_key    YOUR_API_KEY
```

---

### OpenAI (GPT-4o, GPT-4 Turbo, etc.)

```bash
python3 strategy8/scripts/run_inference.py \
    --backend        openai \
    --model          gpt-4o \
    --openai_api_key YOUR_OPENAI_KEY
```

The script uses the `response_format={"type": "json_object"}` mode for GPT-4,
which guarantees valid JSON output.

---

### Anthropic (Claude 3.5 Sonnet, Haiku, etc.)

```bash
python3 strategy8/scripts/run_inference.py \
    --backend           anthropic \
    --model             claude-3-5-sonnet-20241022 \
    --anthropic_api_key YOUR_ANTHROPIC_KEY
```

---

### HuggingFace Inference API

Useful for open models like Mixtral, Llama 3, Mistral, Falcon, etc.

```bash
python3 strategy8/scripts/run_inference.py \
    --backend    hf \
    --model      mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --hf_api_key YOUR_HF_TOKEN
```

Get a free token at https://huggingface.co/settings/tokens

---

### Local OpenAI-Compatible Server (Ollama, LM Studio, vLLM)

If you're running a model locally (e.g. Ollama with Llama 3 70B):

```bash
# Start your local server first, e.g.:
# ollama run llama3:70b

python3 strategy8/scripts/run_inference.py \
    --backend        local \
    --model          llama3:70b \
    --local_base_url http://localhost:11434/v1
```

---

## Advanced Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max_tokens` | `700` | Max tokens to generate per pair. 700 is sufficient for CoT. |
| `--temperature` | `0.0` | Greedy decoding (recommended for classification). |
| `--retry` | `3` | Retries on API errors with exponential back-off. |
| `--no_few_shot` | off | Run zero-shot CoT only (no examples). Useful for ablation. |
| `--input` | dev set | Override with a different JSONL file. |
| `--output` | auto-named | Override the output path. |

### Ablation: Zero-Shot CoT vs. Few-Shot CoT

To measure the pure contribution of few-shot examples:

```bash
# Few-shot CoT (default)
python3 strategy8/scripts/run_inference.py --backend openai --model gpt-4o \
    --output strategy8/results/predictions/preds-few-shot.jsonl

# Zero-shot CoT (no examples)
python3 strategy8/scripts/run_inference.py --backend openai --model gpt-4o \
    --no_few_shot \
    --output strategy8/results/predictions/preds-zero-shot.jsonl

# Score both
python3 strategy8/scripts/analyse_results.py strategy8/results/predictions/preds-few-shot.jsonl
python3 strategy8/scripts/analyse_results.py strategy8/results/predictions/preds-zero-shot.jsonl
```

---

## How the Prompt Works (Technical Detail)

### System Prompt (`SYSTEM_PROMPT` in `cot_prompts.py`)

Sets up the expert persona, defines the two labels and all allowed values,
states the hard rule, and specifies the three reasoning steps and exact JSON
output format. This is passed as the `system` role message.

### User Message (`build_user_message()` in `cot_prompts.py`)

Concatenates:
1. Four worked examples (Q+A format) — these are the **few-shot demonstrations**
2. The actual question: article metadata + person + location + article text

Each few-shot example shows the model:
- The exact JSON field names (`at`, `isAt`, `step1_biographical`, etc.)
- Realistic reasoning for each case
- All possible label combinations

### Output Parsing (`parse_output()` in `run_inference.py`)

1. Strips markdown code fences if the model wraps output in ``` blocks
2. Attempts `json.loads()` on the full response
3. Falls back to regex extraction of the first `{…}` block
4. Validates and normalises all values; applies the hard rule; defaults to
   `FALSE` if extraction fails

---

## Expected Performance

Based on the trajectory of previous strategies:

| Strategy | Approach | Global Macro Recall |
|----------|----------|-------------------|
| Strategy 6.5 | Multi-agent fine-tuning (Qwen 3B) | 0.7204 |
| Strategy 8 (target) | CoT few-shot large LLM | **> 0.76** |

With GPT-4o or Claude 3.5 Sonnet + CoT + few-shot, we anticipate reaching or
exceeding the 0.76 target without any training.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: ibm_watsonx_ai` | `pip install ibm-watsonx-ai` |
| `AuthenticationError` (IBM) | Double-check API key and project ID |
| `RateLimitError` | Increase `--delay` to e.g. `5.0` |
| JSON parse failure | Check `results/predictions/*.jsonl` — the `_cot_*` fields will show what the model actually said |
| All predictions are FALSE | The model may not be following the format; try a larger model |

---

## Contact

Repository: https://github.com/ikshv4ku/HIPE-2026-Team-Hansel-Gretel  
Task lead: Pradyuman Shekhawat (shekhawat.pradyuman10@gmail.com)
