#!/bin/bash
# =============================================================================
# Strategy 8 – Quick-start launcher
# Edit the VARIABLES section below, then run:  bash strategy8/run.sh
# =============================================================================

set -e
cd "$(dirname "$0")/.."   # always run from repo root

# ── EDIT THESE ────────────────────────────────────────────────────────────────
BACKEND="ibm"                                         # ibm | openai | anthropic | hf | local
MODEL="ibm/granite-13b-instruct-v2"                   # model ID for your chosen backend
MAX_TOKENS=700
TEMPERATURE=0.0

# IBM watsonx credentials (leave blank if using a different backend)
IBM_URL="https://us-south.ml.cloud.ibm.com"
IBM_PROJECT_ID=""          # paste your watsonx project ID here
IBM_API_KEY=""             # paste your IBM Cloud API key here

# OpenAI
OPENAI_API_KEY=""

# Anthropic
ANTHROPIC_API_KEY=""

# HuggingFace
HF_TOKEN=""

# Local server
LOCAL_BASE_URL="http://localhost:11434/v1"
# ─────────────────────────────────────────────────────────────────────────────

INPUT="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl"
OUTPUT="strategy8/results/predictions/preds-dev-8-${BACKEND}-$(echo $MODEL | tr '/' '-').jsonl"

mkdir -p strategy8/results/predictions strategy8/results/analysis

echo ""
echo "==========================================="
echo " Strategy 8: CoT Few-Shot Inference"
echo " Backend : $BACKEND"
echo " Model   : $MODEL"
echo "==========================================="

python3 strategy8/scripts/run_inference.py \
    --backend        "$BACKEND" \
    --model          "$MODEL" \
    --input          "$INPUT" \
    --output         "$OUTPUT" \
    --max_tokens     "$MAX_TOKENS" \
    --temperature    "$TEMPERATURE" \
    --ibm_url        "$IBM_URL" \
    --ibm_project_id "$IBM_PROJECT_ID" \
    --ibm_api_key    "$IBM_API_KEY" \
    --openai_api_key "$OPENAI_API_KEY" \
    --anthropic_api_key "$ANTHROPIC_API_KEY" \
    --hf_api_key     "$HF_TOKEN" \
    --local_base_url "$LOCAL_BASE_URL"

echo ""
echo "[S8] Scoring predictions..."
python3 strategy8/scripts/analyse_results.py "$OUTPUT"

echo ""
echo "==========================================="
echo " Strategy 8 complete!"
echo "==========================================="
