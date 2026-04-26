#!/bin/bash
# =============================================================================
# Strategy 7: Multi-Agent Evaluation Pipeline
# Runs 3-agent inference (Historian → Geographer → Arbiter) and scores.
# Uses device_map="auto" — model shards across all visible GPUs.
# =============================================================================
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy7/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Expose all 3 GPUs for device_map="auto" to shard the 7B model
export CUDA_VISIBLE_DEVICES=0,1,2

echo ""
echo "==========================================="
echo " Strategy 7: Multi-Agent Pipeline (Eval)"
echo "==========================================="

mkdir -p strategy7/results/predictions strategy7/results/analysis

echo "[1/2] Running multi-agent inference (Historian → Geographer → Arbiter)..."
python3 strategy7/scripts/agent_inference.py \
    --input    HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl \
    --output   strategy7/results/predictions/preds-dev-7.jsonl \
    --hist_lora strategy7/results/models/historian_lora \
    --geo_lora  strategy7/results/models/geographer_lora \
    --arb_lora  strategy7/results/models/arbiter_lora

echo "[2/2] Scoring with official HIPE-2026 scorer..."
python3 strategy7/scripts/analyse_results.py

echo ""
echo "==========================================="
echo " Strategy 7 evaluation finished!"
echo "==========================================="
