#!/bin/bash
# =============================================================
# Strategy 6: Multi-Agent Agentic Framework for HIPE-2026
# Three agents: Historian → Geographer → Arbiter
# Phase 1: Zero-shot using Strategy 5 LoRA adapter
# =============================================================
set -e
cd "$(dirname "$0")/.."

echo ""
echo "==========================================="
echo " Strategy 6: Multi-Agent Pipeline (Phase 1)"
echo "==========================================="

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy6/scripts
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Run the three-agent inference pipeline on the dev set
echo "[1/2] Running multi-agent inference (Historian → Geographer → Arbiter)..."
mkdir -p strategy6/results/predictions strategy6/results/analysis

python3 strategy6/scripts/agent_inference.py \
    --input  HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl \
    --output strategy6/results/predictions/preds-dev-6.5-specialist.jsonl \
    --hist_lora strategy6/results/models/historian_lora \
    --geo_lora  strategy6/results/models/geographer_lora \
    --arb_lora  strategy6/results/models/arbiter_lora

# Step 2: Run the official HIPE-2026 scorer
echo "[2/2] Scoring with official HIPE-2026 scorer..."
python3 strategy6/scripts/analyse_results.py

echo ""
echo "==========================================="
echo " Strategy 6 execution finished!"
echo "==========================================="
