#!/bin/bash
set -e

echo "==========================================="
echo " Strategy 5: Qwen2.5-3B-Instruct + LoRA SFT"
echo "==========================================="

# Activate env and set cache
source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)/strategy5/scripts

echo "[1/4] Preparing data splits (reuses strategy4 splits if present)..."
python3 strategy5/scripts/prepare_data.py

echo "[2/4] Fine-tuning Gemma-2-2b-it with LoRA..."
python3 strategy5/scripts/train.py \
    --epochs 5 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32

echo "[3/4] Running inference on dev set..."
mkdir -p strategy5/results/predictions
python3 strategy5/scripts/inference.py \
    --input HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl \
    --output strategy5/results/predictions/preds-dev-all.jsonl

echo "[4/4] Analysing results with official scorer..."
mkdir -p strategy5/results/analysis
python3 strategy5/scripts/analyse_results.py

echo "==========================================="
echo " Strategy 5 execution finished!"
echo "==========================================="
