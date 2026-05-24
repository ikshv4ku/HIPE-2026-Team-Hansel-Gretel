#!/bin/bash
set -e

echo "==========================================="
echo " Strategy 4: Qwen Unified PEFT Execution"
echo "==========================================="

echo "[0/5] Setting up environment and cache..."
source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache

echo "[1/5] Fetching and merging datasets..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/strategy4_5/scripts
python3 strategy4_5/scripts/prepare_data.py

echo "[2/5] Starting Strategy 4.5 Unified Training..."
python3 strategy4_5/scripts/train.py --epochs 10 --batch_size 4 --model_name Qwen/Qwen2.5-1.5B --smoothing 0.15

echo "[3/5] Compiling and Running Unified Inference..."
mkdir -p strategy4_5/results/predictions
python3 strategy4_5/scripts/inference.py --input HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl --output strategy4_5/results/predictions/preds-dev-all.jsonl --model_name Qwen/Qwen2.5-1.5B

echo "[4/5] Analysing Resulting Predictions..."
mkdir -p strategy4_5/results/analysis
python3 strategy4_5/scripts/analyse_results.py

echo "==========================================="
echo " Strategy 4.5 execution finished!"
echo "==========================================="
