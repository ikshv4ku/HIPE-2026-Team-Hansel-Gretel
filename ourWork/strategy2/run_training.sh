#!/bin/bash
set -e

echo "==========================================="
echo " Strategy 2: End-to-End Fine-Tuning Setup  "
echo "==========================================="

echo "[0/5] Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[1/5] Installing dependencies..."
pip install transformers datasets accelerate evaluate scikit-learn tqdm
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

echo "[1.5/5] Fetching HIPE-2026 dataset..."
rm -rf HIPE-2026-data
git clone https://github.com/hipe-eval/HIPE-2026-data.git

echo "[2/5] Preparing multilingual data split..."
python3 strategy1.5/scripts/prepare_multilingual_split.py

echo "[3/5] Starting Strategy 2 Training..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/strategy2/scripts
python3 strategy2/scripts/train.py --epochs 10 --batch_size 8

echo "[4/5] Running Inference..."
python3 strategy2/scripts/inference.py --input HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl --output strategy2/results/predictions/preds-dev-all.jsonl

echo "[5/5] Analysing Results..."
python3 strategy2/scripts/analyse_results.py

echo "==========================================="
echo " Strategy 2 execution finished!"
echo "==========================================="
