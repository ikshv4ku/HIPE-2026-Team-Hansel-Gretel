#!/bin/bash
set -e

echo "==========================================="
echo " Strategy 3: Monolingual PEFT Training Setup"
echo "==========================================="

echo "[0/5] Setting up virtual environment..."
source .venv/bin/activate

echo "[1/5] Fetching and slicing datasets..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/strategy3/scripts
python3 strategy3/scripts/split_languages.py

echo "[2/5] Starting Strategy 3 Language Training Series..."
for LANG in en fr de; do
    echo ">>>> Training PEFT model for language: ${LANG} <<<<"
    python3 strategy3/scripts/train.py --language ${LANG} --epochs 10 --batch_size 8
done

echo "[3/5] Compiling and Running Routed Inference..."
mkdir -p strategy3/results/predictions
python3 strategy3/scripts/inference.py --input HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl --output strategy3/results/predictions/preds-dev-all.jsonl

echo "[4/5] Analysing Merged Results..."
mkdir -p strategy3/results/analysis
python3 strategy3/scripts/analyse_results.py

echo "==========================================="
echo " Strategy 3 execution finished!"
echo "==========================================="
