#!/bin/bash
# Strategy 6.5: Parallel Geographer Training
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy6/scripts
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- Starting Geographer Training (Parallel) ---"
python3 strategy6/scripts/train_specialists.py \
  --role geographer \
  --train_file strategy6/data/geographer_sft.jsonl \
  --out_dir strategy6/results/models/geographer_lora \
  --epochs 3
