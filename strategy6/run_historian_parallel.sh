#!/bin/bash
# Strategy 6.5: Parallel Historian Training
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy6/scripts
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- Starting Historian Training (Parallel) ---"
python3 strategy6/scripts/train_specialists.py \
  --role historian \
  --train_file strategy6/data/historian_sft.jsonl \
  --out_dir strategy6/results/models/historian_lora \
  --epochs 3
