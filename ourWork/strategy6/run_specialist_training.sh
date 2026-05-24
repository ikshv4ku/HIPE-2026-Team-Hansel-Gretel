#!/bin/bash
# Strategy 6.5: Specialist Fine-Tuning Orchestration
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy6/scripts
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p strategy6/results/models

# Historian is already running on GPU 0
echo "Historian training is managed separately or currently running."
