#!/bin/bash
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy7/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==========================================="
echo " Resuming Strategy 7 Pipeline from Step 3"
echo "==========================================="

echo "[3/6] Generating Arbiter SFT data (2 shards in parallel on GPU 0 & GPU 2)..."
TOTAL=$(wc -l < strategy7/data/historian_sft.jsonl)
MID=$(( TOTAL / 2 ))

CUDA_VISIBLE_DEVICES=0 python3 strategy7/scripts/generate_arbiter_data.py \
    --start 0 --end $MID --gpu 0 \
    2>&1 | tee strategy7/arbiter_gen_shard1.log &
SHARD1_PID=$!

CUDA_VISIBLE_DEVICES=2 python3 strategy7/scripts/generate_arbiter_data.py \
    --start $MID --end $TOTAL --gpu 2 \
    2>&1 | tee strategy7/arbiter_gen_shard2.log &
SHARD2_PID=$!

wait $SHARD1_PID
wait $SHARD2_PID
echo "  Arbiter data generation complete."

echo "[4/6] Merging Arbiter SFT shards..."
python3 strategy7/scripts/merge_shards.py
echo "  Merged → strategy7/data/arbiter_sft.jsonl"

echo "[5/6] Training Arbiter on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python3 strategy7/scripts/train_specialists.py \
    --role       arbiter \
    --train_file strategy7/data/arbiter_sft.jsonl \
    --out_dir    strategy7/results/models/arbiter_lora \
    --epochs     3 \
    --batch_size 1 \
    --grad_accum 16 \
    2>&1 | tee strategy7/training_arbiter.log
echo "  Arbiter training complete."

echo "[6/6] Running final evaluation pipeline..."
bash strategy7/run_pipeline.sh

echo "==========================================="
echo " Strategy 7 resume pipeline COMPLETE!"
echo "==========================================="
