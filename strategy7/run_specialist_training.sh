 #!/bin/bash
# =============================================================================
# Strategy 7: Specialist Fine-Tuning Orchestration
# Runs Historian and Geographer training IN PARALLEL on separate GPUs,
# then trains the Arbiter sequentially after both finish.
# =============================================================================
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HOME=/mnt/combined/pradyuman/hf_cache
export PYTHONPATH=$(pwd)/strategy7/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p strategy7/data strategy7/results/models strategy7/results/predictions strategy7/results/analysis

echo ""
echo "==========================================="
echo " Strategy 7: Specialist Training Pipeline"
echo "==========================================="

# ── Step 1: Prepare SFT datasets ─────────────────────────────────────────────
echo "[1/6] Preparing Historian & Geographer SFT datasets..."
python3 strategy7/scripts/prepare_specialist_data.py
echo "  Done."

# ── Step 2: Train Historian (GPU 0) & Geographer (GPU 1) in PARALLEL ─────────
echo "[2/6] Launching Historian training on GPU 0 (background)..."
CUDA_VISIBLE_DEVICES=0 python3 strategy7/scripts/train_specialists.py \
    --role       historian \
    --train_file strategy7/data/historian_sft.jsonl \
    --out_dir    strategy7/results/models/historian_lora \
    --epochs     3 \
    --batch_size 1 \
    --grad_accum 16 \
    2>&1 | tee strategy7/training_historian.log &
HIST_PID=$!

echo "[2/6] Launching Geographer training on GPU 1 (background)..."
CUDA_VISIBLE_DEVICES=1 python3 strategy7/scripts/train_specialists.py \
    --role       geographer \
    --train_file strategy7/data/geographer_sft.jsonl \
    --out_dir    strategy7/results/models/geographer_lora \
    --epochs     3 \
    --batch_size 1 \
    --grad_accum 16 \
    2>&1 | tee strategy7/training_geographer.log &
GEO_PID=$!

echo "[2/6] Waiting for both specialist trainings to complete..."
wait $HIST_PID
wait $GEO_PID
echo "  Both specialists trained successfully."

# ── Step 3: Generate Arbiter training data (2 shards in parallel) ─────────────
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

wait $SHARD1_PID && wait $SHARD2_PID
echo "  Arbiter data generation complete."

# ── Step 4: Merge shards ──────────────────────────────────────────────────────
echo "[4/6] Merging Arbiter SFT shards..."
python3 strategy7/scripts/merge_shards.py
echo "  Merged → strategy7/data/arbiter_sft.jsonl"

# ── Step 5: Train Arbiter (GPU 2) ─────────────────────────────────────────────
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

# ── Step 6: Run full eval pipeline ────────────────────────────────────────────
echo "[6/6] Running final evaluation pipeline..."
bash strategy7/run_pipeline.sh

echo ""
echo "==========================================="
echo " Strategy 7 training pipeline COMPLETE!"
echo "==========================================="
