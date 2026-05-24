"""
Strategy 5: Prepare Data
Reuses the same gold+silver merged splits from strategy4 if they already exist.
If the splits don't exist, builds them with gold_weight=5.0.
"""
import json
import glob
import random
import os

SPLIT_DIR = "HIPE-2026-data/data/newspapers/v1.0/splits"
TRAIN_OUT = os.path.join(SPLIT_DIR, "HIPE-2026-v1.0-impresso-train-str4-all.jsonl")
DEV_OUT   = os.path.join(SPLIT_DIR, "HIPE-2026-v1.0-impresso-dev-str4-all.jsonl")


def build_split(gold_dir, silver_dir, output_dir, train_ratio=0.8,
                gold_weight=5.0, silver_weight=1.0):
    gold_files = glob.glob(os.path.join(gold_dir, "HIPE-2026-v1.0-impresso-train-*.jsonl"))
    gold_files = [f for f in gold_files if "randomized" not in f and "all" not in f]

    gold_docs = []
    for f in gold_files:
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    doc = json.loads(line)
                    doc["split_weight"] = gold_weight
                    gold_docs.append(doc)
    print(f"Loaded {len(gold_docs)} GOLD documents.")

    random.seed(42)
    random.shuffle(gold_docs)
    split_idx = int(len(gold_docs) * train_ratio)
    train_gold = gold_docs[:split_idx]
    dev_gold   = gold_docs[split_idx:]

    silver_files = glob.glob(os.path.join(silver_dir, "*.jsonl"))
    silver_docs  = []
    for f in silver_files:
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    doc = json.loads(line)
                    doc["split_weight"] = silver_weight
                    silver_docs.append(doc)
    print(f"Loaded {len(silver_docs)} SILVER documents.")

    train_pool = train_gold + silver_docs
    random.shuffle(train_pool)

    os.makedirs(output_dir, exist_ok=True)
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for d in train_pool:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(DEV_OUT, "w", encoding="utf-8") as f:
        for d in dev_gold:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_pool)} training docs → {TRAIN_OUT}")
    print(f"Saved {len(dev_gold)} dev docs → {DEV_OUT}")


if __name__ == "__main__":
    if os.path.exists(TRAIN_OUT) and os.path.exists(DEV_OUT):
        print("Split files already exist — skipping data preparation.")
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        build_split(
            gold_dir=os.path.join(base, "HIPE-2026-data/data/newspapers/v1.0"),
            silver_dir=os.path.join(base, "HIPE-2026-data/data/sandbox"),
            output_dir=SPLIT_DIR,
        )
