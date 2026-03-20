import json
import glob
import random
import os

def build_split(gold_dir, silver_dir, output_dir, train_ratio=0.8, gold_weight=50.0, silver_weight=1.0):
    # 1. Load Gold Data
    gold_files = glob.glob(os.path.join(gold_dir, "HIPE-2026-v1.0-impresso-train-*.jsonl"))
    gold_files = [f for f in gold_files if "randomized" not in f and "all" not in f]
    
    gold_docs = []
    for f in gold_files:
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    doc = json.loads(line)
                    # Inject extremely high weight for Gold
                    doc["split_weight"] = gold_weight
                    gold_docs.append(doc)
                    
    print(f"Loaded {len(gold_docs)} GOLD documents.")
    
    # Shuffle and split Gold
    random.seed(42)
    random.shuffle(gold_docs)
    
    split_idx = int(len(gold_docs) * train_ratio)
    train_gold = gold_docs[:split_idx]
    dev_gold = gold_docs[split_idx:]
    
    # 2. Load Silver Data
    silver_files = glob.glob(os.path.join(silver_dir, "*.jsonl"))
    silver_docs = []
    for f in silver_files:
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip():
                    doc = json.loads(line)
                    # Inject low weight for Silver
                    doc["split_weight"] = silver_weight
                    silver_docs.append(doc)
                    
    print(f"Loaded {len(silver_docs)} SILVER documents.")
    
    # 3. Form Final Splits
    # Train Pool = 80% of Gold + 100% of Silver
    train_pool = train_gold + silver_docs
    random.shuffle(train_pool)
    
    # Dev Pool = STRICTLY 20% of Gold 
    dev_pool = dev_gold
    
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "HIPE-2026-v1.0-impresso-train-all.jsonl")
    dev_out = os.path.join(output_dir, "HIPE-2026-v1.0-impresso-dev-all.jsonl")
    
    with open(train_out, 'w', encoding='utf-8') as f:
        for d in train_pool:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    with open(dev_out, 'w', encoding='utf-8') as f:
        for d in dev_pool:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(train_pool)} Mined Training docs to {train_out}")
    print(f"Saved {len(dev_pool)} Strict Gold Validation docs to {dev_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/HIPE-2026-data/data/newspapers/v1.0")
    parser.add_argument("--silver_dir", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/HIPE-2026-data/data/sandbox")
    parser.add_argument("--out_dir", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/HIPE-2026-data/data/newspapers/v1.0/splits")
    args = parser.parse_args()
    build_split(args.gold_dir, args.silver_dir, args.out_dir)
