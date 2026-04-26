"""
Strategy 7: Merge arbiter data shards into one file.
Run after all generate_arbiter_data.py shards have completed.
"""
import json
import os
import glob

OUT_DIR   = "strategy7/data"
OUT_FILE  = os.path.join(OUT_DIR, "arbiter_sft.jsonl")
SHARD_PAT = os.path.join(OUT_DIR, "arbiter_sft_shard_*.jsonl")

def main():
    shards = sorted(glob.glob(SHARD_PAT))
    if not shards:
        print(f"No shard files found matching: {SHARD_PAT}")
        return

    total = 0
    with open(OUT_FILE, "w") as out_f:
        for shard in shards:
            with open(shard) as f:
                for line in f:
                    if line.strip():
                        out_f.write(line)
                        total += 1
            print(f"  Merged: {shard}")

    print(f"\nDone. Total samples: {total} → {OUT_FILE}")

if __name__ == "__main__":
    main()
