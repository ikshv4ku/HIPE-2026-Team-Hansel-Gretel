"""
Strategy 7: Data Preparation.
Adapted from strategy6/scripts/prepare_specialist_data.py.
Outputs to strategy7/data/.
"""

import json
import os
import glob
import random

GOLD_TRAIN = "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-train-str4-all.jsonl"
SILVER_DIR = "HIPE-2026-data/data/sandbox"
OUT_DIR    = "strategy7/data"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Reasoning Synthesis Templates ───────────────────────────────────────────

def get_historian_reasoning(at, isat):
    if at == "TRUE":
        if isat == "TRUE":
            return "The article contains explicit contemporary evidence placing the person at this location close to the publication date."
        else:
            return "Historical context confirms the person's past presence or career association with this location, though they are not necessarily there now."
    elif at == "PROBABLE":
        return "The person's institutional role or known biographical timeline strongly implies an association with this location during this period."
    else:
        return "There is no biographical evidence or temporal mention in the text connecting this person's life to this specific location."

def get_geographer_reasoning(at, isat):
    if at == "TRUE":
        if isat == "TRUE":
            return "Regional reporting and spatial context in the text clearly situate the person within this geographic area."
        else:
            return "The text establishes a geographic link between the person and this region, even if the presence is not current."
    elif at == "PROBABLE":
        return "The geographic focus of the source and the regional language markers suggest a likely spatial connection for this person."
    else:
        return "The spatial context and regional framing of the article provide no evidence of the person's presence in this location."

# ── Dataset Builder ──────────────────────────────────────────────────────────

def process_file(path, weight, hist_list, geo_list):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            doc = json.loads(line)

            base_doc = {k: v for k, v in doc.items() if k != "sampled_pairs"}

            for pair in doc.get("sampled_pairs", []):
                at   = (pair.get("at")   or "FALSE").upper()
                isat = (pair.get("isAt") or "FALSE").upper()
                if at   not in {"TRUE", "PROBABLE", "FALSE"}: at   = "FALSE"
                if isat not in {"TRUE", "FALSE"}:             isat = "FALSE"

                hist_list.append({
                    "doc":          base_doc,
                    "pair":         pair,
                    "at_verdict":   at,
                    "isAt_verdict": isat,
                    "reasoning":    get_historian_reasoning(at, isat),
                    "weight":       weight,
                })

                geo_list.append({
                    "doc":          base_doc,
                    "pair":         pair,
                    "at_verdict":   at,
                    "isAt_verdict": isat,
                    "reasoning":    get_geographer_reasoning(at, isat),
                    "weight":       weight,
                })
                count += 1
    return count

def main():
    hist_samples, geo_samples = [], []

    print(f"Loading Gold: {GOLD_TRAIN}")
    g_count = process_file(GOLD_TRAIN, 5.0, hist_samples, geo_samples)
    print(f"  Gold pairs: {g_count}")

    silver_files = glob.glob(f"{SILVER_DIR}/*-train.jsonl")
    s_total = 0
    for sf in sorted(silver_files):
        print(f"Loading Silver: {sf}")
        s_total += process_file(sf, 1.0, hist_samples, geo_samples)
    print(f"  Silver pairs total: {s_total}")

    random.seed(42)
    random.shuffle(hist_samples)
    random.shuffle(geo_samples)

    hist_path = os.path.join(OUT_DIR, "historian_sft.jsonl")
    geo_path  = os.path.join(OUT_DIR, "geographer_sft.jsonl")

    with open(hist_path, "w") as f:
        for s in hist_samples:
            f.write(json.dumps(s) + "\n")

    with open(geo_path, "w") as f:
        for s in geo_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nDone! Saved {len(hist_samples)} samples to {hist_path} and {geo_path}")

if __name__ == "__main__":
    main()
