import json
import glob
import os
import collections

# Path to the dataset inside the cloned repository
data_dir = "./HIPE-2026-data/data/newspapers/v1.0/"
files = glob.glob(os.path.join(data_dir, "*.jsonl"))
files = [f for f in files if "randomized" not in f]

stats = {
    "total_docs": 0,
    "languages": collections.Counter(),
    "total_pairs": 0,
    "at_labels": collections.Counter(),
    "isAt_labels": collections.Counter(),
    "unique_persons": set(),
    "unique_locations": set(),
    "doc_lengths": []
}

for f in files:
    with open(f, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            
            stats["total_docs"] += 1
            stats["languages"][obj.get("language", "unknown")] += 1
            text_len = len(obj.get("text", ""))
            stats["doc_lengths"].append(text_len)
            
            for pair in obj.get("sampled_pairs", []):
                stats["total_pairs"] += 1
                if "at" in pair:
                    stats["at_labels"][str(pair["at"])] += 1
                if "isAt" in pair:
                    stats["isAt_labels"][str(pair["isAt"])] += 1
                    
                p_id = pair.get("pers_wikidata_QID") or pair.get("pers_entity_id")
                l_id = pair.get("loc_wikidata_QID") or pair.get("loc_entity_id")
                
                if p_id: stats["unique_persons"].add(p_id)
                if l_id: stats["unique_locations"].add(l_id)

print("=== HIPE-2026 Dataset EDA ===")
print(f"Total Documents: {stats['total_docs']}")
print(f"Languages: {dict(stats['languages'])}")
if stats['doc_lengths']:
    print(f"Average Document Length: {sum(stats['doc_lengths'])/len(stats['doc_lengths']):.2f} characters")
    print(f"Max Document Length: {max(stats['doc_lengths'])} characters")
    print(f"Min Document Length: {min(stats['doc_lengths'])} characters")
print(f"Total Candidate Pairs: {stats['total_pairs']}")
print(f"Unique Persons: {len(stats['unique_persons'])}")
print(f"Unique Locations: {len(stats['unique_locations'])}")
print(f"'at' Label Distribution: {dict(stats['at_labels'])}")
print(f"'isAt' Label Distribution: {dict(stats['isAt_labels'])}")
