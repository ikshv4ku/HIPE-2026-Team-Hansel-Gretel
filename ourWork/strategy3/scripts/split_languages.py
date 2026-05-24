import json
import os

def split_file(input_path, output_dir, prefix):
    out_files = {}
    total_docs = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            lang = doc.get("language", "unknown").lower()
            
            if lang not in out_files:
                out_path = os.path.join(output_dir, f"{prefix}-{lang}.jsonl")
                out_files[lang] = open(out_path, 'w', encoding='utf-8')
            
            out_files[lang].write(line)
            total_docs += 1
            
    for lang, f in out_files.items():
        f.close()
        print(f"Wrote docs to {prefix}-{lang}.jsonl")
        
    print(f"Total documents processed for {prefix}: {total_docs}")

if __name__ == "__main__":
    base_dir = "/home/pradyuman/HIPE-2026-Team-Hansel-Gretel/HIPE-2026-data/data/newspapers/v1.0"
    train_file = os.path.join(base_dir, "splits", "HIPE-2026-v1.0-impresso-train-all.jsonl")
    dev_file = os.path.join(base_dir, "splits", "HIPE-2026-v1.0-impresso-dev-all.jsonl")
    
    output_dir = os.path.join(base_dir, "splits", "monolingual")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Splitting Train file...")
    split_file(train_file, output_dir, "train")
    
    print("Splitting Dev file...")
    split_file(dev_file, output_dir, "dev")
