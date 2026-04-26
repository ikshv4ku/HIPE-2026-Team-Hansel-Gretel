"""
Strategy 6.5 Phase 3: Generate Arbiter Training Data.
Supports sharding to run in parallel on multiple GPUs.
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
HIST_LORA  = "strategy6/results/models/historian_lora"
GEO_LORA   = "strategy6/results/models/geographer_lora"
TRAIN_DATA = "strategy6/data/historian_sft.jsonl" 
OUT_FILE   = "strategy6/data/arbiter_sft.jsonl"

from prompts import HISTORIAN_SYSTEM, GEOGRAPHER_SYSTEM

def generate(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=960).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

def build_prompt(system, user_content):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=-1)
    parser.add_argument("--gpu",   type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    with open(TRAIN_DATA, "r") as f:
        all_samples = [json.loads(l) for l in f]
    
    if args.end == -1: args.end = len(all_samples)
    samples = all_samples[args.start:args.end]
    print(f"--- Processing shard: {args.start} to {args.end} ({len(samples)} samples) on GPU {args.gpu} ---")

    # 1. Historian
    print("--- Generating Historian Verdicts ---")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, HIST_LORA)
    model = model.merge_and_unload().to(device).eval()

    historian_results = []
    for s in tqdm(samples, desc="Historian Inference"):
        doc, pair = s["doc"], s["pair"]
        date, lang, pub = doc.get("date",""), doc.get("language",""), doc.get("media",{}).get("publication_title","")
        text = doc.get("text","")[:3000]
        p_str = ", ".join(pair.get("pers_mentions_list",[]))
        l_str = ", ".join(pair.get("loc_mentions_list",[]))
        
        user_content = f"Publication date: {date} | Language: {lang} | Source: {pub}\n\nPERSON: {p_str}\nLOCATION: {l_str}\n\nARTICLE TEXT:\n{text}"
        prompt = build_prompt(HISTORIAN_SYSTEM, user_content)
        hist_raw = generate(model, tokenizer, prompt, device)
        historian_results.append(hist_raw)

    # 2. Geographer
    print("--- Generating Geographer Verdicts ---")
    del model, base_model
    torch.cuda.empty_cache()
    
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, GEO_LORA)
    model = model.merge_and_unload().to(device).eval()

    geographer_results = []
    for s in tqdm(samples, desc="Geographer Inference"):
        doc, pair = s["doc"], s["pair"]
        date, lang, pub = doc.get("date",""), doc.get("language",""), doc.get("media",{}).get("publication_title","")
        text = doc.get("text","")[:3000]
        p_str = ", ".join(pair.get("pers_mentions_list",[]))
        l_str = ", ".join(pair.get("loc_mentions_list",[]))
        
        user_content = f"Publication date: {date} | Language: {lang} | Source: {pub}\n\nPERSON: {p_str}\nLOCATION: {l_str}\n\nARTICLE TEXT:\n{text}"
        prompt = build_prompt(GEOGRAPHER_SYSTEM, user_content)
        geo_raw = generate(model, tokenizer, prompt, device)
        geographer_results.append(geo_raw)

    # 3. Save
    shard_file = OUT_FILE.replace(".jsonl", f"_shard_{args.start}_{args.end}.jsonl")
    print(f"--- Saving shard to {shard_file} ---")
    with open(shard_file, "w") as f:
        for i, s in enumerate(samples):
            arb_sample = {
                "doc": s["doc"],
                "pair": s["pair"],
                "hist_output": historian_results[i],
                "geo_output": geographer_results[i],
                "at": s["at_verdict"],
                "isAt": s["isAt_verdict"],
                "weight": s["weight"]
            }
            f.write(json.dumps(arb_sample) + "\n")

if __name__ == "__main__":
    main()
