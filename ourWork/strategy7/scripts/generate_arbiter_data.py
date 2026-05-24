"""
Strategy 7 Phase 3: Generate Arbiter Training Data.
Upgraded to Qwen2.5-7B. Supports sharding for parallel GPU execution.
"""

import json
import os
import sys
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from prompts import HISTORIAN_SYSTEM, GEOGRAPHER_SYSTEM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HIST_LORA  = "strategy7/results/models/historian_lora"
GEO_LORA   = "strategy7/results/models/geographer_lora"
TRAIN_DATA = "strategy7/data/historian_sft.jsonl"
OUT_DIR    = "strategy7/data"


def generate(model, tokenizer, prompt, device, max_new_tokens=120):
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def build_prompt(system, user_content):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0,  help="Start index of shard")
    parser.add_argument("--end",   type=int, default=-1, help="End index of shard (-1 = all)")
    parser.add_argument("--gpu",   type=int, default=0,  help="GPU index (informational)")
    args = parser.parse_args()

    # GPU is pinned via CUDA_VISIBLE_DEVICES externally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S7 ArbiterGen] Running on {device} (GPU {args.gpu})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(TRAIN_DATA, "r") as f:
        all_samples = [json.loads(l) for l in f if l.strip()]

    if args.end == -1:
        args.end = len(all_samples)
    samples = all_samples[args.start:args.end]
    print(f"[S7 ArbiterGen] Processing shard [{args.start}:{args.end}] — {len(samples)} samples")

    # ── Step 1: Historian inference ────────────────────────────────────────────
    print("[S7 ArbiterGen] Loading Historian adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model      = PeftModel.from_pretrained(base_model, HIST_LORA)
    model      = model.merge_and_unload().to(device).eval()

    historian_results = []
    for s in tqdm(samples, desc="Historian Inference"):
        doc, pair = s["doc"], s["pair"]
        date = doc.get("date", "")
        lang = doc.get("language", "")
        pub  = doc.get("media", {}).get("publication_title", "")
        text = doc.get("text", "")[:3000]
        p_str = ", ".join(pair.get("pers_mentions_list", []))
        l_str = ", ".join(pair.get("loc_mentions_list",  []))

        user_content = (
            f"Publication date: {date} | Language: {lang} | Source: {pub}\n\n"
            f"PERSON: {p_str}\nLOCATION: {l_str}\n\nARTICLE TEXT:\n{text}"
        )
        hist_raw = generate(model, tokenizer, build_prompt(HISTORIAN_SYSTEM, user_content), device)
        historian_results.append(hist_raw)

    # ── Step 2: Geographer inference ───────────────────────────────────────────
    print("[S7 ArbiterGen] Swapping to Geographer adapter...")
    del model, base_model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model      = PeftModel.from_pretrained(base_model, GEO_LORA)
    model      = model.merge_and_unload().to(device).eval()

    geographer_results = []
    for s in tqdm(samples, desc="Geographer Inference"):
        doc, pair = s["doc"], s["pair"]
        date = doc.get("date", "")
        lang = doc.get("language", "")
        pub  = doc.get("media", {}).get("publication_title", "")
        text = doc.get("text", "")[:3000]
        p_str = ", ".join(pair.get("pers_mentions_list", []))
        l_str = ", ".join(pair.get("loc_mentions_list",  []))

        user_content = (
            f"Publication date: {date} | Language: {lang} | Source: {pub}\n\n"
            f"PERSON: {p_str}\nLOCATION: {l_str}\n\nARTICLE TEXT:\n{text}"
        )
        geo_raw = generate(model, tokenizer, build_prompt(GEOGRAPHER_SYSTEM, user_content), device)
        geographer_results.append(geo_raw)

    # ── Step 3: Save shard ─────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_file = os.path.join(
        OUT_DIR, f"arbiter_sft_shard_{args.start}_{args.end}.jsonl"
    )
    print(f"[S7 ArbiterGen] Saving shard → {shard_file}")
    with open(shard_file, "w") as f:
        for i, s in enumerate(samples):
            arb_sample = {
                "doc":         s["doc"],
                "pair":        s["pair"],
                "hist_output": historian_results[i],
                "geo_output":  geographer_results[i],
                "at":          s["at_verdict"],
                "isAt":        s["isAt_verdict"],
                "weight":      s["weight"],
            }
            f.write(json.dumps(arb_sample) + "\n")

    print(f"[S7 ArbiterGen] Done. Shard saved: {shard_file}")


if __name__ == "__main__":
    main()
