"""
Strategy 6.5: Specialist SFT Training (Historian, Geographer, Arbiter).
Specializes a Qwen2.5-3B-Instruct model for a specific agent role.
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from prompts import HISTORIAN_SYSTEM, GEOGRAPHER_SYSTEM, ARBITER_SYSTEM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ── Dataset ──────────────────────────────────────────────────────────────

class SpecialistDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, system_prompt, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.samples = []
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                s = json.loads(line)
                
                # Input build
                doc, pair = s["doc"], s["pair"]
                date = doc.get("date", "unknown")
                lang = doc.get("language", "unknown")
                pub  = doc.get("media", {}).get("publication_title", "unknown")
                text = doc.get("text", "")[:3000]
                person_str   = ", ".join(pair.get("pers_mentions_list", []))
                location_str = ", ".join(pair.get("loc_mentions_list", []))

                if "hist_output" in s: # Arbiter specific input
                    user_content = (
                        f"HISTORIAN VERDICT:\n{s['hist_output']}\n\n"
                        f"GEOGRAPHER VERDICT:\n{s['geo_output']}\n\n"
                        f"ARTICLE CONTEXT:\n"
                        f"Publication date: {date} | Language: {lang} | Source: {pub}\n"
                        f"PERSON: {person_str}\n"
                        f"LOCATION: {location_str}\n\n"
                        f"ARTICLE TEXT:\n{text}"
                    )
                else: # Regular specialist input
                    user_content = (
                        f"Publication date: {date} | Language: {lang} | Source: {pub}\n\n"
                        f"PERSON: {person_str}\n"
                        f"LOCATION: {location_str}\n\n"
                        f"ARTICLE TEXT:\n{text}"
                    )
                
                # Update answer checking logic
                if "at_verdict" in s: # Specialist format (from prepare_specialist_data.py)
                    answer = json.dumps({
                        "at_verdict": s["at_verdict"],
                        "isAt_verdict": s["isAt_verdict"],
                        "reasoning": s["reasoning"]
                    })
                else: # Arbiter format (merged shards)
                    answer = json.dumps({
                        "at": s["at"],
                        "isAt": s["isAt"]
                    })                
                prompt = (
                    "<|im_start|>system\n"
                    f"{self.system_prompt}"
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{user_content}"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                
                answer += "<|im_end|>"
                
                self.samples.append({
                    "prompt": prompt,
                    "answer": answer,
                    "weight": s.get("weight", 1.0)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        p_enc = self.tokenizer(s["prompt"], add_special_tokens=False, truncation=True, max_length=self.max_length-150)
        a_enc = self.tokenizer(s["answer"], add_special_tokens=False, truncation=True, max_length=150)
        
        input_ids = p_enc["input_ids"] + a_enc["input_ids"]
        labels    = [-100] * len(p_enc["input_ids"]) + a_enc["input_ids"]
        
        # Padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            input_ids += [pad_id] * pad_len
            labels    += [-100]   * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            labels    = labels[:self.max_length]
            
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1 if x != -100 else 0 for x in labels]), # simple mask
            "labels": torch.tensor(labels),
            "weight": torch.tensor(s["weight"], dtype=torch.float)
        }

# ── Training Loop ─────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role",       choices=["historian", "geographer", "arbiter"], required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--epochs",     type=int,   default=3) # Shorter as we have more data
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4) # Conservative LR for specialists
    args = parser.parse_args()

    prompts = {
        "historian":  HISTORIAN_SYSTEM,
        "geographer": GEOGRAPHER_SYSTEM,
        "arbiter":    ARBITER_SYSTEM
    }
    system_prompt = prompts[args.role]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S6.5] Training {args.role} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset = SpecialistDataset(args.train_file, tokenizer, system_prompt)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Check for existing checkpoint
    checkpoint_path = os.path.join(args.out_dir, "checkpoint_latest")
    if os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")) or os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
        print(f"[S6.5] Resuming from checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0.1*total_steps, total_steps)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
        
        for step, batch in pbar:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            w    = batch["weight"].to(device)

            out = model(input_ids=ids, labels=labs)
            
            # Weighted loss
            loss = out.loss * w.mean() / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            running_loss += out.loss.item()
            pbar.set_postfix({"loss": f"{out.loss.item():.4f}"})

        # Save checkpoint after each epoch
        epoch_dir = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}")
        latest_dir = os.path.join(args.out_dir, "checkpoint_latest")
        model.save_pretrained(epoch_dir)
        model.save_pretrained(latest_dir)
        tokenizer.save_pretrained(latest_dir)
        print(f"Epoch {epoch+1} saved to {epoch_dir}")

    # Final save
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Final model saved to {args.out_dir}")

if __name__ == "__main__":
    train()
