"""
Strategy 5: LoRA SFT Training of Qwen2.5-3B-Instruct for HIPE relation extraction.

Key differences vs strategy4_5:
  - 3B instruct model (2x larger than 1.5B used before)
  - Generative classification via JSON output format
  - Full document context (no 512-token truncation)
  - Answer-only loss masking for clean SFT
  - Wikidata hard rules at inference time
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import HIPEGemmaDataset, build_prompt, AT_MAP_REV, ISAT_MAP_REV

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAVE_DIR   = "strategy5/results/models"


def parse_answer(text: str):
    """Extract at/isAt from generated JSON text."""
    m = re.search(
        r'\{[^{}]*"at"\s*:\s*"([^"]+)"[^{}]*"isAt"\s*:\s*"([^"]+)"[^{}]*\}',
        text, re.IGNORECASE
    )
    if m:
        at   = m.group(1).upper()
        isat = m.group(2).upper()
        at   = at   if at   in {"TRUE", "PROBABLE", "FALSE"} else "FALSE"
        isat = isat if isat in {"TRUE", "FALSE"}             else "FALSE"
        return at, isat
    return "FALSE", "FALSE"


@torch.no_grad()
def evaluate(model, tokenizer, dev_file: str, device, max_samples: int = 300):
    """Evaluate on dev set using greedy decode. Returns at_recall, isat_recall, global."""
    from sklearn.metrics import recall_score

    model.eval()
    pred_at, gold_at     = [], []
    pred_isat, gold_isat = [], []
    count = 0

    with open(dev_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for pair in doc.get("sampled_pairs", []):
                if max_samples and count >= max_samples:
                    break
                at_gold   = (pair.get("at")   or "FALSE").upper()
                isat_gold = (pair.get("isAt") or "FALSE").upper()
                if at_gold   not in {"TRUE", "PROBABLE", "FALSE"}: at_gold   = "FALSE"
                if isat_gold not in {"TRUE", "FALSE"}:             isat_gold = "FALSE"

                prompt = build_prompt(doc, pair)
                enc    = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=984
                ).to(device)

                out = model.generate(
                    **enc,
                    max_new_tokens=25,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                at_pred, isat_pred = parse_answer(generated)

                pred_at.append(at_pred);   gold_at.append(at_gold)
                pred_isat.append(isat_pred); gold_isat.append(isat_gold)
                count += 1

    at_recall   = recall_score(gold_at,   pred_at,   average="macro",
                               labels=["TRUE", "PROBABLE", "FALSE"], zero_division=0)
    isat_recall = recall_score(gold_isat, pred_isat, average="macro",
                               labels=["TRUE", "FALSE"], zero_division=0)
    return at_recall, isat_recall, (at_recall + isat_recall) / 2


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-train-str4-all.jsonl")
    parser.add_argument("--dev_file",   default="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--max_length", type=int,   default=1024)
    parser.add_argument("--lora_r",     type=int,   default=16)
    parser.add_argument("--lora_alpha", type=int,   default=32)
    parser.add_argument("--smoothing",  type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Strategy 5] Using device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"[Strategy 5] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("[Strategy 5] Building datasets …")
    train_ds = HIPEGemmaDataset(
        args.train_file, tokenizer, max_length=args.max_length,
        is_train=True, label_smoothing=args.smoothing,
    )
    print(f"  Train pairs: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    # ── Model + LoRA ─────────────────────────────────────────────────────────
    print(f"[Strategy 5] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=None,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Gradient checkpointing: trades compute for memory (halves peak activation RAM)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()   # required when GC is enabled with PEFT
    model.to(device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_recall = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            weights        = batch["weight"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Scale by mean sample weight (gold=5x) relative to silver baseline
            mean_weight = weights.mean() / 1.0
            scaled_loss = outputs.loss * mean_weight / args.grad_accum
            scaled_loss.backward()

            running_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} avg train loss: {avg_loss:.4f}")

        # ── Dev evaluation ────────────────────────────────────────────────────
        print("Evaluating on dev set …")
        at_r, isat_r, global_r = evaluate(model, tokenizer, args.dev_file, device)
        print(
            f"Epoch {epoch+1}: Global Macro Recall = {global_r:.4f} "
            f"(at: {at_r:.4f}, isAt: {isat_r:.4f})"
        )

        if global_r > best_recall:
            best_recall = global_r
            model.save_pretrained(os.path.join(SAVE_DIR, "best_lora_adapter"))
            tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_lora_adapter"))
            print(f"⭐ New best model saved! (recall={best_recall:.4f})")

    print(f"\nTraining complete. Best global macro recall: {best_recall:.4f}")


if __name__ == "__main__":
    train()
