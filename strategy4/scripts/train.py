import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import recall_score

from data_processing import get_dataloader
from model import MTLFineTuneModel, FocalLoss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-train-str4-all.jsonl")
    parser.add_argument("--dev_file", default="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for 'at' task loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, tokenizer = get_dataloader(args.train_file, batch_size=args.batch_size, is_train=True)
    dev_loader, _ = get_dataloader(args.dev_file, batch_size=args.batch_size, is_train=False)

    # Token IDs for markers
    e1_id = tokenizer.convert_tokens_to_ids("<E1>")
    e2_id = tokenizer.convert_tokens_to_ids("<E2>")

    # Initialize model
    model = MTLFineTuneModel(model_name="Qwen/Qwen2.5-0.5B")
    model.backbone.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Optimizer
    optimizer_grouped_parameters = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": args.lr_heads},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Loss functions
    at_weights = torch.tensor([1.0, 3.0, 9.5]).to(device)
    criterion_at = nn.CrossEntropyLoss(weight=at_weights, reduction='none')
    criterion_isAt = FocalLoss(alpha=0.6, gamma=2)

    best_recall = 0
    os.makedirs("strategy4/results/models", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            at_labels = batch["at_labels"].to(device)
            isAt_labels = batch["isAt_labels"].to(device)
            weights = batch["weight"].to(device)
            
            at_logits, isAt_logits = model(input_ids, attention_mask, e1_id, e2_id)
            
            loss_at = (criterion_at(at_logits, at_labels) * weights).mean()
            loss_isAt = criterion_isAt(isAt_logits, isAt_labels)
            
            batch_loss = args.alpha * loss_at + (1 - args.alpha) * loss_isAt
            
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += batch_loss.item()
            pbar.set_postfix({"loss": batch_loss.item()})

        # Evaluation
        model.eval()
        all_pred_at = []
        all_true_at = []
        all_pred_isAt = []
        all_true_isAt = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                at_labels = batch["at_labels"].to(device)
                isAt_labels = batch["isAt_labels"].to(device)
                
                at_logits, isAt_logits = model(input_ids, attention_mask, e1_id, e2_id)
                
                all_pred_at.extend(torch.argmax(at_logits, dim=1).cpu().numpy())
                all_true_at.extend(at_labels.cpu().numpy())
                all_pred_isAt.extend(torch.argmax(isAt_logits, dim=1).cpu().numpy())
                all_true_isAt.extend(isAt_labels.cpu().numpy())

        at_recall = recall_score(all_true_at, all_pred_at, average='macro', zero_division=0)
        isAt_recall = recall_score(all_true_isAt, all_pred_isAt, average='macro', zero_division=0)
        global_recall = (at_recall + isAt_recall) / 2
        
        print(f"Epoch {epoch+1}: Global Macro Recall = {global_recall:.4f} (at: {at_recall:.4f}, isAt: {isAt_recall:.4f})")
        
        if global_recall > best_recall:
            best_recall = global_recall
            torch.save(model.state_dict(), f"strategy4/results/models/best_ft_model_all.pt")
            print("⭐ New best model saved!")

if __name__ == "__main__":
    train()
