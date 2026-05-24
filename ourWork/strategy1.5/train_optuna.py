"""
Strategy 1.5 — Optuna Hyperparameter Search + Final Retrain

Loads pre-computed embeddings from Strategy 1, builds MTL model, and runs
Optuna to find the best hyperparameters. Then retrains the best config on
the full train set and saves the model.

Run from repo root:
    source HIPE-2026-data/venv/bin/activate
    pip install optuna
    python strategy1.5/train_optuna.py
"""

import pickle
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
import joblib

# Make model importable whether run from repo root or from strategy1.5/
sys.path.insert(0, os.path.dirname(__file__))
from model import MTLRelationClassifier, FocalLoss

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PKL = "results/embeddings/train_all_features.pkl"
DEV_PKL   = "results/embeddings/dev_all_features.pkl"
MODEL_OUT = "strategy1.5/results/best_mtl_model.pt"
STUDY_OUT = "strategy1.5/results/optuna_study.pkl"
os.makedirs("strategy1.5/results", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load embeddings from Strategy 1 ───────────────────────────────────────────
def load_split(path, is_train=True):
    with open(path, "rb") as f:
        data = pickle.load(f)
    X = torch.tensor(data["features"], dtype=torch.float32)
    if is_train:
        y_at   = torch.tensor(data["at_labels"],   dtype=torch.long)
        y_isAt = torch.tensor(data["isAt_labels"], dtype=torch.long)
        w      = torch.tensor(data["weights"],     dtype=torch.float32)
        return X, y_at, y_isAt, w
    return X, data["metadata"]

print("Loading embeddings …")
X_tr, y_at_tr, y_isAt_tr, w_tr = load_split(TRAIN_PKL, is_train=True)
X_dev, dev_meta = load_split(DEV_PKL, is_train=False)

# ── Load gold dev labels from predictions file (gold json) ────────────────────
import json

DEV_GOLD_FILE = "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl"

def load_gold_dev_labels(gold_jsonl, meta):
    gold = {}
    with open(gold_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            for pair in doc.get("sampled_pairs", []):
                key = (doc["document_id"], pair["pers_entity_id"], pair["loc_entity_id"])
                gold[key] = {
                    "at":   str(pair.get("at") or "FALSE"),
                    "isAt": str(pair.get("isAt") or "FALSE"),
                }
    at_map   = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
    isAt_map = {"FALSE": 0, "TRUE": 1}
    y_at_dev, y_isAt_dev = [], []
    for m in meta:
        k = (m["doc_id"], m["pers_id"], m["loc_id"])
        g = gold.get(k, {"at": "FALSE", "isAt": "FALSE"})
        y_at_dev.append(at_map[g["at"]])
        y_isAt_dev.append(isAt_map[g["isAt"]])
    return torch.tensor(y_at_dev, dtype=torch.long), torch.tensor(y_isAt_dev, dtype=torch.long)

y_at_dev, y_isAt_dev = load_gold_dev_labels(DEV_GOLD_FILE, dev_meta)
print(f"Train: {len(X_tr)} | Dev: {len(X_dev)}")

# ── Macro Recall helper ────────────────────────────────────────────────────────
def macro_recall(y_true, y_pred, n_classes):
    recalls = []
    for c in range(n_classes):
        tp = ((y_true == c) & (y_pred == c)).sum().item()
        fn = ((y_true == c) & (y_pred != c)).sum().item()
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))

# ── Training loop ──────────────────────────────────────────────────────────────
def train_and_eval(params, n_trials_mode=True):
    hidden_dim          = params["hidden_dim"]
    dropout             = params["dropout"]
    lr                  = params["lr"]
    alpha               = params["alpha"]          # weight for at loss
    at_weight_true      = params["at_weight_true"]
    at_weight_probable  = params["at_weight_probable"]
    isAt_weight_true    = params["isAt_weight_true"]
    epochs              = params["epochs"]
    batch_size          = params["batch_size"]

    model = MTLRelationClassifier(input_dim=1536, hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights
    at_class_weights   = torch.tensor([1.0, at_weight_probable, at_weight_true], dtype=torch.float32).to(DEVICE)
    isAt_class_weights = torch.tensor([1.0, isAt_weight_true], dtype=torch.float32).to(DEVICE)

    loss_at   = nn.CrossEntropyLoss(weight=at_class_weights, reduction="none")
    loss_isAt = FocalLoss(gamma=2.0, weight=isAt_class_weights)

    # Build dataloader
    ds = TensorDataset(X_tr.to(DEVICE), y_at_tr.to(DEVICE), y_isAt_tr.to(DEVICE), w_tr.to(DEVICE))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_score = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for x_b, y_at_b, y_isAt_b, w_b in dl:
            optimizer.zero_grad()
            at_logits, isAt_logits = model(x_b)
            # Weighted cross entropy for `at` (sample weight applied manually)
            l_at   = (loss_at(at_logits, y_at_b) * w_b).mean()
            l_isAt = loss_isAt(isAt_logits, y_isAt_b)
            loss   = alpha * l_at + (1.0 - alpha) * l_isAt
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Eval on dev every 5 epochs
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                at_l, isAt_l = model(X_dev.to(DEVICE))
                pred_at   = at_l.argmax(dim=1).cpu()
                pred_isAt = isAt_l.argmax(dim=1).cpu()

            # Apply transitivity rule
            pred_isAt[pred_at == 0] = 0

            mr_at   = macro_recall(y_at_dev,   pred_at,   3)
            mr_isAt = macro_recall(y_isAt_dev, pred_isAt, 2)
            score   = (mr_at + mr_isAt) / 2

            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_score, best_state, params

# ── Optuna Objective ───────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "hidden_dim":         trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "dropout":            trial.suggest_float("dropout", 0.1, 0.5),
        "lr":                 trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "alpha":              trial.suggest_float("alpha", 0.2, 0.8),
        "at_weight_true":     trial.suggest_float("at_weight_true", 1.0, 10.0),
        "at_weight_probable": trial.suggest_float("at_weight_probable", 1.0, 5.0),
        "isAt_weight_true":   trial.suggest_float("isAt_weight_true", 1.0, 10.0),
        "epochs":             trial.suggest_int("epochs", 20, 80),
        "batch_size":         trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    score, _, _ = train_and_eval(params)
    return score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()

    print(f"\nStarting Optuna search with {args.n_trials} trials …")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nBest trial:  score={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save study for inspection
    joblib.dump(study, STUDY_OUT)
    print(f"Study saved to {STUDY_OUT}")

    # Final retrain with best hyperparameters on full train data
    print("\nRetraining best config …")
    _, best_state, best_params = train_and_eval(study.best_params, n_trials_mode=False)

    # Save model
    checkpoint = {
        "model_state": best_state,
        "params":      best_params,
        "best_score":  study.best_value,
    }
    torch.save(checkpoint, MODEL_OUT)
    print(f"Best model saved to {MODEL_OUT}")
    print(f"\n✅ Done! Best Global Macro Recall on Dev: {study.best_value:.4f}")
