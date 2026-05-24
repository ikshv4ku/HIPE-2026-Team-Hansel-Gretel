"""
Strategy 1.5 — Results Analysis & Visualisation

Compares Strategy 1.5 predictions vs gold labels and generates charts.
Must be run AFTER inference.py has produced preds-dev-all.jsonl.

Run from repo root:
    source HIPE-2026-data/venv/bin/activate
    python strategy1.5/analyse_results.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)

# ── Paths ─────────────────────────────────────────────────────────────────────
GOLD_FILE = "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl"
PRED_FILE = "strategy1.5/results/preds-dev-all.jsonl"
OUT_DIR   = "strategy1.5/results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)

# ── Load ──────────────────────────────────────────────────────────────────────
def load_pairs(path):
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            lang = doc.get("language", "?")
            for pair in doc.get("sampled_pairs", []):
                key = (doc["document_id"], pair["pers_entity_id"], pair["loc_entity_id"])
                docs[key] = {
                    "at":   str(pair.get("at")   or "FALSE"),
                    "isAt": str(pair.get("isAt") or "FALSE"),
                    "lang": lang,
                }
    return docs

gold  = load_pairs(GOLD_FILE)
preds = load_pairs(PRED_FILE)
common_keys = sorted(set(gold.keys()) & set(preds.keys()))
print(f"Evaluating {len(common_keys)} pairs")

y_at_true   = [gold[k]["at"]   for k in common_keys]
y_at_pred   = [preds[k]["at"]  for k in common_keys]
y_isAt_true = [gold[k]["isAt"] for k in common_keys]
y_isAt_pred = [preds[k]["isAt"] for k in common_keys]
langs       = [gold[k]["lang"] for k in common_keys]

def macro_recall(y_true, y_pred, labels):
    recalls = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))

# ── 1. Confusion Matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Strategy 1.5 (MTL + Optuna) — Confusion Matrices", fontsize=15, fontweight="bold")
for ax, yt, yp, title, labels in [
    (axes[0], y_at_true,   y_at_pred,   "`at` Relation",   ["FALSE", "PROBABLE", "TRUE"]),
    (axes[1], y_isAt_true, y_isAt_pred, "`isAt` Relation", ["FALSE", "TRUE"]),
]:
    cm = confusion_matrix(yt, yp, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved confusion matrices")

# ── 2. Per-Class Metrics ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Per-Class Precision, Recall & F1 — Strategy 1.5", fontsize=15, fontweight="bold")
for ax, yt, yp, title, labels in [
    (axes[0], y_at_true,   y_at_pred,   "`at` Relation",   ["FALSE", "PROBABLE", "TRUE"]),
    (axes[1], y_isAt_true, y_isAt_pred, "`isAt` Relation", ["FALSE", "TRUE"]),
]:
    P, R, F, _ = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    x = np.arange(len(labels)); w = 0.25
    for vals, label, color, shift in [(P, "Precision", "#3498DB", -w), (R, "Recall", "#E67E22", 0), (F, "F1", "#9B59B6", w)]:
        bars = ax.bar(x + shift, vals, w, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h+0.02, f"{h:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.1); ax.set_title(title, fontsize=13, fontweight="bold"); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_metrics.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved per-class metrics")

# ── 3. Comparison: Strategy 1 vs Strategy 1.5 ────────────────────────────────
mr_at   = macro_recall(y_at_true,   y_at_pred,   ["FALSE", "PROBABLE", "TRUE"])
mr_isAt = macro_recall(y_isAt_true, y_isAt_pred, ["FALSE", "TRUE"])
global_mr = (mr_at + mr_isAt) / 2

labels_cmp = ["Random\nBaseline", "Strategy 1\n(RF+LGB)", "Strategy 1.5\n(MTL+Optuna)"]
at_scores   = [0.33,   0.5200, mr_at]
isAt_scores = [0.50,   0.6835, mr_isAt]
global_sc   = [0.4242, 0.6018, global_mr]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels_cmp)); w = 0.28
b1 = ax.bar(x - w,   at_scores,   w, label="`at` macro-recall",     color="#3498DB", alpha=0.85)
b2 = ax.bar(x,       isAt_scores, w, label="`isAt` macro-recall",   color="#E67E22", alpha=0.85)
b3 = ax.bar(x + w,   global_sc,   w, label="Global macro-recall",   color="#27AE60", alpha=0.85)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.015, f"{h:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels_cmp, fontsize=12)
ax.set_ylim(0, 1.0); ax.set_ylabel("Macro Recall", fontsize=12)
ax.set_title("Strategy Comparison — Macro Recall", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/strategy_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved strategy comparison chart")

# ── 4. Per-Language Breakdown ─────────────────────────────────────────────────
lang_labels = sorted(set(langs))
lang_at, lang_isAt = [], []
for lg in lang_labels:
    idx = [i for i, l in enumerate(langs) if l == lg]
    lang_at.append(macro_recall([y_at_true[i] for i in idx], [y_at_pred[i] for i in idx], ["FALSE","PROBABLE","TRUE"]))
    lang_isAt.append(macro_recall([y_isAt_true[i] for i in idx], [y_isAt_pred[i] for i in idx], ["FALSE","TRUE"]))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(lang_labels))
ax.bar(x - 0.2, lang_at,   0.38, label="`at`",   color="#3498DB", alpha=0.85)
ax.bar(x + 0.2, lang_isAt, 0.38, label="`isAt`", color="#E67E22", alpha=0.85)
for i, (a, b) in enumerate(zip(lang_at, lang_isAt)):
    ax.text(i-0.2, a+0.015, f"{a:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(i+0.2, b+0.015, f"{b:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"{l.upper()}" for l in lang_labels], fontsize=12)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro Recall"); ax.legend(fontsize=11)
ax.set_title("Per-Language Macro Recall — Strategy 1.5", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_language_recall.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved per-language chart")

# ── 5. Print & Save detailed text report ─────────────────────────────────────
print("\n" + "═"*62)
print("   STRATEGY 1.5 — DETAILED CLASSIFICATION REPORT")
print("═"*62)
print("\n`at` Relation:")
print(classification_report(y_at_true, y_at_pred, labels=["FALSE","PROBABLE","TRUE"], zero_division=0, digits=4))
print(f"Macro Recall (`at`)  : {mr_at:.4f}")
print("\n`isAt` Relation:")
print(classification_report(y_isAt_true, y_isAt_pred, labels=["FALSE","TRUE"], zero_division=0, digits=4))
print(f"Macro Recall (`isAt`): {mr_isAt:.4f}")
print(f"\nGlobal Macro Recall  : {global_mr:.4f}")
print(f"vs. Strategy 1       : 0.6018  (Δ = {global_mr-0.6018:+.4f})")
print("═"*62)

with open(f"{OUT_DIR}/detailed_report.txt", "w") as f:
    f.write("STRATEGY 1.5 — DETAILED EVALUATION REPORT\n" + "="*60 + "\n\n")
    f.write("`at` Relation:\n")
    f.write(classification_report(y_at_true, y_at_pred, labels=["FALSE","PROBABLE","TRUE"], zero_division=0, digits=4))
    f.write(f"\nMacro Recall (`at`): {mr_at:.4f}\n\n")
    f.write("`isAt` Relation:\n")
    f.write(classification_report(y_isAt_true, y_isAt_pred, labels=["FALSE","TRUE"], zero_division=0, digits=4))
    f.write(f"\nMacro Recall (`isAt`): {mr_isAt:.4f}\n")
    f.write(f"\nGlobal Macro Recall: {global_mr:.4f}\n")
    f.write(f"vs. Strategy 1     : 0.6018  (Δ = {global_mr-0.6018:+.4f})\n\n")
    f.write("Per-Language Breakdown:\n")
    for lg, a, b in zip(lang_labels, lang_at, lang_isAt):
        f.write(f"  {lg.upper()}: at={a:.4f}, isAt={b:.4f}\n")
print(f"Saved detailed report to {OUT_DIR}/detailed_report.txt")
print("All charts saved! Open strategy1.5/results/analysis/ to view them.")
