"""
Detailed analysis of Strategy 1 predictions vs gold labels.
Generates confusion matrices, per-class metrics and charts.

Run from repo root:
    source HIPE-2026-data/venv/bin/activate
    pip install matplotlib seaborn scikit-learn
    python scripts/analyse_results.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

# ── Paths ─────────────────────────────────────────────────────────────────────
GOLD_FILE  = "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl"
PRED_FILE  = "results/predictions/preds-dev-all.jsonl"
OUT_DIR    = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {"TRUE": "#2ECC71", "FALSE": "#E74C3C", "PROBABLE": "#F39C12"}
sns.set_theme(style="whitegrid", font_scale=1.15)

# ── Load Data ──────────────────────────────────────────────────────────────────
def load_pairs(path, is_gold=True):
    """
    Returns a dict keyed by (doc_id, pers_entity_id, loc_entity_id)
    with values {at, isAt, lang, date}.
    """
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            lang = doc.get("language", "?")
            date = doc.get("date", "?")
            for pair in doc.get("sampled_pairs", []):
                key = (doc["document_id"], pair["pers_entity_id"], pair["loc_entity_id"])
                docs[key] = {
                    "at":   str(pair.get("at")   or "FALSE"),
                    "isAt": str(pair.get("isAt") or "FALSE"),
                    "lang": lang,
                    "date": date,
                }
    return docs

gold  = load_pairs(GOLD_FILE)
preds = load_pairs(PRED_FILE)

# Only evaluate pairs that exist in both
common_keys = sorted(set(gold.keys()) & set(preds.keys()))
print(f"Evaluating {len(common_keys)} pairs (gold={len(gold)}, preds={len(preds)})")

y_at_true   = [gold[k]["at"]   for k in common_keys]
y_at_pred   = [preds[k]["at"]  for k in common_keys]
y_isAt_true = [gold[k]["isAt"] for k in common_keys]
y_isAt_pred = [preds[k]["isAt"] for k in common_keys]
langs       = [gold[k]["lang"] for k in common_keys]

# ── Helpers ────────────────────────────────────────────────────────────────────
def macro_recall(y_true, y_pred, labels):
    recalls = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))

# ── 1. Confusion Matrices ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Strategy 1 — Confusion Matrices (Dev Set, 21 Gold Docs)", fontsize=15, fontweight="bold", y=1.01)

for ax, y_true, y_pred, title, labels in [
    (axes[0], y_at_true,   y_at_pred,   "`at` Relation",   ["FALSE", "PROBABLE", "TRUE"]),
    (axes[1], y_isAt_true, y_isAt_pred, "`isAt` Relation", ["FALSE", "TRUE"]),
]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="lightgrey", cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Ground-Truth Label", fontsize=11)

plt.tight_layout()
path_cm = os.path.join(OUT_DIR, "confusion_matrices.png")
plt.savefig(path_cm, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path_cm}")

# ── 2. Per-class Precision / Recall / F1 bar chart ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Per-Class Precision, Recall & F1 — Strategy 1", fontsize=15, fontweight="bold")

for ax, y_true, y_pred, title, labels in [
    (axes[0], y_at_true,   y_at_pred,   "`at` Relation",   ["FALSE", "PROBABLE", "TRUE"]),
    (axes[1], y_isAt_true, y_isAt_pred, "`isAt` Relation", ["FALSE", "TRUE"]),
]:
    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    x = np.arange(len(labels))
    w = 0.25
    bars_p = ax.bar(x - w,   P, w, label="Precision", color="#3498DB", alpha=0.85)
    bars_r = ax.bar(x,       R, w, label="Recall",    color="#E67E22", alpha=0.85)
    bars_f = ax.bar(x + w,   F, w, label="F1",        color="#9B59B6", alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=10)
    # value labels
    for bar_group in [bars_p, bars_r, bars_f]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

plt.tight_layout()
path_prf = os.path.join(OUT_DIR, "per_class_metrics.png")
plt.savefig(path_prf, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path_prf}")

# ── 3. Label Distribution: Gold vs Predicted ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Label Distribution: Gold vs Predicted", fontsize=15, fontweight="bold")

for ax, y_true, y_pred, title, all_labels in [
    (axes[0], y_at_true,   y_at_pred,   "`at` Relation",   ["FALSE", "PROBABLE", "TRUE"]),
    (axes[1], y_isAt_true, y_isAt_pred, "`isAt` Relation", ["FALSE", "TRUE"]),
]:
    n = len(all_labels)
    x = np.arange(n)
    gold_counts = [y_true.count(l) for l in all_labels]
    pred_counts = [y_pred.count(l) for l in all_labels]
    ax.bar(x - 0.2, gold_counts, 0.4, label="Gold (True)", color="#2C3E50", alpha=0.8)
    ax.bar(x + 0.2, pred_counts, 0.4, label="Predicted",   color="#E74C3C", alpha=0.8)
    for i, (g, p) in enumerate(zip(gold_counts, pred_counts)):
        ax.text(i - 0.2, g + 0.5, str(g), ha="center", fontsize=11, fontweight="bold")
        ax.text(i + 0.2, p + 0.5, str(p), ha="center", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=12)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(gold_counts), max(pred_counts)) * 1.2)

plt.tight_layout()
path_dist = os.path.join(OUT_DIR, "label_distribution.png")
plt.savefig(path_dist, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path_dist}")

# ── 4. Macro-Recall Progression Chart ─────────────────────────────────────────
runs   = ["Random\nBaseline", "Gold-Only\n(no silver)", "Gold + Silver\nWeighted (Ours)"]
scores = [0.4242, 0.4459, 0.6018]
colors = ["#BDC3C7", "#F39C12", "#27AE60"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(runs, scores, color=colors, width=0.45, edgecolor="white", linewidth=1.5)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{score:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.axhline(0.5, color="#E74C3C", linestyle="--", linewidth=1.4, label="0.50 threshold")
ax.set_ylim(0, 0.75)
ax.set_ylabel("Global Macro Recall", fontsize=12)
ax.set_title("Strategy 1 — Macro Recall Progression", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.tick_params(axis='x', labelsize=11)
plt.tight_layout()
path_prog = os.path.join(OUT_DIR, "macro_recall_progression.png")
plt.savefig(path_prog, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path_prog}")

# ── 5. Per-Language Breakdown ──────────────────────────────────────────────────
lang_labels = sorted(set(langs))
lang_at_recall, lang_isAt_recall = [], []
for lg in lang_labels:
    idx = [i for i, l in enumerate(langs) if l == lg]
    at_t  = [y_at_true[i]   for i in idx]
    at_p  = [y_at_pred[i]   for i in idx]
    i_t   = [y_isAt_true[i] for i in idx]
    i_p   = [y_isAt_pred[i] for i in idx]
    lang_at_recall.append(macro_recall(at_t, at_p, ["FALSE", "PROBABLE", "TRUE"]))
    lang_isAt_recall.append(macro_recall(i_t, i_p, ["FALSE", "TRUE"]))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(lang_labels))
ax.bar(x - 0.2, lang_at_recall,   0.38, label="`at` macro-recall",   color="#3498DB", alpha=0.85)
ax.bar(x + 0.2, lang_isAt_recall, 0.38, label="`isAt` macro-recall", color="#E67E22", alpha=0.85)
for i, (a, b) in enumerate(zip(lang_at_recall, lang_isAt_recall)):
    ax.text(i - 0.2, a + 0.015, f"{a:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(i + 0.2, b + 0.015, f"{b:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Lang: {l.upper()}" for l in lang_labels], fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Macro Recall", fontsize=12)
ax.set_title("Per-Language Macro Recall — Strategy 1 (Dev Set)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
path_lang = os.path.join(OUT_DIR, "per_language_recall.png")
plt.savefig(path_lang, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path_lang}")

# ── 6. Print Full Text Report ──────────────────────────────────────────────────
print("\n" + "═"*62)
print("   DETAILED CLASSIFICATION REPORT — STRATEGY 1 (Dev Set)")
print("═"*62)

print("\n┌─── `at` Relation ──────────────────────────────────────┐")
print(classification_report(y_at_true, y_at_pred,
      labels=["FALSE","PROBABLE","TRUE"], zero_division=0,
      digits=4))
print(f"  Macro Recall (`at`): {macro_recall(y_at_true, y_at_pred, ['FALSE','PROBABLE','TRUE']):.4f}")

print("\n┌─── `isAt` Relation ────────────────────────────────────┐")
print(classification_report(y_isAt_true, y_isAt_pred,
      labels=["FALSE","TRUE"], zero_division=0, digits=4))
print(f"  Macro Recall (`isAt`): {macro_recall(y_isAt_true, y_isAt_pred, ['FALSE','TRUE']):.4f}")

total_correct = sum(1 for t, p in zip(y_at_true + y_isAt_true, y_at_pred + y_isAt_pred) if t == p)
total = len(y_at_true) + len(y_isAt_true)
global_mr = (macro_recall(y_at_true, y_at_pred, ["FALSE","PROBABLE","TRUE"]) +
             macro_recall(y_isAt_true, y_isAt_pred, ["FALSE","TRUE"])) / 2
print(f"\n  Global Macro Recall : {global_mr:.4f}")
print(f"  Total Correct Pairs : {total_correct}/{total}")
print("═"*62 + "\n")

# Save text report
with open(os.path.join(OUT_DIR, "detailed_report.txt"), "w") as f:
    f.write(f"STRATEGY 1 — DETAILED EVALUATION REPORT\n{'='*60}\n\n")
    f.write(f"`at` Classification Report:\n")
    f.write(classification_report(y_at_true, y_at_pred,
            labels=["FALSE","PROBABLE","TRUE"], zero_division=0, digits=4))
    f.write(f"\nMacro Recall (`at`): {macro_recall(y_at_true, y_at_pred, ['FALSE','PROBABLE','TRUE']):.4f}\n\n")
    f.write(f"`isAt` Classification Report:\n")
    f.write(classification_report(y_isAt_true, y_isAt_pred,
            labels=["FALSE","TRUE"], zero_division=0, digits=4))
    f.write(f"\nMacro Recall (`isAt`): {macro_recall(y_isAt_true, y_isAt_pred, ['FALSE','TRUE']):.4f}\n")
    f.write(f"\nGlobal Macro Recall: {global_mr:.4f}\n")
    f.write(f"Total Correct Pairs: {total_correct}/{total}\n\n")
    f.write("Per-Language Breakdown:\n")
    for lg, a, b in zip(lang_labels, lang_at_recall, lang_isAt_recall):
        f.write(f"  {lg.upper()}: at={a:.4f}, isAt={b:.4f}\n")
print("Saved: results/analysis/detailed_report.txt")
print("All done! Open results/analysis/ to view the charts.")
