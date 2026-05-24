import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import argparse

def analyze(gold_file, pred_file):
    with open(gold_file, 'r') as f:
        gold_docs = [json.loads(line) for line in f]
    with open(pred_file, 'r') as f:
        pred_docs = [json.loads(line) for line in f]

    gold_at, pred_at = [], []
    gold_isAt, pred_isAt = [], []

    for gd, pd in zip(gold_docs, pred_docs):
        for g_pair, p_pair in zip(gd['sampled_pairs'], pd['sampled_pairs']):
            gold_at.append(g_pair.get('at', 'FALSE'))
            pred_at.append(p_pair.get('at', 'FALSE'))
            gold_isAt.append(g_pair.get('isAt', 'FALSE'))
            pred_isAt.append(p_pair.get('isAt', 'FALSE'))

    os.makedirs("strategy2.5/results/analysis", exist_ok=True)
    
    # 1. Classification Reports
    report_at = classification_report(gold_at, pred_at, output_dict=True)
    report_isAt = classification_report(gold_isAt, pred_isAt, output_dict=True)
    
    with open("strategy2.5/results/analysis/detailed_report.txt", "w") as f:
        f.write("=== STRATEGY 2.5: PEFT MTL REPORT ===\n\n")
        f.write("AT RELATION:\n")
        f.write(classification_report(gold_at, pred_at))
        f.write("\nIS_AT RELATION:\n")
        f.write(classification_report(gold_isAt, pred_isAt))
        
    # 2. Confusion Matrices
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    labels_at = ["FALSE", "PROBABLE", "TRUE"]
    cm_at = confusion_matrix(gold_at, pred_at, labels=labels_at)
    sns.heatmap(cm_at, annot=True, fmt='d', xticklabels=labels_at, yticklabels=labels_at, ax=ax[0], cmap='Blues')
    ax[0].set_title("at Relation Confusion Matrix")
    
    labels_isAt = ["FALSE", "TRUE"]
    cm_isAt = confusion_matrix(gold_isAt, pred_isAt, labels=labels_isAt)
    sns.heatmap(cm_isAt, annot=True, fmt='d', xticklabels=labels_isAt, yticklabels=labels_isAt, ax=ax[1], cmap='Greens')
    ax[1].set_title("isAt Relation Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig("strategy2.5/results/analysis/confusion_matrices.png")
    
    # 3. Bar Chart Comparison
    # Strategy 1.5 baseline results
    s15_macro = 0.6249
    s2_macro = (report_at['macro avg']['recall'] + report_isAt['macro avg']['recall']) / 2
    
    plt.figure(figsize=(10, 6))
    strategies = ['Strategy 1.5 (Frozen)', 'Strategy 2 (End-to-End)', 'Strategy 2.5 (PEFT/LoRA)']
    recalls = [0.6249, 0.4167, s2_macro]
    sns.barplot(x=strategies, y=recalls, hue=strategies, palette='viridis', legend=False)
    plt.ylim(0.3, 0.8)
    plt.ylabel("Global Macro Recall")
    plt.title("Evolution of Model Performance")
    for i, v in enumerate(recalls):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig("strategy2.5/results/analysis/strategy_comparison.png")

    print(f"Strategy 2.5 Global Macro Recall: {s2_macro:.4f} (vs Strategy 1.5: 0.6249)")

if __name__ == "__main__":
    analyze("HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl", 
            "strategy2.5/results/predictions/preds-dev-all.jsonl")
