"""
Strategy 5: Analyse dev-set predictions using the official HIPE-2026 scorer.
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

GOLD_FILE = os.path.join(ROOT, "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl")
PRED_FILE = os.path.join(ROOT, "strategy5/results/predictions/preds-dev-all.jsonl")
SCORER    = os.path.join(ROOT, "HIPE-2026-data/scripts/file_scorer_evaluation.py")
OUT_DIR   = os.path.join(ROOT, "strategy5/results/analysis")


def run_scorer(gold: str, pred: str) -> None:
    schema    = os.path.join(ROOT, "HIPE-2026-data/schemas/hipe-2026-data.schema.json")
    result_txt = os.path.join(OUT_DIR, "evaluation_results.txt")

    cmd = [
        sys.executable, SCORER,
        "--schema_file", schema,
        "--gold_data_file", gold,
        "--predictions_file", pred
    ]
    print(f"\n[Strategy 5] Running scorer:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    print(output)
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Results written to {result_txt}")


if __name__ == "__main__":
    if not os.path.exists(PRED_FILE):
        print(f"[Error] Prediction file not found: {PRED_FILE}")
        print("Run inference.py first.")
        sys.exit(1)
    run_scorer(GOLD_FILE, PRED_FILE)
