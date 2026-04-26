"""
Strategy 8: Score predictions with the official HIPE-2026 scorer.

Run after run_inference.py has produced a prediction file.
"""
import subprocess
import os
import sys

# Paths
VENV_PYTHON   = ".venv/bin/python3"
SCORER_SCRIPT = "HIPE-2026-data/scripts/file_scorer_evaluation.py"
SCHEMA_FILE   = "HIPE-2026-data/schemas/hipe-2026-data.schema.json"
GOLD_FILE     = "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl"

# Accept prediction file as argument, default to strategy8
PRED_FILE   = sys.argv[1] if len(sys.argv) > 1 else "strategy8/results/predictions/preds-dev-8.jsonl"
OUT_RESULTS = PRED_FILE.replace(".jsonl", "_evaluation.txt").replace("predictions/", "analysis/")

os.makedirs(os.path.dirname(OUT_RESULTS), exist_ok=True)

cmd = [
    VENV_PYTHON, SCORER_SCRIPT,
    "--schema_file",      SCHEMA_FILE,
    "--gold_data_file",   GOLD_FILE,
    "--predictions_file", PRED_FILE,
]

print(f"\n[S8 Scorer] Running:\n  {' '.join(cmd)}\n")
result = subprocess.run(cmd, capture_output=True, text=True)
output = result.stdout + result.stderr

print(output)
with open(OUT_RESULTS, "w") as f:
    f.write(output)

print(f"\n[S8 Scorer] Results saved → {OUT_RESULTS}")
