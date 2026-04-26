"""
Strategy 7: Score predictions with official HIPE-2026 scorer.
"""
import subprocess
import os

VENV_PYTHON   = ".venv/bin/python3"
SCORER_SCRIPT = "HIPE-2026-data/scripts/file_scorer_evaluation.py"
SCHEMA_FILE   = "HIPE-2026-data/schemas/hipe-2026-data.schema.json"
GOLD_FILE     = "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl"
PRED_FILE     = "strategy7/results/predictions/preds-dev-7.jsonl"
OUT_RESULTS   = "strategy7/results/analysis/evaluation_results.txt"

os.makedirs("strategy7/results/analysis", exist_ok=True)

cmd = [
    VENV_PYTHON, SCORER_SCRIPT,
    "--schema_file",      SCHEMA_FILE,
    "--gold_data_file",   GOLD_FILE,
    "--predictions_file", PRED_FILE,
]

print(f"\n[Strategy 7] Running scorer:\n  {' '.join(cmd)}\n")
result = subprocess.run(cmd, capture_output=True, text=True)

output = result.stdout + result.stderr
print(output)

with open(OUT_RESULTS, "w") as f:
    f.write(output)

print(f"\nResults written to {OUT_RESULTS}")
