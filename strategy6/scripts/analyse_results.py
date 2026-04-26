"""
Strategy 6: Official scorer wrapper — delegates to HIPE-2026 scorer.
(Adapted from strategy5/scripts/analyse_results.py)
"""
import os
import subprocess
import sys

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCORER  = os.path.join(ROOT, "HIPE-2026-data/scripts/file_scorer_evaluation.py")
GOLD    = os.path.join(ROOT, "HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl")
PRED    = os.path.join(ROOT, "strategy6/results/predictions/preds-dev-6.5-specialist.jsonl")
SCHEMA  = os.path.join(ROOT, "HIPE-2026-data/schemas/hipe-2026-data.schema.json")
OUT_DIR = os.path.join(ROOT, "strategy6/results/analysis")


def run_scorer():
    os.makedirs(OUT_DIR, exist_ok=True)
    result_txt = os.path.join(OUT_DIR, "evaluation_results.txt")

    cmd = [
        sys.executable, SCORER,
        "--schema_file",       SCHEMA,
        "--gold_data_file",    GOLD,
        "--predictions_file",  PRED,
    ]
    print(f"\n[Strategy 6] Running scorer:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    with open(result_txt, "w") as f:
        f.write(output)

    print(output)
    print(f"Results written to {result_txt}")


if __name__ == "__main__":
    run_scorer()
