"""
Strategy 5: Inference with Qwen2.5-3B-Instruct + LoRA adapter.

Pipeline:
  1. Load base model + LoRA adapter (merged for fast inference)
  2. For each (person, location) pair: build prompt, greedy decode, parse JSON answer
  3. Post-processing: if at=FALSE → force isAt=FALSE
  4. Wikidata hard rule: if person's death date is >30 days before pub date → force isAt=FALSE
  5. Write fully compliant HIPE-2026 output JSONL
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_prompt

MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "strategy5/results/models/best_lora_adapter"

AT_VALID   = {"TRUE", "PROBABLE", "FALSE"}
ISAT_VALID = {"TRUE", "FALSE"}

# ── Wikidata death-date cache ─────────────────────────────────────────────────
_wikidata_cache: dict = {}
WIKIDATA_CACHE_FILE = "strategy5/results/wikidata_cache.json"


def _load_wikidata_cache():
    global _wikidata_cache
    if os.path.exists(WIKIDATA_CACHE_FILE):
        with open(WIKIDATA_CACHE_FILE, "r") as f:
            _wikidata_cache = json.load(f)


def _save_wikidata_cache():
    os.makedirs(os.path.dirname(WIKIDATA_CACHE_FILE), exist_ok=True)
    with open(WIKIDATA_CACHE_FILE, "w") as f:
        json.dump(_wikidata_cache, f)


def get_death_date(qid: str):
    """Fetch person death date from Wikidata. Returns 'YYYY-MM-DD' or None."""
    if not qid:
        return None
    if qid in _wikidata_cache:
        return _wikidata_cache[qid]
    try:
        import urllib.request, urllib.parse
        query = f'SELECT ?d WHERE {{ wd:{qid} wdt:P570 ?d . }} LIMIT 1'
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
            {"query": query, "format": "json"}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "HIPE2026Bot/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        bindings = data.get("results", {}).get("bindings", [])
        result = bindings[0]["d"]["value"][:10] if bindings else None
    except Exception:
        result = None
    _wikidata_cache[qid] = result
    return result


def person_dead_before_pub(pers_qid: str, pub_date_str: str, window_days: int = 30) -> bool:
    """Returns True if person died >window_days before the pub date → isAt must be FALSE."""
    if not pers_qid or not pub_date_str:
        return False
    death = get_death_date(pers_qid)
    if not death:
        return False
    try:
        d_death = datetime.strptime(death[:10], "%Y-%m-%d")
        d_pub   = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
        return d_death < d_pub - timedelta(days=window_days)
    except ValueError:
        return False


# ── Answer parsing ─────────────────────────────────────────────────────────────

def parse_answer(text: str):
    m = re.search(
        r'\{[^{}]*"at"\s*:\s*"([^"]+)"[^{}]*"isAt"\s*:\s*"([^"]+)"[^{}]*\}',
        text, re.IGNORECASE
    )
    if m:
        at   = m.group(1).upper()
        isat = m.group(2).upper()
        at   = at   if at   in AT_VALID   else "FALSE"
        isat = isat if isat in ISAT_VALID else "FALSE"
        return at, isat
    return "FALSE", "FALSE"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--model_name",   default=MODEL_NAME)
    parser.add_argument("--adapter_dir",  default=ADAPTER_DIR)
    parser.add_argument("--use_wikidata", action="store_true", default=True)
    parser.add_argument("--no_wikidata",  dest="use_wikidata", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Strategy 5 Inference] Device: {device}")

    print(f"Loading tokenizer from {args.adapter_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.model_name} …")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=None,
    )
    print(f"Loading LoRA adapter from {args.adapter_dir} …")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    print("Model ready.")

    if args.use_wikidata:
        _load_wikidata_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        lines = [l for l in f_in if l.strip()]
        for line in tqdm(lines, desc="Inference"):
            doc      = json.loads(line.strip())
            pub_date = doc.get("date", "")
            out_pairs = []

            for pair in doc.get("sampled_pairs", []):
                prompt = build_prompt(doc, pair)
                enc    = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=984
                ).to(device)

                with torch.no_grad():
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

                # Rule 1: transitivity
                if at_pred == "FALSE":
                    isat_pred = "FALSE"

                # Rule 2: Wikidata death-date
                if args.use_wikidata and isat_pred == "TRUE":
                    pers_qid = pair.get("pers_wikidata_QID", "")
                    if person_dead_before_pub(pers_qid, pub_date):
                        isat_pred = "FALSE"
                        print(f"  [Wikidata] {pers_qid} dead before {pub_date} → isAt=FALSE")

                pair = dict(pair)
                pair["at"]   = at_pred
                pair["isAt"] = isat_pred
                out_pairs.append(pair)

            doc = dict(doc)
            doc["sampled_pairs"] = out_pairs
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    if args.use_wikidata:
        _save_wikidata_cache()
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    run_inference()
