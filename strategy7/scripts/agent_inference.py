"""
Strategy 7: Multi-Agent Inference Pipeline.

Upgrades vs Strategy 6:
  - Base model: Qwen2.5-7B-Instruct
  - device_map="auto" for multi-GPU inference (shards 7B across available GPUs)
  - Adapters point to strategy7/results/models/
  - 3-shot prompts from strategy7/scripts/prompts.py
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
from prompts import HISTORIAN_SYSTEM, GEOGRAPHER_SYSTEM, ARBITER_SYSTEM

MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_HIST    = "strategy7/results/models/historian_lora"
DEFAULT_GEO     = "strategy7/results/models/geographer_lora"
DEFAULT_ARB     = "strategy7/results/models/arbiter_lora"

AT_VALID   = {"TRUE", "PROBABLE", "FALSE"}
ISAT_VALID = {"TRUE", "FALSE"}

# ── Wikidata death-date cache ─────────────────────────────────────────────────
_wikidata_cache: dict = {}
WIKIDATA_CACHE_FILE = "strategy7/results/wikidata_cache.json"


def _load_wikidata_cache():
    global _wikidata_cache
    if os.path.exists(WIKIDATA_CACHE_FILE):
        with open(WIKIDATA_CACHE_FILE) as f:
            _wikidata_cache = json.load(f)


def _save_wikidata_cache():
    os.makedirs(os.path.dirname(WIKIDATA_CACHE_FILE), exist_ok=True)
    with open(WIKIDATA_CACHE_FILE, "w") as f:
        json.dump(_wikidata_cache, f)


def get_death_date(qid: str):
    if not qid:
        return None
    if qid in _wikidata_cache:
        return _wikidata_cache[qid]
    try:
        import urllib.request, urllib.parse
        query = f"SELECT ?d WHERE {{ wd:{qid} wdt:P570 ?d . }} LIMIT 1"
        url   = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
            {"query": query, "format": "json"}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "HIPE2026Bot/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        bindings = data.get("results", {}).get("bindings", [])
        result   = bindings[0]["d"]["value"][:10] if bindings else None
    except Exception:
        result = None
    _wikidata_cache[qid] = result
    return result


def person_dead_before_pub(qid: str, pub_date_str: str, window_days: int = 30) -> bool:
    if not qid or not pub_date_str:
        return False
    death = get_death_date(qid)
    if not death:
        return False
    try:
        d_death = datetime.strptime(death[:10], "%Y-%m-%d")
        d_pub   = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
        return d_death < d_pub - timedelta(days=window_days)
    except ValueError:
        return False


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_agent_prompt(system: str, user_content: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_document_user_content(doc: dict, pair: dict, max_text_chars: int = 3000) -> str:
    date = doc.get("date", "unknown")
    lang = doc.get("language", "unknown")
    pub  = doc.get("media", {}).get("publication_title", "unknown")
    text = doc.get("text", "")[:max_text_chars]
    person_str   = ", ".join(pair.get("pers_mentions_list", []))
    location_str = ", ".join(pair.get("loc_mentions_list", []))

    return (
        f"Publication date: {date} | Language: {lang} | Source: {pub}\n\n"
        f"PERSON: {person_str}\n"
        f"LOCATION: {location_str}\n\n"
        f"ARTICLE TEXT:\n{text}"
    )


def build_arbiter_user_content(
    doc: dict, pair: dict,
    hist_raw: str, geo_raw: str,
    hist_reasoning: str, geo_reasoning: str,
    hist_at: str, hist_isat: str,
    geo_at: str, geo_isat: str,
    max_text_chars: int = 2000,
) -> str:
    date = doc.get("date", "unknown")
    lang = doc.get("language", "unknown")
    pub  = doc.get("media", {}).get("publication_title", "unknown")
    text = doc.get("text", "")[:max_text_chars]
    person_str   = ", ".join(pair.get("pers_mentions_list", []))
    location_str = ", ".join(pair.get("loc_mentions_list", []))

    return (
        f"Publication date: {date} | Language: {lang} | Source: {pub}\n"
        f"PERSON: {person_str} | LOCATION: {location_str}\n\n"
        f"--- HISTORIAN ANALYSIS ---\n"
        f"Verdict: at={hist_at}, isAt={hist_isat}\n"
        f"Reasoning: {hist_reasoning}\n\n"
        f"--- GEOGRAPHER ANALYSIS ---\n"
        f"Verdict: at={geo_at}, isAt={geo_isat}\n"
        f"Reasoning: {geo_reasoning}\n\n"
        f"--- ARTICLE TEXT (excerpt) ---\n{text}"
    )


# ── Output parsers ────────────────────────────────────────────────────────────

def parse_specialist_output(text: str):
    m = re.search(
        r'\{\s*"at_verdict"\s*:\s*"([^"]+)"\s*,\s*"isAt_verdict"\s*:\s*"([^"]+)"\s*,'
        r'\s*"reasoning"\s*:\s*"([^"]*)"',
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        at   = m.group(1).upper()
        isat = m.group(2).upper()
        reas = m.group(3).strip()
        return (
            at   if at   in AT_VALID   else "FALSE",
            isat if isat in ISAT_VALID else "FALSE",
            reas,
        )
    # Fallback
    at_m   = re.search(r'"at_verdict"\s*:\s*"([^"]+)"',   text, re.IGNORECASE)
    isat_m = re.search(r'"isAt_verdict"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    at   = (at_m.group(1).upper()   if at_m   else "FALSE")
    isat = (isat_m.group(1).upper() if isat_m else "FALSE")
    return (
        at   if at   in AT_VALID   else "FALSE",
        isat if isat in ISAT_VALID else "FALSE",
        "No reasoning extracted.",
    )


def parse_arbiter_output(text: str):
    m = re.search(
        r'\{[^{}]*"at"\s*:\s*"([^"]+)"[^{}]*"isAt"\s*:\s*"([^"]+)"[^{}]*\}',
        text, re.IGNORECASE,
    )
    if m:
        at   = m.group(1).upper()
        isat = m.group(2).upper()
        return (
            at   if at   in AT_VALID   else "FALSE",
            isat if isat in ISAT_VALID else "FALSE",
        )
    return "FALSE", "FALSE"


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    first_device = next(model.parameters()).device
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(first_device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--model_name",   default=MODEL_NAME)
    parser.add_argument("--hist_lora",    default=DEFAULT_HIST)
    parser.add_argument("--geo_lora",     default=DEFAULT_GEO)
    parser.add_argument("--arb_lora",     default=DEFAULT_ARB)
    parser.add_argument("--use_wikidata", action="store_true", default=True)
    parser.add_argument("--no_wikidata",  dest="use_wikidata", action="store_false")
    args = parser.parse_args()

    print(f"[S7] Base model: {args.model_name}")
    print(f"[S7] Loading with device_map='auto' (multi-GPU if needed)...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"[S7] Loading Historian adapter: {args.hist_lora}")
    model = PeftModel.from_pretrained(base_model, args.hist_lora, adapter_name="historian")

    print(f"[S7] Loading Geographer adapter: {args.geo_lora}")
    model.load_adapter(args.geo_lora, adapter_name="geographer")

    print(f"[S7] Loading Arbiter adapter: {args.arb_lora}")
    model.load_adapter(args.arb_lora, adapter_name="arbiter")

    model.eval()
    print("[S7] Multi-adapter model ready.")

    if args.use_wikidata:
        _load_wikidata_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tallies = {"hist_TRUE": 0, "geo_TRUE": 0, "agree": 0, "disagree": 0}

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:

        lines = [l for l in f_in if l.strip()]

        for line in tqdm(lines, desc="[S7] Documents"):
            doc      = json.loads(line.strip())
            pub_date = doc.get("date", "")
            out_pairs = []

            for pair in doc.get("sampled_pairs", []):
                user_content = build_document_user_content(doc, pair)

                # ── Agent 1: Historian ──────────────────────────────────────
                model.set_adapter("historian")
                hist_raw = generate(model, tokenizer,
                                    build_agent_prompt(HISTORIAN_SYSTEM, user_content))
                hist_at, hist_isat, hist_reasoning = parse_specialist_output(hist_raw)
                if hist_at == "TRUE":
                    tallies["hist_TRUE"] += 1

                # ── Agent 2: Geographer ─────────────────────────────────────
                model.set_adapter("geographer")
                geo_raw = generate(model, tokenizer,
                                   build_agent_prompt(GEOGRAPHER_SYSTEM, user_content))
                geo_at, geo_isat, geo_reasoning = parse_specialist_output(geo_raw)
                if geo_at == "TRUE":
                    tallies["geo_TRUE"] += 1

                if hist_at == geo_at:
                    tallies["agree"] += 1
                else:
                    tallies["disagree"] += 1

                # ── Agent 3: Arbiter ────────────────────────────────────────
                model.set_adapter("arbiter")
                arbiter_content = build_arbiter_user_content(
                    doc, pair,
                    hist_raw=hist_raw, geo_raw=geo_raw,
                    hist_reasoning=hist_reasoning, geo_reasoning=geo_reasoning,
                    hist_at=hist_at, hist_isat=hist_isat,
                    geo_at=geo_at,   geo_isat=geo_isat,
                )
                arb_raw = generate(model, tokenizer,
                                   build_agent_prompt(ARBITER_SYSTEM, arbiter_content),
                                   max_new_tokens=30)
                at_pred, isat_pred = parse_arbiter_output(arb_raw)

                if at_pred == "FALSE":
                    isat_pred = "FALSE"

                if args.use_wikidata and isat_pred == "TRUE":
                    pers_qid = pair.get("pers_wikidata_QID", "")
                    if person_dead_before_pub(pers_qid, pub_date):
                        isat_pred = "FALSE"
                        print(f"  [Wikidata] {pers_qid} dead before {pub_date} → isAt=FALSE")

                pair = dict(pair)
                pair["at"]               = at_pred
                pair["isAt"]             = isat_pred
                pair["_hist_at"]         = hist_at
                pair["_hist_isat"]       = hist_isat
                pair["_hist_reasoning"]  = hist_reasoning
                pair["_geo_at"]          = geo_at
                pair["_geo_isat"]        = geo_isat
                pair["_geo_reasoning"]   = geo_reasoning
                out_pairs.append(pair)

            doc = dict(doc)
            doc["sampled_pairs"] = out_pairs
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    if args.use_wikidata:
        _save_wikidata_cache()

    print(f"\n[S7] Predictions written to {args.output}")
    print(f"[S7] Agent agreement stats: {tallies}")


if __name__ == "__main__":
    run_pipeline()
