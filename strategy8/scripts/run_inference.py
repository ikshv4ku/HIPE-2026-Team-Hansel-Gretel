"""
Strategy 8: Multi-backend LLM Inference.

Supported backends
──────────────────
  ibm        → IBM watsonx.ai  (ibm-watsonx-ai SDK)
  openai     → OpenAI API      (openai SDK)
  anthropic  → Anthropic API   (anthropic SDK)
  hf         → Hugging Face Inference API (requests)
  local      → Any OpenAI-compatible local server (LM Studio, Ollama, vLLM…)

Usage examples
──────────────
  # IBM watsonx
  python3 scripts/run_inference.py \\
      --backend ibm \\
      --model ibm/granite-13b-instruct-v2 \\
      --ibm_url https://us-south.ml.cloud.ibm.com \\
      --ibm_project_id <YOUR_PROJECT_ID> \\
      --ibm_api_key   <YOUR_API_KEY>

  # OpenAI GPT-4o
  python3 scripts/run_inference.py \\
      --backend openai \\
      --model gpt-4o \\
      --openai_api_key <YOUR_API_KEY>

  # Anthropic Claude 3.5 Sonnet
  python3 scripts/run_inference.py \\
      --backend anthropic \\
      --model claude-3-5-sonnet-20241022 \\
      --anthropic_api_key <YOUR_API_KEY>

  # HuggingFace Inference API (e.g. Mixtral, Llama-3)
  python3 scripts/run_inference.py \\
      --backend hf \\
      --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --hf_api_key <YOUR_HF_TOKEN>

  # Local OpenAI-compatible server (e.g. Ollama on port 11434)
  python3 scripts/run_inference.py \\
      --backend local \\
      --model llama3:70b \\
      --local_base_url http://localhost:11434/v1
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cot_prompts import SYSTEM_PROMPT, build_user_message

# ── Constants ─────────────────────────────────────────────────────────────────

AT_VALID   = {"TRUE", "PROBABLE", "FALSE"}
ISAT_VALID = {"TRUE", "FALSE"}

DEFAULT_INPUT  = (
    "HIPE-2026-data/data/newspapers/v1.0/splits/"
    "HIPE-2026-v1.0-impresso-dev-str4-all.jsonl"
)
DEFAULT_OUTPUT = "strategy8/results/predictions/preds-dev-8.jsonl"


# ══════════════════════════════════════════════════════════════════════════════
# Backend adapters
# Each adapter returns (response_text: str) given the system + user messages.
# ══════════════════════════════════════════════════════════════════════════════

def call_ibm(system: str, user: str, args) -> str:
    """IBM watsonx.ai via the ibm-watsonx-ai SDK."""
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

    credentials = Credentials(url=args.ibm_url, api_key=args.ibm_api_key)
    client      = APIClient(credentials, project_id=args.ibm_project_id)

    model = ModelInference(
        model_id   = args.model,
        api_client = client,
        params     = {
            Params.MAX_NEW_TOKENS: args.max_tokens,
            Params.TEMPERATURE:    args.temperature,
            Params.STOP_SEQUENCES: ["}\n\n", "}\n---"],
        },
    )

    # watsonx uses a single string prompt — format as chat template
    prompt = (
        f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    )
    result = model.generate(prompt=prompt)
    return result["results"][0]["generated_text"]


def call_openai(system: str, user: str, args) -> str:
    """OpenAI Chat Completions API."""
    from openai import OpenAI
    client = OpenAI(api_key=args.openai_api_key)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_format={"type": "json_object"},   # enforces JSON output
    )
    return resp.choices[0].message.content


def call_anthropic(system: str, user: str, args) -> str:
    """Anthropic Messages API."""
    import anthropic
    client = anthropic.Anthropic(api_key=args.anthropic_api_key)
    msg = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=args.temperature,
    )
    return msg.content[0].text


def call_hf(system: str, user: str, args) -> str:
    """Hugging Face Inference API (serverless or dedicated endpoints)."""
    import requests

    combined = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    headers  = {"Authorization": f"Bearer {args.hf_api_key}",
                 "Content-Type":  "application/json"}
    payload  = {
        "inputs":      combined,
        "parameters": {
            "max_new_tokens":    args.max_tokens,
            "temperature":       max(args.temperature, 0.01),
            "return_full_text":  False,
        },
    }
    url  = f"https://api-inference.huggingface.co/models/{args.model}"
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data[0].get("generated_text", "")
    return data.get("generated_text", str(data))


def call_local(system: str, user: str, args) -> str:
    """OpenAI-compatible local server (Ollama, LM Studio, vLLM, etc.)."""
    from openai import OpenAI
    client = OpenAI(
        api_key  = "local",
        base_url = args.local_base_url,
    )
    resp = client.chat.completions.create(
        model    = args.model,
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature = args.temperature,
        max_tokens  = args.max_tokens,
    )
    return resp.choices[0].message.content


BACKENDS = {
    "ibm":       call_ibm,
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "hf":        call_hf,
    "local":     call_local,
}


# ══════════════════════════════════════════════════════════════════════════════
# Output parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_output(text: str) -> dict:
    """
    Extract the JSON object from the model's response.
    Tries full JSON parse first, then regex extraction as fallback.
    Returns a dict with keys: at, isAt, step1…step3, raw.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Attempt full JSON parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    at   = str(parsed.get("at",   "FALSE")).upper()
    isat = str(parsed.get("isAt", "FALSE")).upper()

    at   = at   if at   in AT_VALID   else "FALSE"
    isat = isat if isat in ISAT_VALID else "FALSE"

    # Enforce hard rule
    if at == "FALSE":
        isat = "FALSE"

    return {
        "at":                at,
        "isAt":              isat,
        "step1_biographical": parsed.get("step1_biographical", ""),
        "step2_geographic":   parsed.get("step2_geographic",   ""),
        "step3_synthesis":    parsed.get("step3_synthesis",    ""),
        "raw":                text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run():
    parser = argparse.ArgumentParser(
        description="Strategy 8 – CoT Few-Shot LLM inference for HIPE-2026"
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument("--input",  default=DEFAULT_INPUT,
                        help="Path to input JSONL (dev or test set)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Path to write prediction JSONL")

    # ── Backend & Model ───────────────────────────────────────────────────────
    parser.add_argument("--backend", required=True,
                        choices=list(BACKENDS.keys()),
                        help="LLM backend to use")
    parser.add_argument("--model", required=True,
                        help="Model ID (e.g. ibm/granite-13b-instruct-v2, gpt-4o, …)")

    # ── Generation params ─────────────────────────────────────────────────────
    parser.add_argument("--max_tokens",  type=int,   default=700,
                        help="Maximum new tokens to generate per call")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--retry",       type=int,   default=3,
                        help="Number of retries on API errors")
    parser.add_argument("--delay",       type=float, default=1.0,
                        help="Seconds between retries")
    parser.add_argument("--no_few_shot", action="store_true",
                        help="Disable few-shot examples (zero-shot CoT only)")

    # ── IBM-specific ──────────────────────────────────────────────────────────
    parser.add_argument("--ibm_url",        default=os.getenv("IBM_WATSONX_URL",        ""))
    parser.add_argument("--ibm_api_key",    default=os.getenv("IBM_WATSONX_API_KEY",    ""))
    parser.add_argument("--ibm_project_id", default=os.getenv("IBM_WATSONX_PROJECT_ID", ""))

    # ── OpenAI-specific ───────────────────────────────────────────────────────
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY", ""))

    # ── Anthropic-specific ────────────────────────────────────────────────────
    parser.add_argument("--anthropic_api_key", default=os.getenv("ANTHROPIC_API_KEY", ""))

    # ── HuggingFace-specific ──────────────────────────────────────────────────
    parser.add_argument("--hf_api_key", default=os.getenv("HF_TOKEN", ""))

    # ── Local-specific ────────────────────────────────────────────────────────
    parser.add_argument("--local_base_url", default="http://localhost:11434/v1",
                        help="Base URL for local OpenAI-compatible server")

    args = parser.parse_args()

    # Validate required credentials per backend
    if args.backend == "ibm" and not (args.ibm_url and args.ibm_api_key and args.ibm_project_id):
        print("ERROR: IBM backend requires --ibm_url, --ibm_api_key, --ibm_project_id "
              "(or environment variables IBM_WATSONX_URL, IBM_WATSONX_API_KEY, IBM_WATSONX_PROJECT_ID)")
        sys.exit(1)
    if args.backend == "openai" and not args.openai_api_key:
        print("ERROR: OpenAI backend requires --openai_api_key (or OPENAI_API_KEY env var)")
        sys.exit(1)
    if args.backend == "anthropic" and not args.anthropic_api_key:
        print("ERROR: Anthropic backend requires --anthropic_api_key (or ANTHROPIC_API_KEY env var)")
        sys.exit(1)
    if args.backend == "hf" and not args.hf_api_key:
        print("ERROR: HF backend requires --hf_api_key (or HF_TOKEN env var)")
        sys.exit(1)

    call_fn = BACKENDS[args.backend]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]

    total_pairs = sum(len(d.get("sampled_pairs", [])) for d in docs)
    print(f"\n[S8] Backend : {args.backend}  |  Model: {args.model}")
    print(f"[S8] Input   : {args.input}  ({len(docs)} docs, {total_pairs} pairs)")
    print(f"[S8] Output  : {args.output}")
    print(f"[S8] Few-shot: {'enabled' if not args.no_few_shot else 'disabled (zero-shot CoT)'}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    processed = 0
    errors    = 0
    t_start   = time.time()

    with open(args.output, "w", encoding="utf-8") as f_out:
        for doc_idx, doc in enumerate(docs):
            out_pairs = []

            for pair in doc.get("sampled_pairs", []):
                person_str   = ", ".join(pair.get("pers_mentions_list", []))
                location_str = ", ".join(pair.get("loc_mentions_list",  []))
                text         = doc.get("text", "")[:3500]

                user_msg = build_user_message(
                    date         = doc.get("date",     "unknown"),
                    language     = doc.get("language", "unknown"),
                    source       = doc.get("media", {}).get("publication_title", "unknown"),
                    person       = person_str,
                    location     = location_str,
                    text         = text,
                    include_few_shot = not args.no_few_shot,
                )

                # Retry loop
                result = None
                for attempt in range(1, args.retry + 1):
                    try:
                        raw_text = call_fn(SYSTEM_PROMPT, user_msg, args)
                        result   = parse_output(raw_text)
                        break
                    except Exception as e:
                        print(f"  [WARN] Attempt {attempt}/{args.retry} failed for "
                              f"doc {doc_idx} | {person_str} @ {location_str}: {e}")
                        if attempt < args.retry:
                            time.sleep(args.delay * attempt)

                if result is None:
                    errors += 1
                    result = {
                        "at": "FALSE", "isAt": "FALSE",
                        "step1_biographical": "API error",
                        "step2_geographic":   "API error",
                        "step3_synthesis":    "API error",
                        "raw": "",
                    }

                processed += 1
                elapsed = time.time() - t_start
                rate    = processed / elapsed

                print(
                    f"  [{processed:>5}/{total_pairs}] "
                    f"{person_str[:30]:<30} @ {location_str[:20]:<20} "
                    f"→ at={result['at']}, isAt={result['isAt']}  "
                    f"({rate:.2f} pairs/s)"
                )

                pair = dict(pair)
                pair["at"]                  = result["at"]
                pair["isAt"]                = result["isAt"]
                pair["_cot_biographical"]   = result["step1_biographical"]
                pair["_cot_geographic"]     = result["step2_geographic"]
                pair["_cot_synthesis"]      = result["step3_synthesis"]
                out_pairs.append(pair)

            doc_out = dict(doc)
            doc_out["sampled_pairs"] = out_pairs
            f_out.write(json.dumps(doc_out, ensure_ascii=False) + "\n")
            f_out.flush()

    elapsed = time.time() - t_start
    print(f"\n[S8] Done. {processed} pairs in {elapsed:.1f}s "
          f"({processed/elapsed:.2f} pairs/s). Errors: {errors}.")
    print(f"[S8] Predictions → {args.output}")


if __name__ == "__main__":
    run()
