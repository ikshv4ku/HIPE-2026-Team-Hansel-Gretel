"""
Strategy 8: Zero-Shot / Few-Shot Chain-of-Thought LLM Inference for HIPE-2026.

Follows the IBM RITS API pattern (llmaj.py reference).
Uses OpenAI-compatible client pointed at the RITS inference gateway.

Supported judge models (via RITS):
  - meta-llama/llama-3-3-70b-instruct
  - ibm-granite/granite-3.2-8b-instruct
  - meta-llama/llama-4-maverick-17b-128e-instruct-fp8
  - openai/gpt-oss-120b

Usage:
  python hipe_inference.py -input_file HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl -model llama70b
"""

from collections import defaultdict
import os
import sys
import json
import argparse
from tqdm import tqdm
from dotenv import dotenv_values
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


# ══════════════════════════════════════════════════════════════════════════════
# Utility: clean special tokens from LLM output
# ══════════════════════════════════════════════════════════════════════════════

def clean_special_tokens(text):
    """Remove special tokens from generated text"""
    if not text:
        return ""

    special_tokens = [
        r"<\|END_OF_TURN_TOKEN\|>",
        r"<\|END_RESPONSE\|>",
        r"<\|END_RESPONSE\|><EOS_TOKEN>",
        r"<end_of_turn>",
        r"<turn\|>",
        r"<pad>",
        r"<EOS_TOKEN>",
    ]

    cleaned_text = text
    for token in special_tokens:
        cleaned_text = re.sub(token, "", cleaned_text)

    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# RITS API helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_rits_model_list(api_key):
    url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
    response = requests.get(url, headers={"RITS_API_KEY": api_key})
    if response.status_code == 200:
        return {m["model_name"]: m["endpoint"] for m in response.json()}
    else:
        raise Exception(f"Failed getting RITS model list:\n\n{response.text}")


# ══════════════════════════════════════════════════════════════════════════════
# Prompt definitions
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert historian and geographer specialising in 19th–20th century European
newspapers and their historical context. Your task is to determine, for a given
PERSON–LOCATION pair extracted from a historical newspaper article, two relation labels:

  • at   – Was the person EVER at this location before the article date?
             TRUE     = there is explicit or strong inferential evidence.
             PROBABLE = the evidence is suggestive but not definitive.
             FALSE    = no credible evidence links them to this place.

  • isAt – Was the person at this location WITHIN ~1 MONTH of the publication date?
             TRUE  = the text or context confirms very recent / current presence.
             FALSE = there is no such evidence (or at=FALSE, which forces isAt=FALSE).

Hard rule: if at=FALSE then isAt MUST be FALSE.

──────────────────────────────────────────────────────────────────────────────
REASONING PROCESS (Chain-of-Thought)
──────────────────────────────────────────────────────────────────────────────
For each pair you MUST work through THREE explicit steps before answering:

Step 1 – BIOGRAPHICAL & TEMPORAL ANALYSIS
  Consider the person's known career, role, nationality, and life timeline
  relative to the publication date. Do dates mentioned constrain presence?

Step 2 – GEOGRAPHIC & CONTEXTUAL ANALYSIS
  Consider the article's publication language and source, named places,
  institutional affiliations, and any spatial language in the text.
  Does the regional context imply the person was there?

Step 3 – SYNTHESIS & DECISION
  Weigh both analyses. Apply the hard rule. State the final verdicts.

──────────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT (strictly JSON, no extra text after the JSON block)
──────────────────────────────────────────────────────────────────────────────
{
  "step1_biographical": "<your biographical/temporal reasoning>",
  "step2_geographic":   "<your geographic/contextual reasoning>",
  "step3_synthesis":    "<your synthesis and decision>",
  "at":   "<TRUE|PROBABLE|FALSE>",
  "isAt": "<TRUE|FALSE>"
}
"""

# ── Few-shot examples ──────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "input": {
            "date": "1902-03-15", "language": "fr",
            "source": "Le Journal de Genève",
            "person": "Émile Zola", "location": "Paris",
            "text": (
                "M. Émile Zola est de retour à Paris après son séjour forcé en Angleterre. "
                "Il a été aperçu hier soir au théâtre de l'Odéon en compagnie de plusieurs "
                "journalistes parisiens. L'écrivain a déclaré reprendre ses travaux littéraires."
            ),
        },
        "output": {
            "step1_biographical": "Émile Zola (1840–1902) was a French novelist who lived in Paris for most of his career. He fled to England in 1898 after the Dreyfus affair but returned to France in 1899. By March 1902 he was back in Paris, consistent with his timeline.",
            "step2_geographic": "The article is a French-language Swiss newspaper reporting Zola's return TO Paris. The phrase 'est de retour à Paris' and the specific sighting at the Odéon theatre spatially anchor him in Paris within the last 24 hours of the publication date.",
            "step3_synthesis": "Both analyses concur: Zola is explicitly in Paris and was seen there the day before publication. at=TRUE (associated with Paris). isAt=TRUE (confirmed presence within days of the article date).",
            "at": "TRUE", "isAt": "TRUE",
        },
    },
    {
        "input": {
            "date": "1885-11-04", "language": "de",
            "source": "Neue Zürcher Zeitung",
            "person": "Otto von Bismarck", "location": "Wien",
            "text": (
                "Der Reichskanzler Bismarck hat in seiner langen Karriere mehrfach diplomatische "
                "Verhandlungen mit österreichischen Vertretern geführt, zuletzt im Rahmen des "
                "Drei-Kaiser-Abkommens. Seine Außenpolitik gilt als Meisterwerk europäischer "
                "Diplomatie. Derzeit weilt er auf seinem Gut in Friedrichsruh."
            ),
        },
        "output": {
            "step1_biographical": "Bismarck (1815–1898) was German Chancellor and conducted extensive diplomatic relations with Austria, including the Three Emperors' League. Such negotiations would plausibly have required visits to Vienna. However, the article states he is currently on his estate in Friedrichsruh.",
            "step2_geographic": "The article references past diplomatic meetings with Austrian representatives and the Three Emperors' League, implying historical contact with Wien. No current geographic anchor to Wien is given.",
            "step3_synthesis": "Historical diplomatic role implies PROBABLE past presence in Wien (at=PROBABLE). He is currently at Friedrichsruh, ruling out recent presence (isAt=FALSE).",
            "at": "PROBABLE", "isAt": "FALSE",
        },
    },
    {
        "input": {
            "date": "1910-06-22", "language": "en",
            "source": "The Manchester Guardian",
            "person": "Victor Hugo", "location": "London",
            "text": (
                "The city mourns the passing of a great labour reformer. Local trade unions "
                "gathered at Hyde Park yesterday to demand better working conditions for textile "
                "workers in Manchester. Several MPs have pledged support for the new bill."
            ),
        },
        "output": {
            "step1_biographical": "Victor Hugo (1802–1885) was a French poet and novelist who died 25 years before this article. While he did live in exile in the Channel Islands and briefly in Brussels, he is not biographically linked to London. His death in 1885 rules out any presence in 1910.",
            "step2_geographic": "The article is entirely about a Manchester labour rally at Hyde Park. Victor Hugo is not mentioned anywhere in the text. There is no geographic or contextual signal connecting him to London.",
            "step3_synthesis": "No biographical or textual evidence connects Victor Hugo to London. He is not mentioned in the article. at=FALSE; isAt=FALSE (forced by the hard rule).",
            "at": "FALSE", "isAt": "FALSE",
        },
    },
    {
        "input": {
            "date": "1898-07-10", "language": "fr",
            "source": "Journal de Genève",
            "person": "Ferdinand de Lesseps", "location": "Le Caire",
            "text": (
                "On commémore aujourd'hui le vingtième anniversaire de l'inauguration du Canal "
                "de Suez, grande œuvre à laquelle Ferdinand de Lesseps a consacré sa vie. "
                "Le ingénieur, aujourd'hui décédé, reste une figure incontournable de l'histoire "
                "de l'Égypte moderne."
            ),
        },
        "output": {
            "step1_biographical": "Ferdinand de Lesseps (1805–1894) directed the construction of the Suez Canal and spent many years in Egypt, headquartered in or near Cairo. He died in 1894, four years before this article.",
            "step2_geographic": "The article commemorates the Suez Canal anniversary and calls de Lesseps a key figure in modern Egyptian history. Cairo is the capital closest to his operations.",
            "step3_synthesis": "De Lesseps was historically and biographically present in Cairo during his Suez Canal work (at=TRUE). He died in 1894, so he cannot be there in 1898 (isAt=FALSE).",
            "at": "TRUE", "isAt": "FALSE",
        },
    },
]


def build_user_message(date, language, source, person, location, text,
                       include_few_shot=True):
    """Build the user-turn message, optionally prepending few-shot examples."""
    parts = []

    if include_few_shot:
        parts.append("DEMONSTRATION EXAMPLES\n" + "=" * 60)
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            inp = ex["input"]
            out = ex["output"]
            parts.append(
                f"\n--- Example {i} ---\n"
                f"Publication date: {inp['date']} | Language: {inp['language']} "
                f"| Source: {inp['source']}\n"
                f"PERSON: {inp['person']}\n"
                f"LOCATION: {inp['location']}\n"
                f"ARTICLE TEXT:\n{inp['text']}\n\n"
                f"RESPONSE:\n{json.dumps(out, indent=2, ensure_ascii=False)}"
            )
        parts.append("\n" + "=" * 60 + "\nNOW CLASSIFY THE FOLLOWING:\n")

    parts.append(
        f"Publication date: {date} | Language: {language} | Source: {source}\n"
        f"PERSON: {person}\n"
        f"LOCATION: {location}\n\n"
        f"ARTICLE TEXT:\n{text}\n\n"
        f"Respond ONLY with a valid JSON object. No extra text outside the JSON."
    )

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Core inference function (following llmaj.py pattern)
# ══════════════════════════════════════════════════════════════════════════════

AT_VALID   = {"TRUE", "PROBABLE", "FALSE"}
ISAT_VALID = {"TRUE", "FALSE"}


def parse_output(text):
    """Extract JSON from LLM response, validate and normalise labels."""
    text = clean_special_tokens(text)
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
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
        "at": at,
        "isAt": isat,
        "step1_biographical": parsed.get("step1_biographical", ""),
        "step2_geographic":   parsed.get("step2_geographic",   ""),
        "step3_synthesis":    parsed.get("step3_synthesis",    ""),
        "raw": text,
    }


def hipe_classify(client, model, system_prompt, user_message):
    """Classify a single person-location pair via the RITS LLM API."""
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    try:
        completion = (
            client.chat.completions.create(
                model=model,
                messages=chat_prompt,
                max_completion_tokens=1024,
                temperature=0,
            )
            .choices[0]
            .message.content
        )
        result = parse_output(completion)
    except Exception as e:
        result = {
            "at": "FALSE", "isAt": "FALSE",
            "step1_biographical": f"Error: {e}",
            "step2_geographic":   "Error",
            "step3_synthesis":    "Error",
            "raw": "",
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline (following llmaj.py pattern)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME_MAP = {
    "llama70b": "meta-llama/llama-3-3-70b-instruct",
    "granite":  "ibm-granite/granite-3.2-8b-instruct",
    "llama4":   "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "gpt":      "openai/gpt-oss-120b",
}

MODEL_ENDPOINT_OVERRIDES = {
    "llama70b": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct",
    "gpt":      "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b",
}


def run_hipe_inference(input_file, model_key, no_few_shot=False):
    """
    Run HIPE-2026 relation extraction on the input JSONL file using
    a RITS-hosted LLM as the backbone.
    """
    # ── Load RITS credentials from .env ──────────────────────────────────
    config = dotenv_values(".env")
    if "RITS_API_KEY" not in config:
        # Fall back to environment variable
        api_key = os.environ.get("RITS_API_KEY", "")
        if not api_key:
            print("ERROR: RITS_API_KEY not found in .env or environment variables.")
            print("Create a .env file in the repo root with: RITS_API_KEY=your_key_here")
            sys.exit(1)
    else:
        api_key = config["RITS_API_KEY"]

    mname = MODEL_NAME_MAP[model_key]

    # ── Resolve endpoint URL ─────────────────────────────────────────────
    if model_key in MODEL_ENDPOINT_OVERRIDES:
        url = MODEL_ENDPOINT_OVERRIDES[model_key]
    else:
        minfo = get_rits_model_list(api_key)
        if mname not in minfo:
            print(f"ERROR: Model '{mname}' not found in RITS. Available models:")
            for k in sorted(minfo.keys()):
                print(f"  - {k}")
            sys.exit(1)
        url = minfo[mname]

    base_url = f"{url}/v1"
    print(f"\n[Strategy 8] Model  : {mname} ({model_key})")
    print(f"[Strategy 8] URL    : {base_url}")
    print(f"[Strategy 8] Input  : {input_file}")
    print(f"[Strategy 8] Few-shot: {'enabled' if not no_few_shot else 'disabled'}\n")

    # ── Create OpenAI-compatible client ──────────────────────────────────
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"RITS_API_KEY": api_key},
    )

    # ── Load HIPE-2026 data ──────────────────────────────────────────────
    with open(input_file, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    total_pairs = sum(len(d.get("sampled_pairs", [])) for d in docs)
    print(f"[Strategy 8] Loaded {len(docs)} documents, {total_pairs} pairs\n")

    # ── Run inference ────────────────────────────────────────────────────
    output_file = f"strategy8/results/predictions/preds-dev-8-{model_key}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed = 0
    stats = defaultdict(int)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for doc in tqdm(docs, desc="[Strategy 8] Documents"):
            out_pairs = []

            for pair in doc.get("sampled_pairs", []):
                person_str   = ", ".join(pair.get("pers_mentions_list", []))
                location_str = ", ".join(pair.get("loc_mentions_list",  []))
                text         = doc.get("text", "")[:3500]

                user_msg = build_user_message(
                    date     = doc.get("date", "unknown"),
                    language = doc.get("language", "unknown"),
                    source   = doc.get("media", {}).get("publication_title", "unknown"),
                    person   = person_str,
                    location = location_str,
                    text     = text,
                    include_few_shot = not no_few_shot,
                )

                result = hipe_classify(client, mname, SYSTEM_PROMPT, user_msg)

                processed += 1
                stats[f"at_{result['at']}"] += 1
                stats[f"isAt_{result['isAt']}"] += 1

                pair = dict(pair)
                pair["at"]                = result["at"]
                pair["isAt"]              = result["isAt"]
                pair["_cot_biographical"] = result["step1_biographical"]
                pair["_cot_geographic"]   = result["step2_geographic"]
                pair["_cot_synthesis"]    = result["step3_synthesis"]
                out_pairs.append(pair)

            doc_out = dict(doc)
            doc_out["sampled_pairs"] = out_pairs
            f_out.write(json.dumps(doc_out, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"\n[Strategy 8] Inference complete: {processed} pairs processed.")
    print(f"[Strategy 8] Label distribution: {dict(stats)}")
    print(f"[Strategy 8] Predictions → {output_file}")

    # ── Score with official HIPE-2026 scorer ──────────────────────────────
    print("\n[Strategy 8] Running official scorer...\n")
    import subprocess
    scorer_cmd = [
        sys.executable,
        "HIPE-2026-data/scripts/file_scorer_evaluation.py",
        "--schema_file",      "HIPE-2026-data/schemas/hipe-2026-data.schema.json",
        "--gold_data_file",   input_file,
        "--predictions_file", output_file,
    ]
    result = subprocess.run(scorer_cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)

    # Save evaluation results
    eval_file = f"strategy8/results/analysis/evaluation_{model_key}.txt"
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    with open(eval_file, "w") as f:
        f.write(output)
    print(f"[Strategy 8] Evaluation → {eval_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Strategy 8 — HIPE-2026 Relation Extraction via RITS LLMs"
    )

    parser.add_argument(
        "-input_file", "--input_file",
        type=str,
        default="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-str4-all.jsonl",
        help="Path to input JSONL file (HIPE-2026 dev or test set)",
    )
    parser.add_argument(
        "-model", "--model",
        default="llama70b",
        choices=["llama70b", "granite", "llama4", "gpt"],
        help="RITS model to use as backbone",
    )
    parser.add_argument(
        "-no_few_shot", "--no_few_shot",
        action="store_true",
        help="Disable few-shot examples (zero-shot CoT only)",
    )

    args = parser.parse_args()

    run_hipe_inference(
        input_file=args.input_file,
        model_key=args.model,
        no_few_shot=args.no_few_shot,
    )


if __name__ == "__main__":
    main()
