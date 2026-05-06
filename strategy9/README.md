# Strategy 9 — CoT Few-Shot Inference via RITS LLMs (with Wikidata Injection)

> **Status:** Ready to run — no GPU, no fine-tuning required.  
> **API:** IBM RITS (Research Internal Technology Services)

---

## What This Strategy Does

This is an upgraded version of Strategy 8. It attacks the HIPE-2026 relation-extraction task **without any model training** using **Chain-of-Thought (CoT) Few-Shot Prompting** through IBM's RITS inference gateway (e.g. Llama 3.3 70B).

### The Upgrade: Wikidata Knowledge Injection
Unlike Strategy 8, Strategy 9 dynamically queries the Wikidata SPARQL endpoint during inference. For every pair, it fetches:
- **Person Facts:** Biographical descriptions, birth dates, and death dates.
- **Location Facts:** Geographic descriptions and country contexts.

These facts are injected directly into the LLM prompt as a `[WIKIDATA FACTS]` block, allowing the model to ground its temporal reasoning in hard facts rather than relying solely on parametric memory.

---

## Step-by-Step Setup Guide

**Important:** Please follow these steps exactly to ensure you have the latest code, the newly released test data, and the required packages.

### Step 1 — Clone the Repository & Submodules

Open your terminal and clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/ikshv4ku/HIPE-2026-Team-Hansel-Gretel.git
cd HIPE-2026-Team-Hansel-Gretel
```

### Step 2 — Fetch the Latest Official Test Data

Ensure the data submodule is fully up to date:

```bash
cd HIPE-2026-data
git checkout main
git pull origin main
cd ..
```

### Step 3 — Set Up the Environment

Set up a virtual environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r strategy9/requirements.txt
```

### Step 4 — Set Up Your RITS API Key

Create a `.env` file in the **repository root**:

```bash
echo "RITS_API_KEY=your_actual_rits_api_key" > .env
```
*(Request access via the IBM RITS portal at `https://rits.fmaas.res.ibm.com` if you don't have a key)*

### Step 5 — Run the 4 Official Inference Commands

Execute the following four commands. They will query the Llama 70B model via RITS, inject the Wikidata facts, and output the exact files we need for the official submission.

**Command 1 (German Test Set):**
```bash
python strategy9/hipe_inference.py \
  -model llama70b \
  -input_file HIPE-2026-data/official_test_unlabeled/HIPE-2026-v1.0-impresso-test-de.jsonl \
  > Hansel\&Gretel_HIPE-2026-v1.0-impresso-test-de_run3.jsonl
```

**Command 2 (English Test Set):**
```bash
python strategy9/hipe_inference.py \
  -model llama70b \
  -input_file HIPE-2026-data/official_test_unlabeled/HIPE-2026-v1.0-impresso-test-en.jsonl \
  > Hansel\&Gretel_HIPE-2026-v1.0-impresso-test-en_run3.jsonl
```

**Command 3 (French Test Set):**
```bash
python strategy9/hipe_inference.py \
  -model llama70b \
  -input_file HIPE-2026-data/official_test_unlabeled/HIPE-2026-v1.0-impresso-test-fr.jsonl \
  > Hansel\&Gretel_HIPE-2026-v1.0-impresso-test-fr_run3.jsonl
```

**Command 4 (French Surprise Literary Set):**
```bash
python strategy9/hipe_inference.py \
  -model llama70b \
  -input_file HIPE-2026-data/official_test_unlabeled/HIPE-2026-v1.0-surprise-test-fr.jsonl \
  > Hansel\&Gretel_HIPE-2026-v1.0-surprise-test-fr_run3.jsonl
```

### Step 6 — Send the Files!

Once those commands finish, you will have four `.jsonl` files in your root directory. Please ZIP them up and send them over so they can be merged into the master `Hansel&Gretel.zip` submission archive.
