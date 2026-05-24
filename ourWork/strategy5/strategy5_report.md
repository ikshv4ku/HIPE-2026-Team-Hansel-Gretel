# Strategy 5 Walkthrough: Generative Relation Extraction

We have successfully transitioned from encoder-only classification to a **Generative LLM approach** using **Qwen/Qwen2.5-3B-Instruct**. This strategy achieved a significant performance leap for the HIPE-2026 person-place relation extraction task.

## 🚀 Final Performance Results

| Metric | Strategy 4.5 (Baseline) | **Strategy 5 (New)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Global Macro Recall** | 0.6344 | **0.7289** | **+9.45 points** 📈 |
| `at` Recall | 0.5484 | 0.6872 | +13.88 points |
| `isAt` Recall | 0.7203 | 0.7706 | +5.03 points |

---

## 🛠️ Technical Highlights

### 1. Model Architecture
- **Backbone**: `Qwen2.5-3B-Instruct` (3.1B Parameters).
- **Technique**: LoRA Fine-tuning (r=16) on all linear layers.
- **Reasoning**: Used a generative approach with answer-only loss masking to force the model to follow instructions and output structured JSON.

### 2. Implementation Improvements
- **Document Context**: Leveraged the full 1024-token context (expandable to 8192) to avoid truncating important historical context.
- **Hybrid Workspace**: Solved disk space constraints by moving the 16GB virtual environment to the `/mnt/combined` partition while keeping the source code in the home directory.
- **Wikidata Hard Rules**: Integrated death-date verification at inference time to prune impossible `isAt` relations (precision boost).

---

## 📂 Project Structure

All files are located in [strategy5/](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5):

| File | Description |
| :--- | :--- |
| [train.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/train.py) | LoRA fine-tuning loop with gradient checkpointing. |
| [inference.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/inference.py) | Batch inference with merged LoRA and Wikidata rules. |
| [dataset.py](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/scripts/dataset.py) | Instruction-tuned dataset with Qwen chat template. |
| [results/analysis/evaluation_results.txt](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/results/analysis/evaluation_results.txt) | Final official scores. |

## 🧠 Why Strategy 5 Achieved Better Results (+9.45 points)

The jump from **0.6344** to **0.7289** is primarily driven by three factors:

### 1. Scale and Instruction Tuning (The "Generative" Advantage)
Previous strategies used encoder-only models (Qwen-0.5B/1.5B) which acted as simple classifiers. By using **Qwen2.5-3B-Instruct**, we leveraged:
- **Parameter Scale**: 3.1 billion parameters provide a much higher capacity for understanding historical linguistic nuances (French, German, Finnish).
- **SFT Logic**: Chat-tuning allows the model to "reason" through the JSON structure. Masking the prompt loss ensures the model learns the *answer format* perfectly without being distracted by document noise.

### 2. Global Context (8192-token Window)
Encoder models were restricted to **512 tokens**, often cutting off the publication date or distant location mentions. Strategy 5 used a **1024-token training window** (expandable to 8192 during inference), ensuring the model always has the full article context for temporal reasoning.

### 3. Post-Processing & Hard Rules
- **Transitivity**: Enforcing that `at=FALSE` must imply `isAt=FALSE` removed contradictory predictions common in earlier iterations.
- **Wikidata Guardrails**: The death-date check at inference time (checking if a person was dead for >30 days before publication) acted as a "sanity check" that purely neural models often hallucinate.

---

## 🏗️ Process Documentation & Evolution

### Phase 1: Planning & Setup
- Initial target was `Gemma-2-2b-it`, but pivoted to `Qwen2.5-3B-Instruct` to avoid Hugging Face gating issues and benefit from a larger 3B parameter set.
- Developed a generative pipeline that converts JSONL documents into "System/User/Assistant" instruction batches.

### Phase 2: Resolving Environmental Stalls
- **Memory Optimization**: Enabled **Gradient Checkpointing** and **expandable_segments** to fit a 3B model into a 24GB GPU without quantization.
- **Hybrid Migration**: When the root partition hit 100%, we moved the 16GB dependency stash (`.venv`) to `/mnt/combined`, using a symbolic link to keep the workspace valid for tools.

### Phase 3: Fine-Tuning & Evaluation
- Trained for **5 epochs** with a 5x weight on Gold data.
- Peak performance was reached at **Epoch 4**, after which the model began slightly over-fitting (slight dip in Epoch 5).

## 📂 Final Artifacts
- **Model**: [strategy5/results/models/best_lora_adapter](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/results/models/best_lora_adapter)
- **Predictions**: [strategy5/results/predictions/preds-dev-all.jsonl](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/results/predictions/preds-dev-all.jsonl)
- **Official Scorer Output**: [evaluation_results.txt](file:///home/pradyuman/HIPE-2026-Team-Hansel-Gretel/strategy5/results/analysis/evaluation_results.txt)
