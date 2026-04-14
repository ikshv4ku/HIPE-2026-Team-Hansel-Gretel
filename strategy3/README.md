# Strategy 3: Monolingual PEFT Routing

## Architecture & Concept

Following the structural interference experienced in **Strategy 2.5** (where French, German, and English historical dialects clashed violently inside the same attention adapters due to data scarcity and high variance), Strategy 3 implements a **Language-Routed Multi-Adapter Architecture**.

Rather than feeding the model an unsorted multilingual batch, the full `HIPE-2026-v1.0-impresso-train-all.jsonl` dataset (6.1MB, ~14,000 documents including the pseudo-labelled silver data) was partitioned into three distinct, monolingual training pools (`en`, `fr`, `de`).

We independently fine-tuned three separate **LoRA** (Low-Rank Adaptation) adapters over the frozen `xlm-roberta-base` backbone for each language.

During inference, our router (`inference.py`) dynamically reads the language metadata of the evaluated historical document, loads the correct monolingual LoRA adapter into VRAM, and evaluates the text. This prevents cross-lingual gradient confusion while retaining LoRA's robust protection against catastrophic forgetting.

---

## Training Configuration
- **Model**: `xlm-roberta-base` + PEFT / LoRA (Rank: 8, Alpha: 16, Dropout: 0.1)
- **Epochs**: 10 per language
- **Batch Size**: 8
- **Learning Rate**: `1e-4`
- **Hardware**: RTX A5000 
- **Target Dataset**: `HIPE-2026-v1.0-impresso-train-all.jsonl` (6.1MB)
  
---

## Strategy 3 Results and Inferences

**Global Macro Recall:** `0.4232` (vs Strategy 2.5: `0.4550` & Strategy 1.5: `0.6249`)

### Performance Breakdown
* **`at` Relation**
  * Macro Recall: 0.35
  * Notes: High recall for `TRUE` (0.95), but catastrophic failure on `FALSE` (0.09) and `PROBABLE` (0.00), meaning the model is over-predicting `TRUE`.
* **`isAt` Relation**
  * Macro Recall: 0.50
  * Notes: The model failed to predict a single `TRUE` instance for this relation (TRUE Recall: 0.00).

### Analysis
Contrary to expectations, routing inference to distinct monolingual LoRA adapters **degraded** performance compared to the multilingual PEFT approach (Strategy 2.5). 

**Why didn't this work?**
By partitioning the already scarce `HIPE-2026-v1.0` dataset into three separate language pools (`en`, `fr`, `de`), each individual PEFT model suffered severely from data starvation. While Strategy 3 eliminated cross-lingual interference, it also prevented the models from sharing learned representations of named entities and textual structures across languages. 

The catastrophic failure to detect `isAt` (0.00 true recall) and the massive over-prediction of `at` indicates that the monolingual datasets alone are simply too small to allow the LoRA adapters to generalize effectively. The single multilingual model in Strategy 2.5, despite suffering from cross-lingual interference, benefited more from the combined volume of training data.