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

*(Training currently in progress... results and metrics will be documented here once the `analyse_results.py` script completes.)*