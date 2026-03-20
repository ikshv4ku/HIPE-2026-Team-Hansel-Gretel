# HIPE-2026 Strategy & Brainstorming Guide

This document outlines the brainstorming and strategic decisions for the HIPE-2026 Shared Task on Person-Place Relation Extraction, tailored for a balance of **Model Accuracy**, **Model Size (Efficiency)**, and **Generalization**.

---

## 1. Brainstorming Approaches (Simple to Complex)

### Approach A: Heuristic / Traditional ML (TF-IDF + Random Forest)
* **Description**: Extract sentences mentioning both entities, convert to TF-IDF vectors, and engineer features (e.g., word distance between entities).
* **Merits**: Extremely lightweight (a few MBs), lightning-fast inference. Perfect for the Efficiency profile.
* **Demerits**: Fails to capture semantic meaning or temporal nuance. Will struggle significantly on the unseen literary generalization test set.

### Approach B: Frozen Contextual Embeddings (BERT) + Classifier
* **Description**: Pass the text through a frozen, pre-trained transformer (like `xlm-roberta-base`) to extract dense vector embeddings of the `[person]` and `[location]`. Train a LightGBM or Random Forest classifier on these vectors.
* **Merits**: Fast to train because the large transformer isn't updated. Captures deep semantic meaning while keeping the trained classifier tiny.
* **Demerits**: The embeddings are generic and not fine-tuned for relation extraction, limiting peak accuracy.

### Approach C: Full End-to-End Fine-Tuning of a Transformer
* **Description**: Add special marker tokens (e.g., `<E1> Napoleon </E1>` and `<E2> Paris </E2>`) to the text. Fine-tune a multilingual transformer (`xlm-roberta-base` or `mBERT`) to predict the relation directly via a classification head.
* **Merits**: Exceptional accuracy. The model learns exactly what a "person-place" relation looks like across languages. Excellent generalization.
* **Demerits**: Larger model footprint (~270MB to 1GB) and longer training times.

### Approach D: Large Language Models (LLMs) via Few-Shot Prompting
* **Description**: Ask an LLM (like GPT-4, Claude, or a local Llama-3 model) to classify the relation by providing a few examples in the prompt.
* **Merits**: Incredible out-of-the-box spatial and temporal reasoning. Top-tier accuracy.
* **Demerits**: Completely fails the "Efficiency" profile (massive model sizes). Very slow inference.

---

## 2. Handling Multiple Languages
**Verdict: Use a Single Multilingual Model.**
Instead of training three separate models for English, French, and German, we should train one single multilingual model (e.g., `xlm-roberta-base`).
* **Why?** Cross-lingual transfer learning! The model learns what an `isAt` relation looks like in English and automatically applies that knowledge to French and German.
* **Efficiency**: Maintaining one 300MB model is much better for the Efficiency profile than maintaining three 300MB models (900MB total).

---

## 3. Addressing the Professor's Ideas

### 3.1 The Baseline Ensemble (BERT -> RF for AT, LightGBM for IS_AT, NN for both)
* **Explanation**: This approach uses frozen BERT embeddings to power three different models. Since `isAt` is highly dependent on `at` (if `at` is FALSE, `isAt` must be FALSE), training a separate model for each, plus a joint NN, is a robust strategy.
* **How to combine**: You would use "Soft Voting." The neural network (NN) acts as the primary predictor for both labels. Then, you average its probabilities with the Random Forest (for `at`) and LightGBM (for `isAt`) to get a highly stable, consensus decision.

### 3.2 Bi-LSTM based approaches?
* **Explanation**: Bi-LSTMs (Bidirectional Long Short-Term Memory) are older Recurrent Neural Networks used before Transformers (BERT) took over NLP. 
* **Verdict**: We should skip this. Transformers process entire sequences simultaneously and understand context much better than Bi-LSTMs. If we want a lightweight model, a tiny Transformer (like `MiniLM`) is far superior to a Bi-LSTM.

### 3.3 BERT? Will it come in handy?
* **Verdict**: **Absolutely.** BERT (specifically Multilingual BERT or XLM-RoBERTa) should be the absolute backbone of our architecture. 

### 3.4 Ensemble Approaches?
* **Verdict**: Ensembling (averaging the predictions of 3-5 different neural networks) almost always boosts accuracy by 1-3%. **However**, it multiplies your model size! For the Efficiency profile, a massive ensemble is penalized. We should stick to exactly *one* highly optimized model to win the efficiency track, but perhaps use an ensemble for the pure "Accuracy" track.

### 3.5 Soft-labelling instead of hard-labelling?
* **Explanation**: Hard labeling says a relation is `[1, 0, 0]` (100% TRUE). Soft labeling (Label Smoothing) says it's `[0.85, 0.10, 0.05]`.
* **Verdict**: Highly recommended! Historical texts are noisy and ambiguous. Label smoothing prevents the model from becoming overconfident, which dramatically improves its ability to generalize to the unseen literary dataset.

### 3.6 Pretraining on unlabelled/silver-labelled data in the `sandbox/` folder?
* **Explanation**: The organizers provided noisy, automatically generated "silver" data. We can train our BERT model on this massive dataset *first*, and *then* fine-tune it on the tiny, high-quality "gold" training set.
* **Verdict**: **This is a winning strategy!** It is called "Intermediate Task Transfer Learning." It teaches the model the unique, archaic vocabulary and OCR errors of historical newspapers before we ask it to learn the exact relations.

### 3.8 Finetuning and few-shot prompting different LLMs
* **Verdict**: Excellent for the Accuracy profile, but we must be careful. If the organizers strictly score Model Size in GBs, an 8-Billion parameter Llama model will be heavily penalized. If we go this route, we must use heavily quantized models (e.g., 4-bit quantization via LoRA/PEFT) to keep the footprint small.

---

## 4. How to Leverage the Silver-Labelled Data
We can use the `sandbox/` data in two steps:
1. **Masked Language Modeling (MLM)**: Continue pre-training the XLM-RoBERTa model on the raw text so it learns historical spelling variations (e.g., "Müſchen" -> "München").
2. **Silver Fine-Tuning**: Fine-tune the model to predict the relations on the noisy silver data. Then, reset the learning rate to a very small value, and fine-tune it *again* on the clean, gold training data.

---

## 5. How to use Wikidata QIDs
The dataset provides a `QID` (e.g., `Q807` for a location). This uniquely identifies an entity in Wikipedia/Wikidata!
* **Knowledge Injection**: We can literally write a script to query the Wikidata API for every QID. We can fetch the entity's description (e.g., "Paris: capital of France") and its properties (e.g., "Person birth date: 1802").
* **How to feed to the model**: We concatenate this text to the end of our input! 
  * *Original*: `<E1>Napoleon</E1> arrived in <E2>Paris</E2>.`
  * *With Wikidata*: `<E1>Napoleon</E1> arrived in <E2>Paris</E2>. [SEP] Napoleon: French emperor (born 1769). Paris: capital of France.`
* This gives the model external knowledge to easily determine if the person could have possibly been at that location at the time of publication!

---
This 1-month window is a crucial constraint! It means our model can't just guess isAt = TRUE just because a location is mentioned next to a person. The model has to learn to read the context and check if the event occurred recently relative to the newspaper's date metadata.

This is also why I suggested fetching the Wikidata QIDs for the persons to get their birth and death dates—if a person died more than a month before the article was published, we can automatically force the isAt prediction to FALSE (since they definitely can't be at that location currently), which is an easy way to boost your accuracy!

---

## 6. Advanced Agentic & Teacher-Student Frameworks

### 6.1 Distillation from 8B LLMs to 1B-3B LLMs
* **Explanation**: Rather than deploying a massive 8B parameter model, we use it exclusively offline as a "Teacher" to label data and generate rationales. We then train a smaller 1B-3B model (the "Student") to mimic the Teacher's outputs.
* **Merits**: This directly targets the **Efficiency Profile**. We capture the complex reasoning capabilities of an 8B model but retain the lightning-fast inference and tiny memory footprint of a 1B model!

### 6.2 Agentic Framework & Gemma Multilingual Prompt Engineering
* **Explanation**: Instead of a simple one-pass classification, we design an intelligent agent pipeline. The agent determines what it needs (e.g., querying Wikidata, checking dates against the 1-month window) before finalizing its prediction.
* **Language Strategy**: Because models like Google's Gemma architectures are highly multilingual and efficient, we only need to write our prompts and fine-tune our agents in **English**. Gemma's powerful cross-lingual capabilities will naturally understand and process the French and German texts, saving us the massive headache of maintaining translated prompt chains!

### 6.3 Reasoning-First Pipeline (Explain-Then-Predict)
* **Explanation**: The official HIPE-2026 schema allows for optional `at_explanation` and `isAt_explanation` fields. We shouldn't just feed raw labels between models.
* **The Teacher-Student Flow**:
  1. We prompt the Teacher LLM to generate a detailed, logical explanation *before* outputting its final label (Chain-of-Thought prompting).
  2. We train the Student model to output both the reasoning *and* the label.
* **Merits**: Forcing the model to explicitly state its reasoning ("Because this person died 10 years before this paper was published, the relation is FALSE.") drastically reduces hallucinations and improves accuracy on ambiguous edge-cases compared to a classification-only approach.