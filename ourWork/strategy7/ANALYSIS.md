# Strategy 7: Comprehensive Analysis & Cross-Strategy Comparison

**Team Hansel & Gretel | Date: April 2026**

---

## 1. Executive Summary

Strategy 7 implements a **Multi-Agent Specialist Architecture** using a **Qwen2.5-7B-Instruct** backbone with three LoRA-fine-tuned agents (Historian, Geographer, Arbiter). It achieves a **Global Macro Recall of 0.7280**, representing a significant improvement over most prior strategies and a competitive result within the project's trajectory.

| Metric | Value |
|--------|-------|
| **Global Macro Recall** | **0.7280** |
| `at` Macro Recall | 0.6742 |
| `isAt` Macro Recall | 0.7817 |
| `at` Accuracy | 69.29% (176/254) |
| `isAt` Accuracy | 91.34% (232/254) |
| Total Correct Pairs | 408 / 508 |

---

## 2. Cross-Strategy Performance Comparison

### 2.1 Results Summary Table

| Strategy | Architecture | Base Model | Global MR | `at` MR | `isAt` MR | Δ vs Previous |
|----------|-------------|-----------|-----------|---------|-----------|---------------|
| **Baseline** | Dummy random | — | 0.4242 | 0.3211 | 0.5273 | — |
| **S1** | XLM-R frozen + RF/LGBM | xlm-roberta-base (278M) | 0.6018 | 0.5200 | 0.6835 | +0.1776 |
| **S1.5** | XLM-R frozen + RF/LGBM (tuned) | xlm-roberta-base (278M) | 0.6249 | 0.5486 | 0.7013 | +0.0231 |
| **S3** | Monolingual LoRA routing | xlm-roberta-base (278M) | 0.4232 | — | — | −0.2017 |
| **S4** | MTL LoRA fine-tune | Qwen2.5-0.5B (500M) | 0.6264 | 0.5313 | 0.7111 | +0.2032 |
| **S4.5** | MTL LoRA + soft labels | Qwen2.5-0.5B (500M) | 0.6212 | 0.5313 | 0.7111 | −0.0052 |
| **S5** | Generative SFT (single model) | Qwen2.5-3B-Instruct (3B) | 0.7289 | 0.6872 | 0.7706 | +0.1077 |
| **S6** | Multi-agent (3 specialists) | Qwen2.5-3B-Instruct (3B) | 0.7204 | 0.7019 | 0.7388 | −0.0085 |
| **S7** | Multi-agent + 7B + class weights | **Qwen2.5-7B-Instruct (7B)** | **0.7280** | 0.6742 | **0.7817** | +0.0076 |

### 2.2 Performance Progression Chart

```
Global Macro Recall Progression
═══════════════════════════════════════════════════════════
Baseline ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.4242
S1       ████████████████████████░░░░░░░░░░░░░░░░  0.6018
S1.5     █████████████████████████░░░░░░░░░░░░░░░  0.6249
S3       ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.4232
S4       █████████████████████████░░░░░░░░░░░░░░░  0.6264
S4.5     ████████████████████████░░░░░░░░░░░░░░░░  0.6212
S5       ████████████████████████████░░░░░░░░░░░░  0.7289
S6       ████████████████████████████░░░░░░░░░░░░  0.7204
S7       ████████████████████████████░░░░░░░░░░░░  0.7280
═══════════════════════════════════════════════════════════
         0.0    0.2    0.4    0.6    0.8    1.0
```

---

## 3. Strategy Evolution & Architectural Analysis

### Phase 1: Feature Extraction (S1 – S1.5)
**Approach**: Frozen XLM-RoBERTa used as a feature extractor. Entity marker tokens (`<E1>`, `<E2>`) were injected into documents, and concatenated hidden states were fed to classical classifiers (RandomForest for `at`, LightGBM for `isAt`).

**Key Insight**: This approach established a strong baseline (0.6018 → 0.6249) by leveraging pre-trained multilingual representations without fine-tuning, avoiding overfitting on the small gold dataset (104 documents). The class-weighted sampling and calibrated decision boundaries in S1.5 provided marginal but meaningful gains.

### Phase 2: Discriminative Fine-Tuning (S3 – S4.5)
**Approach**: Moved to LoRA fine-tuning of transformer backbones with classification heads.

- **S3** (Monolingual LoRA Routing): Catastrophic regression to 0.4232. Splitting an already small dataset by language created severe data scarcity per adapter.
- **S4** (Qwen2.5-0.5B MTL): Recovered to 0.6264 by using a single multilingual adapter with multi-task heads. The move from encoder-only (XLM-R) to decoder-only (Qwen) was architecturally lateral.
- **S4.5** (Soft Labels): No improvement (0.6212). The soft labeling scheme added noise without enough signal to compensate.

**Key Insight**: Discriminative approaches plateau around ~0.63 on this task. The small dataset and inherent label ambiguity (especially for PROBABLE) limit supervised classification heads.

### Phase 3: Generative SFT (S5)
**Approach**: Paradigm shift to **generative classification** — the model generates structured JSON outputs containing both predictions and reasoning. Used Qwen2.5-3B-Instruct with LoRA SFT, full document context (no 512-token truncation), and answer-only loss masking.

**Key Insight**: This produced the largest single-step improvement (+0.1077 to 0.7289). Generating reasoning alongside predictions forces the model to develop deeper document understanding, and the instruction-tuned backbone contributes strong zero-shot multilingual reasoning capability.

### Phase 4: Multi-Agent Architecture (S6 – S7)
**Approach**: Split the generative model into three specialized agents:
- **Historian**: Focuses on temporal/biographical reasoning
- **Geographer**: Focuses on spatial/geographic reasoning  
- **Arbiter**: Synthesizes both expert opinions into a final decision

#### S6 (3B, no class weights, zero-shot prompts)
- Global MR: 0.7204 — slight regression from S5's single-model approach
- **Critical Issue**: Agent agreement was 246/254 with hist_TRUE=0, geo_TRUE=0, meaning specialists collapsed to always predicting FALSE and the Arbiter was essentially making all decisions unilaterally
- The specialist models lacked diversity in their predictions

#### S7 (7B, 3× class weights, 3-shot prompts)
- Global MR: 0.7280 — recovered and exceeded S6
- **Key Improvements over S6**:
  1. **Model Scale**: 7B vs 3B backbone provides richer reasoning and multilingual capability
  2. **Class-Weighted Loss** (3× for TRUE/PROBABLE): Fixed the specialist collapse — agents now produce meaningful TRUE predictions (hist_TRUE=53, geo_TRUE=47)
  3. **3-Shot Prompts**: Grounded examples in each agent's system prompt improve calibration and output formatting consistency
  4. **Larger Context Window**: max_length increased from 1024 to 2048 tokens

---

## 4. Why Strategy 7 Improved Over S6

### 4.1 Specialist Collapse Fixed
The most critical improvement. In S6, both Historian and Geographer always predicted FALSE (hist_TRUE=0, geo_TRUE=0), making the Arbiter the sole decision-maker. The 3× class-weighted loss in S7 forced specialists to learn meaningful TRUE/PROBABLE patterns:

| Metric | S6 | S7 |
|--------|----|----|
| Historian TRUE predictions | 0 | 53 |
| Geographer TRUE predictions | 0 | 47 |
| Agent agreement cases | 246/254 | 236/254 |
| Disagreement (Arbiter needed) | 8/254 | 18/254 |

The increased disagreement rate (3.1% → 7.1%) indicates genuine specialist diversity, allowing the Arbiter to serve its intended role.

### 4.2 Model Scale (3B → 7B)
The Qwen2.5-7B-Instruct model provides:
- **Better multilingual understanding** (critical for DE/EN/FR historical texts)
- **Stronger reasoning chains** in generated JSON explanations
- **More robust few-shot learning** from the 3-shot prompt examples
- **Higher capacity LoRA adapters** that can encode more nuanced patterns

### 4.3 Few-Shot Prompting
S6 used zero-shot system prompts. S7 added 3 carefully crafted examples per agent covering:
- TRUE case (explicit evidence — Zola in Paris)
- PROBABLE case (implied evidence — Bismarck and Vienna)
- FALSE case (no evidence — Hugo and London)

Each example is tailored to the agent's reasoning lens (temporal for Historian, spatial for Geographer), improving output calibration.

### 4.4 isAt Improvement
S7's isAt macro recall (0.7817) exceeds all other strategies, including S5 (0.7706). The multi-agent architecture appears particularly effective at temporal reasoning (is the person at the location *right now*?), where the Historian's biographical expertise complements the Geographer's spatial analysis.

---

## 5. Why S7 Slightly Trails S5

Despite architectural improvements, S7's Global MR (0.7280) is marginally below S5 (0.7289) — a difference of just **0.0009**. Analysis of the specific sub-metrics reveals:

| Metric | S5 | S7 | Δ |
|--------|----|----|---|
| `at` MR | **0.6872** | 0.6742 | −0.0130 |
| `isAt` MR | 0.7706 | **0.7817** | +0.0111 |
| Global MR | **0.7289** | 0.7280 | −0.0009 |

- S7 **wins on isAt** (+0.0111) but **loses on at** (−0.0130)
- The `at` regression is likely due to the multi-agent pipeline introducing additional inference steps where parsing errors or conservative Arbiter defaults can flip borderline PROBABLE predictions to FALSE
- S5's single-model approach has fewer failure modes in the inference chain

---

## 6. Recommendations for Further Improvement

### 6.1 High-Impact (Likely to push past 0.75)

1. **Ensemble S5 + S7**: Combine single-model and multi-agent predictions via majority voting or confidence-weighted fusion. S5 is better on `at`, S7 on `isAt` — an ensemble could capture the best of both.

2. **Arbiter Training Data Quality**: Currently the Arbiter trains on Historian/Geographer predictions against gold labels. Augmenting with *contrastive* examples (where one agent is right and the other wrong) would improve the Arbiter's conflict resolution capability.

3. **Curriculum Training for Specialists**: Train specialists in two phases — first on easy/obvious cases (strong TRUE/FALSE signals), then on ambiguous PROBABLE cases. This prevents early loss from noisy gradients.

4. **Per-Language Specialist Tuning**: Add language-conditioned LoRA layers or train separate adapters per language within each specialist, given the significant performance variance across DE/EN/FR.

### 6.2 Medium-Impact (Potential 0.01–0.03 gain)

5. **Increase Arbiter Disagreement Exposure**: Only 18/254 cases (7.1%) trigger the Arbiter's conflict resolution. Artificially introducing disagreement during training (e.g., randomly flipping one specialist's prediction) would make the Arbiter more robust.

6. **Temperature-Scaled Inference**: Use different temperatures for specialists (higher = more diverse predictions) vs. the Arbiter (lower = more conservative synthesis).

7. **Wikidata Hard Rules at Inference**: S5 applied Wikidata-based post-processing rules. Adding these to S7's pipeline could catch systematic errors (e.g., deceased persons cannot have isAt=TRUE).

8. **LoRA Rank/Alpha Tuning**: Experiment with higher LoRA ranks (16 or 32 instead of 8) to give specialists more parameter capacity for nuanced pattern encoding.

### 6.3 Exploratory (Research-level)

9. **Self-Consistency Decoding**: Generate multiple reasoning chains per agent and take the majority vote prediction. This reduces variance from single-sample generation.

10. **Iterative Debate**: Allow Historian and Geographer to see each other's reasoning and revise their predictions in a second round before the Arbiter synthesizes.

11. **RAG-Augmented Agents**: Give the Historian access to an external biographical knowledge base and the Geographer access to a historical gazetteer, enabling factual grounding beyond the article text.

---

## 7. Training Configuration (Strategy 7)

### 7.1 Model & LoRA
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-7B-Instruct |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Target Modules | q_proj, v_proj |
| TRUE/PROBABLE Loss Weight | 3.0× |

### 7.2 Training
| Parameter | Value |
|-----------|-------|
| Epochs (Historian/Geographer) | 2 |
| Epochs (Arbiter) | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-5 |
| Scheduler | Cosine with Warmup |
| Max Sequence Length | 2048 |
| Hardware | 3× NVIDIA RTX A5000 (24GB each) |

### 7.3 Data
| Dataset | Samples |
|---------|---------|
| Historian SFT | 15,418 |
| Geographer SFT | 15,418 |
| Arbiter SFT | 15,418 (merged from 2 shards) |

### 7.4 Inference
- **device_map="auto"**: 7B model sharded across all 3 GPUs
- **Multi-adapter loading**: All 3 LoRA adapters loaded simultaneously
- **Pipeline**: Historian → Geographer → Arbiter (sequential per document)
- **Total inference time**: ~30 minutes for 21 dev documents

---

## 8. Conclusion

Strategy 7 represents the culmination of the multi-agent approach for HIPE-2026 relation extraction. By scaling from 3B to 7B parameters, introducing class-weighted loss to fix specialist collapse, and adding few-shot prompt examples, it achieves a **Global Macro Recall of 0.7280** — competitive with the best single-model result (S5: 0.7289) while demonstrating the potential of multi-agent architectures for structured NLP tasks.

The multi-agent framework's primary value lies in its **interpretability** (each agent provides explicit reasoning) and **modularity** (individual specialists can be improved independently). The tight score with S5 suggests that an **ensemble** of both approaches is the most promising path to exceeding 0.75.

---

*Generated: 2026-05-01 | Strategy 7 evaluation pipeline completed successfully.*
