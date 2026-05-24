# HIPE-2026 — Team Hansel & Gretel

**Team 9 (Hansel & Gretel)** — [HIPE-2026 Shared Task](https://hipe-eval.github.io/HIPE-2026/) on Person–Place Relation Extraction from Historical Newspapers.

**Authors:**
- Pradyuman Singh Shekhawat — pradyuman_ss@cs.iitr.ac.in
- Rohan Gupta — rohan_g@cs.iitr.ac.in

**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Technology Roorkee

---

## Repository Structure

```
.
├── problem_statement/       # Conference details & problem definition
│   └── baseline/            # Organiser-provided baseline model
├── HIPE-2026-data/          # Official dataset
├── ourWork/                 # Our strategies & experiments
├── submission/              # Final submitted runs
├── evaluation/              # Evaluation framework & scripts
│   └── hipe-2026-eval/      # Official evaluation repository
├── results/                 # Published evaluation results
├── notebook-paper/          # Working notes paper
│   └── CLEF2026-template/   # CEUR-WS LaTeX template
└── README.md
```

---

### 1. `problem_statement/`

Contains documentation about the HIPE-2026 shared task, including the official guidelines, the exact problem statement (Person–Place relation extraction from historical newspaper articles), and our exploratory data analysis notes.

- `about-hipe.txt` — Overview of the HIPE-2026 conference and task description.
- `hipe-2026-official-guidlines.txt` — Full official guidelines from the organisers.
- `eda_results.txt` — Exploratory data analysis findings.
- `reproduction_steps.md` — Steps to reproduce the baseline.

#### 1.1 `problem_statement/baseline/`

Contains the **HIPE-2026 LLM Baseline** provided by the organisers (`HIPE-2026-llm-baseline`). This is the reference zero-shot LLM-based system that all participant systems are compared against.

---

### 2. `HIPE-2026-data/`

The official dataset provided by the HIPE-2026 organisers. Data is in JSONL format and organized as follows:

| Split | Source | Description |
|-------|--------|-------------|
| **Gold (Train)** | `data/newspapers/` | Gold-standard annotated newspaper articles (DE, EN, FR) |
| **Silver (Sandbox)** | `data/sandbox/` | Silver-standard sandbox data for development |
| **Test** | `official_test_unlabeled/` | Unlabeled test sets — both newspaper (impresso) and surprise domain |
| **Samples** | `data/sample_submissions/` | Example submission files for format reference |

Refer to `HIPE-2026-data/README.md` for full schema documentation and data format details.

---

### 3. `ourWork/`

Contains all the strategies we developed and deployed during the shared task. Each strategy is in its own subdirectory:

| Strategy | Description |
|----------|-------------|
| `planning/` | Initial planning and approach design |
| `strategy1/` – `strategy9/` | Iterative experimental strategies |

Our final submitted runs were based on:
- **Run 1 (Strategy 5):** Fine-tuned Qwen2.5-3B-Instruct with LoRA (~3B params)
- **Run 2 (Strategy 7):** Multi-agent fine-tuned Qwen2.5-7B-Instruct with 3 LoRA adapters (~7B params)
- **Run 3 (Strategy 9):** Zero-training few-shot GPT-OSS 120B with Wikidata knowledge injection (~120B params)

---

### 4. `submission/`

Contains our final submitted prediction files and related documentation.

- **12 prediction files** (4 test sets × 3 runs) in JSONL format, validated against the official schema.
- `submission-strategy.txt` — Details of each submitted run and the submission confirmation.
- `submission-confirmation.pdf` — Official receipt from the organisers.

Test sets covered: `impresso-test-de`, `impresso-test-en`, `impresso-test-fr`, `surprise-test-fr`.

---

### 5. `evaluation/`

Documentation and tools related to how the organisers carried out evaluations.

- `about-evaluation.txt` — Summary of the evaluation methodology.
- `hipe-2026-eval/` — The **official HIPE-2026 evaluation repository** containing evaluation scripts, reference data, all team submissions, and the Makefile-based evaluation pipeline.

---

### 6. `results/`

Final evaluation results as published by the organisers. **Our team is Team 9 (Hansel & Gretel).**

- `about-results` — Overview and interpretation of the results.
- `official_evaluation_report.txt` — Full official evaluation report across all teams and metrics.
- `binary-at-evaluation-report.txt` — Binary @-mention evaluation results.

---

### 7. `notebook-paper/`

Working notes paper for CLEF 2026, describing our system, experiments, and results.

- `hipe2026_ceurart.tex` — Our paper source (LaTeX).
- `hipe2026_ceurart.pdf` — Compiled PDF.
- `references.bib` — Bibliography.
- `guidelines.txt` — Paper writing guidelines from the organisers.

#### 7.1 `notebook-paper/CLEF2026-template/`

The official **CEUR-WS one-column LaTeX template** (`ceurart`) that must be used for formatting the working notes paper. See its `README.md` for usage instructions.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ikshv4ku/HIPE-2026-Team-Hansel-Gretel.git
cd HIPE-2026-Team-Hansel-Gretel

# Explore the problem statement
cat problem_statement/about-hipe.txt

# Look at our strategies
ls ourWork/

# Check the results (we are Team 9)
cat results/official_evaluation_report.txt
```

---

## Acknowledgements

This work was carried out as part of the [HIPE-2026 Shared Task](https://hipe-eval.github.io/HIPE-2026/) at [CLEF 2026](https://clef2026.clef-initiative.eu/), organized by the Digital Humanities Lab (EPFL) and the Institute of Computational Linguistics (University of Zürich).
