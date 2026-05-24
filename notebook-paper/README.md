# CLEF HIPE-2026 Notebook Paper — Team Hansel&Gretel

**"From Frozen Encoders to Multi-Agent LLM Pipelines: An Iterative Journey through Person–Place Relation Extraction in Multilingual Historical Texts"**

*Team Hansel&Gretel (Pradyuman Singh Shekhawat & Rohan Gupta) — IIT Roorkee*

---

## Repository Contents

| File | Description |
|---|---|
| `hipe2026_ceurart.tex` | Main LaTeX source (768 lines) |
| `hipe2026_ceurart.pdf` | Pre-compiled PDF — 14 pages, ready to read |
| `references.bib` | BibTeX bibliography (17 entries) |
| `ceurart.cls` | CEUR-WS document class (required to compile) |
| `review_debate_log.md` | Simulated 3-reviewer panel critique and response |

---

## How to Compile

### Prerequisites

You need a LaTeX distribution installed. Choose one:

- **Linux**: TeX Live
  ```bash
  sudo apt-get install texlive-latex-extra texlive-science \
       texlive-fonts-recommended texlive-fonts-extra texlive-publishers
  ```
- **macOS**: [MacTeX](https://www.tug.org/mactex/) (full install recommended)
- **Windows**: [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

> **Note:** The paper uses `pgfplots`, `booktabs`, `tikz`, `multirow`, `tabularx`, `amsmath`, `listings`, and the CEUR `doclicense` package (which requires `ccicons`). All of these are included in a standard TeX Live full or `texlive-latex-extra` + `texlive-fonts-extra` install.

---

### Compile Steps

Run the following commands **in order** from this directory:

```bash
# Step 1 — first LaTeX pass (generates .aux file)
pdflatex hipe2026_ceurart.tex

# Step 2 — resolve bibliography
bibtex hipe2026_ceurart

# Step 3 — second LaTeX pass (embeds references)
pdflatex hipe2026_ceurart.tex

# Step 4 — third LaTeX pass (resolves all cross-references)
pdflatex hipe2026_ceurart.tex
```

The output PDF will be at `hipe2026_ceurart.pdf`.

---

### One-liner with latexmk (easiest)

If you have `latexmk` installed (comes with most TeX Live distributions), it handles all passes automatically:

```bash
latexmk -pdf hipe2026_ceurart.tex
```

To clean up auxiliary files afterwards:

```bash
latexmk -c
```

---

### Expected Output

- **Pages**: 14
- **File size**: ~717 KB
- **Compilation time**: ~30–60 seconds
- **Errors**: 0
- **Warnings**: Minor BibTeX warnings about empty optional fields (`booktitle`, `pages`) — these are harmless and do not affect the rendered output

---

## Troubleshooting

| Error | Fix |
|---|---|
| `File 'ceurart.cls' not found` | Make sure you are running `pdflatex` from this directory, where `ceurart.cls` lives |
| `File 'ccicons.sty' not found` | Run `sudo apt-get install texlive-fonts-extra` (Linux) or update your MiKTeX packages |
| `elsarticle-num-names.bst not found` | Run `sudo apt-get install texlive-publishers` |
| `compat=1.18 unknown` (pgfplots) | Already fixed to `compat=1.16` in this repo; if it still appears, update your TeX Live to 2020+ |
| Citations show as `[?]` | You skipped the `bibtex` step — run all 4 steps in order |
| Cross-references show as `??` | Run `pdflatex` a third time |

---

## Paper Summary

This paper documents Team Hansel&Gretel's complete experimental journey through **9 strategies** for the CLEF HIPE-2026 Shared Task on Person–Place Relation Extraction from multilingual historical documents (German, French, English).

| Run | Strategy | Model | Official Test Score |
|---|---|---|---|
| Run 1 | S5 — Generative SFT | Qwen2.5-3B-Instruct | 0.5458 |
| Run 2 | S7 — Multi-Agent | Qwen2.5-7B-Instruct (3 agents) | 0.5788 |
| **Run 3** | **S9 — CoT + Wikidata** | **GPT-OSS 120B via IBM RITS** | **0.6221 (rank 13/46)** |

Best dev score: **0.7289** (S5, internal evaluation). Best official test score: **0.6221** (S9/Run 3).
