# HIPE-2026 Shared Task - Step-by-Step Reproduction Guide

This document outlines the exact command-line instructions used to set up the environment, run the baseline prediction scripts, evaluate the results, and perform Exploratory Data Analysis (EDA) on the HIPE-2026 dataset.

## 1. Environment Setup

First, navigate to the dataset repository and set up a Python virtual environment to cleanly install all required dependencies.

```bash
cd HIPE-2026-data
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Running the Random Baseline Prediction

The repository provides a script to generate random predictions as a baseline. Specify the input data and where to save the dummy predictions:

```bash
mkdir -p scripts/tmp
teamname=RANDOM

python scripts/dummy_predict.py \
  --input_path data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl \
  --output_path scripts/tmp/${teamname}_HIPE-2026-v1.0-impresso-train-de.jsonl
```

## 3. Running the Evaluation Scorer

Next, evaluate the generated random predictions against the gold-standard labels using the official scorer provided by the organizers.

```bash
python scripts/file_scorer_evaluation.py \
  --gold_data_file data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl \
  --predictions_file scripts/tmp/${teamname}_HIPE-2026-v1.0-impresso-train-de.jsonl
```
*Note: The actual evaluation results from this run are saved in `../results/evaluation_results.txt`.*

## 4. Testing the Dropout Baseline Script

A custom script was created (`scripts/create_random_baseline_including_dropout.py`) to simulate incomplete predictions (e.g., dropped entity pairs and documents). This tests the robustness of the official scorer.

```bash
cd data/newspapers/v1.0/
python ../../../scripts/create_random_baseline_including_dropout.py
cd ../../../

# Evaluate the dropout predictions
python scripts/file_scorer_evaluation.py \
  --gold_data_file data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl \
  --predictions_file data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de-randomized.jsonl
```

## 5. Exploratory Data Analysis (EDA)

A custom EDA script (`scripts/eda.py`) was created to compute dataset statistics such as the number of documents, average length, unique entities, and label distributions across the languages.

```bash
# Run from the root of your project
python scripts/eda.py
```
*Note: The outputs from this script are saved in `results/eda_results.txt`.*
