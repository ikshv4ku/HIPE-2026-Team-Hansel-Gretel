"""
Strategy 1.5 — Inference Script

Loads the best MTL model checkpoint from Optuna search and runs it on any
input JSONL file, writing official-format predictions.

Run from repo root:
    source HIPE-2026-data/venv/bin/activate
    python strategy1.5/inference.py \
        --input  HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl \
        --output strategy1.5/results/preds-dev-all.jsonl
"""

import argparse
import json
import os
import sys
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MTLRelationClassifier

# Also need the Strategy 1 DataLoader to extract embeddings on the fly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data_processing import get_dataloader
from transformers import XLMRobertaModel

MODEL_PATH = "strategy1.5/results/best_mtl_model.pt"

AT_MAP_REV   = {0: "FALSE", 1: "PROBABLE", 2: "TRUE"}
ISAT_MAP_REV = {0: "FALSE", 1: "TRUE"}


def run_inference(input_jsonl, output_jsonl, tokenizer_name="xlm-roberta-base"):
    print(f"Loading MTL model from {MODEL_PATH} …")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    params = ckpt["params"]
    model = MTLRelationClassifier(
        input_dim=1536,
        hidden_dim=params["hidden_dim"],
        dropout=0.0,   # No dropout at inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract embeddings on the fly using XLM-RoBERTa (same as Strategy 1)
    print("Setting up XLM-RoBERTa embedder …")
    embedder = XLMRobertaModel.from_pretrained(tokenizer_name)
    dataloader, tokenizer = get_dataloader(
        input_jsonl, batch_size=8, tokenizer_name=tokenizer_name, is_train=False
    )
    embedder.resize_token_embeddings(len(tokenizer))
    embedder.to(device)
    embedder.eval()

    e1_id = tokenizer.convert_tokens_to_ids("<E1>")
    e2_id = tokenizer.convert_tokens_to_ids("<E2>")

    predictions = {}

    print("Running inference …")
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = embedder(input_ids=input_ids, attention_mask=attention_mask)
            hidden         = outputs.last_hidden_state

            bsz = input_ids.size(0)
            for i in range(bsz):
                seq = input_ids[i].cpu().numpy()
                h   = hidden[i]

                e1_idx = np.where(seq == e1_id)[0]
                e2_idx = np.where(seq == e2_id)[0]
                e1_emb = h[e1_idx[0]] if len(e1_idx) > 0 else h[0]
                e2_emb = h[e2_idx[0]] if len(e2_idx) > 0 else h[0]
                feat   = torch.cat([e1_emb, e2_emb]).unsqueeze(0)  # (1, 1536)

                at_logits, isAt_logits = model(feat.to(device))
                pred_at   = at_logits.argmax(dim=1).item()
                pred_isAt = isAt_logits.argmax(dim=1).item()

                # Transitivity rule
                if pred_at == 0:
                    pred_isAt = 0

                doc_id = batch["doc_id"][i]
                if doc_id not in predictions:
                    predictions[doc_id] = []
                predictions[doc_id].append({
                    "pers_entity_id": batch["pers_id"][i],
                    "loc_entity_id":  batch["loc_id"][i],
                    "at":             AT_MAP_REV[pred_at],
                    "isAt":           ISAT_MAP_REV[pred_isAt],
                })

    print(f"Writing predictions to {output_jsonl} …")
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(input_jsonl, encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc["document_id"]
            if doc_id in predictions:
                for pair in doc.get("sampled_pairs", []):
                    for p in predictions[doc_id]:
                        if (p["pers_entity_id"] == pair["pers_entity_id"] and
                                p["loc_entity_id"] == pair["loc_entity_id"]):
                            pair["at"]   = p["at"]
                            pair["isAt"] = p["isAt"]
                            break
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="HIPE-2026-data/data/newspapers/v1.0/splits/HIPE-2026-v1.0-impresso-dev-all.jsonl")
    parser.add_argument("--output", default="strategy1.5/results/preds-dev-all.jsonl")
    args = parser.parse_args()
    run_inference(args.input, args.output)
