import json
import torch
import numpy as np
import os
import joblib
from transformers import XLMRobertaModel
import sys

try:
    from data_processing import get_dataloader
except ImportError:
    from scripts.data_processing import get_dataloader
    
from tqdm import tqdm

def run_inference(input_jsonl, output_jsonl, model_dir, tokenizer_name="xlm-roberta-base"):
    print(f"Loading trained classifiers from {model_dir}...")
    rf_at = joblib.load(os.path.join(model_dir, "rf_at_model.joblib"))
    lgbm_isAt = joblib.load(os.path.join(model_dir, "lgbm_isAt_model.joblib"))
    
    print(f"Setting up XLM-RoBERTa Embedder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = XLMRobertaModel.from_pretrained(tokenizer_name)
    
    # We load data using is_train=False to disable shuffling and padding
    dataloader, tokenizer = get_dataloader(input_jsonl, batch_size=8, tokenizer_name=tokenizer_name, is_train=False)
    
    # Resize embeddings to handle <E1> and <E2>
    embedder.resize_token_embeddings(len(tokenizer))
    embedder.to(device)
    embedder.eval()
    
    e1_id = tokenizer.convert_tokens_to_ids('<E1>')
    e2_id = tokenizer.convert_tokens_to_ids('<E2>')

    at_map_reverse = {0: "FALSE", 1: "PROBABLE", 2: "TRUE"}
    isAt_map_reverse = {0: "FALSE", 1: "TRUE"}
    
    predictions = {}
    
    print("Extracting features and generating predictions in parallel...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = embedder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            
            batch_size_actual = input_ids.size(0)
            for i in range(batch_size_actual):
                seq_ids = input_ids[i].cpu().numpy()
                hidden = last_hidden_states[i]
                
                e1_indices = np.where(seq_ids == e1_id)[0]
                e1_emb = hidden[e1_indices[0]] if len(e1_indices) > 0 else hidden[0]
                
                e2_indices = np.where(seq_ids == e2_id)[0]
                e2_emb = hidden[e2_indices[0]] if len(e2_indices) > 0 else hidden[0]
                
                feature_vec = torch.cat((e1_emb, e2_emb), dim=0).cpu().numpy().reshape(1, -1)
                
                # Predict relations
                pred_at_idx = rf_at.predict(feature_vec)[0]
                pred_isAt_idx = lgbm_isAt.predict(feature_vec)[0]
                
                # Knowledge Rule Logic Override:
                # If a person is NOT AT a location, they cannot be AT the location within the 1-month window
                if pred_at_idx == 0:  # 0 corresponds to FALSE
                    pred_isAt_idx = 0 # force isAt = FALSE
                
                doc_id = batch["doc_id"][i]
                if doc_id not in predictions:
                    predictions[doc_id] = []
                    
                predictions[doc_id].append({
                    "pers_entity_id": batch["pers_id"][i],
                    "loc_entity_id": batch["loc_id"][i],
                    "at": at_map_reverse[pred_at_idx],
                    "isAt": isAt_map_reverse[pred_isAt_idx]
                })

    print(f"Writing fully compliant JSON Predictions to {output_jsonl}...")
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    with open(input_jsonl, 'r', encoding='utf-8') as fin, open(output_jsonl, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            doc = json.loads(line)
            doc_id = doc["document_id"]
            
            if doc_id in predictions:
                doc_preds = predictions[doc_id]
                for pair in doc.get("sampled_pairs", []):
                    # Match the prediction perfectly back to the pair by entity ID
                    for p in doc_preds:
                        if p["pers_entity_id"] == pair["pers_entity_id"] and p["loc_entity_id"] == pair["loc_entity_id"]:
                            pair["at"] = p["at"]
                            pair["isAt"] = p["isAt"]
                            break
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    print("Inference Pipeline Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl")
    parser.add_argument("--output", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/results/predictions/preds-train-de.jsonl")
    parser.add_argument("--model_dir", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/results/models/")
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model_dir)
