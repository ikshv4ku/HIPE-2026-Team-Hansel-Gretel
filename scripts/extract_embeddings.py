import os
import torch
import numpy as np
import pickle
from transformers import XLMRobertaModel
from tqdm import tqdm
import sys

# Ensure data_processing can be imported if run from root or inside scripts/
try:
    from data_processing import get_dataloader
except ImportError:
    from scripts.data_processing import get_dataloader

def extract_features(jsonl_file, output_pkl, tokenizer_name="xlm-roberta-base", batch_size=4, is_train=True, sample_weight=1.0):
    print(f"Extracting features for {jsonl_file}...")
    
    dataloader, tokenizer = get_dataloader(jsonl_file, batch_size=batch_size, tokenizer_name=tokenizer_name, is_train=is_train, sample_weight=sample_weight)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = XLMRobertaModel.from_pretrained(tokenizer_name)
    
    # Critical: Resize embeddings to account for <E1>, </E1>, <E2>, </E2> added by the dataloader
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    model.eval()
    
    # We need the token IDs for <E1> and <E2> to find them in the sequence
    e1_id = tokenizer.convert_tokens_to_ids('<E1>')
    e2_id = tokenizer.convert_tokens_to_ids('<E2>')

    all_features = []
    all_at_labels = []
    all_isAt_labels = []
    all_weights = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state  # (B, L, H)
            
            # Extract E1 and E2 embeddings
            batch_size_actual = input_ids.size(0)
            for i in range(batch_size_actual):
                seq_ids = input_ids[i].cpu().numpy()
                hidden = last_hidden_states[i] # (L, H)
                
                # Find first occurrence of <E1>
                e1_indices = np.where(seq_ids == e1_id)[0]
                if len(e1_indices) > 0:
                    e1_emb = hidden[e1_indices[0]]
                else:
                    e1_emb = hidden[0]  # Fallback to [CLS] token context if cropped
                    
                # Find first occurrence of <E2>
                e2_indices = np.where(seq_ids == e2_id)[0]
                if len(e2_indices) > 0:
                    e2_emb = hidden[e2_indices[0]]
                else:
                    e2_emb = hidden[0]  # Fallback to [CLS] token context if cropped
                    
                # Concatenate 
                feature_vec = torch.cat((e1_emb, e2_emb), dim=0).cpu().numpy()
                all_features.append(feature_vec)
                
                if is_train:
                    all_at_labels.append(batch["at_labels"][i].item())
                    all_isAt_labels.append(batch["isAt_labels"][i].item())
                    all_weights.append(batch["weight"][i].item())
                    
                all_metadata.append({
                    "doc_id": batch["doc_id"][i],
                    "pers_id": batch["pers_id"][i],
                    "loc_id": batch["loc_id"][i]
                })

    # Save to disk
    dataset_dict = {
        "features": np.array(all_features),
        "metadata": all_metadata
    }
    if is_train:
        dataset_dict["at_labels"] = np.array(all_at_labels)
        dataset_dict["isAt_labels"] = np.array(all_isAt_labels)
        dataset_dict["weights"] = np.array(all_weights)
        
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    print(f"Saved {len(all_features)} feature vectors with dimension {len(all_features[0])} to {output_pkl}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/HIPE-2026-data/data/newspapers/v1.0/HIPE-2026-v1.0-impresso-train-de.jsonl")
    parser.add_argument("--output", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/results/embeddings/train_de_features.pkl")
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--weight", type=float, default=1.0)
    args = parser.parse_args()
    
    extract_features(args.input, args.output, is_train=not args.is_test, sample_weight=args.weight)
