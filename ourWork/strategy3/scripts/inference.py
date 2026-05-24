import json
import torch
import argparse
import os
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast
from model import MTLFineTuneModel

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--tokenizer_name", default="xlm-roberta-base")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and markers
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.tokenizer_name)
    special_tokens_dict = {'additional_special_tokens': ['<E1>', '</E1>', '<E2>', '</E2>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    e1_id = tokenizer.convert_tokens_to_ids("<E1>")
    e2_id = tokenizer.convert_tokens_to_ids("<E2>")

    # Load the 3 distinct models
    languages = ["en", "fr", "de"]
    models = {}
    for lang in languages:
        model = MTLFineTuneModel()
        model.backbone.resize_token_embeddings(len(tokenizer))
        model_path = f"strategy3/results/models/best_ft_model_{lang}.pt"
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.to(device)
            model.eval()
            models[lang] = model
            print(f"Loaded {lang} model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model for {lang} not found at {model_path}. Expect terrible predictions for this language.")
            model.to(device)
            model.eval()
            models[lang] = model

    at_map_rev = {0: "FALSE", 1: "PROBABLE", 2: "TRUE"}
    isAt_map_rev = {0: "FALSE", 1: "TRUE"}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.input, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Running routed inference"):
            doc = json.loads(line.strip())
            marked_pairs = []
            lang = doc.get("language", "unknown").lower()
            
            # Route to the appropriate model, fallback to 'en'
            active_model = models.get(lang, models["en"])
            
            for pair in doc.get("sampled_pairs", []):
                pers_mentions = pair.get("pers_mentions_list", [])
                loc_mentions = pair.get("loc_mentions_list", [])
                
                text = doc["text"]
                # Sorting mentions to avoid partial replacement bugs
                pers_mentions.sort(key=len, reverse=True)
                loc_mentions.sort(key=len, reverse=True)
                
                for pm in pers_mentions:
                    if pm: text = text.replace(pm, f"<E1>{pm}</E1>")
                for lm in loc_mentions:
                    if lm: text = text.replace(lm, f"<E2>{lm}</E2>")
                
                encoding = tokenizer(
                    text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    at_logits, isAt_logits = active_model(encoding["input_ids"], encoding["attention_mask"], e1_id, e2_id)
                
                at_pred_idx = torch.argmax(at_logits, dim=1).item()
                isAt_pred_idx = torch.argmax(isAt_logits, dim=1).item()
                
                at_pred = at_map_rev[at_pred_idx]
                isAt_pred = isAt_map_rev[isAt_pred_idx]
                
                # Rule Injection: at=FALSE implies isAt=FALSE
                if at_pred == "FALSE":
                    isAt_pred = "FALSE"
                
                pair["at"] = at_pred
                pair["isAt"] = isAt_pred
                marked_pairs.append(pair)
            
            doc["sampled_pairs"] = marked_pairs
            f_out.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    run_inference()
