import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizerFast
import re

class HIPEFineTuneDataset(Dataset):
    """
    Dataset for End-to-End Fine-tuning of XLM-RoBERTa for Relation Extraction.
    Injects <E1>, </E1>, <E2>, </E2> markers and returns raw token IDs.
    """
    def __init__(self, jsonl_file, tokenizer_name="xlm-roberta-base", max_length=512, is_train=True, sample_weight=1.0):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.is_train = is_train
        self.sample_weight = sample_weight
        self.samples = []
        
        # Add special tokens for entities
        special_tokens_dict = {'additional_special_tokens': ['<E1>', '</E1>', '<E2>', '</E2>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self._load_data(jsonl_file)

    def _load_data(self, jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text", "")
                
                for pair in doc.get("sampled_pairs", []):
                    pers_mentions = pair.get("pers_mentions_list", [])
                    loc_mentions = pair.get("loc_mentions_list", [])
                    
                    # Sort by length descending to avoid substring replacement bugs
                    pers_mentions.sort(key=len, reverse=True)
                    loc_mentions.sort(key=len, reverse=True)

                    marked_text = text
                    
                    # Inject <E1> around person mentions
                    for pm in pers_mentions:
                        if pm:
                            escaped_pm = re.escape(pm)
                            marked_text = re.sub(f'(?i)({escaped_pm})', r'<E1>\1</E1>', marked_text)

                    # Inject <E2> around location mentions
                    for lm in loc_mentions:
                        if lm:
                            escaped_lm = re.escape(lm)
                            marked_text = re.sub(f'(?i)({escaped_lm})', r'<E2>\1</E2>', marked_text)

                    # Labels mappings
                    at_label = pair.get("at", "FALSE") if pair.get("at") is not None else "FALSE"
                    isAt_label = pair.get("isAt", "FALSE") if pair.get("isAt") is not None else "FALSE"

                    at_map = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
                    isAt_map = {"FALSE": 0, "TRUE": 1}

                    self.samples.append({
                        "doc_id": doc.get("document_id"),
                        "pers_id": pair.get("pers_entity_id"),
                        "loc_id": pair.get("loc_entity_id"),
                        "text": marked_text,
                        "at_label": at_map.get(at_label, 0),
                        "isAt_label": isAt_map.get(isAt_label, 0),
                        "weight": doc.get("split_weight", self.sample_weight)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add labels and metadata
        item["at_labels"] = torch.tensor(sample["at_label"], dtype=torch.long)
        item["isAt_labels"] = torch.tensor(sample["isAt_label"], dtype=torch.long)
        item["weight"] = torch.tensor(sample["weight"], dtype=torch.float)
        item["doc_id"] = sample["doc_id"]
        item["pers_id"] = sample["pers_id"]
        item["loc_id"] = sample["loc_id"]
        
        return item

def get_dataloader(jsonl_file, batch_size=4, tokenizer_name="xlm-roberta-base", is_train=True, sample_weight=1.0):
    dataset = HIPEFineTuneDataset(jsonl_file, tokenizer_name, is_train=is_train, sample_weight=sample_weight)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train), dataset.tokenizer
