"""
Strategy 5: Dataset for Qwen2.5-3B-Instruct Generative Fine-tuning.

Each training sample uses the Qwen2.5 chat format:
  <|im_start|>system\n{system}<|im_end|>\n
  <|im_start|>user\n{user}<|im_end|>\n
  <|im_start|>assistant\n{"at": "...", "isAt": "..."}<|im_end|>

Loss is masked so only the JSON answer tokens contribute to gradient updates.
Gold samples get 5x higher weight; silver samples get soft labels via data noise.
"""

import json
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ─── Prompt template ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert historian specializing in 19th–20th century newspapers. "
    "Your task is to classify person–place relations in historical news articles.\n\n"
    "For each (person, location) pair, respond ONLY with a JSON object:\n"
    '{"at": "<TRUE|PROBABLE|FALSE>", "isAt": "<TRUE|FALSE>"}\n\n'
    "Definitions:\n"
    "- at: Was the person EVER at this location before the publication date?\n"
    "  TRUE = explicit evidence | PROBABLE = implied/inferred | FALSE = no evidence\n"
    "- isAt: Was the person at this location WITHIN ~1 MONTH of the publication date?\n"
    "  TRUE = recent presence confirmed | FALSE = otherwise\n"
    "Rule: if at=FALSE, then isAt MUST be FALSE.\n"
    "Respond ONLY with the JSON object and nothing else."
)


def build_prompt(doc: dict, pair: dict, max_text_chars: int = 3600) -> str:
    """Build the full chat prompt for one (doc, pair). Returns the prompt text only (no answer)."""
    date = doc.get("date", "unknown")
    lang = doc.get("language", "unknown")
    pub  = doc.get("media", {}).get("publication_title", "unknown")
    text = doc.get("text", "")[:max_text_chars]

    person_str   = ", ".join(pair.get("pers_mentions_list", []))
    location_str = ", ".join(pair.get("loc_mentions_list", []))

    user_content = (
        f"Publication date: {date} | Language: {lang} | Source: {pub}\n\n"
        f"PERSON: {person_str}\n"
        f"LOCATION: {location_str}\n\n"
        f"ARTICLE TEXT:\n{text}"
    )

    # Qwen2.5 chat template (applied manually so we can mask prompt tokens)
    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def build_answer(at_label: str, is_at_label: str) -> str:
    return f'{{"at": "{at_label}", "isAt": "{is_at_label}"}}<|im_end|>'


AT_MAP       = {"FALSE": 0, "PROBABLE": 1, "TRUE": 2}
ISAT_MAP     = {"FALSE": 0, "TRUE": 1}
AT_MAP_REV   = {v: k for k, v in AT_MAP.items()}
ISAT_MAP_REV = {v: k for k, v in ISAT_MAP.items()}


class HIPEGemmaDataset(Dataset):
    """
    Generative fine-tuning dataset for Qwen2.5-3B-Instruct.

    Each item:
      input_ids  : full prompt + answer tokens
      labels     : -100 for prompt tokens (masked), answer tokens for loss
      weight     : float, used to scale loss (gold=5, silver=1)
    """

    def __init__(
        self,
        jsonl_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        is_train: bool = True,
        label_smoothing: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.label_smoothing = label_smoothing
        self.samples = []
        self._load(jsonl_file)

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                weight = float(doc.get("split_weight", 1.0))

                for pair in doc.get("sampled_pairs", []):
                    at_raw   = (pair.get("at")   or "FALSE").upper()
                    isat_raw = (pair.get("isAt") or "FALSE").upper()

                    if at_raw   not in AT_MAP:   at_raw   = "FALSE"
                    if isat_raw not in ISAT_MAP: isat_raw = "FALSE"

                    prompt = build_prompt(doc, pair)
                    answer = build_answer(at_raw, isat_raw)

                    self.samples.append({
                        "prompt":   prompt,
                        "answer":   answer,
                        "at_idx":   AT_MAP[at_raw],
                        "isat_idx": ISAT_MAP[isat_raw],
                        "weight":   weight,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt_enc = self.tokenizer(
            s["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 40,
        )
        answer_enc = self.tokenizer(
            s["answer"],
            add_special_tokens=False,
            truncation=True,
            max_length=40,
        )

        input_ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
        # Mask prompt tokens from loss; only compute loss on answer tokens
        labels = [-100] * len(prompt_enc["input_ids"]) + answer_enc["input_ids"]

        total_len = len(input_ids)
        if total_len > self.max_length:
            overflow  = total_len - self.max_length
            input_ids = input_ids[overflow:]
            labels    = labels[overflow:]
        else:
            pad_len   = self.max_length - total_len
            pad_id    = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            input_ids = input_ids + [pad_id] * pad_len
            labels    = labels    + [-100]   * pad_len

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if t != pad_id else 0 for t in input_ids],
                dtype=torch.long,
            ),
            "labels":   torch.tensor(labels,          dtype=torch.long),
            "weight":   torch.tensor(s["weight"],     dtype=torch.float),
            "at_idx":   torch.tensor(s["at_idx"],     dtype=torch.long),
            "isat_idx": torch.tensor(s["isat_idx"],   dtype=torch.long),
        }
