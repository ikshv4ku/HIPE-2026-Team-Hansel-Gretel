import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class MTLFineTuneModel(nn.Module):
    """
    MTL Model with a trainable XLM-RoBERTa backbone.
    Pools information from <E1> and <E2> tokens.
    """
    def __init__(self, model_name="xlm-roberta-base", num_at_labels=3, num_isAt_labels=2, dropout_prob=0.1):
        super(MTLFineTuneModel, self).__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Shared pooling representation (E1 + E2)
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Task heads
        self.at_head = nn.Linear(512, num_at_labels)
        self.is_at_head = nn.Linear(512, num_isAt_labels)

    def forward(self, input_ids, attention_mask, e1_token_id, e2_token_id):
        # Forward through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [B, T, H]
        
        # Find e1 and e2 token positions
        batch_size = input_ids.size(0)
        e1_features = []
        e2_features = []
        
        for i in range(batch_size):
            # Find index of first <E1> and <E2> in input_ids
            # Note: in reality, we should pass these indices from the dataset for speed, 
            # but searching here is safer for varying tokenizations.
            idx1 = (input_ids[i] == e1_token_id).nonzero(as_tuple=True)[0]
            idx2 = (input_ids[i] == e2_token_id).nonzero(as_tuple=True)[0]
            
            # If marker not found (e.g. truncated), default to CLS or mean?
            # Markers should exist as they were injected, but truncation might cut them.
            f1 = last_hidden_state[i, idx1[0]] if len(idx1) > 0 else last_hidden_state[i, 0]
            f2 = last_hidden_state[i, idx2[0]] if len(idx2) > 0 else last_hidden_state[i, 0]
            
            e1_features.append(f1)
            e2_features.append(f2)
            
        e1_features = torch.stack(e1_features)
        e2_features = torch.stack(e2_features)
        
        # Concatenate and pool
        combined = torch.cat([e1_features, e2_features], dim=1)
        shared_out = self.shared_layer(combined)
        
        # Task outputs
        at_logits = self.at_head(shared_out)
        isAt_logits = self.is_at_head(shared_out)
        
        return at_logits, isAt_logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()
