"""
Strategy 1.5 — Multi-Task Learning Model
Single PyTorch neural network with two classification heads:
  - Head A: at   (3 classes: FALSE, PROBABLE, TRUE)
  - Head B: isAt (2 classes: FALSE, TRUE)

Both heads share a common dense layer on top of the pre-computed
1536-dim XLM-RoBERTa embeddings saved by Strategy 1's extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTLRelationClassifier(nn.Module):
    """
    Multi-Task Learning classifier with two prediction heads.

    Input:  (batch, 1536)  — concatenated E1 + E2 XLM-R embeddings
    Output: tuple of logits
             at_logits   (batch, 3)
             isAt_logits (batch, 2)
    """

    def __init__(self, input_dim=1536, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head A — `at` relation (FALSE=0, PROBABLE=1, TRUE=2)
        self.head_at = nn.Linear(hidden_dim, 3)

        # Head B — `isAt` relation (FALSE=0, TRUE=1)
        self.head_isAt = nn.Linear(hidden_dim, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.shared(x)
        return self.head_at(shared), self.head_isAt(shared)


class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance on isAt."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
