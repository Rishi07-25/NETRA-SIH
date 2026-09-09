"""PyTorch LSTM Temporal World Model for NETRA Stage 3B.1.

Consumes historical network-state sequences:
    X in R^(N x L x D)
where:
    L = 5 temporal state windows
    D = 19 behavioral network-state features

Projects future threat dynamics at horizon t + H via four decoupled prediction heads:
1. future_attack_binary (2-class classification: benign vs attack)
2. future_attack_type (multi-class canonical attack category)
3. future_attack_stage (multi-class NETRA operational taxonomy stage)
4. future_attack_risk_score (continuous empirical attack density in [0.0, 1.0])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ml.forecasting.schema import (
    STATE_PREDICTIVE_FEATURES,
    TAXONOMY_DISCLAIMER,
    VALID_ATTACK_STAGES,
)

logger = logging.getLogger(__name__)

# Default canonical attack categories codified in Stage 3A
CANONICAL_ATTACK_TYPES: List[str] = [
    "BENIGN",
    "Reconnaissance",
    "Brute Force",
    "DoS",
    "DDoS",
    "Web Attack",
    "Infiltration",
    "Botnet",
    "Heartbleed",
]


class TemporalWorldModel(nn.Module):
    """LSTM-based Multi-Head Temporal World Model."""

    def __init__(
        self,
        input_size: int = 19,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_attack_types: int = len(CANONICAL_ATTACK_TYPES),
        num_attack_stages: int = len(VALID_ATTACK_STAGES),
        attack_types: Optional[List[str]] = None,
        attack_stages: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.feature_names = feature_names or list(STATE_PREDICTIVE_FEATURES)
        self.attack_types = attack_types or list(CANONICAL_ATTACK_TYPES)
        self.attack_stages = attack_stages or list(VALID_ATTACK_STAGES)
        self.num_attack_types = len(self.attack_types)
        self.num_attack_stages = len(self.attack_stages)

        # 1. Temporal Encoder: 2-layer LSTM
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(p=dropout)

        # 2. Prediction Heads
        # Head 1: Binary Attack Classification (2 logits)
        self.head_binary = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 2),
        )

        # Head 2: Attack Type Multi-class Classification
        self.head_type = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, self.num_attack_types),
        )

        # Head 3: Operational Attack Stage Classification
        self.head_stage = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, self.num_attack_stages),
        )

        # Head 4: Attack Risk Score Regression (bounded in [0.0, 1.0] via Sigmoid)
        self.head_risk = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Head 5 (Stage 3B.2): Standardized Next-State Prediction (19 continuous features)
        # Predicts continuous standardized S_(t+1) in R^19; no Sigmoid applied.
        self.head_state = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, input_size),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over sequence tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of head predictions:
            - "embedding": (batch_size, hidden_size)
            - "binary_logits": (batch_size, 2)
            - "type_logits": (batch_size, num_attack_types)
            - "stage_logits": (batch_size, num_attack_stages)
            - "risk_score": (batch_size,) in [0.0, 1.0]
            - "next_state": (batch_size, input_size) continuous standardized state
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch_size, seq_len, input_size), got {x.shape}")
        if x.size(2) != self.input_size:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_size}, got {x.size(2)}"
            )

        # LSTM output: (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final time step hidden state as network-state summary embedding
        last_step_embedding = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        embedding = self.dropout(last_step_embedding)

        binary_logits = self.head_binary(embedding)
        type_logits = self.head_type(embedding)
        stage_logits = self.head_stage(embedding)
        risk_score = self.head_risk(embedding).squeeze(-1)  # shape: (batch_size,)
        next_state = self.head_state(embedding)  # shape: (batch_size, input_size)

        return {
            "embedding": last_step_embedding,
            "binary_logits": binary_logits,
            "type_logits": type_logits,
            "stage_logits": stage_logits,
            "risk_score": risk_score,
            "next_state": next_state,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture metadata."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "num_attack_types": self.num_attack_types,
            "num_attack_stages": self.num_attack_stages,
            "attack_types": self.attack_types,
            "attack_stages": self.attack_stages,
            "feature_names": self.feature_names,
            "taxonomy_disclaimer": TAXONOMY_DISCLAIMER,
        }
