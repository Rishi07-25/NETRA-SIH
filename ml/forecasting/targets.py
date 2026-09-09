"""Future forecast target generation for NETRA Stage 3A.

Constructs future prediction targets over horizon t + H:
- future_attack_binary (0 / 1)
- future_attack_type (canonical attack category)
- future_attack_stage (NETRA operational stage)
- future_attack_risk_score (fraction of attack flows in [0.0, 1.0])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.forecasting.schema import (
    STATE_TARGET_COLUMNS,
    TAXONOMY_DISCLAIMER,
    map_label_to_attack_stage,
)

logger = logging.getLogger(__name__)


def extract_future_targets(
    states_df: pd.DataFrame,
    target_indices: List[int],
) -> pd.DataFrame:
    """Extract and align future targets for sequence observations.

    Parameters
    ----------
    states_df : pd.DataFrame
        Complete chronological DataFrame of network states.
    target_indices : List[int]
        Indices in states_df corresponding to target window t + H for each sequence.

    Returns
    -------
    pd.DataFrame
        DataFrame of aligned future targets:
        - future_attack_binary (int: 0 or 1)
        - future_attack_type (str: canonical attack name)
        - future_attack_stage (str: NETRA operational stage)
        - future_attack_risk_score (float: in [0.0, 1.0])
    """
    if not target_indices:
        return pd.DataFrame(columns=[
            "future_attack_binary",
            "future_attack_type",
            "future_attack_stage",
            "future_attack_risk_score",
        ])

    target_rows = states_df.iloc[target_indices].reset_index(drop=True)

    # 1. Binary target: 1 if attack present, 0 if benign
    if "has_attack" in target_rows.columns:
        binary_target = target_rows["has_attack"].apply(lambda v: 1 if bool(v) else 0)
    elif "dominant_label" in target_rows.columns:
        binary_target = target_rows["dominant_label"].apply(
            lambda lbl: 0 if str(lbl).upper() == "BENIGN" else 1
        )
    else:
        raise ValueError("states_df missing 'has_attack' or 'dominant_label' column.")

    # 2. Attack type
    attack_type = (
        target_rows["dominant_label"].astype(str)
        if "dominant_label" in target_rows.columns
        else pd.Series(["BENIGN"] * len(target_rows))
    )

    # 3. Operational stage
    if "attack_stage" in target_rows.columns:
        attack_stage = target_rows["attack_stage"].astype(str)
    else:
        attack_stage = attack_type.apply(map_label_to_attack_stage)

    # 4. Continuous risk score (ratio of attack flows in [0.0, 1.0])
    if "attack_flow_ratio" in target_rows.columns:
        risk_score = target_rows["attack_flow_ratio"].astype(float).clip(0.0, 1.0)
    else:
        risk_score = binary_target.astype(float)

    targets_df = pd.DataFrame({
        "future_attack_binary": binary_target.to_numpy(dtype=np.int32),
        "future_attack_type": attack_type.to_numpy(),
        "future_attack_stage": attack_stage.to_numpy(),
        "future_attack_risk_score": np.round(risk_score.to_numpy(dtype=np.float32), 4),
    })

    logger.info(
        f"Extracted {len(targets_df)} future targets "
        f"(Attacks: {int(targets_df['future_attack_binary'].sum())}, "
        f"Benign: {int((targets_df['future_attack_binary'] == 0).sum())})."
    )
    return targets_df


def validate_target_alignment(
    X_sequences: np.ndarray,
    targets_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    """Validate strict alignment and temporal separation between sequences and future targets.

    Raises
    ------
    ValueError
        If length mismatch, risk score bounds violated, or temporal overlap detected.
    """
    if len(X_sequences) != len(targets_df):
        raise ValueError(
            f"Row count mismatch between sequences ({len(X_sequences)}) and targets ({len(targets_df)})."
        )

    if len(targets_df) != len(metadata_df):
        raise ValueError(
            f"Row count mismatch between targets ({len(targets_df)}) and metadata ({len(metadata_df)})."
        )

    # Check risk score range
    scores = targets_df["future_attack_risk_score"].to_numpy()
    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("future_attack_risk_score contains values outside [0.0, 1.0].")

    # Check binary target values
    binaries = set(targets_df["future_attack_binary"].unique())
    if not binaries.issubset({0, 1}):
        raise ValueError(f"future_attack_binary contains non-binary values: {binaries}")

    # Check temporal separation: observation_end <= target_start
    if "observation_end" in metadata_df.columns and "target_start" in metadata_df.columns:
        overlap = metadata_df[metadata_df["observation_end"] > metadata_df["target_start"]]
        if not overlap.empty:
            raise ValueError(
                f"Detected temporal overlap in {len(overlap)} sequence(s): observation_end > target_start. "
                "Future information leakage detected."
            )
