"""Data cleaning and normalization routines for network flow records.

Supports CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, and custom flow formats.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard canonical label dictionary
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    # Benign variations
    "benign": "BENIGN",
    "normal": "BENIGN",
    "0": "BENIGN",
    # Reconnaissance / Scans
    "portscan": "Reconnaissance",
    "port_scan": "Reconnaissance",
    "ipsweep": "Reconnaissance",
    "portsweep": "Reconnaissance",
    "reconnaissance": "Reconnaissance",
    "nmap": "Reconnaissance",
    # Brute Force / Credential attacks
    "ssh-bruteforce": "Brute Force",
    "ftp-patator": "Brute Force",
    "ssh-patator": "Brute Force",
    "bruteforce": "Brute Force",
    "brute_force": "Brute Force",
    # DoS / DDoS
    "dos": "DoS",
    "ddos": "DDoS",
    "ddos-synflood": "DDoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "heartbleed": "Exploitation",
    # Infiltration / Botnet
    "bot": "Botnet",
    "infiltration": "Infiltration",
    "web attack": "Web Attack",
}


def normalize_column_names(columns: List[str]) -> List[str]:
    """Normalize raw dataframe column names into standardized snake_case format.

    Strips whitespace, converts to lowercase, replaces punctuation and spaces with underscores.
    """
    normalized = []
    for col in columns:
        cleaned = str(col).strip().lower()
        cleaned = re.sub(r"[^\w\s]", "_", cleaned)
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        normalized.append(cleaned)
    return normalized


def identify_label_column(df: pd.DataFrame) -> Optional[str]:
    """Dynamically identify the ground-truth attack or label column in the dataframe."""
    candidate_names = [
        "label",
        "attack_cat",
        "attack",
        "target",
        "class",
        "label_name",
        "threat_class",
    ]
    for col in df.columns:
        if col.lower() in candidate_names:
            return col
    # Fallback to fuzzy search
    for col in df.columns:
        if "label" in col.lower() or "attack" in col.lower():
            return col
    return None


def identify_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Dynamically identify the timestamp column in the dataframe."""
    candidates = [
        "timestamp",
        "time",
        "start_time",
        "flow_start_time",
        "sttl",
        "stime",
        "datetime",
    ]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            return col
    return None


def normalize_labels(label_series: pd.Series) -> pd.Series:
    """Normalize polymorphic label representations into standard unified categories."""
    def _map_val(val):
        if pd.isna(val):
            return "BENIGN"
        s_val = str(val).strip().lower()
        # Direct dictionary match
        if s_val in LABEL_NORMALIZATION_MAP:
            return LABEL_NORMALIZATION_MAP[s_val]
        # Partial substring match
        for key, target in LABEL_NORMALIZATION_MAP.items():
            if key in s_val:
                return target
        return str(val).strip()

    return label_series.apply(_map_val)


def clean_network_dataframe(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    drop_duplicates: bool = True,
    numeric_fill_strategy: str = "median",
) -> Tuple[pd.DataFrame, str]:
    """Clean and normalize a raw network-flow DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    label_col : Optional[str]
        Explicit name of the label column. Auto-detected if None.
    timestamp_col : Optional[str]
        Explicit name of the timestamp column. Auto-detected if None.
    drop_duplicates : bool
        Whether to drop identical flow records.
    numeric_fill_strategy : str
        Strategy to replace missing or infinite values ('median', 'mean', or 'zero').

    Returns
    -------
    Tuple[pd.DataFrame, str]
        Cleaned dataframe and identified label column name.
    """
    cleaned = df.copy()

    # 1. Normalize Column Names
    cleaned.columns = normalize_column_names(cleaned.columns)

    # 2. Identify and Preserve Label Column
    detected_label = label_col
    if detected_label is None or detected_label not in cleaned.columns:
        detected_label = identify_label_column(cleaned)

    if detected_label is None:
        raise ValueError("Could not detect a label or attack column in the provided dataset.")

    # 3. Handle Timestamps if present
    detected_time = timestamp_col
    if detected_time is None or detected_time not in cleaned.columns:
        detected_time = identify_timestamp_column(cleaned)

    if detected_time and detected_time in cleaned.columns:
        try:
            cleaned[detected_time] = pd.to_datetime(cleaned[detected_time])
        except Exception as err:
            logger.warning(f"Could not convert timestamp column '{detected_time}' to datetime: {err}")

    # 4. Remove Duplicates
    if drop_duplicates:
        initial_rows = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        dropped_count = initial_rows - len(cleaned)
        if dropped_count > 0:
            logger.info(f"Removed {dropped_count} duplicate rows.")

    # 5. Handle Infinite and NaN Values in Numeric Columns
    for col in cleaned.columns:
        if col == detected_label or col == detected_time:
            continue

        # Try converting string/object numbers into numeric
        if cleaned[col].dtype == object:
            # Check if column is an IP or text
            sample_val = str(cleaned[col].dropna().iloc[0]) if not cleaned[col].dropna().empty else ""
            if "." in sample_val and any(c.isalpha() for c in sample_val):
                continue
            if "." in sample_val and sample_val.count(".") == 3:  # Likely IPv4
                continue
            # Try numeric coercion
            converted = pd.to_numeric(cleaned[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(converted):
                cleaned[col] = converted

        if pd.api.types.is_numeric_dtype(cleaned[col]):
            # Replace infinities with NaN first
            cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)
            if cleaned[col].isna().any():
                if numeric_fill_strategy == "median":
                    fill_val = cleaned[col].median()
                elif numeric_fill_strategy == "mean":
                    fill_val = cleaned[col].mean()
                else:
                    fill_val = 0.0
                fill_val = 0.0 if pd.isna(fill_val) else fill_val
                cleaned[col] = cleaned[col].fillna(fill_val)

    # 6. Normalize Labels
    cleaned[detected_label] = normalize_labels(cleaned[detected_label])

    return cleaned, detected_label


def load_and_clean_dataset(
    filepath: Union[str, Path],
    label_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Load a CSV dataset from disk and apply standard cleaning routines."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    logger.info(f"Loading dataset from {path}")
    # Read with low_memory=False to prevent mixed type warnings
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    cleaned_df, detected_label = clean_network_dataframe(
        df,
        label_col=label_col,
        timestamp_col=timestamp_col,
        drop_duplicates=drop_duplicates,
    )
    logger.info(f"Cleaning complete. Output shape: {cleaned_df.shape}. Label column: '{detected_label}'")
    return cleaned_df, detected_label
