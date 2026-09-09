"""Real Network Intrusion Dataset Ingestion for NETRA Stage 3A.

Supports single CSV files or multi-file directories (such as CIC-IDS2017 multi-day
captures). Normalizes columns, timestamps, labels, and provides comprehensive
data hygiene accounting without silently dropping records.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml.forecasting.schema import (
    CIC_IDS2017_COLUMN_MAP,
    LABEL_NORMALIZATION_MAP,
    UNSW_NB15_COLUMN_MAP,
    map_label_to_attack_stage,
)
from ml.preprocessing.clean_data import normalize_column_names
from ml.preprocessing.feature_engineering import compute_derived_flow_features

logger = logging.getLogger(__name__)


def parse_flexible_timestamps(time_series: pd.Series) -> pd.Series:
    """Parse timestamps from multiple formats (ISO strings, slash dates, epoch s/ms).

    Returns a tz-naive pd.Series of datetime64[ns] with invalid formats as NaT.
    """
    if pd.api.types.is_datetime64_any_dtype(time_series):
        return pd.to_datetime(time_series)

    # Check if numeric epoch timestamps
    if pd.api.types.is_numeric_dtype(time_series):
        sample_val = time_series.dropna().iloc[0] if not time_series.dropna().empty else 0
        if sample_val > 1e11:
            return pd.to_datetime(time_series, unit="ms", errors="coerce")
        elif sample_val > 1e8:
            return pd.to_datetime(time_series, unit="s", errors="coerce")
        else:
            return pd.to_datetime(time_series, unit="s", errors="coerce")

    # String parsing: first inspect if string contains numeric epoch
    first_clean = str(time_series.dropna().iloc[0]).strip() if not time_series.dropna().empty else ""
    if first_clean.replace(".", "", 1).isdigit():
        num_series = pd.to_numeric(time_series, errors="coerce")
        num_val = float(first_clean)
        if num_val > 1e11:
            return pd.to_datetime(num_series, unit="ms", errors="coerce")
        elif num_val > 1e8:
            return pd.to_datetime(num_series, unit="s", errors="coerce")

    # Try standard flexible parsing (handles ISO, dayfirst, monthfirst, mixed)
    try:
        parsed = pd.to_datetime(time_series, errors="coerce", format="mixed")
    except Exception:
        parsed = pd.to_datetime(time_series, errors="coerce", dayfirst=True)
        if parsed.isna().sum() > len(time_series) * 0.5:
            parsed = pd.to_datetime(time_series, errors="coerce", dayfirst=False)

    return parsed


def normalize_dataset_labels(labels: pd.Series) -> pd.Series:
    """Normalize polymorphic raw attack labels into canonical NETRA labels."""
    def _map_single_label(val: Any) -> str:
        if pd.isna(val):
            return "BENIGN"
        s = str(val).strip().lower()
        if s in LABEL_NORMALIZATION_MAP:
            return LABEL_NORMALIZATION_MAP[s]
        for key, target in LABEL_NORMALIZATION_MAP.items():
            if key in s:
                return target
        return str(val).strip()

    return labels.apply(_map_single_label)


class RealDatasetLoader:
    """Loader and normalizer for real-world intrusion datasets (e.g. CIC-IDS2017)."""

    def __init__(self, dataset_format: str = "CIC-IDS2017"):
        self.dataset_format = dataset_format.upper()
        if "CIC" in self.dataset_format:
            self.alias_map = CIC_IDS2017_COLUMN_MAP
        elif "UNSW" in self.dataset_format:
            self.alias_map = UNSW_NB15_COLUMN_MAP
        else:
            self.alias_map = {}

    def load_dataset(
        self,
        source_path: Union[str, Path],
        max_rows: Optional[int] = None,
        drop_duplicates: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and normalize flow records from a file or folder of CSVs.

        Parameters
        ----------
        source_path : str or Path
            Path to single CSV file or directory of CSV files.
        max_rows : Optional[int]
            Optional cap on loaded rows (useful for fast verification).
        drop_duplicates : bool
            Whether to drop duplicate flow records.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Cleaned, chronologically sorted DataFrame and comprehensive audit summary.
        """
        p = Path(source_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset path not found: {p}")

        files: List[Path] = []
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(list(p.glob("*.csv")) + list(p.glob("*.parquet")))
            if not files:
                raise ValueError(f"No CSV or Parquet files found in directory: {p}")
        else:
            raise ValueError(f"Invalid path type for {p}")

        logger.info(f"Loading {len(files)} file(s) from {p}...")
        dfs = []
        total_input_rows = 0

        for f in files:
            if f.suffix.lower() == ".parquet":
                df_chunk = pd.read_parquet(f)
            else:
                df_chunk = pd.read_csv(f, low_memory=False, nrows=max_rows)

            total_input_rows += len(df_chunk)
            dfs.append(df_chunk)
            if max_rows and total_input_rows >= max_rows:
                break

        raw_df = pd.concat(dfs, ignore_index=True)
        if max_rows and len(raw_df) > max_rows:
            raw_df = raw_df.iloc[:max_rows].copy()

        # 1. Normalize Column Names
        raw_df.columns = normalize_column_names(list(raw_df.columns))

        # Apply schema alias mapping
        renamed_cols = {}
        for col in raw_df.columns:
            cleaned_col = col.lower().strip()
            if cleaned_col in self.alias_map:
                renamed_cols[col] = self.alias_map[cleaned_col]
        if renamed_cols:
            raw_df = raw_df.rename(columns=renamed_cols)

        # 2. Identify and Normalize Timestamp Column
        time_col = None
        for candidate in ["timestamp", "time", "stime", "flow_start_time"]:
            if candidate in raw_df.columns:
                time_col = candidate
                break

        if time_col is None:
            raise ValueError("Required timestamp column not found in dataset.")

        raw_df["timestamp"] = parse_flexible_timestamps(raw_df[time_col])
        invalid_timestamps = int(raw_df["timestamp"].isna().sum())
        df_valid = raw_df.dropna(subset=["timestamp"]).copy()

        # 3. Identify and Normalize Label Column
        label_col = None
        for candidate in ["label", "attack_cat", "class", "target"]:
            if candidate in df_valid.columns:
                label_col = candidate
                break

        if label_col is None:
            raise ValueError("Required attack label column not found in dataset.")

        df_valid["label"] = normalize_dataset_labels(df_valid[label_col])

        # 4. Remove Duplicate Rows if requested
        initial_valid = len(df_valid)
        duplicate_count = 0
        if drop_duplicates:
            df_valid = df_valid.drop_duplicates().reset_index(drop=True)
            duplicate_count = initial_valid - len(df_valid)

        # 5. Clean Numeric Values & Infs
        numeric_cols = [c for c in df_valid.columns if c not in {"timestamp", "label", "src_ip", "dst_ip"}]
        invalid_numeric_count = 0
        for col in numeric_cols:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
            # Count infinities/NaNs
            inf_mask = np.isinf(df_valid[col].to_numpy())
            invalid_numeric_count += int(inf_mask.sum())
            df_valid[col] = df_valid[col].replace([np.inf, -np.inf], np.nan)

        # Impute missing numeric values with 0.0 or median
        df_valid[numeric_cols] = df_valid[numeric_cols].fillna(0.0)

        # 6. Compute Derived Ratio Features if prerequisite columns exist
        df_valid = compute_derived_flow_features(df_valid)

        # 7. Sort Chronologically
        df_valid = df_valid.sort_values(by="timestamp").reset_index(drop=True)

        # 8. Compute Operational Attack Stage
        df_valid["attack_stage"] = df_valid["label"].apply(map_label_to_attack_stage)

        time_min = str(df_valid["timestamp"].min()) if not df_valid.empty else "N/A"
        time_max = str(df_valid["timestamp"].max()) if not df_valid.empty else "N/A"

        dropped_rows = total_input_rows - len(df_valid)
        summary: Dict[str, Any] = {
            "source_path": str(source_path),
            "dataset_format": self.dataset_format,
            "total_input_rows": total_input_rows,
            "valid_rows": len(df_valid),
            "dropped_rows": dropped_rows,
            "duplicate_rows": duplicate_count,
            "invalid_timestamp_rows": invalid_timestamps,
            "invalid_numeric_values": invalid_numeric_count,
            "time_range": {"start": time_min, "end": time_max},
            "label_distribution": df_valid["label"].value_counts().to_dict(),
            "attack_stage_distribution": df_valid["attack_stage"].value_counts().to_dict(),
        }

        logger.info(
            f"Dataset loaded: {len(df_valid)} valid rows ({dropped_rows} dropped) across [{time_min} -> {time_max}]."
        )
        return df_valid, summary


def load_network_dataset(
    source_path: Union[str, Path],
    dataset_format: str = "CIC-IDS2017",
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience functional loader for real-world intrusion datasets."""
    loader = RealDatasetLoader(dataset_format=dataset_format)
    return loader.load_dataset(source_path, max_rows=max_rows)
