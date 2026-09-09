"""End-to-end pipeline execution runner.

Loads network-flow data, executes preprocessing and cleaning, extracts
the ML feature matrix, generates temporal sliding windows, and saves
processed outputs to data/processed/ and data/features/.
"""

import argparse
import logging
from pathlib import Path

from ml.preprocessing.clean_data import load_and_clean_dataset
from ml.preprocessing.create_time_windows import create_sliding_time_windows
from ml.preprocessing.feature_engineering import prepare_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NETRA-DataPipeline")


def run_data_pipeline(
    input_csv: str = "data/samples/synthetic_network_flows.csv",
    output_processed: str = "data/processed/cleaned_flows.csv",
    output_features: str = "data/features/features_X.csv",
    output_labels: str = "data/features/labels_y.csv",
    output_windows: str = "data/features/time_windows.csv",
    window_size_sec: int = 60,
    stride_sec: int = 30,
):
    """Execute complete Stage 1 data pipeline."""
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"=== Stage 1: Starting Data Ingestion & Cleaning on {input_path} ===")
    cleaned_df, label_col = load_and_clean_dataset(input_path)

    # Save cleaned dataset
    processed_path = Path(output_processed)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    logger.info(f"Cleaned dataset saved to: {processed_path} (Shape: {cleaned_df.shape})")

    logger.info("=== Stage 2: Feature Matrix Extraction & Validation ===")
    X, y, feature_names = prepare_feature_matrix(cleaned_df, label_col=label_col)

    feat_path = Path(output_features)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(feat_path, index=False)
    y.to_csv(output_labels, index=False)
    logger.info(f"Feature matrix saved to: {feat_path} (Shape: {X.shape}, Features: {len(feature_names)})")

    logger.info("=== Stage 3: Sliding Temporal Window Generation ===")
    windows_df, window_labels_df = create_sliding_time_windows(
        cleaned_df,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
    )

    if not windows_df.empty:
        windows_path = Path(output_windows)
        windows_df.to_csv(windows_path, index=False)
        logger.info(f"Temporal windows saved to: {windows_path} ({len(windows_df)} windows generated)")

    logger.info("=== Stage 1 Data Pipeline Complete: Ready for ML Modeling ===")
    return cleaned_df, X, y, windows_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NETRA Stage 1 Data Pipeline")
    parser.add_argument("--input", default="data/samples/synthetic_network_flows.csv", help="Input CSV path")
    parser.add_argument("--window-size", type=int, default=60, help="Window size in seconds")
    parser.add_argument("--stride", type=int, default=30, help="Stride in seconds")
    args = parser.parse_args()

    run_data_pipeline(
        input_csv=args.input,
        window_size_sec=args.window_size,
        stride_sec=args.stride,
    )
