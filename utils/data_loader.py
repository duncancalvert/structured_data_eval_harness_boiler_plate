"""Utilities for loading and preprocessing data files."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def load_dataframe(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV or Excel file
        sheet_name: Name of the sheet to load (for Excel files). If None, loads first sheet.

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def align_columns(
    generated_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    column_mapping: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Align columns between generated and ground truth DataFrames.

    Args:
        generated_df: DataFrame with generated data
        ground_truth_df: DataFrame with ground truth data
        column_mapping: Optional dict mapping generated column names to ground truth column names.
                       If None, assumes column names match.

    Returns:
        Tuple of (aligned_generated_df, aligned_ground_truth_df, column_names)
    """
    if column_mapping is None:
        # Find common columns
        common_columns = list(set(generated_df.columns) & set(ground_truth_df.columns))
        if not common_columns:
            raise ValueError(
                "No common columns found. Please provide column_mapping or ensure column names match."
            )
        column_mapping = {col: col for col in common_columns}
    else:
        # Validate that all mapped columns exist
        for gen_col, gt_col in column_mapping.items():
            if gen_col not in generated_df.columns:
                raise ValueError(f"Column '{gen_col}' not found in generated data")
            if gt_col not in ground_truth_df.columns:
                raise ValueError(f"Column '{gt_col}' not found in ground truth data")

    # Create aligned dataframes
    aligned_generated = pd.DataFrame()
    aligned_ground_truth = pd.DataFrame()

    column_names = []

    for gen_col, gt_col in column_mapping.items():
        aligned_generated[gen_col] = generated_df[gen_col]
        aligned_ground_truth[gen_col] = ground_truth_df[gt_col]
        column_names.append(gen_col)

    return aligned_generated, aligned_ground_truth, column_names


def prepare_comparison_pairs(
    generated_series: pd.Series,
    ground_truth_series: pd.Series,
    match_strategy: str = "index",
) -> List[Tuple[any, any]]:
    """Prepare pairs of values for comparison, handling size mismatches.

    Args:
        generated_series: Series with generated values
        ground_truth_series: Series with ground truth values
        match_strategy: How to match values:
            - "index": Match by index (default)
            - "all_pairs": Compare all generated values with all ground truth values
            - "truncate": Truncate longer series to match shorter one

    Returns:
        List of tuples (generated_value, ground_truth_value)
    """
    if match_strategy == "index":
        # Align by index, filling missing values with NaN
        aligned = pd.DataFrame(
            {
                "generated": generated_series,
                "ground_truth": ground_truth_series,
            }
        )
        pairs = [
            (row["generated"], row["ground_truth"])
            for _, row in aligned.iterrows()
        ]
        return pairs

    elif match_strategy == "all_pairs":
        # Compare every generated value with every ground truth value
        pairs = []
        for gen_val in generated_series:
            for gt_val in ground_truth_series:
                pairs.append((gen_val, gt_val))
        return pairs

    elif match_strategy == "truncate":
        # Truncate to the shorter length
        min_len = min(len(generated_series), len(ground_truth_series))
        pairs = list(
            zip(
                generated_series.iloc[:min_len],
                ground_truth_series.iloc[:min_len],
            )
        )
        return pairs

    else:
        raise ValueError(
            f"Unknown match_strategy: {match_strategy}. "
            "Choose from: 'index', 'all_pairs', 'truncate'"
        )
