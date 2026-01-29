"""Main evaluation harness for structured data comparison."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path

from utils.nlp_metrics import calculate_all_metrics
from utils.data_loader import (
    load_dataframe,
    align_columns,
    prepare_comparison_pairs,
)


class StructuredDataEvaluator:
    """Evaluator for comparing generated structured data against ground truth."""

    def __init__(
        self,
        generated_data_path: str,
        ground_truth_path: str,
        column_mapping: Optional[Dict[str, str]] = None,
        match_strategy: str = "index",
        key_column: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            generated_data_path: Path to CSV or Excel file with generated data
            ground_truth_path: Path to CSV or Excel file with ground truth data
            column_mapping: Optional dict mapping generated column names to ground truth names.
                           If None, assumes column names match.
            match_strategy: How to match rows between datasets:
                           - "index": Match by index (default)
                           - "all_pairs": Compare all generated with all ground truth
                           - "truncate": Truncate longer dataset to match shorter one
                           - "key": Match rows by key column value (requires key_column)
            key_column: Column name to use for key-based matching. If provided and match_strategy="key",
                       rows will be matched based on matching values in this column.
        """
        self.generated_data_path = generated_data_path
        self.ground_truth_path = ground_truth_path
        self.column_mapping = column_mapping
        self.match_strategy = match_strategy
        self.key_column = key_column

        # Load dataframes
        self.generated_df = load_dataframe(generated_data_path)
        self.ground_truth_df = load_dataframe(ground_truth_path)

        # Align columns
        (
            self.aligned_generated,
            self.aligned_ground_truth,
            self.column_names,
        ) = align_columns(self.generated_df, self.ground_truth_df, column_mapping)
        
        # Initialize matched_pairs
        self.matched_pairs = None
        
        # Create key-based matching if key_column is provided
        if key_column and match_strategy == "key":
            self._create_key_matching()

    def _create_key_matching(self):
        """Create key-based matching between generated and ground truth dataframes."""
        if self.key_column not in self.generated_df.columns:
            raise ValueError(f"Key column '{self.key_column}' not found in generated data")
        
        # Find the corresponding key column in ground truth
        if self.column_mapping:
            # Check if key_column is mapped
            gt_key_column = None
            for gen_col, gt_col in self.column_mapping.items():
                if gen_col == self.key_column:
                    gt_key_column = gt_col
                    break
            if gt_key_column is None:
                gt_key_column = self.key_column  # Try same name
        else:
            gt_key_column = self.key_column
        
        if gt_key_column not in self.ground_truth_df.columns:
            raise ValueError(f"Key column '{gt_key_column}' not found in ground truth data")
        
        # Create matching based on key values
        gen_keys = self.generated_df[self.key_column].values
        gt_keys = self.ground_truth_df[gt_key_column].values
        
        # Create a mapping: key -> (gen_index, gt_index)
        self.matched_pairs = []
        for gen_idx, gen_key in enumerate(gen_keys):
            # Find matching ground truth rows
            gt_indices = [i for i, gt_key in enumerate(gt_keys) if str(gen_key).strip().lower() == str(gt_key).strip().lower()]
            if gt_indices:
                # Use first match if multiple exist
                self.matched_pairs.append((gen_idx, gt_indices[0], gen_key))
            else:
                # No match found - still include but mark as unmatched
                self.matched_pairs.append((gen_idx, None, gen_key))

    def evaluate_column(
        self,
        column_name: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single column.

        Args:
            column_name: Name of the column to evaluate
            metrics: List of specific metrics to calculate. If None, calculates all metrics.

        Returns:
            Dictionary with evaluation results
        """
        if column_name not in self.column_names:
            raise ValueError(
                f"Column '{column_name}' not found in aligned columns. "
                f"Available columns: {self.column_names}"
            )

        generated_series = self.aligned_generated[column_name]
        ground_truth_series = self.aligned_ground_truth[column_name]

        # Prepare comparison pairs based on match strategy
        if self.match_strategy == "key" and self.matched_pairs is not None:
            # Use key-based matching
            pairs = []
            for gen_idx, gt_idx, key_value in self.matched_pairs:
                if gt_idx is not None:
                    gen_val = generated_series.iloc[gen_idx] if gen_idx < len(generated_series) else None
                    gt_val = ground_truth_series.iloc[gt_idx] if gt_idx < len(ground_truth_series) else None
                    if gen_val is not None and gt_val is not None:
                        pairs.append((gen_val, gt_val))
        else:
            # Use standard matching strategies
            pairs = prepare_comparison_pairs(
                generated_series,
                ground_truth_series,
                match_strategy=self.match_strategy,
            )

        # Calculate metrics for each pair
        all_results = []
        for gen_val, gt_val in pairs:
            if metrics is None:
                result = calculate_all_metrics(gen_val, gt_val)
            else:
                result = {}
                if "cosine_similarity" in metrics:
                    from utils.nlp_metrics import calculate_cosine_similarity

                    result["cosine_similarity"] = calculate_cosine_similarity(
                        gen_val, gt_val
                    )
                if "edit_distance" in metrics:
                    from utils.nlp_metrics import calculate_edit_distance

                    result["edit_distance"] = calculate_edit_distance(gen_val, gt_val)
                if "levenshtein_distance" in metrics:
                    from utils.nlp_metrics import calculate_levenshtein_distance

                    result["levenshtein_distance"] = calculate_levenshtein_distance(
                        gen_val, gt_val
                    )
                if "jaccard_similarity" in metrics:
                    from utils.nlp_metrics import calculate_jaccard_similarity

                    result["jaccard_similarity"] = calculate_jaccard_similarity(
                        gen_val, gt_val
                    )

            all_results.append(result)

        # Aggregate results
        if not all_results:
            return {
                "column_name": column_name,
                "num_comparisons": 0,
                "metrics": {},
            }

        # Calculate statistics for each metric
        aggregated = {"column_name": column_name, "num_comparisons": len(all_results)}

        # Get all metric names
        if metrics is None:
            metric_names = list(all_results[0].keys())
        else:
            metric_names = metrics

        for metric_name in metric_names:
            values = [r[metric_name] for r in all_results if metric_name in r]
            if values:
                aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

        return aggregated

    def evaluate_all_columns(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate all columns.

        Args:
            metrics: List of specific metrics to calculate. If None, calculates all metrics.

        Returns:
            Dictionary with evaluation results for all columns
        """
        results = {}
        for column_name in self.column_names:
            results[column_name] = self.evaluate_column(column_name, metrics)

        # Calculate overall statistics
        overall = self._calculate_overall_stats(results)
        results["overall"] = overall

        return results

    def _calculate_overall_stats(self, column_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall statistics across all columns."""
        overall = {
            "num_columns": len(self.column_names),
            "total_comparisons": sum(
                r.get("num_comparisons", 0) for r in column_results.values()
            ),
        }

        # Aggregate metrics across all columns
        metric_names = set()
        for result in column_results.values():
            if "metrics" in result:
                metric_names.update(result["metrics"].keys())
            else:
                # Extract metric names from the result dict
                for key in result.keys():
                    if key not in ["column_name", "num_comparisons"]:
                        metric_names.add(key)

        for metric_name in metric_names:
            metric_values = []
            for result in column_results.values():
                if metric_name in result and isinstance(result[metric_name], dict):
                    metric_values.append(result[metric_name]["mean"])

            if metric_values:
                overall[metric_name] = {
                    "mean": float(np.mean(metric_values)),
                    "std": float(np.std(metric_values)),
                    "min": float(np.min(metric_values)),
                    "max": float(np.max(metric_values)),
                }

        return overall

    def get_summary_report(self, results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate a summary report as a DataFrame.

        Args:
            results: Evaluation results. If None, runs evaluation first.

        Returns:
            DataFrame with summary statistics
        """
        if results is None:
            results = self.evaluate_all_columns()

        summary_rows = []
        for column_name, column_result in results.items():
            if column_name == "overall":
                continue

            row = {"column": column_name}
            row["num_comparisons"] = column_result.get("num_comparisons", 0)

            # Add metric means
            for key, value in column_result.items():
                if key not in ["column_name", "num_comparisons"]:
                    if isinstance(value, dict) and "mean" in value:
                        row[key] = value["mean"]
                    elif not isinstance(value, dict):
                        row[key] = value

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)
    
    def get_matched_comparisons(self, column_name: str) -> pd.DataFrame:
        """Get detailed comparisons for matched rows based on key column.
        
        Args:
            column_name: Name of the column to compare
            
        Returns:
            DataFrame with matched rows and their metrics
        """
        if self.match_strategy != "key" or self.matched_pairs is None:
            raise ValueError("Key-based matching not enabled. Set match_strategy='key' and provide key_column.")
        
        if column_name not in self.column_names:
            raise ValueError(
                f"Column '{column_name}' not found in aligned columns. "
                f"Available columns: {self.column_names}"
            )
        
        generated_series = self.aligned_generated[column_name]
        ground_truth_series = self.aligned_ground_truth[column_name]
        
        comparison_rows = []
        for gen_idx, gt_idx, key_value in self.matched_pairs:
            if gt_idx is not None:
                gen_val = generated_series.iloc[gen_idx] if gen_idx < len(generated_series) else None
                gt_val = ground_truth_series.iloc[gt_idx] if gt_idx < len(ground_truth_series) else None
                
                if gen_val is not None and gt_val is not None:
                    # Calculate all metrics
                    metrics = calculate_all_metrics(gen_val, gt_val)
                    
                    row = {
                        "key": key_value,
                        "generated_index": gen_idx,
                        "generated_value": gen_val,
                        "ground_truth_index": gt_idx,
                        "ground_truth_value": gt_val,
                        **metrics
                    }
                    comparison_rows.append(row)
        
        if not comparison_rows:
            return pd.DataFrame()
        
        # Create dataframe
        df = pd.DataFrame(comparison_rows)
        
        # Reorder columns for better readability
        metric_cols = [col for col in df.columns if col not in 
                       ["key", "generated_index", "generated_value", "ground_truth_index", "ground_truth_value"]]
        column_order = ["key", "generated_index", "generated_value", "ground_truth_index", "ground_truth_value"] + metric_cols
        df = df[column_order]
        
        return df
