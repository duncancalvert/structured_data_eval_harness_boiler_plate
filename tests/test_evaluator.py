"""Tests for StructuredDataEvaluator."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from eval_harness.evaluator import StructuredDataEvaluator


@pytest.fixture
def sample_generated_data():
    """Create sample generated data file."""
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "London", "Paris"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_ground_truth_data():
    """Create sample ground truth data file."""
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David"],  # Extra row
            "age": [25, 30, 35, 40],
            "city": ["New York", "London", "Paris", "Tokyo"],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)
    yield temp_path
    os.unlink(temp_path)


class TestStructuredDataEvaluator:
    """Tests for StructuredDataEvaluator class."""

    def test_initialization(self, sample_generated_data, sample_ground_truth_data):
        """Test evaluator initialization."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )
        assert evaluator.generated_data_path == sample_generated_data
        assert evaluator.ground_truth_path == sample_ground_truth_data
        assert len(evaluator.column_names) > 0

    def test_initialization_with_mapping(
        self, sample_generated_data, sample_ground_truth_data
    ):
        """Test evaluator initialization with column mapping."""
        # Create data with different column names
        df1 = pd.DataFrame({"gen_name": ["Alice"], "gen_age": [25]})
        df2 = pd.DataFrame({"gt_name": ["Alice"], "gt_age": [25]})

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f1:
            path1 = f1.name
            df1.to_csv(path1, index=False)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f2:
            path2 = f2.name
            df2.to_csv(path2, index=False)

        try:
            mapping = {"gen_name": "gt_name", "gen_age": "gt_age"}
            evaluator = StructuredDataEvaluator(
                generated_data_path=path1,
                ground_truth_path=path2,
                column_mapping=mapping,
            )
            assert len(evaluator.column_names) == 2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_evaluate_column(self, sample_generated_data, sample_ground_truth_data):
        """Test evaluating a single column."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )

        column_name = evaluator.column_names[0]
        results = evaluator.evaluate_column(column_name)

        assert "column_name" in results
        assert "num_comparisons" in results
        assert results["column_name"] == column_name
        assert results["num_comparisons"] > 0

    def test_evaluate_column_invalid(self, sample_generated_data, sample_ground_truth_data):
        """Test evaluating a non-existent column."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )

        with pytest.raises(ValueError):
            evaluator.evaluate_column("nonexistent_column")

    def test_evaluate_all_columns(self, sample_generated_data, sample_ground_truth_data):
        """Test evaluating all columns."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )

        results = evaluator.evaluate_all_columns()

        assert "overall" in results
        for column_name in evaluator.column_names:
            assert column_name in results

    def test_evaluate_with_specific_metrics(
        self, sample_generated_data, sample_ground_truth_data
    ):
        """Test evaluating with specific metrics only."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )

        metrics = ["cosine_similarity", "edit_distance"]
        results = evaluator.evaluate_all_columns(metrics=metrics)

        # Check that only specified metrics are present
        for column_name in evaluator.column_names:
            column_result = results[column_name]
            for key in column_result.keys():
                if key not in ["column_name", "num_comparisons"]:
                    assert key in metrics

    def test_get_summary_report(self, sample_generated_data, sample_ground_truth_data):
        """Test generating summary report."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )

        results = evaluator.evaluate_all_columns()
        summary_df = evaluator.get_summary_report(results)

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(evaluator.column_names)
        assert "column" in summary_df.columns
        assert "num_comparisons" in summary_df.columns

    def test_get_summary_report_no_results(
        self, sample_generated_data, sample_ground_truth_data
    ):
        """Test generating summary report without providing results."""
        evaluator = StructuredDataEvaluator(
            generated_data_path=sample_generated_data,
            ground_truth_path=sample_ground_truth_data,
        )

        summary_df = evaluator.get_summary_report()

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(evaluator.column_names)
