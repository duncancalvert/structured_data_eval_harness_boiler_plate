"""Tests for data loading utilities."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from utils.data_loader import load_dataframe, align_columns, prepare_comparison_pairs


class TestLoadDataframe:
    """Tests for load_dataframe function."""

    def test_load_csv(self):
        """Test loading a CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\nvalue1,value2\nvalue3,value4\n")
            temp_path = f.name

        try:
            df = load_dataframe(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ["col1", "col2"]
        finally:
            os.unlink(temp_path)

    def test_load_excel(self):
        """Test loading an Excel file."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
            df.to_excel(temp_path, index=False)

        try:
            loaded_df = load_dataframe(temp_path)
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) == 2
            assert list(loaded_df.columns) == ["col1", "col2"]
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_dataframe("nonexistent_file.csv")

    def test_unsupported_format(self):
        """Test that ValueError is raised for unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
            f.write(b"test content")

        try:
            with pytest.raises(ValueError):
                load_dataframe(temp_path)
        finally:
            os.unlink(temp_path)


class TestAlignColumns:
    """Tests for align_columns function."""

    def test_common_columns(self):
        """Test aligning columns with common column names."""
        df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})

        aligned1, aligned2, columns = align_columns(df1, df2)
        assert list(columns) == ["col1", "col2"]
        assert list(aligned1.columns) == ["col1", "col2"]
        assert list(aligned2.columns) == ["col1", "col2"]

    def test_column_mapping(self):
        """Test aligning columns with explicit mapping."""
        df1 = pd.DataFrame({"gen_col1": [1, 2], "gen_col2": [3, 4]})
        df2 = pd.DataFrame({"gt_col1": [5, 6], "gt_col2": [7, 8]})

        mapping = {"gen_col1": "gt_col1", "gen_col2": "gt_col2"}
        aligned1, aligned2, columns = align_columns(df1, df2, column_mapping=mapping)
        assert list(columns) == ["gen_col1", "gen_col2"]
        assert list(aligned1.columns) == ["gen_col1", "gen_col2"]
        assert list(aligned2.columns) == ["gen_col1", "gen_col2"]

    def test_no_common_columns(self):
        """Test that ValueError is raised when no common columns exist."""
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col2": [3, 4]})

        with pytest.raises(ValueError):
            align_columns(df1, df2)

    def test_missing_column_in_generated(self):
        """Test that ValueError is raised when mapped column missing in generated data."""
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col2": [3, 4]})

        mapping = {"missing_col": "col2"}
        with pytest.raises(ValueError):
            align_columns(df1, df2, column_mapping=mapping)

    def test_missing_column_in_ground_truth(self):
        """Test that ValueError is raised when mapped column missing in ground truth."""
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col2": [3, 4]})

        mapping = {"col1": "missing_col"}
        with pytest.raises(ValueError):
            align_columns(df1, df2, column_mapping=mapping)


class TestPrepareComparisonPairs:
    """Tests for prepare_comparison_pairs function."""

    def test_index_strategy(self):
        """Test index-based matching strategy."""
        series1 = pd.Series([1, 2, 3], index=[0, 1, 2])
        series2 = pd.Series([4, 5, 6], index=[0, 1, 2])

        pairs = prepare_comparison_pairs(series1, series2, match_strategy="index")
        assert len(pairs) == 3
        assert pairs[0] == (1, 4)
        assert pairs[1] == (2, 5)
        assert pairs[2] == (3, 6)

    def test_index_strategy_mismatched_lengths(self):
        """Test index strategy with mismatched lengths."""
        series1 = pd.Series([1, 2], index=[0, 1])
        series2 = pd.Series([4, 5, 6], index=[0, 1, 2])

        pairs = prepare_comparison_pairs(series1, series2, match_strategy="index")
        assert len(pairs) == 3
        assert pairs[0] == (1, 4)
        assert pairs[1] == (2, 5)
        assert pd.isna(pairs[2][0])  # Missing value filled with NaN

    def test_truncate_strategy(self):
        """Test truncate matching strategy."""
        series1 = pd.Series([1, 2, 3])
        series2 = pd.Series([4, 5])

        pairs = prepare_comparison_pairs(series1, series2, match_strategy="truncate")
        assert len(pairs) == 2
        assert pairs[0] == (1, 4)
        assert pairs[1] == (2, 5)

    def test_all_pairs_strategy(self):
        """Test all-pairs matching strategy."""
        series1 = pd.Series([1, 2])
        series2 = pd.Series([3, 4])

        pairs = prepare_comparison_pairs(series1, series2, match_strategy="all_pairs")
        assert len(pairs) == 4  # 2 * 2 = 4 pairs
        assert (1, 3) in pairs
        assert (1, 4) in pairs
        assert (2, 3) in pairs
        assert (2, 4) in pairs

    def test_invalid_strategy(self):
        """Test that ValueError is raised for invalid strategy."""
        series1 = pd.Series([1, 2])
        series2 = pd.Series([3, 4])

        with pytest.raises(ValueError):
            prepare_comparison_pairs(series1, series2, match_strategy="invalid")
