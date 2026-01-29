"""Tests for NLP metrics functions."""

import pytest
import numpy as np
from utils.nlp_metrics import (
    calculate_cosine_similarity,
    calculate_edit_distance,
    calculate_jaccard_similarity,
    calculate_levenshtein_distance,
    calculate_all_metrics,
)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_strings(self):
        """Test that identical strings have cosine similarity of 1."""
        result = calculate_cosine_similarity("hello world", "hello world")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_different_strings(self):
        """Test that different strings have cosine similarity < 1."""
        result = calculate_cosine_similarity("hello", "world")
        assert 0 <= result < 1

    def test_empty_strings(self):
        """Test handling of empty strings."""
        result = calculate_cosine_similarity("", "")
        assert result == pytest.approx(1.0, abs=0.01)

    def test_one_empty_string(self):
        """Test handling when one string is empty."""
        result = calculate_cosine_similarity("hello", "")
        assert result == pytest.approx(0.0, abs=0.01)

    def test_nan_values(self):
        """Test handling of NaN values."""
        result = calculate_cosine_similarity(float("nan"), float("nan"))
        assert result == pytest.approx(1.0, abs=0.01)

    def test_numeric_values(self):
        """Test handling of numeric values."""
        result = calculate_cosine_similarity(123, 123)
        assert result == pytest.approx(1.0, abs=0.01)


class TestEditDistance:
    """Tests for edit distance calculation."""

    def test_identical_strings_normalized(self):
        """Test that identical strings have normalized edit distance of 0."""
        result = calculate_edit_distance("hello", "hello", normalized=True)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_different_strings_normalized(self):
        """Test normalized edit distance for different strings."""
        result = calculate_edit_distance("hello", "world", normalized=True)
        assert 0 < result <= 1

    def test_identical_strings_raw(self):
        """Test that identical strings have raw edit distance of 0."""
        result = calculate_edit_distance("hello", "hello", normalized=False)
        assert result == 0

    def test_empty_strings(self):
        """Test handling of empty strings."""
        result = calculate_edit_distance("", "", normalized=True)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_one_empty_string(self):
        """Test handling when one string is empty."""
        result = calculate_edit_distance("hello", "", normalized=True)
        assert result == pytest.approx(1.0, abs=0.01)


class TestLevenshteinDistance:
    """Tests for Levenshtein distance calculation."""

    def test_identical_strings(self):
        """Test that identical strings have Levenshtein distance of 0."""
        result = calculate_levenshtein_distance("hello", "hello")
        assert result == 0

    def test_one_character_difference(self):
        """Test Levenshtein distance for one character difference."""
        result = calculate_levenshtein_distance("hello", "hallo")
        assert result == 1

    def test_empty_strings(self):
        """Test handling of empty strings."""
        result = calculate_levenshtein_distance("", "")
        assert result == 0


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""

    def test_identical_strings(self):
        """Test that identical strings have Jaccard similarity of 1."""
        result = calculate_jaccard_similarity("hello world", "hello world", ngram_size=1)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_different_strings(self):
        """Test Jaccard similarity for different strings."""
        result = calculate_jaccard_similarity("hello", "world", ngram_size=1)
        assert 0 <= result < 1

    def test_empty_strings(self):
        """Test handling of empty strings."""
        result = calculate_jaccard_similarity("", "", ngram_size=1)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_bigram_similarity(self):
        """Test Jaccard similarity with bigrams."""
        result = calculate_jaccard_similarity("hello", "hello", ngram_size=2)
        assert result == pytest.approx(1.0, abs=0.01)


class TestAllMetrics:
    """Tests for calculate_all_metrics function."""

    def test_all_metrics_output(self):
        """Test that calculate_all_metrics returns all expected metrics."""
        result = calculate_all_metrics("hello", "world")
        assert "cosine_similarity" in result
        assert "cosine_similarity_tfidf" in result
        assert "levenshtein_distance" in result
        assert "edit_distance" in result
        assert "edit_distance_normalized" in result
        assert "jaccard_similarity_unigram" in result
        assert "jaccard_similarity_bigram" in result

    def test_all_metrics_types(self):
        """Test that all metrics return numeric values."""
        result = calculate_all_metrics("hello", "world")
        for key, value in result.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
