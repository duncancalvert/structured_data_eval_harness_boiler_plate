"""NLP text distance and similarity metrics for comparing text data."""

import numpy as np
from typing import Union, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


def preprocess_text(text: Union[str, float, int]) -> str:
    """Preprocess text by converting to string and handling NaN values.

    Args:
        text: Input text (can be string, float, int, or NaN)

    Returns:
        Preprocessed text string
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).strip().lower()


def calculate_cosine_similarity(
    text1: Union[str, float, int],
    text2: Union[str, float, int],
    use_tfidf: bool = True,
) -> float:
    """Calculate cosine similarity between two texts.

    Args:
        text1: First text to compare
        text2: Second text to compare
        use_tfidf: If True, use TF-IDF vectorization; otherwise use simple count vectorization

    Returns:
        Cosine similarity score between 0 and 1
    """
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    if not text1_processed and not text2_processed:
        return 1.0  # Both empty, consider them identical
    if not text1_processed or not text2_processed:
        return 0.0  # One empty, one not

    try:
        vectorizer = TfidfVectorizer() if use_tfidf else TfidfVectorizer(use_idf=False)
        vectors = vectorizer.fit_transform([text1_processed, text2_processed])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)
    except Exception:
        # Fallback to simple character-based comparison
        return 1.0 if text1_processed == text2_processed else 0.0


def calculate_levenshtein_distance(
    text1: Union[str, float, int],
    text2: Union[str, float, int],
) -> int:
    """Calculate Levenshtein (edit) distance between two texts.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Levenshtein distance (number of edits needed)
    """
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    return Levenshtein.distance(text1_processed, text2_processed)


def calculate_edit_distance(
    text1: Union[str, float, int],
    text2: Union[str, float, int],
    normalized: bool = True,
) -> float:
    """Calculate normalized edit distance between two texts.

    Args:
        text1: First text to compare
        text2: Second text to compare
        normalized: If True, return normalized distance (0-1), otherwise raw distance

    Returns:
        Edit distance (normalized or raw)
    """
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    if not text1_processed and not text2_processed:
        return 0.0 if normalized else 0

    max_len = max(len(text1_processed), len(text2_processed))
    if max_len == 0:
        return 0.0 if normalized else 0

    distance = Levenshtein.distance(text1_processed, text2_processed)

    if normalized:
        return distance / max_len
    return float(distance)


def calculate_jaccard_similarity(
    text1: Union[str, float, int],
    text2: Union[str, float, int],
    ngram_size: int = 1,
) -> float:
    """Calculate Jaccard similarity between two texts using n-grams.

    Args:
        text1: First text to compare
        text2: Second text to compare
        ngram_size: Size of n-grams to use (1 for unigrams, 2 for bigrams, etc.)

    Returns:
        Jaccard similarity score between 0 and 1
    """
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)

    if not text1_processed and not text2_processed:
        return 1.0
    if not text1_processed or not text2_processed:
        return 0.0

    def get_ngrams(text: str, n: int) -> set:
        """Extract n-grams from text."""
        if n == 1:
            return set(text.split())
        return set(text[i : i + n] for i in range(len(text) - n + 1))

    set1 = get_ngrams(text1_processed, ngram_size)
    set2 = get_ngrams(text2_processed, ngram_size)

    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def calculate_all_metrics(
    text1: Union[str, float, int],
    text2: Union[str, float, int],
) -> Dict[str, Any]:
    """Calculate all available NLP metrics for two texts.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Dictionary containing all metric scores
    """
    return {
        "cosine_similarity": calculate_cosine_similarity(text1, text2, use_tfidf=False),  # TF only (no IDF)
        "cosine_similarity_tfidf": calculate_cosine_similarity(text1, text2, use_tfidf=True),  # Full TF-IDF
        "levenshtein_distance": calculate_levenshtein_distance(text1, text2),
        "edit_distance": calculate_edit_distance(text1, text2, normalized=False),
        "edit_distance_normalized": calculate_edit_distance(text1, text2, normalized=True),
        "jaccard_similarity_unigram": calculate_jaccard_similarity(text1, text2, ngram_size=1),
        "jaccard_similarity_bigram": calculate_jaccard_similarity(text1, text2, ngram_size=2),
    }
