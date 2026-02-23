"""Tests for the embedding canonicalizer.

These tests use the greedy clustering fallback (no hdbscan needed)
and mock the sentence-transformers model to avoid downloading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trustgate.canonicalize.embedding import EmbeddingCanonicalizer, _greedy_cluster


# ---------------------------------------------------------------------------
# Greedy clustering (no external deps)
# ---------------------------------------------------------------------------


class TestGreedyCluster:
    def test_identical_vectors_same_cluster(self) -> None:
        emb = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
        labels = _greedy_cluster(emb, threshold=0.9)
        assert len(set(labels)) == 1

    def test_orthogonal_vectors_different_clusters(self) -> None:
        emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        labels = _greedy_cluster(emb, threshold=0.9)
        assert len(set(labels)) == 3

    def test_two_clusters(self) -> None:
        emb = np.array([
            [1, 0, 0],
            [0.95, 0.05, 0],
            [0, 1, 0],
            [0.05, 0.95, 0],
        ], dtype=float)
        labels = _greedy_cluster(emb, threshold=0.8)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_single_vector(self) -> None:
        emb = np.array([[1, 0, 0]], dtype=float)
        labels = _greedy_cluster(emb, threshold=0.9)
        assert labels == [0]

    def test_empty_array(self) -> None:
        emb = np.zeros((0, 3), dtype=float)
        labels = _greedy_cluster(emb, threshold=0.9)
        assert labels == []


# ---------------------------------------------------------------------------
# EmbeddingCanonicalizer (with mocked model)
# ---------------------------------------------------------------------------


class TestEmbeddingCanonicalizer:
    def test_import_error_without_package(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                EmbeddingCanonicalizer()

    def test_canonicalize_batch_similar(self) -> None:
        """Similar answers should cluster together."""
        mock_model = MagicMock()
        # Return near-identical embeddings for similar answers
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
        ])

        with patch("trustgate.canonicalize.embedding.EmbeddingCanonicalizer.__init__", return_value=None):
            canon = EmbeddingCanonicalizer.__new__(EmbeddingCanonicalizer)
            canon.model = mock_model
            canon.min_cluster_size = 2

        labels = canon.canonicalize_batch("Q?", ["answer1", "answer2", "answer3"])
        assert len(labels) == 3
        # All should be in the same cluster
        assert labels[0] == labels[1] == labels[2]

    def test_canonicalize_batch_different(self) -> None:
        """Different answers should get different labels."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        with patch("trustgate.canonicalize.embedding.EmbeddingCanonicalizer.__init__", return_value=None):
            canon = EmbeddingCanonicalizer.__new__(EmbeddingCanonicalizer)
            canon.model = mock_model
            canon.min_cluster_size = 2

        labels = canon.canonicalize_batch("Q?", ["cat", "dog", "fish"])
        assert len(labels) == 3
        assert len(set(labels)) == 3  # all different

    def test_canonicalize_batch_single(self) -> None:
        """Single answer should return a label without error."""
        with patch("trustgate.canonicalize.embedding.EmbeddingCanonicalizer.__init__", return_value=None):
            canon = EmbeddingCanonicalizer.__new__(EmbeddingCanonicalizer)
            canon.model = MagicMock()
            canon.min_cluster_size = 2

        labels = canon.canonicalize_batch("Q?", ["only answer"])
        assert labels == ["cluster_0"]

    def test_canonicalize_single_returns_text(self) -> None:
        """The single-answer canonicalize() returns preprocessed text."""
        with patch("trustgate.canonicalize.embedding.EmbeddingCanonicalizer.__init__", return_value=None):
            canon = EmbeddingCanonicalizer.__new__(EmbeddingCanonicalizer)

        assert canon.canonicalize("Q?", "  hello world  ") == "hello world"

    def test_canonicalize_empty(self) -> None:
        with patch("trustgate.canonicalize.embedding.EmbeddingCanonicalizer.__init__", return_value=None):
            canon = EmbeddingCanonicalizer.__new__(EmbeddingCanonicalizer)

        assert canon.canonicalize("Q?", "") == "empty"
