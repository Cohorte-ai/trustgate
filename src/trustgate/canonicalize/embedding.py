"""Embedding-based clustering canonicalization."""

from __future__ import annotations

from trustgate.canonicalize import Canonicalizer, register_canonicalizer


@register_canonicalizer("embedding")
class EmbeddingCanonicalizer(Canonicalizer):
    """Cluster answers by semantic similarity using embeddings + HDBSCAN.

    Requires the ``trustgate[embedding]`` extra (sentence-transformers, hdbscan).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 2,
        **kwargs: object,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "EmbeddingCanonicalizer requires sentence-transformers. "
                "Install with: pip install 'trustgate[embedding]'"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.min_cluster_size = min_cluster_size

    def canonicalize(self, question: str, answer: str) -> str:
        """Single-answer fallback — returns the preprocessed text as its own label."""
        return self.preprocess(answer) or "empty"

    def canonicalize_batch(self, question: str, answers: list[str]) -> list[str]:
        """Cluster a batch of answers and return cluster labels.

        Each label is ``"cluster_N"`` where N is the cluster index.
        Noise points (HDBSCAN label -1) get unique singleton labels.
        """
        import numpy as np

        cleaned = [self.preprocess(a) or "empty" for a in answers]

        if len(cleaned) <= 1:
            return ["cluster_0"] * len(cleaned)

        embeddings = self.model.encode(cleaned, show_progress_bar=False)

        try:
            import hdbscan as hdbscan_mod  # type: ignore[import-not-found]

            clusterer = hdbscan_mod.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric="cosine",
            )
            labels: list[int] = list(clusterer.fit_predict(embeddings))
        except ImportError:
            labels = _greedy_cluster(np.asarray(embeddings), threshold=0.8)

        # Convert int labels to string labels; rename noise (-1) to unique singletons
        result: list[str] = []
        noise_counter = 0
        max_label = max(labels) if len(labels) > 0 else -1
        for lbl in labels:
            if lbl == -1:
                noise_counter += 1
                result.append(f"cluster_{max_label + noise_counter}")
            else:
                result.append(f"cluster_{lbl}")
        return result


def _greedy_cluster(
    embeddings: object, threshold: float = 0.8
) -> list[int]:
    """Simple greedy cosine-similarity clustering (no hdbscan dependency)."""
    import numpy as np

    emb = np.asarray(embeddings)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = emb / norms

    n = len(normed)
    labels = [-1] * n
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cluster_id
        for j in range(i + 1, n):
            if labels[j] != -1:
                continue
            sim = float(np.dot(normed[i], normed[j]))
            if sim >= threshold:
                labels[j] = cluster_id
        cluster_id += 1

    return labels
