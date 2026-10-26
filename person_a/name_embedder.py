"""Semantic embeddings for normalized column names.

Uses sentence-transformers (all-MiniLM-L6-v2) by default with lazy singleton loading,
per-string embedding cache, and L2 normalization. If offline or unavailable,
explicitly falls back to a character n-gram TF-IDF embedder and logs loudly.
"""

from typing import Dict, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Singleton holder
_GLOBAL_EMBEDDER: Optional["NameEmbedder"] = None


class FallbackTfidfEmbedder:
    """Explicit deterministic fallback embedder using character 3-gram hashing into 384 dimensions."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            clean = f" {text.strip().lower()} "
            if not clean.strip():
                continue
            # 3-gram feature hashing
            for j in range(len(clean) - 2):
                gram = clean[j : j + 3]
                idx = abs(hash(gram)) % self.dim
                vectors[i, idx] += 1.0

            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class NameEmbedder:
    """Embedding generator with lazy loading, L2-normalization, caching, and fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_dim = 384
        self._cache: Dict[str, np.ndarray] = {}
        self.backend_name = "sentence-transformers"
        self._model = None
        self._fallback_embedder: Optional[FallbackTfidfEmbedder] = None

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Initializing SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self.embedding_dim = int(self._model.get_embedding_dimension())
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
            self.backend_name = f"sentence-transformers:{self.model_name}"
            logger.info("SentenceTransformer loaded. Dim=%d", self.embedding_dim)
        except Exception as exc:
            logger.warning(
                "CRITICAL: sentence-transformers failed to load (%s). "
                "Activating FALLBACK char-ngram TF-IDF embedder. Embeddings will not be neural.",
                exc,
            )
            self._model = None
            self._fallback_embedder = FallbackTfidfEmbedder(dim=self.embedding_dim)
            self.backend_name = "fallback:tfidf-char-ngram"

    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            norm = float(np.linalg.norm(arr))
            return (arr / norm) if norm > 0 else arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    def embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Convert a list of normalized tokens to a single (384,) unit vector."""
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        text = " ".join(tokens).strip()
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if text in self._cache:
            return self._cache[text].copy()

        if self._model is not None:
            raw = self._model.encode(text, convert_to_numpy=True)
        else:
            raw = self._fallback_embedder.encode([text])[0]

        vec = self._l2_normalize(np.asarray(raw, dtype=np.float32))
        self._cache[text] = vec
        return vec.copy()

    def embed_raw_name(self, raw_name: str) -> np.ndarray:
        """Embed a raw column name directly."""
        tokens = [t for t in str(raw_name).strip().split() if t]
        return self.embed_tokens(tokens)

    def embed_batch(self, token_lists: List[List[str]]) -> np.ndarray:
        """Batch encode a list of token lists into an (n, 384) unit matrix."""
        if not token_lists:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        texts = [" ".join(tokens).strip() for tokens in token_lists]
        n = len(texts)
        result = np.zeros((n, self.embedding_dim), dtype=np.float32)

        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for i, text in enumerate(texts):
            if not text:
                continue
            if text in self._cache:
                result[i] = self._cache[text]
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            if self._model is not None:
                encoded = self._model.encode(missing_texts, convert_to_numpy=True)
            else:
                encoded = self._fallback_embedder.encode(missing_texts)

            normalized = self._l2_normalize(np.asarray(encoded, dtype=np.float32))
            for idx, text, vec in zip(missing_indices, missing_texts, normalized):
                self._cache[text] = vec
                result[idx] = vec

        return result


def get_name_embedder(model_name: str = "all-MiniLM-L6-v2") -> NameEmbedder:
    """Return the lazy singleton instance of NameEmbedder."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None or _GLOBAL_EMBEDDER.model_name != model_name:
        _GLOBAL_EMBEDDER = NameEmbedder(model_name=model_name)
    return _GLOBAL_EMBEDDER