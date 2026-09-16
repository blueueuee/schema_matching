"""
Semantic embeddings for column names using Sentence Transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class NameEmbedder:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.embedding_dim = (
            self.model.get_sentence_embedding_dimension()
        )

        print(
            f"✓ Model loaded. "
            f"Embedding dimension: {self.embedding_dim}"
        )

    def embed_tokens(self, tokens):
        """
        Convert normalized tokens into an embedding.
        """

        if not tokens:
            return np.zeros(
                self.embedding_dim,
                dtype=np.float32
            )

        text = " ".join(tokens)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

    def embed_raw_name(self, raw_name):
        """
        Generate embedding directly from raw name.
        """

        embedding = self.model.encode(
            raw_name,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

    def embed_batch(self, token_lists):
        """
        Generate embeddings for multiple token lists.
        """

        texts = [
            " ".join(tokens)
            for tokens in token_lists
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)


if __name__ == "__main__":

    embedder = NameEmbedder()

    tokens = [
        "customer",
        "identifier"
    ]

    embedding = embedder.embed_tokens(tokens)

    print()
    print("Embedding shape:", embedding.shape)
    print("First 10 values:", embedding[:10])