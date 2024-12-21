from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np


class BaseEmbeddings(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name/identifier of the embedding model."""
        pass

    @abstractmethod
    def supports_batch_encoding(self) -> bool:
        """Whether the model supports batch encoding of texts."""
        pass