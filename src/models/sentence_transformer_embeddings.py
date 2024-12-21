from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer  # Corrected import
# from sentence_transformers import SentenceTransformerEmbeddings  # Corrected import
import torch
from .base_embeddings import BaseEmbeddings


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Implementation of BaseEmbeddings using sentence-transformers."""

    def __init__(
            self,
            model_name: str = "all-mpnet-base-v2",
            device: str = None,
            batch_size: int = 32
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        # Encode all texts in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    async def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        return self.model_name# return self.model.get_model_card()

    def supports_batch_encoding(self) -> bool:
        return True