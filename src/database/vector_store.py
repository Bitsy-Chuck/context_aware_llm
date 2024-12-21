from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
import pickle
import os
import logging
from pathlib import Path


class VectorStore:
    def __init__(self, dimension: int, index_path: str = "data/vector_indices"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.logger = logging.getLogger(__name__)

        # Create index directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.stored_embeddings: Dict[int, Dict] = {}

        # Load existing index if available
        self._load_index()

    def _get_index_file(self) -> Path:
        return self.index_path / "faiss_index.bin"

    def _get_metadata_file(self) -> Path:
        return self.index_path / "metadata.pkl"

    def _load_index(self):
        """Load existing index and metadata if available."""
        index_file = self._get_index_file()
        metadata_file = self._get_metadata_file()

        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, 'rb') as f:
                    self.stored_embeddings = pickle.load(f)
                self.logger.info(f"Loaded {self.index.ntotal} vectors from disk")
            except Exception as e:
                self.logger.error(f"Error loading index: {str(e)}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.stored_embeddings = {}

    def _save_index(self):
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(self._get_index_file()))
            with open(self._get_metadata_file(), 'wb') as f:
                pickle.dump(self.stored_embeddings, f)
            self.logger.info(f"Saved {self.index.ntotal} vectors to disk")
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")

    async def add_embeddings(
            self,
            embeddings: np.ndarray,
            texts: List[str],
            metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add embeddings to the vector store."""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        if metadata and len(metadata) != len(texts):
            raise ValueError("If provided, metadata must match number of texts")

        # Get starting index for new embeddings
        start_idx = self.index.ntotal

        # Add embeddings to FAISS index
        self.index.add(embeddings)

        # Store metadata
        ids = list(range(start_idx, start_idx + len(embeddings)))
        for i, idx in enumerate(ids):
            self.stored_embeddings[idx] = {
                "text": texts[i],
                "metadata": metadata[i] if metadata else None
            }

        # Save to disk
        self._save_index()

        return ids

    async def search(
            self,
            query_embedding: np.ndarray,
            k: int = 5,
            filter_criteria: Optional[Dict] = None
    ) -> List[Tuple[int, float, Dict]]:
        """Search for similar vectors."""
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            metadata = self.stored_embeddings.get(int(idx))
            if metadata is None:
                continue

            # Apply filters if provided
            if filter_criteria and not self._matches_filter(metadata, filter_criteria):
                continue

            results.append((int(idx), float(distance), metadata))

        return results

    def _matches_filter(self, metadata: Dict, filter_criteria: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        if not metadata.get("metadata"):
            return False

        for key, value in filter_criteria.items():
            if metadata["metadata"].get(key) != value:
                return False
        return True