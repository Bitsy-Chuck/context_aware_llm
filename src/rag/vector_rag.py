from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

from .base_rag import BaseRAG, RAGDocument, RAGQuery, RAGResult
from ..models.base_embeddings import BaseEmbeddings
from ..database.vector_store import VectorStore
from ..database.db_manager import DatabaseManager
from ..indexing.document_processor import DocumentProcessor, ProcessedChunk


class VectorRAG(BaseRAG):
    """Vector-based implementation of RAG using FAISS."""

    def __init__(
            self,
            embedding_model: BaseEmbeddings,
            vector_store: VectorStore,
            db_manager: DatabaseManager,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        """
        Initialize VectorRAG.

        Args:
            embedding_model: Model for generating embeddings
            vector_store: Vector store for similarity search
            db_manager: Database manager for metadata storage
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.db_manager = db_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

    async def add_documents(self, documents: List[RAGDocument]) -> List[str]:
        """Add documents to the vector store."""
        try:
            # Generate embeddings for documents
            texts = [doc.content for doc in documents]
            embeddings = await self.embedding_model.embed_texts(texts)

            # Add to vector store
            metadata = [
                {
                    "text": doc.content,
                    "metadata": {
                        **doc.metadata,
                        "added_at": datetime.now().isoformat(),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap
                    }
                }
                for doc in documents
            ]

            vector_ids = await self.vector_store.add_embeddings(
                embeddings=embeddings,
                texts=[doc.content for doc in documents],
                metadata=metadata
            )

            # Store document metadata in database
            doc_ids = []
            for i, doc in enumerate(documents):
                doc_id = f"doc_{vector_ids[i]}"
                source_file = doc.metadata.get("source_file", "unknown")

                await self.db_manager.save_indexed_file(
                    file_id=doc_id,
                    file_path=source_file,
                    file_type=doc.metadata.get("mime_type", "text/plain"),
                    metadata={
                        "vector_id": vector_ids[i],
                        **doc.metadata
                    },
                    embedding_model=self.embedding_model.get_model_name(),
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                doc_ids.append(doc_id)

            return doc_ids

        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise RuntimeError(f"Failed to add documents: {str(e)}")

    async def search(self, query: RAGQuery) -> RAGResult:
        """Search for relevant documents."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.embed_query(query.query_text)

            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                k=query.top_k,
                filter_criteria=query.filters
            )

            # Convert results to RAGDocuments
            documents = []
            for idx, distance, metadata in results:
                doc = RAGDocument(
                    content=metadata["text"],
                    metadata={
                        **metadata["metadata"],
                        "vector_id": idx,
                        "score": 1.0 / (1.0 + distance)  # Convert distance to similarity score
                    },
                    doc_id=f"doc_{idx}",
                    score=1.0 / (1.0 + distance)
                )
                documents.append(doc)

            return RAGResult(
                documents=documents,
                query=query,
                metadata={
                    "total_vectors_searched": self.vector_store.index.ntotal,
                    "embedding_model": self.embedding_model.get_model_name()
                }
            )

        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the system."""
        try:
            success = True
            for doc_id in doc_ids:
                # Delete from database
                if not await self.db_manager.delete_indexed_file(doc_id):
                    success = False
                    self.logger.warning(f"Failed to delete document {doc_id} from database")

                # Note: Current FAISS implementation doesn't support deletion
                # This is a limitation that would need to be addressed for production use
                self.logger.warning("Document deletion from vector store is not supported")

            return success

        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}")

    async def update_document(self, doc_id: str, document: RAGDocument) -> bool:
        """Update an existing document."""
        try:
            # Since FAISS doesn't support updates, we'll delete and re-add
            await self.delete_documents([doc_id])
            new_ids = await self.add_documents([document])
            return len(new_ids) > 0

        except Exception as e:
            self.logger.error(f"Error updating document: {str(e)}")
            raise RuntimeError(f"Failed to update document: {str(e)}")

    async def get_document(self, doc_id: str) -> Optional[RAGDocument]:
        """Retrieve a specific document."""
        try:
            # Get document metadata from database
            doc_info = await self.db_manager.get_indexed_files()
            for doc in doc_info:
                if doc["file_id"] == doc_id:
                    vector_id = doc["metadata"].get("vector_id")
                    if vector_id is not None and vector_id in self.vector_store.stored_embeddings:
                        stored_doc = self.vector_store.stored_embeddings[vector_id]
                        return RAGDocument(
                            content=stored_doc["text"],
                            metadata={**doc["metadata"], **stored_doc.get("metadata", {})},
                            doc_id=doc_id
                        )
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving document: {str(e)}")
            raise RuntimeError(f"Failed to retrieve document: {str(e)}")

    async def clear(self) -> bool:
        """Clear all documents from the system."""
        try:
            # Reset vector store
            self.vector_store.index = self.vector_store.index.reset()
            self.vector_store.stored_embeddings = {}
            self.vector_store._save_index()

            # Clear database records
            docs = await self.db_manager.get_indexed_files()
            for doc in docs:
                await self.db_manager.delete_indexed_file(doc["file_id"])

            return True

        except Exception as e:
            self.logger.error(f"Error clearing system: {str(e)}")
            raise RuntimeError(f"Failed to clear system: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_vectors": self.vector_store.index.ntotal,
            "dimension": self.vector_store.dimension,
            "embedding_model": self.embedding_model.get_model_name(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "supports_deletion": False,  # FAISS limitation
            "backend_type": self.get_backend_type()
        }

    def supports_metadata_search(self) -> bool:
        """Whether metadata search is supported."""
        return True

    def get_backend_type(self) -> str:
        """Get the type of RAG implementation."""
        return "vector"

    async def save_state(self, path: Path) -> bool:
        """Save the current state."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.vector_store._save_index()
            return True

        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise RuntimeError(f"Failed to save state: {str(e)}")

    async def load_state(self, path: Path) -> bool:
        """Load a previously saved state."""
        try:
            if not path.exists():
                raise FileNotFoundError(f"State directory not found: {path}")

            self.vector_store._load_index()
            return True

        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            raise RuntimeError(f"Failed to load state: {str(e)}")
