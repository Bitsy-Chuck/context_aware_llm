from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass

from ..models.base_embeddings import BaseEmbeddings
from ..database.db_manager import DatabaseManager
from ..database.vector_store import VectorStore
from ..indexing.document_processor import DocumentProcessor, ProcessedChunk
from ..rag.factory import RAGFactory, RAGType
from ..rag.base_rag import RAGDocument, RAGQuery
from ..rag.vector_rag import VectorRAG


@dataclass
class IndexedFile:
    file_id: str
    file_path: str
    file_type: str
    num_chunks: int
    vector_ids: List[int]  # Changed from List[str] to List[int] to match VectorStore
    embedding_model: str
    indexed_at: datetime
    metadata: Dict


class IndexManager:
    def __init__(
            self,
            embedding_model: BaseEmbeddings,
            db_manager: DatabaseManager,
            vector_store: Optional[VectorStore] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            rag_type: RAGType = RAGType.GRAPH,
            rag_config: Optional[Dict[str, Any]] = None
    ):
        self.embedding_model = embedding_model
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)

        # Initialize RAG factory and create RAG implementation
        self.rag_factory = RAGFactory()
        self.rag_type = rag_type
        self.rag_config = rag_config or {}
        self.rag = None  # Will be initialized in initialize()

    async def initialize(self) -> None:
        """Initialize the RAG implementation."""
        try:
            self.rag = await self.rag_factory.create_rag(
                rag_type=self.rag_type,
                embedding_model=self.embedding_model,
                db_manager=self.db_manager,
                vector_store=self.vector_store,
                config={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    **self.rag_config
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG: {str(e)}")
            raise RuntimeError(f"Failed to initialize RAG: {str(e)}")

    async def index_file(self, file_path: Union[str, Path]) -> str:
        """
        Index a single file.

        Args:
            file_path: Path to the file to be indexed

        Returns:
            str: The file_id of the indexed file

        Raises:
            ValueError: If no content could be extracted from the file
            Exception: If any error occurs during indexing
        """
        if not self.rag:
            await self.initialize()

        try:
            # Process file into chunks using RAG's file processor
            file_id = await self.rag.add_file(file_path)

            # Maintain backward compatibility with IndexedFile structure
            if isinstance(self.rag, VectorRAG):
                # For VectorRAG, maintain the original structure
                doc = await self.rag.get_document(file_id)
                if doc:
                    indexed_file = IndexedFile(
                        file_id=file_id,
                        file_path=str(file_path),
                        file_type=doc.metadata.get("mime_type", "unknown"),
                        num_chunks=1,  # Since we're using whole documents now
                        vector_ids=[int(doc.metadata.get("vector_id", 0))],
                        embedding_model=self.embedding_model.get_model_name(),
                        indexed_at=datetime.now(),
                        metadata=doc.metadata
                    )
                    return indexed_file.file_id

            return file_id

        except Exception as e:
            self.logger.error(f"Error indexing file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to index file: {str(e)}")

    async def search_similar(
            self,
            query: str,
            k: int = 5,
            filter_criteria: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar content using the query.

        Args:
            query: The search query
            k: Number of results to return
            filter_criteria: Optional filters for the search

        Returns:
            List[Dict]: List of search results with metadata and scores
        """
        if not self.rag:
            await self.initialize()

        try:
            # Create RAGQuery
            rag_query = RAGQuery(
                query_text=query,
                filters=filter_criteria,
                top_k=k
            )

            # Perform search
            results = await self.rag.search(rag_query)

            # Format results to maintain backward compatibility
            formatted_results = []
            for doc in results.documents:
                formatted_results.append({
                    "content": doc.content,
                    "metadata": {
                        **doc.metadata,
                        "similarity_score": doc.score if doc.score is not None else 0.0,
                        "vector_id": doc.metadata.get("vector_id", None)
                    }
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")

    async def get_indexed_files_stats(self) -> List[Dict]:
        """
        Get statistics about indexed files.

        Returns:
            List[Dict]: List of indexed files with their statistics
        """
        if not self.rag:
            await self.initialize()

        try:
            # Get basic stats from RAG implementation
            rag_stats = self.rag.get_stats()

            # Get files from database
            files = await self.db_manager.get_indexed_files()

            # Enhance with additional stats
            enhanced_files = []
            for file in files:
                try:
                    file_path = Path(file["file_path"])
                    stats = {
                        **file,
                        "exists": file_path.exists(),
                        "size": file_path.stat().st_size if file_path.exists() else None,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_path.exists() else None,
                        "rag_backend": self.rag.get_backend_type(),
                        "rag_stats": {
                            "embedding_model": self.embedding_model.get_model_name(),
                            "supports_metadata_search": self.rag.supports_metadata_search(),
                            **rag_stats
                        }
                    }
                    enhanced_files.append(stats)
                except Exception as e:
                    self.logger.warning(f"Error getting stats for file {file['file_id']}: {str(e)}")
                    enhanced_files.append(file)

            return enhanced_files

        except Exception as e:
            self.logger.error(f"Error getting indexed files stats: {str(e)}")
            raise RuntimeError(f"Failed to get stats: {str(e)}")